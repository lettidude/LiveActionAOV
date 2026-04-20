"""Position-from-depth post-processor (design §10.4 — auto-derived P pass).

When any pass in the job produces a depth channel (`Z`), this post-processor
runs automatically after the pass loop and writes three new channels into
every frame:

    P.x = (u - cx) / fx * Z
    P.y = (v - cy) / fy * Z
    P.z = Z

These are camera-space position coordinates in the same units as Z (metres
when Depth Pro runs, "relative" when DA-V2 / Video-Depth-Anything runs
with per-clip normalisation). Any Nuke relight gizmo or EnvRelight node
needs a position pass — deriving it from Z + intrinsics is cheap and
lets us ship P without extra model infrastructure.

Intrinsics source priority (first hit wins):

1. Explicit `fx` / `fy` / `cx` / `cy` params passed to the post-processor
   (the GUI's per-shot intrinsics dropdown will populate these).
2. `Shot.fx` / `Shot.fy` / `Shot.cx` / `Shot.cy` if the shot carries them.
3. Camera-track solver output (Phase 2a — not yet implemented; will feed
   through the same shot attributes).
4. Approximate intrinsics derived from plate resolution assuming
   `horizontal_fov_deg` (default 60°, a reasonable wide-normal guess for
   stock VFX footage). `cx = W/2`, `cy = H/2`.

Whatever source supplied the intrinsics is stamped into the sidecar
metadata (`utility/position/intrinsics_source`) so downstream tools
can tell an approximate P from a solved one.

Output range note
-----------------

When Z is per-clip-normalised to [0, 1] (DA-V2, VDA default), P values are
relative — not metres. That's still useful for gizmos that need a position
field shape (parallax-aware effects, lens flares, defocus falloff). When
Z is metric (Depth Pro, or scaled from camera-track solve), P is in metres
and can drive real-world relighting.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from live_action_aov.io.channels import CH_P_X, CH_P_Y, CH_P_Z, CH_Z
from live_action_aov.shared.optical_flow.cache import FlowCache


class PositionFromDepth:
    """Post-processor: derive P.x/P.y/P.z from Z + intrinsics."""

    name = "position_from_depth"
    algorithm = "pinhole_unproject_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        # Explicit intrinsics — if any are None, fall back to shot attrs,
        # else to approximate-from-HFOV.
        "fx": None,
        "fy": None,
        "cx": None,
        "cy": None,
        # Fallback when no explicit intrinsics and no shot attrs.
        "horizontal_fov_deg": 60.0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        # Stamped by `apply` so the executor can write it into metadata.
        self.intrinsics_source: str = "unknown"
        self.resolved_intrinsics: tuple[float, float, float, float] | None = None

    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: FlowCache,
        shot_id: str,
        *,
        shot: Any = None,
    ) -> dict[int, dict[str, np.ndarray]]:
        """For every frame that has a Z channel, add P.x / P.y / P.z.

        Frames without Z are left alone. Existing P channels are
        overwritten — the rationale: if a future camera-track solver
        emits P directly, it'll register under a different post-processor
        that runs *after* this one, so the overwrite order is
        approximate → solved, never the other way around.

        `flow_cache` and `shot_id` are unused but kept for interface
        parity with `TemporalSmoother`. `shot` is optional and used to
        read pre-existing intrinsics when the params don't override.
        """
        out: dict[int, dict[str, np.ndarray]] = {
            f: dict(channels) for f, channels in per_frame_channels.items()
        }
        any_z = next((c for c in out.values() if CH_Z in c), None)
        if any_z is None:
            return out

        h, w = any_z[CH_Z].shape
        fx, fy, cx, cy = self._resolve_intrinsics(w, h, shot)
        self.resolved_intrinsics = (fx, fy, cx, cy)

        # Build pixel-grid once — same for every frame since plate res is
        # uniform within a shot.
        u = np.arange(w, dtype=np.float32)[None, :]  # (1, W)
        v = np.arange(h, dtype=np.float32)[:, None]  # (H, 1)
        inv_fx = np.float32(1.0 / fx)
        inv_fy = np.float32(1.0 / fy)
        u_norm = (u - np.float32(cx)) * inv_fx  # (1, W)
        v_norm = (v - np.float32(cy)) * inv_fy  # (H, 1)

        for _f, channels in out.items():
            if CH_Z not in channels:
                continue
            z = channels[CH_Z].astype(np.float32, copy=False)
            if z.shape != (h, w):
                # Mixed-resolution clip — rare, but the pixel grid we built
                # above won't match. Rebuild on the fly for safety.
                fh, fw = z.shape
                fu = np.arange(fw, dtype=np.float32)[None, :]
                fv = np.arange(fh, dtype=np.float32)[:, None]
                fu_norm = (fu - np.float32(cx)) * inv_fx
                fv_norm = (fv - np.float32(cy)) * inv_fy
                px = (fu_norm * z).astype(np.float32, copy=False)
                py = (fv_norm * z).astype(np.float32, copy=False)
            else:
                px = (u_norm * z).astype(np.float32, copy=False)
                py = (v_norm * z).astype(np.float32, copy=False)
            channels[CH_P_X] = px
            channels[CH_P_Y] = py
            channels[CH_P_Z] = z.astype(np.float32, copy=False)

        return out

    def _resolve_intrinsics(
        self, width: int, height: int, shot: Any
    ) -> tuple[float, float, float, float]:
        """Resolve (fx, fy, cx, cy) from params / shot / approximate."""
        p_fx = self.params.get("fx")
        p_fy = self.params.get("fy")
        p_cx = self.params.get("cx")
        p_cy = self.params.get("cy")
        if p_fx is not None and p_fy is not None and p_cx is not None and p_cy is not None:
            self.intrinsics_source = "params"
            return float(p_fx), float(p_fy), float(p_cx), float(p_cy)

        s_fx = getattr(shot, "fx", None) if shot is not None else None
        s_fy = getattr(shot, "fy", None) if shot is not None else None
        s_cx = getattr(shot, "cx", None) if shot is not None else None
        s_cy = getattr(shot, "cy", None) if shot is not None else None
        if s_fx is not None and s_fy is not None and s_cx is not None and s_cy is not None:
            self.intrinsics_source = "shot_metadata"
            return float(s_fx), float(s_fy), float(s_cx), float(s_cy)

        # Approximate — cx/cy = principal point at image centre, fx = fy
        # from horizontal FOV.
        hfov_deg = float(self.params.get("horizontal_fov_deg", 60.0))
        hfov_rad = math.radians(hfov_deg)
        fx = float(width) / (2.0 * math.tan(hfov_rad / 2.0))
        fy = fx  # square pixels, matches our display/no-squeeze assumption
        cx = width / 2.0
        cy = height / 2.0
        self.intrinsics_source = "approximate_from_hfov"
        return fx, fy, cx, cy


__all__ = ["PositionFromDepth"]
