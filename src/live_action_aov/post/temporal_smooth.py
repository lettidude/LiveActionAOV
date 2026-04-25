# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Flow-guided temporal smoothing (design §9.1).

Post-processor — NOT a pass. Runs after all passes complete, mutating the
per-frame channel dict in place with a one-sided EMA:

    smoothed[t] = alpha * warp(channel[t-1], backward_flow[t]) + (1 - alpha) * channel[t]

For pixels where forward/backward flow disagrees (F-B consistency > threshold
px) the warp is rejected and the raw value is kept. This closes ~70% of the
per-frame flicker seen in DA V2 / DSINE without full video models (design §9).

Design decisions:
- v1 is one-sided (only looks at t-1). v2+ blends both neighbours.
- Works on any channel name declared in `applied_to`. A channel absent at
  frame t-1 falls through untouched (covers clip endpoints).
- Requires both `forward_flow` at t-1 and `backward_flow` at t from the
  flow cache. Frames missing either stay unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z
from live_action_aov.shared.optical_flow.cache import FlowCache


class TemporalSmoother:
    """Flow-guided EMA post-processor (design §9.1)."""

    name = "temporal_smooth"
    algorithm = "flow_guided_ema_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "applied_to": [],  # list of channel names to smooth
        "alpha": 0.4,
        "fb_threshold_px": 1.0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: FlowCache,
        shot_id: str,
        **_kwargs: Any,
    ) -> dict[int, dict[str, np.ndarray]]:
        """Return a new per-frame channel dict with the selected channels smoothed.

        The input dict is not mutated; callers can choose to replace it with
        the return value.
        """
        applied_to: list[str] = list(self.params.get("applied_to") or [])
        if not applied_to:
            return per_frame_channels
        alpha = float(self.params["alpha"])
        threshold = float(self.params["fb_threshold_px"])

        out: dict[int, dict[str, np.ndarray]] = {
            f: dict(channels) for f, channels in per_frame_channels.items()
        }
        frames_sorted = sorted(out.keys())
        for i, f in enumerate(frames_sorted):
            if i == 0:
                continue  # first frame — nothing to warp from
            prev = frames_sorted[i - 1]
            bwd = flow_cache.get(shot_id, f, "backward")  # f → f-1
            fwd_prev = flow_cache.get(shot_id, prev, "forward")  # f-1 → f
            if bwd is None or fwd_prev is None:
                continue
            occlusion = _fb_occlusion(fwd_prev, bwd, threshold_px=threshold)
            # occlusion: (H, W) float in [0, 1] — 1 means occluded (reject warp).
            for ch_name in applied_to:
                if ch_name not in out[prev] or ch_name not in out[f]:
                    continue
                prev_ch = out[prev][ch_name]
                cur_ch = out[f][ch_name]
                if prev_ch.shape != cur_ch.shape:
                    continue
                warped = _warp_backward(prev_ch, bwd)
                weight = (1.0 - occlusion) * alpha  # per-pixel smoothing weight
                out[f][ch_name] = (weight * warped + (1.0 - weight) * cur_ch).astype(
                    np.float32, copy=False
                )
            # If we just blended N.x/N.y/N.z, renormalize the triplet. Bilinear
            # blending of unit-length normals breaks the unit constraint, and
            # Nuke's Relight node assumes |N| = 1 per pixel. Only touch frames
            # where all three were actually smoothed.
            _renormalize_normal_triplet_if_present(out[f], applied_to)
        return out


# ----------------------------------------------------------------------
# Helpers: warp + occlusion (torch-backed for grid_sample convenience)
# ----------------------------------------------------------------------


def _warp_backward(frame_prev: np.ndarray, bwd_at_cur: np.ndarray) -> np.ndarray:
    """Sample `frame_prev` at (p + bwd_at_cur[p]) per pixel p.

    `frame_prev`: (H, W) float32 value from frame t-1
    `bwd_at_cur`:  (2, H, W) float32 = flow from frame t → t-1 (pixels)
    Returns: (H, W) float32 warped into the frame-t grid.
    """
    import torch

    assert frame_prev.ndim == 2 and bwd_at_cur.ndim == 3 and bwd_at_cur.shape[0] == 2
    h, w = frame_prev.shape
    fp = torch.from_numpy(frame_prev.astype(np.float32, copy=False))[None, None]
    b = torch.from_numpy(bwd_at_cur.astype(np.float32, copy=False))
    y_coords = torch.arange(h, dtype=torch.float32)
    x_coords = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    src_x = xx + b[0]
    src_y = yy + b[1]
    grid = torch.stack([src_x / max(w - 1, 1) * 2 - 1, src_y / max(h - 1, 1) * 2 - 1], dim=-1)[None]
    warped = torch.nn.functional.grid_sample(
        fp, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped[0, 0].numpy().astype(np.float32, copy=False)


def _fb_occlusion(fwd_prev: np.ndarray, bwd_cur: np.ndarray, threshold_px: float) -> np.ndarray:
    """Compute per-pixel occlusion [0,1] at frame t.

    Given `fwd_prev` (flow t-1 → t) and `bwd_cur` (flow t → t-1), a pixel p
    at frame t is consistent iff bwd_cur[p] ≈ -fwd_prev[p - bwd_cur[p]]. We
    return 1 where the round-trip residual exceeds `threshold_px`, 0 elsewhere
    (with a soft transition around the threshold).
    """
    import torch

    assert fwd_prev.shape == bwd_cur.shape and fwd_prev.ndim == 3 and fwd_prev.shape[0] == 2
    _, h, w = fwd_prev.shape
    fp = torch.from_numpy(fwd_prev.astype(np.float32, copy=False))[None]
    bc = torch.from_numpy(bwd_cur.astype(np.float32, copy=False))[None]
    y_coords = torch.arange(h, dtype=torch.float32)
    x_coords = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    # Sample fwd_prev at the predicted source location (p + bwd_cur[p]).
    src_x = xx + bc[0, 0]
    src_y = yy + bc[0, 1]
    grid = torch.stack([src_x / max(w - 1, 1) * 2 - 1, src_y / max(h - 1, 1) * 2 - 1], dim=-1)[None]
    fwd_at_src = torch.nn.functional.grid_sample(
        fp, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    # Consistent iff bc[t][p] + fp[t-1][p + bc[t][p]] ≈ 0.
    residual = bc + fwd_at_src
    err = torch.sqrt(residual[0, 0] ** 2 + residual[0, 1] ** 2)
    # Soft: 0 at err=0, → 1 at err=threshold (and beyond).
    occlusion = torch.clamp(err / max(threshold_px, 1e-6), 0.0, 1.0)
    return occlusion.numpy().astype(np.float32, copy=False)


_NORMAL_TRIPLET = (CH_N_X, CH_N_Y, CH_N_Z)


def _renormalize_normal_triplet_if_present(
    frame_channels: dict[str, np.ndarray], applied_to: list[str]
) -> None:
    """If N.x/N.y/N.z are all in `applied_to` and present in the frame output,
    rescale them so each pixel's (Nx, Ny, Nz) has unit length.

    Mutates `frame_channels` in place.
    """
    if not all(c in applied_to and c in frame_channels for c in _NORMAL_TRIPLET):
        return
    nx = frame_channels[CH_N_X]
    ny = frame_channels[CH_N_Y]
    nz = frame_channels[CH_N_Z]
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    mag = np.maximum(mag, 1e-6)
    frame_channels[CH_N_X] = (nx / mag).astype(np.float32, copy=False)
    frame_channels[CH_N_Y] = (ny / mag).astype(np.float32, copy=False)
    frame_channels[CH_N_Z] = (nz / mag).astype(np.float32, copy=False)


__all__ = ["TemporalSmoother"]
