"""Screen-space ambient occlusion — derived from Z + N, no model.

Post-processor that reads the per-frame `Z` + `N.x/N.y/N.z` channels that
upstream depth + normals passes already produced, and writes a single
`ao` channel where 1.0 = fully lit and 0.0 = fully occluded. Runs on
every frame that carries the required inputs; frames missing either Z
or any of the three normal channels pass through untouched (safe for
mixed pipelines where one pass failed on a frame).

Algorithm (vectorised numpy hemisphere SSAO):

    For each pixel p with surface normal n:
      1. Build a tangent frame (T, B, N=n) from n plus a world-up fallback.
      2. For each of `samples` deterministic hemisphere offsets, rotate
         into camera space via the TBN basis.
      3. Screen offset: `(u, v) += offset.xy * radius_px`, where
         `radius_px = radius * image_width / 2`. This is a deliberate
         approximation — we don't thread camera intrinsics into the post
         stage yet, and width-relative pixel offset is a good-enough
         scale for tunable SSAO. Intrinsic-aware AO lands with the
         CameraPass metadata plumb.
      4. Expected depth at sample: `Z[p] + offset.z * radius` (in the
         same units as Z, normalized or metric).
      5. Sample Z at `(u', v')` via nearest-neighbour clamp.
      6. Occluded if `actual_Z > expected_Z + bias` — i.e. geometry is
         nearer than the hemisphere sample point in our "Z=1 is near"
         convention.
    AO = clip(1 - (occluded_fraction * intensity), 0, 1).

Parameters mirror the spec:
    radius    (0.5)   : world-space sample radius (Z units — normalized
                       for relative-depth backends, metres for Depth Pro)
    samples   (16)    : hemisphere sample count
    bias      (0.025) : minimum depth delta to count as occlusion
    intensity (1.0)   : output strength (>1 crushes shadows harder)

Metadata stamped by the executor:
    liveActionAOV/ao/derived_from = "depth+normals"
    liveActionAOV/ao/algorithm    = "ssao_hemisphere_v1"
    liveActionAOV/ao/radius       = <float>
    liveActionAOV/ao/samples      = <int>
    liveActionAOV/ao/bias         = <float>
    liveActionAOV/ao/intensity    = <float>
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z, CH_Z

_AO_CHANNEL = "ao"


class SSAOPost:
    """Pure-numpy hemisphere SSAO. Reads Z + N, writes `ao`."""

    name = "ssao"
    algorithm = "ssao_hemisphere_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "radius": 0.5,
        "samples": 16,
        "bias": 0.025,
        "intensity": 1.0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)

    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: Any,  # unused — signature matches TemporalSmoother so the executor calls uniformly
        shot_id: str,  # unused
        **_kwargs: Any,
    ) -> dict[int, dict[str, np.ndarray]]:
        del flow_cache, shot_id
        radius = float(self.params["radius"])
        samples = int(self.params["samples"])
        bias = float(self.params["bias"])
        intensity = float(self.params["intensity"])

        if samples < 1:
            return per_frame_channels

        # Fibonacci hemisphere + seeded radius jitter. Seed is fixed so
        # the sample pattern is stable across frames — otherwise AO
        # would flicker temporally even on identical surfaces.
        offsets = _hemisphere_samples(samples, seed=42)

        for channels in per_frame_channels.values():
            if CH_Z not in channels:
                continue
            if not all(c in channels for c in (CH_N_X, CH_N_Y, CH_N_Z)):
                continue
            Z = channels[CH_Z].astype(np.float32, copy=False)
            N = np.stack(
                [
                    channels[CH_N_X].astype(np.float32, copy=False),
                    channels[CH_N_Y].astype(np.float32, copy=False),
                    channels[CH_N_Z].astype(np.float32, copy=False),
                ],
                axis=-1,
            )
            channels[_AO_CHANNEL] = _compute_ssao(Z, N, offsets, radius, bias, intensity)

        return per_frame_channels


def _hemisphere_samples(n: int, *, seed: int) -> np.ndarray:
    """Return (n, 3) hemisphere offsets on the unit ball with z ≥ 0.

    Fibonacci spiral places the points evenly, and a seeded radius
    jitter keeps the sampling from forming visible concentric rings in
    the AO output.
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    idx = np.arange(n)
    z = (idx + 0.5) / n  # [0, 1) — +Z hemisphere
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = phi * idx
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    base = np.stack([x, y, z], axis=-1).astype(np.float32)
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.1, 1.0, size=(n, 1)).astype(np.float32)
    return base * scale


def _tangent_basis(N: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build per-pixel (T, B) so that (T, B, N) is orthonormal RH.

    Picks world +Y as the auxiliary axis, swapping to world +X on
    pixels where the normal is nearly ±Y (avoids `cross(n, up) ≈ 0`).
    """
    up_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    up_x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    near_y = np.abs(N[..., 1]) > 0.99  # (H, W)
    up = np.where(near_y[..., None], up_x, up_y)  # (H, W, 3)
    T = np.cross(up, N)
    T /= np.linalg.norm(T, axis=-1, keepdims=True) + 1e-9
    B = np.cross(N, T)
    return T, B


def _compute_ssao(
    Z: np.ndarray,
    N: np.ndarray,
    offsets: np.ndarray,
    radius: float,
    bias: float,
    intensity: float,
) -> np.ndarray:
    """Vectorised SSAO inner loop. See module docstring."""
    H, W = Z.shape
    radius_px = max(1.0, radius * W * 0.5)

    T, B = _tangent_basis(N)
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))  # (H, W)

    occluded_sum = np.zeros((H, W), dtype=np.float32)
    for off in offsets:
        # offset_world = T*off.x + B*off.y + N*off.z, per-pixel rotation.
        offset_world = T * off[0] + B * off[1] + N * off[2]
        sample_u = np.clip(
            np.round(u_grid + offset_world[..., 0] * radius_px).astype(np.int32),
            0,
            W - 1,
        )
        sample_v = np.clip(
            np.round(v_grid + offset_world[..., 1] * radius_px).astype(np.int32),
            0,
            H - 1,
        )
        expected_z = Z + offset_world[..., 2] * radius
        actual_z = Z[sample_v, sample_u]
        occluded_sum += (actual_z > expected_z + bias).astype(np.float32)

    occluded_frac = occluded_sum / float(len(offsets))
    ao = 1.0 - occluded_frac * intensity
    return np.clip(ao, 0.0, 1.0).astype(np.float32)


__all__ = ["SSAOPost"]
