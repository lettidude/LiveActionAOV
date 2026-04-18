"""Pass-type-aware resize: flow scaling, normal renormalization, nearest for
discrete, intrinsics scaling."""

from __future__ import annotations

import numpy as np

from live_action_aov.io.resize import (
    scale_intrinsics,
    upscale,
)


def test_flow_vectors_scale_with_upscale_ratio() -> None:
    # 2-channel (u, v) flow at half resolution; upscale by 2 should double
    # each vector's magnitude (trap 1).
    flow = np.ones((4, 4, 2), dtype=np.float32)
    out = upscale(flow, target=(8, 8), channel_type="flow_vector")
    assert out.shape == (8, 8, 2)
    assert np.allclose(out[:, :, 0], 2.0)  # x scaled
    assert np.allclose(out[:, :, 1], 2.0)  # y scaled


def test_normal_vectors_renormalized_to_unit_length() -> None:
    # Non-unit input; after resize the output must be unit length per pixel
    # (trap 2).
    n = np.tile(np.array([0.8, 0.0, 0.6], dtype=np.float32), (4, 4, 1))
    # Scale the length so the renormalize has something to do.
    n *= 2.5
    out = upscale(n, target=(8, 8), channel_type="normal_vector")
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_nearest_upscale_preserves_discrete_values() -> None:
    # Distinct "class IDs" — must not be blended (trap 6).
    ids = np.array([[0, 1], [2, 3]], dtype=np.float32)
    out = upscale(ids, target=(4, 4), channel_type="discrete")
    assert set(np.unique(out).tolist()) == {0.0, 1.0, 2.0, 3.0}


def test_intrinsics_scale_with_resolution() -> None:
    k = {"fx": 2000.0, "fy": 2000.0, "cx": 1920.0, "cy": 1080.0}
    scaled = scale_intrinsics(k, from_res=(3840, 2160), to_res=(960, 540))
    assert scaled["fx"] == 500.0
    assert scaled["fy"] == 500.0
    assert scaled["cx"] == 480.0
    assert scaled["cy"] == 270.0
