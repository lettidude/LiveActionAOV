"""Unit tests for the SSAO post-processor.

Fast numeric tests — no plate I/O, no torch. We build small synthetic
(Z, N) fields where the expected AO behaviour is obvious by inspection:

- Flat plane facing the camera should AO ≈ 1 everywhere (no occlusion).
- A sharp depth step (wall) should darken the base of the wall.
- `intensity=0` should produce AO = 1 everywhere regardless of input.
- Missing channels should pass through untouched.
- Output should be deterministic frame-to-frame (stable sample pattern).
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import CH_AO, CH_N_X, CH_N_Y, CH_N_Z, CH_Z
from live_action_aov.post.ssao import SSAOPost


def _flat_frame(h: int = 16, w: int = 16, z: float = 0.5) -> dict[str, np.ndarray]:
    """A frame with constant depth and normals pointing at camera (+Z)."""
    return {
        CH_Z: np.full((h, w), z, dtype=np.float32),
        CH_N_X: np.zeros((h, w), dtype=np.float32),
        CH_N_Y: np.zeros((h, w), dtype=np.float32),
        CH_N_Z: np.ones((h, w), dtype=np.float32),
    }


def test_output_channel_named_ao_and_matches_z_shape():
    post = SSAOPost({"radius": 0.1, "samples": 8})
    frames = {0: _flat_frame(12, 20)}
    out = post.apply(frames, flow_cache=None, shot_id="s")
    assert CH_AO in out[0]
    assert out[0][CH_AO].shape == out[0][CH_Z].shape
    assert out[0][CH_AO].dtype == np.float32


def test_output_is_in_unit_range():
    post = SSAOPost({"radius": 0.4, "samples": 16, "intensity": 2.0})
    frames = {0: _flat_frame(16, 16)}
    out = post.apply(frames, flow_cache=None, shot_id="s")
    ao = out[0][CH_AO]
    assert float(ao.min()) >= 0.0
    assert float(ao.max()) <= 1.0


def test_flat_surface_facing_camera_is_unoccluded():
    # Constant-depth flat plane + normals pointing at camera: samples
    # above the surface never find closer geometry, so AO ≈ 1 everywhere.
    post = SSAOPost({"radius": 0.3, "samples": 16, "bias": 0.0})
    frames = {0: _flat_frame(32, 32)}
    out = post.apply(frames, flow_cache=None, shot_id="s")
    ao = out[0][CH_AO]
    # Exactly 1.0 in the interior; edge pixels can clamp to the same Z
    # as the sampled neighbour, which with `bias=0` produces borderline
    # equal-to-expected reads. Keep the assertion generous.
    assert float(ao.mean()) > 0.95


def test_intensity_zero_leaves_ao_at_one():
    # Any geometry — even a sharp step — should still produce AO=1
    # when the intensity knob zeroes out the darkening.
    h, w = 16, 16
    Z = np.full((h, w), 0.5, dtype=np.float32)
    Z[:, w // 2 :] = 0.9  # step
    frame = {
        CH_Z: Z,
        CH_N_X: np.zeros((h, w), dtype=np.float32),
        CH_N_Y: np.zeros((h, w), dtype=np.float32),
        CH_N_Z: np.ones((h, w), dtype=np.float32),
    }
    out = SSAOPost({"intensity": 0.0}).apply({0: frame}, flow_cache=None, shot_id="s")
    assert np.allclose(out[0][CH_AO], 1.0)


def test_depth_step_produces_occlusion_near_the_wall():
    # Left half is far (Z=0.2), right half is near (Z=0.8). In our
    # convention Z=1 means near-camera, so the right half is a wall
    # sticking out toward the camera. Pixels on the LEFT side of the
    # wall (i.e. left-half pixels near the seam) should see occlusion
    # from the near wall and therefore darken.
    h, w = 24, 48
    Z = np.full((h, w), 0.2, dtype=np.float32)
    Z[:, w // 2 :] = 0.8
    frame = {
        CH_Z: Z,
        CH_N_X: np.zeros((h, w), dtype=np.float32),
        CH_N_Y: np.zeros((h, w), dtype=np.float32),
        CH_N_Z: np.ones((h, w), dtype=np.float32),
    }
    post = SSAOPost({"radius": 0.3, "samples": 32, "bias": 0.005, "intensity": 1.0})
    ao = post.apply({0: frame}, flow_cache=None, shot_id="s")[0][CH_AO]

    # Left-side pixels near the seam should be meaningfully darker than
    # left-side pixels far from the seam.
    near_seam = ao[:, w // 2 - 4 : w // 2]
    far_from_seam = ao[:, :4]
    assert float(near_seam.mean()) < float(far_from_seam.mean()) - 0.05


def test_missing_z_skips_frame_without_adding_ao():
    frame = {
        # no CH_Z
        CH_N_X: np.zeros((8, 8), dtype=np.float32),
        CH_N_Y: np.zeros((8, 8), dtype=np.float32),
        CH_N_Z: np.ones((8, 8), dtype=np.float32),
    }
    out = SSAOPost({}).apply({0: frame}, flow_cache=None, shot_id="s")
    assert CH_AO not in out[0]


def test_missing_one_normal_component_skips_frame():
    frame = {
        CH_Z: np.full((8, 8), 0.5, dtype=np.float32),
        CH_N_X: np.zeros((8, 8), dtype=np.float32),
        # no N_Y
        CH_N_Z: np.ones((8, 8), dtype=np.float32),
    }
    out = SSAOPost({}).apply({0: frame}, flow_cache=None, shot_id="s")
    assert CH_AO not in out[0]


def test_deterministic_across_frames():
    # Same inputs on two different frames should produce identical AO
    # — the hemisphere sampling is seeded so nothing flickers from frame
    # to frame when the geometry is constant.
    frames = {
        0: _flat_frame(16, 16, z=0.5),
        1: _flat_frame(16, 16, z=0.5),
    }
    out = SSAOPost({"samples": 16}).apply(frames, flow_cache=None, shot_id="s")
    assert np.array_equal(out[0][CH_AO], out[1][CH_AO])


def test_defaults_are_spec_compliant():
    # These are the defaults the user specified in the PR brief —
    # changing them silently could confuse artists tuning the pass.
    assert SSAOPost.DEFAULT_PARAMS == {
        "radius": 0.5,
        "samples": 16,
        "bias": 0.025,
        "intensity": 1.0,
    }
    assert SSAOPost.name == "ssao"
    assert SSAOPost.algorithm == "ssao_hemisphere_v1"


def test_zero_samples_is_noop():
    # Edge case: `samples=0` should just not run — not crash on a
    # divide-by-zero when computing the occluded fraction.
    frames = {0: _flat_frame(8, 8)}
    out = SSAOPost({"samples": 0}).apply(frames, flow_cache=None, shot_id="s")
    assert CH_AO not in out[0]


def test_apply_returns_same_dict_identity_for_in_place_mutation():
    # The post mutates channels in place (matches TemporalSmoother's
    # contract where the executor assigns the returned dict back). This
    # saves a full H×W copy per frame on large plates.
    frames = {0: _flat_frame(8, 8)}
    out = SSAOPost({}).apply(frames, flow_cache=None, shot_id="s")
    assert out is frames
