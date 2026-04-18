"""RAFT optical-flow pass — end-to-end smoke test.

Runs RAFT (torchvision) on a tiny synthetic pair and asserts:
  1. All five flow channels are present with plate-resolution shape
  2. Flow magnitudes are non-zero on a visibly-moving subject
  3. Forward-backward confidence is bounded in [0, 1]
  4. Artifacts include forward/backward flow and a parallax scalar

CPU-safe: uses `raft_small`, inference_resolution=128, num_flow_updates=6.
Marked `slow` so CI can run everything else quickly.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from live_action_aov.io.channels import (  # noqa: E402
    CH_BACK_X,
    CH_BACK_Y,
    CH_BACKWARD_U,
    CH_BACKWARD_V,
    CH_FLOW_CONFIDENCE,
    CH_FORWARD_U,
    CH_FORWARD_V,
    CH_MOTION_X,
    CH_MOTION_Y,
)
from live_action_aov.passes.flow.raft import RAFTPass  # noqa: E402


pytestmark = pytest.mark.slow


def _make_pair(h: int = 64, w: int = 96, shift_x: int = 8) -> np.ndarray:
    """Build a 2-frame pair with a bright square translated by `shift_x` px."""
    rng = np.random.default_rng(42)
    bg = rng.uniform(0.15, 0.25, size=(h, w, 3)).astype(np.float32)
    a = bg.copy()
    b = bg.copy()
    y0, y1 = h // 3, 2 * h // 3
    x0, x1 = w // 4, w // 4 + w // 6
    a[y0:y1, x0:x1, :] = 0.9
    b[y0:y1, x0 + shift_x : x1 + shift_x, :] = 0.9
    return np.stack([a, b], axis=0)


def test_raft_forward_motion_is_nonzero_on_moving_subject() -> None:
    pair = _make_pair()
    pass_ = RAFTPass(
        {
            "backend": "raft_small",
            "inference_resolution": 128,
            "num_flow_updates": 6,
            "fb_threshold_px": 2.0,
        }
    )
    model_in = pass_.preprocess(pair)
    model_out = pass_.infer(model_in)
    channels = pass_.postprocess(model_out)

    # All declared channels must be present at plate resolution — both the
    # spec-mandated names and the Nuke-native aliases.
    plate_h, plate_w = pair.shape[1], pair.shape[2]
    expected = (
        CH_MOTION_X, CH_MOTION_Y, CH_BACK_X, CH_BACK_Y, CH_FLOW_CONFIDENCE,
        CH_FORWARD_U, CH_FORWARD_V, CH_BACKWARD_U, CH_BACKWARD_V,
    )
    for ch in expected:
        assert ch in channels, f"missing {ch}"
        assert channels[ch].shape == (plate_h, plate_w), f"{ch} wrong shape"
        assert channels[ch].dtype == np.float32

    # Aliases must exactly equal their canonical counterparts.
    assert np.array_equal(channels[CH_FORWARD_U], channels[CH_MOTION_X])
    assert np.array_equal(channels[CH_FORWARD_V], channels[CH_MOTION_Y])
    assert np.array_equal(channels[CH_BACKWARD_U], channels[CH_BACK_X])
    assert np.array_equal(channels[CH_BACKWARD_V], channels[CH_BACK_Y])

    # Forward motion on the square: we shifted +8 px in x, so motion.x should
    # be positive on average inside the moving region.
    motion_x = channels[CH_MOTION_X]
    subject = motion_x[plate_h // 3 : 2 * plate_h // 3, plate_w // 4 : plate_w // 2]
    assert subject.mean() > 1.0, f"expected >1 px forward motion, got {subject.mean()}"

    # Confidence is bounded in [0, 1].
    conf = channels[CH_FLOW_CONFIDENCE]
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_raft_run_shot_produces_per_frame_channels_and_artifacts() -> None:
    """run_shot drives pair iteration + artifact emission."""

    class _FakeReader:
        def __init__(self, frames: list[np.ndarray]) -> None:
            self._frames = frames
            self._first = 1

        def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
            return self._frames[idx - self._first], {}

    # 3-frame sequence: square shifts +6 px between each pair.
    pair = _make_pair()
    f1 = pair[0]
    f2 = pair[1]
    rng = np.random.default_rng(42)
    bg = rng.uniform(0.15, 0.25, size=f1.shape).astype(np.float32)
    f3 = bg.copy()
    y0, y1 = f1.shape[0] // 3, 2 * f1.shape[0] // 3
    x0 = f1.shape[1] // 4 + 16
    x1 = x0 + f1.shape[1] // 6
    f3[y0:y1, x0:x1, :] = 0.9

    reader = _FakeReader([f1, f2, f3])
    pass_ = RAFTPass(
        {
            "backend": "raft_small",
            "inference_resolution": 128,
            "num_flow_updates": 6,
        }
    )
    per_frame = pass_.run_shot(reader, frame_range=(1, 3))

    # Three frames, all nine channels each (5 spec + 4 Nuke aliases).
    assert sorted(per_frame.keys()) == [1, 2, 3]
    for f in (1, 2, 3):
        for ch in (
            CH_MOTION_X, CH_MOTION_Y, CH_BACK_X, CH_BACK_Y, CH_FLOW_CONFIDENCE,
            CH_FORWARD_U, CH_FORWARD_V, CH_BACKWARD_U, CH_BACKWARD_V,
        ):
            assert ch in per_frame[f]

    # Endpoint frames have zero motion for the missing direction:
    # frame 1 has no backward pair → back.x/y all zero
    assert np.allclose(per_frame[1][CH_BACK_X], 0.0)
    assert np.allclose(per_frame[1][CH_BACK_Y], 0.0)
    # frame 3 has no forward pair → motion.x/y all zero
    assert np.allclose(per_frame[3][CH_MOTION_X], 0.0)
    assert np.allclose(per_frame[3][CH_MOTION_Y], 0.0)

    artifacts = pass_.emit_artifacts()
    assert "forward_flow" in artifacts and "backward_flow" in artifacts
    assert "occlusion_mask" in artifacts and "parallax_estimate" in artifacts
    # Per-frame forward flow shapes are (2, H, W).
    any_fwd = next(iter(artifacts["forward_flow"].values()))
    assert any_fwd.shape == (2, f1.shape[0], f1.shape[1])
    # Parallax is a non-negative scalar expressed as a length-1 array.
    parallax_arr = next(iter(artifacts["parallax_estimate"].values()))
    assert parallax_arr.shape == (1,) and parallax_arr[0] >= 0.0
