"""Sliding-window planning + overlap-blend stitching contract tests.

Pure-numpy math — no torch, no diffusers. These are the tests that let us
refactor DepthCrafter / NormalCrafter's stitching without burning GPU time.
"""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.shared.video_clip import (
    plan_window_starts,
    stitch_windowed_predictions,
    trapezoid_weight,
)

# --- plan_window_starts ---


def test_plan_covers_all_frames_single_window_case() -> None:
    # n_frames <= window → one window starting at 0.
    assert plan_window_starts(50, 110, 25) == [0]


def test_plan_backtracks_last_window_to_end_exactly() -> None:
    # Example from the docstring: 250 frames, window=110, overlap=25.
    starts = plan_window_starts(250, 110, 25)
    assert starts == [0, 85, 140]
    # Last window ends exactly at n_frames.
    assert starts[-1] + 110 == 250


def test_plan_no_duplicate_starts_when_stride_divides_evenly() -> None:
    # 200 / stride 50 would yield [0, 50, 100, 150] and last_start = 90 —
    # we must not double-enter 90 twice.
    starts = plan_window_starts(200, 100, 50)
    assert starts[-1] + 100 == 200
    assert len(starts) == len(set(starts))


def test_plan_rejects_overlap_at_or_beyond_window() -> None:
    with pytest.raises(ValueError, match="overlap must be"):
        plan_window_starts(200, 100, 100)
    with pytest.raises(ValueError, match="overlap must be"):
        plan_window_starts(200, 100, 150)


def test_plan_empty_clip_returns_empty() -> None:
    assert plan_window_starts(0, 110, 25) == []


# --- trapezoid_weight ---


def test_trapezoid_interior_is_flat_one() -> None:
    w = trapezoid_weight(window=20, overlap=5)
    # Interior (indices 5..14) should all be 1.
    assert np.allclose(w[5:15], 1.0)


def test_trapezoid_edges_ramp_from_1_over_overlap_to_one() -> None:
    w = trapezoid_weight(window=10, overlap=5)
    # k=0 → d_start = 1/5; k=4 → d_start = 5/5 = 1.
    assert w[0] == pytest.approx(0.2)
    assert w[4] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(0.2)
    assert w[-5] == pytest.approx(1.0)


def test_trapezoid_overlap_zero_is_all_ones() -> None:
    w = trapezoid_weight(window=7, overlap=0)
    assert np.allclose(w, 1.0)


# --- stitch_windowed_predictions ---


def _fake_window(values: float, window: int, shape=(2, 2)) -> np.ndarray:
    """Return a (window, *shape) array filled with `values`."""
    return np.full((window, *shape), values, dtype=np.float32)


def test_stitch_single_window_passthrough() -> None:
    pred = _fake_window(3.0, window=5)
    out = stitch_windowed_predictions([pred], starts=[0], n_frames=5, overlap=0)
    assert out.shape == (5, 2, 2)
    assert np.allclose(out, 3.0)


def test_stitch_overlap_crossfades_between_two_windows() -> None:
    """Two windows of 10 frames with overlap=4. Window A fills with 0.0,
    window B with 10.0. In the 4-frame overlap the stitched output should
    be a monotonic ramp from ~0 to ~10 (linear crossfade)."""
    window, overlap = 10, 4
    # First window covers frames 0..9, second covers frames 6..15.
    # n_frames = 16, starts = [0, 6].
    starts = plan_window_starts(16, window, overlap)
    assert starts == [0, 6]

    a = _fake_window(0.0, window, shape=(1, 1))
    b = _fake_window(10.0, window, shape=(1, 1))
    out = stitch_windowed_predictions([a, b], starts, n_frames=16, overlap=overlap)
    flat = out[:, 0, 0]

    # Endpoint pin: first frame is entirely window A (endpoint_unramped), last
    # frame entirely window B.
    assert flat[0] == pytest.approx(0.0)
    assert flat[-1] == pytest.approx(10.0)
    # Overlap frames 6..9: weighted average must be strictly monotonic.
    overlap_region = flat[6:10]
    assert np.all(np.diff(overlap_region) > 0)
    assert overlap_region[0] < 5.0 < overlap_region[-1]


def test_stitch_endpoint_unramped_pins_clip_edges_to_source() -> None:
    """With endpoint_unramped=True the absolute first/last frames should read
    the source window exactly (weight = 1)."""
    window, overlap = 8, 3
    a = _fake_window(1.0, window)
    b = _fake_window(7.0, window)
    starts = [0, 8 - overlap]  # [0, 5]
    n_frames = 8 + (window - overlap)  # 13

    out = stitch_windowed_predictions([a, b], starts, n_frames, overlap)
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[-1], 7.0)


def test_stitch_rejects_mismatched_trailing_shape() -> None:
    a = _fake_window(0.0, 5, shape=(4, 4))
    b = _fake_window(0.0, 5, shape=(4, 3))  # different W
    with pytest.raises(ValueError, match="trailing shape"):
        stitch_windowed_predictions([a, b], starts=[0, 3], n_frames=8, overlap=2)


def test_stitch_rejects_length_mismatch() -> None:
    a = _fake_window(0.0, 4)
    with pytest.raises(ValueError, match="length mismatch"):
        stitch_windowed_predictions([a], starts=[0, 3], n_frames=7, overlap=1)


def test_stitch_empty_raises() -> None:
    with pytest.raises(ValueError, match="No predictions"):
        stitch_windowed_predictions([], starts=[], n_frames=5, overlap=0)
