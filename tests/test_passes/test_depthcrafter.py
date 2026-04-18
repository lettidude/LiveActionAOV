"""DepthCrafterPass — VIDEO_CLIP contract tests with a bypassed model.

Verifies the sliding-window → stitch → per-clip-normalize path without
touching diffusers. `_load_model` is a no-op; `_infer_window` returns a
deterministic per-window depth tensor so we can reason about stitching
and normalization analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.io.channels import CH_Z, CH_Z_RAW  # noqa: E402
from live_action_aov.passes.depth.depthcrafter import DepthCrafterPass  # noqa: E402


class _FakeDepthCrafter(DepthCrafterPass):
    """Short-circuit the diffusers pipeline.

    `_infer_window` returns a depth tensor that's a simple function of
    (frame index, image mean). This makes the stitched result predictable
    and lets us test that the sliding window actually covers every frame.
    """

    def _load_model(self) -> None:  # type: ignore[override]
        if self._pipeline is not None:
            return
        self._pipeline = object()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _infer_window(self, tensor):  # type: ignore[override]
        # Input video shape: (N, 3, h, w). Return (N, h, w) depth where each
        # frame's depth field = image mean + (in-window frame index) * 0.01.
        video = tensor["video"]
        n, _, h, w = video.shape
        depth = torch.zeros((n, h, w), dtype=torch.float32)
        for k in range(n):
            depth[k] = float(video[k].mean()) + k * 0.01
        return depth


class _FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._first = 1

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _varying_frames(n: int, h: int = 16, w: int = 24) -> list[np.ndarray]:
    """Frames whose mean intensity ramps linearly across the shot."""
    return [
        np.full((h, w, 3), 0.1 + 0.8 * (i / max(n - 1, 1)), dtype=np.float32)
        for i in range(n)
    ]


def test_license_is_noncommercial_gated() -> None:
    lic = DepthCrafterPass.declared_license()
    assert lic.commercial_use is False
    assert "SVD" in lic.spdx
    assert "Stable Video Diffusion" in lic.notes


def test_overlap_geq_window_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        DepthCrafterPass({"window": 10, "overlap": 10})
    with pytest.raises(ValueError, match="overlap"):
        DepthCrafterPass({"window": 10, "overlap": 20})


def test_run_shot_covers_every_frame_and_normalizes_per_clip() -> None:
    """Clip longer than one window — exercises the sliding-window path."""
    frames = _varying_frames(n=30)
    pass_ = _FakeDepthCrafter(
        {"window": 12, "overlap": 4, "inference_short_edge": 16}
    )
    reader = _FakeReader(frames)
    out = pass_.run_shot(reader, frame_range=(1, 30))

    # Every frame in range produced output.
    assert sorted(out) == list(range(1, 31))
    for f in out:
        assert CH_Z in out[f] and CH_Z_RAW in out[f]
        assert out[f][CH_Z].shape == (16, 24)
        assert out[f][CH_Z].dtype == np.float32

    # Per-clip normalization: combined Z min/max spans [0, 1] exactly.
    stacked = np.stack([out[f][CH_Z] for f in out])
    assert stacked.min() == pytest.approx(0.0, abs=1e-5)
    assert stacked.max() == pytest.approx(1.0, abs=1e-5)
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0


def test_run_shot_clip_shorter_than_window_is_single_window() -> None:
    frames = _varying_frames(n=5)
    pass_ = _FakeDepthCrafter({"window": 110, "overlap": 25, "inference_short_edge": 16})
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    assert sorted(out) == [1, 2, 3, 4, 5]
    for f in out:
        assert out[f][CH_Z].shape == (16, 24)


def test_z_raw_is_unflipped_and_monotonic_with_fake_depth() -> None:
    """Our fake depth is `image_mean + k*0.01`; image means ramp with frame,
    so raw depth means should be monotonically increasing across the clip.
    Z (after flip) should be monotonically decreasing."""
    frames = _varying_frames(n=20)
    pass_ = _FakeDepthCrafter({"window": 10, "overlap": 3, "inference_short_edge": 16})
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 20))

    raw_means = [float(out[f][CH_Z_RAW].mean()) for f in range(1, 21)]
    z_means = [float(out[f][CH_Z].mean()) for f in range(1, 21)]
    # Not strictly monotonic at overlap boundaries (blending adds tiny
    # jitter) but the overall trend must hold.
    assert raw_means[-1] > raw_means[0]
    assert z_means[0] > z_means[-1]


def test_emit_artifacts_exposes_depth_normalization_constants() -> None:
    frames = _varying_frames(n=8)
    pass_ = _FakeDepthCrafter({"window": 6, "overlap": 2, "inference_short_edge": 16})
    pass_.run_shot(_FakeReader(frames), frame_range=(1, 8))
    artifacts = pass_.emit_artifacts()
    assert "depth_norm_min" in artifacts and "depth_norm_max" in artifacts
    min_arr = next(iter(artifacts["depth_norm_min"].values()))
    max_arr = next(iter(artifacts["depth_norm_max"].values()))
    assert min_arr.shape == (1,) and max_arr.shape == (1,)
    assert min_arr[0] < max_arr[0]


def test_smoothable_channels_is_empty_for_video_clip() -> None:
    """VIDEO_CLIP passes are already temporally coherent — the auto-smoother
    must not attach to them. Empty `smoothable_channels` is how we signal
    that to the executor (spec §13.1)."""
    assert DepthCrafterPass.smoothable_channels == []
