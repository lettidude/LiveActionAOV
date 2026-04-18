"""NormalCrafterPass — VIDEO_CLIP contract tests with a bypassed model.

Verifies the sliding-window + unit-length renormalize + axis-convert path
without touching diffusers. The fake model returns synthetic camera-space
normals so we can reason about the output.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z  # noqa: E402
from live_action_aov.passes.normals.normalcrafter import NormalCrafterPass  # noqa: E402


class _FakeNormalCrafter(NormalCrafterPass):
    """Returns a deterministic (N, 3, h, w) normals tensor.

    Every window returns +X unit normals — trivially unit-length and makes
    the axis-convention test easy (OpenCV +X maps to OpenGL +X, both
    unchanged; OpenCV +Y/+Z flip).
    """

    def _load_model(self) -> None:  # type: ignore[override]
        if self._pipeline is not None:
            return
        self._pipeline = object()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _infer_window(self, tensor):  # type: ignore[override]
        video = tensor["video"]
        n, _, h, w = video.shape
        normals = torch.zeros((n, 3, h, w), dtype=torch.float32)
        normals[:, 0] = 1.0          # +X everywhere
        return normals


class _FakeReader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._first = 1

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _flat_frames(n: int, h: int = 16, w: int = 24) -> list[np.ndarray]:
    return [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n)]


def test_license_is_noncommercial_gated() -> None:
    lic = NormalCrafterPass.declared_license()
    assert lic.commercial_use is False
    assert "SVD" in lic.spdx
    assert "Stable Video Diffusion" in lic.notes


def test_overlap_geq_window_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        NormalCrafterPass({"window": 14, "overlap": 14})


def test_run_shot_produces_unit_length_normals_per_frame() -> None:
    """Spec §10.3 + trap 2: every pixel in the stitched clip must have
    |N| = 1, even after the overlap blend."""
    frames = _flat_frames(n=30)
    pass_ = _FakeNormalCrafter(
        {"window": 10, "overlap": 3, "inference_short_edge": 16}
    )
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 30))

    assert sorted(out) == list(range(1, 31))
    for f in out:
        nx = out[f][CH_N_X]
        ny = out[f][CH_N_Y]
        nz = out[f][CH_N_Z]
        assert nx.shape == ny.shape == nz.shape == (16, 24)
        assert nx.dtype == np.float32
        mag = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        assert np.allclose(mag, 1.0, atol=1e-4)


def test_axis_conversion_opencv_to_opengl_preserves_x() -> None:
    """Fake emits +X in OpenCV. OpenCV→OpenGL flips Y and Z but leaves X,
    so CH_N_X should be ≈ +1 everywhere."""
    frames = _flat_frames(n=8)
    pass_ = _FakeNormalCrafter(
        {"window": 5, "overlap": 1, "inference_short_edge": 16}
    )
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 8))
    for f in out:
        assert np.allclose(out[f][CH_N_X], 1.0, atol=1e-4)
        assert np.allclose(out[f][CH_N_Y], 0.0, atol=1e-4)
        assert np.allclose(out[f][CH_N_Z], 0.0, atol=1e-4)


def test_axis_conversion_opencv_pass_through_does_not_flip() -> None:
    """Override to skip axis conversion — fake +Z normals must stay +Z."""
    class _FakeZ(_FakeNormalCrafter):
        def _infer_window(self, tensor):  # type: ignore[override]
            video = tensor["video"]
            n, _, h, w = video.shape
            normals = torch.zeros((n, 3, h, w), dtype=torch.float32)
            normals[:, 2] = 1.0   # +Z
            return normals

    frames = _flat_frames(n=5)
    pass_ = _FakeZ(
        {"window": 5, "overlap": 0, "output_axes": "opencv",
         "inference_short_edge": 16}
    )
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    for f in out:
        assert np.allclose(out[f][CH_N_Z], 1.0, atol=1e-4)

    # With default opengl output axes, +Z OpenCV flips to -Z.
    pass2 = _FakeZ({"window": 5, "overlap": 0, "inference_short_edge": 16})
    out2 = pass2.run_shot(_FakeReader(frames), frame_range=(1, 5))
    for f in out2:
        assert np.allclose(out2[f][CH_N_Z], -1.0, atol=1e-4)


def test_smoothable_channels_is_empty_for_video_clip() -> None:
    assert NormalCrafterPass.smoothable_channels == []
