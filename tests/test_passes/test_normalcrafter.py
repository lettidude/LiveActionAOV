"""NormalCrafterPass — VIDEO_CLIP contract tests with a bypassed model.

Verifies the unit-length + axis-convert + upscale paths without touching
diffusers or the vendored NormalCrafter pipeline. The fake model returns
synthetic camera-space normals of a known shape/direction so we can
reason about the output.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z  # noqa: E402
from live_action_aov.passes.normals.normalcrafter import NormalCrafterPass  # noqa: E402


class _FakeNormalCrafter(NormalCrafterPass):
    """Returns a deterministic (N, H, W, 3) normals array.

    Every frame returns +X unit normals in OpenCV camera space — trivially
    unit-length, and makes the axis-convention test easy (OpenCV +X maps
    to OpenGL +X, both unchanged; OpenCV +Y/+Z flip).
    """

    def _load_model(self) -> None:  # type: ignore[override]
        # Skip the real diffusers loader entirely.
        self._pipeline = object()
        self._device = torch.device("cpu")
        self._dtype = torch.float32

    def _infer_clip(self, frames_pil):  # type: ignore[override]
        # frames_pil is a list of PIL.Image. Use the first to learn the
        # inference-time spatial shape, then emit +X everywhere.
        w, h = frames_pil[0].size
        n = len(frames_pil)
        normals = np.zeros((n, h, w, 3), dtype=np.float32)
        normals[..., 0] = 1.0   # +X everywhere
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
    """Spec §10.3 + trap 2: every pixel in the returned clip must have
    |N| = 1, even after axis conversion and upscale."""
    frames = _flat_frames(n=30)
    pass_ = _FakeNormalCrafter({"max_res": 1024})
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


def test_default_passthrough_does_not_flip_normals() -> None:
    """The real NormalCrafter output is already OpenGL-space (first
    CAT_070_0030 run revealed this), so the pass default is
    input_axes=output_axes=opengl → identity. A fake emitting +X in that
    frame should come out as +X unchanged."""
    frames = _flat_frames(n=8)
    pass_ = _FakeNormalCrafter({"max_res": 1024})
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 8))
    for f in out:
        assert np.allclose(out[f][CH_N_X], 1.0, atol=1e-4)
        assert np.allclose(out[f][CH_N_Y], 0.0, atol=1e-4)
        assert np.allclose(out[f][CH_N_Z], 0.0, atol=1e-4)


def test_axis_conversion_flips_y_and_z_when_explicitly_requested() -> None:
    """Users who want OpenCV-output normals (legacy DSINE-compatible) can
    opt in with `output_axes="opencv"`. The conversion helper flips Y and
    Z; verify by feeding a fake +Z OpenGL normal and asking for OpenCV."""

    class _FakeZ(_FakeNormalCrafter):
        def _infer_clip(self, frames_pil):  # type: ignore[override]
            w, h = frames_pil[0].size
            n = len(frames_pil)
            normals = np.zeros((n, h, w, 3), dtype=np.float32)
            normals[..., 2] = 1.0   # +Z
            return normals

    frames = _flat_frames(n=5)
    # Default input_axes=opengl; explicit output=opencv → flip Y and Z.
    pass_ = _FakeZ({"output_axes": "opencv", "max_res": 1024})
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    for f in out:
        assert np.allclose(out[f][CH_N_Z], -1.0, atol=1e-4)

    # Default axes → identity → +Z stays +Z.
    pass2 = _FakeZ({"max_res": 1024})
    out2 = pass2.run_shot(_FakeReader(frames), frame_range=(1, 5))
    for f in out2:
        assert np.allclose(out2[f][CH_N_Z], 1.0, atol=1e-4)


def test_smoothable_channels_is_empty_for_video_clip() -> None:
    assert NormalCrafterPass.smoothable_channels == []
