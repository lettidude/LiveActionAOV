"""Crop-on-read (`io.readers.crop`) and the crop's EXR data window."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from live_action_aov.io.oiio_io import HAS_OIIO
from live_action_aov.io.readers.base import ImageSequenceReader
from live_action_aov.io.readers.crop import CropReader, clamp_crop, wrap_if_crop
from live_action_aov.io.readers.proxy import wrap_if_proxy


class _Ramp(ImageSequenceReader):
    """A 64x32 plate whose R channel is the x coordinate and G the y."""

    def __init__(self) -> None:
        super().__init__(Path("."), "ramp.####.exr")

    def frame_range(self) -> tuple[int, int]:
        return (1, 2)

    def resolution(self) -> tuple[int, int]:
        return (64, 32)

    def pixel_aspect(self) -> float:
        return 2.0

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        yy, xx = np.mgrid[0:32, 0:64]
        px = np.stack([xx, yy, np.full_like(xx, frame)], axis=-1).astype(np.float32)
        return px, {"frame": frame}


def test_crop_reader_returns_the_box_only() -> None:
    r = CropReader(_Ramp(), (10, 4, 20, 8))
    assert r.resolution() == (20, 8)
    assert r.frame_range() == (1, 2)
    assert r.pixel_aspect() == 2.0
    px, attrs = r.read_frame(2)
    assert px.shape == (8, 20, 3)
    assert px[0, 0, 0] == 10 and px[0, 0, 1] == 4  # top-left of the crop
    assert px[-1, -1, 0] == 29 and px[-1, -1, 1] == 11
    assert attrs == {"frame": 2}
    assert px.flags["C_CONTIGUOUS"]


def test_crop_is_clamped_to_the_plate_and_rejected_when_empty() -> None:
    assert clamp_crop((-5, -5, 20, 10), (64, 32)) == (0, 0, 15, 5)
    assert clamp_crop((60, 30, 20, 20), (64, 32)) == (60, 30, 4, 2)
    assert clamp_crop((10.4, 3.6, 7.5, 2.5), (64, 32)) == (10, 4, 8, 2)  # rounded, not truncated
    with pytest.raises(ValueError):
        clamp_crop((100, 0, 10, 10), (64, 32))
    assert wrap_if_crop(_Ramp(), None).__class__ is _Ramp


def test_proxy_scales_the_crop_not_the_plate() -> None:
    r = wrap_if_proxy(wrap_if_crop(_Ramp(), (0, 0, 40, 20)), 20)
    assert r.resolution() == (20, 10)
    assert r.read_frame(1)[0].shape == (10, 20, 3)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_exr_data_window_places_the_crop_in_the_plate(tmp_path: Path) -> None:
    import OpenImageIO as oiio

    from live_action_aov.io.oiio_io import read_exr
    from live_action_aov.io.writers.exr import ExrSidecarWriter

    out = tmp_path / "crop.exr"
    px = np.full((8, 20), 0.5, np.float32)
    ExrSidecarWriter().write_frame(out, {"Z": px}, origin=(10, 4), full_size=(64, 32))
    spec = oiio.ImageInput.open(str(out)).spec()
    assert (spec.x, spec.y, spec.width, spec.height) == (10, 4, 20, 8)
    assert (spec.full_x, spec.full_y, spec.full_width, spec.full_height) == (0, 0, 64, 32)
    back, _ = read_exr(out)
    assert back.shape[:2] == (8, 20)
