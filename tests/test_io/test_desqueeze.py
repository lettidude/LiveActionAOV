"""Anamorphic de-squeeze (`io.readers.desqueeze`), the per-type write-back and
the tracker proxy's resampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from live_action_aov.executors.local import _upscale_channels_to_plate, tracker_proxy_channels
from live_action_aov.io.readers.base import ImageSequenceReader
from live_action_aov.io.readers.desqueeze import (
    DesqueezeReader,
    desqueezed_size,
    is_squeezed,
    wrap_if_desqueeze,
)
from live_action_aov.io.readers.proxy import wrap_if_proxy


class _Plate(ImageSequenceReader):
    """A 64x48 anamorphic plate: R = x / 64, G = y / 48."""

    def __init__(self, par: float = 2.0) -> None:
        super().__init__(Path("."), "plate.####.exr")
        self._par = par

    def frame_range(self) -> tuple[int, int]:
        return (1, 2)

    def resolution(self) -> tuple[int, int]:
        return (64, 48)

    def pixel_aspect(self) -> float:
        return self._par

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        ys, xs = np.mgrid[0:48, 0:64].astype(np.float32)
        img = np.stack([xs / 64, ys / 48, np.full_like(xs, 0.5)], axis=-1)
        return img, {"frame": frame}


def test_size_never_grows_an_axis() -> None:
    assert desqueezed_size(4096, 3432, 2.0) == (4096, 1716)
    assert desqueezed_size(1000, 800, 0.5) == (500, 800)
    assert desqueezed_size(1920, 1080, 1.0) == (1920, 1080)
    assert desqueezed_size(1920, 1080, 1.0004) == (1920, 1080)
    assert not is_squeezed(1.0) and not is_squeezed(None) and is_squeezed(1.33)


def test_reader_desqueezes_and_keeps_values() -> None:
    r = DesqueezeReader(_Plate(), 2.0)
    assert r.resolution() == (64, 24) and r.pixel_aspect() == 1.0
    assert r.plate_resolution == (64, 48) and r.plate_pixel_aspect == 2.0
    img, attrs = r.read_frame(1)
    assert img.shape == (24, 64, 3) and attrs == {"frame": 1}
    src, _ = _Plate().read_frame(1)
    assert np.allclose(img.mean(axis=(0, 1)), src.mean(axis=(0, 1)), atol=1e-5)
    assert np.allclose(img[:, :, 0], src[:24, :, 0])  # x untouched


def test_wrap_is_a_no_op_when_off_or_square() -> None:
    plate = _Plate()
    assert wrap_if_desqueeze(plate, 2.0, False) is plate
    assert wrap_if_desqueeze(plate, 1.0, True) is plate
    assert isinstance(wrap_if_desqueeze(plate, 2.0, True), DesqueezeReader)


def test_proxy_scales_the_desqueezed_frame() -> None:
    r = wrap_if_proxy(DesqueezeReader(_Plate(), 2.0), 32)
    assert r.resolution() == (32, 12)
    assert r.read_frame(1)[0].shape == (12, 32, 3)


def test_write_back_per_channel_type() -> None:
    """De-squeezed (64x24) -> plate (64x48): flow y doubles and x stays, ids
    stay ids, normals stay unit length."""
    h, w = 24, 64
    ids = np.where(np.arange(w)[None, :] < 31, 1.5e-3, -2.25e7).astype(np.float32)
    ids = np.repeat(ids, h, axis=0)
    ids[10:14, :] = 7.0  # a horizontal band: y edges get resampled
    n = np.zeros((h, w, 3), np.float32)
    n[:, : w // 2] = (1.0, 0.0, 0.0)
    n[:, w // 2 :] = (0.0, 0.0, 1.0)
    ch = {
        "motion.x": np.full((h, w), 3.0, np.float32),
        "motion.y": np.full((h, w), 2.0, np.float32),
        "motion_fb.y": np.full((h, w), 0.5, np.float32),
        "CryptoObject00.R": ids,
        "N.x": n[..., 0],
        "N.y": n[..., 1],
        "N.z": n[..., 2],
        "Z": np.linspace(1, 2, h * w, dtype=np.float32).reshape(h, w),
    }
    out = _upscale_channels_to_plate(
        ch, 48, 64, nearest={"CryptoObject00.R"}, renormalize_normals=True
    )
    assert all(a.shape == (48, 64) for a in out.values())
    assert np.allclose(out["motion.x"], 3.0) and np.allclose(out["motion.y"], 4.0)
    assert np.allclose(out["motion_fb.y"], 1.0)
    assert set(np.unique(out["CryptoObject00.R"])) == set(np.unique(ids))
    length = np.sqrt(out["N.x"] ** 2 + out["N.y"] ** 2 + out["N.z"] ** 2)
    assert np.allclose(length, 1.0, atol=1e-5)
    assert abs(float(out["Z"].mean()) - float(ch["Z"].mean())) < 1e-3


def test_write_back_default_is_unchanged() -> None:
    """Without the new arguments the proxy write-back is the historic one."""
    ids = np.repeat(np.where(np.arange(16) < 8, 1.0, 3.0)[None, :], 8, axis=0).astype(np.float32)
    out = _upscale_channels_to_plate({"CryptoObject00.R": ids}, 16, 32)
    assert len(np.unique(out["CryptoObject00.R"])) > 2  # bilinear blends, as before


def test_tracker_proxy_channels() -> None:
    h, w = 48, 64
    z = np.where(np.arange(w)[None, :] < 30, 1.0, 10.0).astype(np.float32).repeat(h, 0)
    ch = {
        "motion.x": np.full((h, w), 8.0, np.float32),
        "back.y": np.full((h, w), -4.0, np.float32),
        "Z": z,
        "mask.person": np.ones((h, w), np.float32),
        "matte.r": np.zeros((h, w), np.float32),
        "N.x": np.zeros((h, w), np.float32),
        "CryptoObject00.R": np.zeros((h, w), np.float32),
    }
    small, size = tracker_proxy_channels(ch, 0.25)
    assert size == (16, 12)
    assert set(small) == {"motion.x", "back.y", "Z", "mask.person", "matte.r"}
    assert all(a.shape == (12, 16) for a in small.values())
    assert np.allclose(small["motion.x"], 8.0)  # plate pixels, not rescaled
    assert np.allclose(small["back.y"], -4.0)
    assert set(np.unique(small["Z"])) <= {1.0, 10.0}  # point-sampled, no mixed depth


def test_tracker_proxy_nothing_to_carry() -> None:
    assert tracker_proxy_channels({"N.x": np.zeros((8, 8), np.float32)}, 0.25) == ({}, (0, 0))


def test_fb_residual_is_zero_when_flows_agree() -> None:
    torch = pytest.importorskip("torch")
    from live_action_aov.passes.flow.raft import _fb_residual

    a = torch.zeros((1, 2, 16, 20))
    a[:, 0] = 3.0
    a[:, 1] = -1.0
    b = -a
    assert float(_fb_residual(a, b).abs().max()) < 1e-5
    bad = b.clone()
    bad[:, 0] = 0.0  # the reverse flow forgot the x motion
    err = _fb_residual(a, bad)
    assert torch.allclose(err[:, 0], torch.full_like(err[:, 0], 3.0))
    assert torch.allclose(err[:, 1], torch.zeros_like(err[:, 1]), atol=1e-5)
