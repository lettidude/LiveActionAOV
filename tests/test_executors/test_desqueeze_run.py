"""End to end: an anamorphic plate run de-squeezed (`Shot.desqueeze`) and the
tracker proxy (`Shot.tracker_proxy`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.registry import get_registry
from live_action_aov.io.oiio_io import HAS_OIIO, read_exr, write_exr

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed"),
]

ID_A, ID_B = 1.5e-3, -2.25e7
W, H = 64, 48


class _Probe(UtilityPass):
    """Emits channels whose write-back is easy to check; records what it saw."""

    name = "probe_desqueeze"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="test")
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels: ClassVar[list[ChannelSpec]] = [ChannelSpec(name="Z", description="t")]
    seen: ClassVar[list[tuple[int, int]]] = []

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        img = tensor[0] if tensor.ndim == 4 else tensor
        h, w = img.shape[:2]
        _Probe.seen.append((w, h))
        ids = np.where(np.arange(w)[None, :] < w // 2, ID_A, ID_B).astype(np.float32)
        return {
            "Z": np.ascontiguousarray(img[..., 0]),
            "motion.x": np.full((h, w), 1.0, np.float32),
            "motion.y": np.full((h, w), 1.0, np.float32),
            "mask.person": np.ascontiguousarray(img[..., 1]),
            "CryptoObject00.R": np.repeat(ids, h, axis=0),
            "CryptoObject00.G": np.ones((h, w), np.float32),
        }


class _ProbeCamera(_Probe):
    name = "probe_desqueeze_camera"
    needs_full_frame = True


@pytest.fixture(autouse=True)
def _register() -> None:
    get_registry().register_pass(_Probe.name, _Probe)
    get_registry().register_pass(_ProbeCamera.name, _ProbeCamera)
    _Probe.seen.clear()


@pytest.fixture
def anamorphic_plate(tmp_path: Path) -> Path:
    folder = tmp_path / "plate"
    folder.mkdir()
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    for f in (1, 2):
        img = np.stack([xs / W, ys / H, np.full_like(xs, 0.25)], axis=-1)
        write_exr(folder / f"ana.{f:04d}.exr", img, ["R", "G", "B"], pixel_aspect=2.0)
    return folder


def _run(plate: Path, out: Path, passes: list[str] | None = None, **kw: Any) -> Shot:
    passes = passes or [_Probe.name]
    shot = Shot(
        name="ana",
        folder=plate,
        output_dir=out,
        sequence_pattern="ana.####.exr",
        frame_range=(1, 2),
        resolution=(W, H),
        pixel_aspect=2.0,
        passes_enabled=passes,
        **kw,
    )
    live_action_aov.run(Job(shot=shot, passes=[PassConfig(name=p) for p in passes]))
    return shot


def _channels(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    px, attrs = read_exr(path)
    return {n: px[..., i] for i, n in enumerate(attrs["channelnames"])}, attrs


def test_desqueezed_run_writes_back_at_the_plate_squeeze(
    anamorphic_plate: Path, tmp_path: Path
) -> None:
    _run(anamorphic_plate, tmp_path / "on", desqueeze=True)
    assert _Probe.seen[0] == (W, H // 2)  # the pass saw square pixels
    ch, attrs = _channels(tmp_path / "on" / "passes" / "ana.utility.0001.exr")
    assert (attrs["width"], attrs["height"]) == (W, H)
    assert attrs["pixelAspectRatio"] == pytest.approx(2.0)
    assert attrs["liveaov/input/processing"] == "desqueezed"
    assert np.allclose(ch["motion.x"], 1.0)  # x was not resampled
    assert np.allclose(ch["motion.y"], 2.0)  # y doubled back
    assert set(np.unique(ch["CryptoObject00.R"])) == {np.float32(ID_A), np.float32(ID_B)}
    assert abs(float(ch["Z"].mean()) - float((np.arange(W) / W).mean())) < 1e-3
    assert abs(float(ch["mask.person"].mean()) - float((np.arange(H) / H).mean())) < 2e-2


def test_off_keeps_the_plate_pixels(anamorphic_plate: Path, tmp_path: Path) -> None:
    _run(anamorphic_plate, tmp_path / "off", desqueeze=False)
    assert _Probe.seen[0] == (W, H)
    ch, attrs = _channels(tmp_path / "off" / "passes" / "ana.utility.0001.exr")
    assert np.allclose(ch["motion.y"], 1.0)
    assert "liveaov/input/processing" not in attrs


def test_camera_runs_stay_squeezed(anamorphic_plate: Path, tmp_path: Path) -> None:
    _run(anamorphic_plate, tmp_path / "cam", passes=[_ProbeCamera.name], desqueeze=True)
    assert _Probe.seen[0] == (W, H)


def test_tracker_proxy(anamorphic_plate: Path, tmp_path: Path) -> None:
    out = tmp_path / "trk"
    _run(anamorphic_plate, out, desqueeze=True, tracker_proxy=True)
    path = out / "passes" / "tracker" / "ana.tracker.0001.exr"
    ch, attrs = _channels(path)
    assert (attrs["width"], attrs["height"]) == (W // 4, H // 4)
    assert set(ch) == {"Z", "motion.x", "motion.y", "mask.person"}
    assert np.allclose(ch["motion.y"], 2.0)  # plate pixels
    assert attrs["liveaov/tracker_proxy/scale_x"] == pytest.approx(0.25)
    assert attrs["liveaov/tracker_proxy/source_size"] == f"{W}x{H}"
    assert attrs["pixelAspectRatio"] == pytest.approx(2.0)
    assert not any(str(k).startswith("cryptomatte/") for k in attrs)


def test_no_tracker_proxy_by_default(anamorphic_plate: Path, tmp_path: Path) -> None:
    _run(anamorphic_plate, tmp_path / "plain")
    assert not (tmp_path / "plain" / "passes" / "tracker").exists()
