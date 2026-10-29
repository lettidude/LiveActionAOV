"""A pass that fails because of the footage (camera solve on a pan) must not
throw away the rest of the run: the mattes / depth are still written, the
passes depending on it are skipped, and the failure is reported."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import ChannelSpec, License, PassType, TemporalMode, UtilityPass
from live_action_aov.core.registry import get_registry
from live_action_aov.io.channels import CH_Z
from live_action_aov.io.oiio_io import HAS_OIIO

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")


class _Base(UtilityPass):
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}


class _Depth(_Base):
    name = "ff_depth"
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name=CH_Z)]

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        h, w = tensor.shape[1:3] if tensor.ndim == 4 else tensor.shape[:2]
        return {CH_Z: np.full((h, w), 0.5, dtype=np.float32)}


class _Camera(_Base):
    name = "ff_camera"
    temporal_mode = TemporalMode.VIDEO_CLIP
    provides_artifacts = ["camera_solve"]
    footage_dependent = True
    unloaded = False

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        raise RuntimeError("Camera track: only 10/100 frames could be solved")

    def unload(self) -> None:
        type(self).unloaded = True


class _Splat(_Base):
    name = "ff_splat"
    temporal_mode = TemporalMode.VIDEO_CLIP
    requires_artifacts = ["camera_solve"]
    footage_dependent = True
    ran = False

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        type(self).ran = True
        return {}


class _Broken(_Base):
    name = "ff_broken"
    temporal_mode = TemporalMode.VIDEO_CLIP

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        raise RuntimeError("a real bug")


@pytest.fixture(autouse=True)
def _register() -> None:
    for cls in (_Depth, _Camera, _Splat, _Broken):
        get_registry().register_pass(cls.name, cls)


def _job(plate: Path, out: Path, names: list[str]) -> Job:
    shot = Shot(
        name="ff",
        folder=plate,
        output_dir=out,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, 3),
        resolution=(640, 360),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=names,
    )
    return Job(shot=shot, passes=[PassConfig(name=n) for n in names])


def test_camera_failure_keeps_the_rest(test_plate_1080p: Path, tmp_path: Path) -> None:
    _Splat.ran = False
    _Camera.unloaded = False
    job = _job(test_plate_1080p, tmp_path, ["ff_depth", "ff_camera", "ff_splat"])
    live_action_aov.run(job)
    assert job.shot.status == "done"
    assert len(list((tmp_path / "passes").glob("ff.utility.*.exr"))) == 3  # depth kept
    assert (tmp_path / "ff.comp.nk").exists()
    assert not _Splat.ran  # its camera never came
    assert _Camera.unloaded  # temp files / model still released
    assert "ff_camera: Camera track: only 10/100" in job.shot.notes
    assert "ff_splat: needs camera_solve" in job.shot.notes


def test_a_real_bug_still_fails_the_run(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, ["ff_depth", "ff_broken"])
    with pytest.raises(RuntimeError, match="a real bug"):
        live_action_aov.run(job)
