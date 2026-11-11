"""The executor writes `logs/<shot>.run_report.json` on every run: passes that
ran, failed, were skipped; a pass's own fields read before `unload`; a run
that fails still leaves its report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import ChannelSpec, License, PassType, TemporalMode, UtilityPass
from live_action_aov.core.registry import get_registry
from live_action_aov.core.run_report import load_report
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
    name = "rr_depth"
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name=CH_Z)]

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        h, w = tensor.shape[1:3] if tensor.ndim == 4 else tensor.shape[:2]
        return {CH_Z: np.full((h, w), 0.5, dtype=np.float32)}


class _Refiner(_Base):
    """Falls back, and its report needs state that `unload` clears."""

    name = "rr_refiner"
    temporal_mode = TemporalMode.VIDEO_CLIP

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        self._engine = "plan_b"
        return {}

    def report_fields(self) -> dict[str, Any]:
        return {
            "engine": self._engine,
            "fallbacks": [{"wanted": "plan_a", "used": self._engine, "reason": "server down"}],
            "section": "refiner",
            "summary": {"engine": self._engine},
        }

    def unload(self) -> None:
        self._engine = None


class _Camera(_Base):
    name = "rr_camera"
    temporal_mode = TemporalMode.VIDEO_CLIP
    provides_artifacts = ["camera_solve"]
    footage_dependent = True

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        raise RuntimeError("only 10/100 frames could be solved")


class _Splat(_Base):
    name = "rr_splat"
    temporal_mode = TemporalMode.VIDEO_CLIP
    requires_artifacts = ["camera_solve"]
    footage_dependent = True


class _Broken(_Base):
    name = "rr_broken"
    temporal_mode = TemporalMode.VIDEO_CLIP

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        raise RuntimeError("a real bug")


@pytest.fixture(autouse=True)
def _register() -> None:
    for cls in (_Depth, _Refiner, _Camera, _Splat, _Broken):
        get_registry().register_pass(cls.name, cls)


def _job(plate: Path, out: Path, names: list[str]) -> Job:
    shot = Shot(
        name="rr",
        folder=plate,
        output_dir=out / "rr" / "v002",
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, 3),
        resolution=(640, 360),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=names,
    )
    return Job(shot=shot, passes=[PassConfig(name=n) for n in names])


def test_report_after_a_run(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, ["rr_depth", "rr_refiner", "rr_camera", "rr_splat"])
    live_action_aov.run(job)
    path = tmp_path / "rr" / "v002" / "logs" / "rr_v002.run_report.json"
    assert path.is_file()
    assert job.shot.sidecars["run_report"] == path
    rep = load_report(path)
    by = {p.name: p for p in rep.passes}
    assert rep.status == "done"
    assert by["rr_depth"].status == "ran" and by["rr_depth"].channels == [CH_Z]
    assert by["rr_depth"].frames == (1, 3) and by["rr_depth"].seconds >= 0.0
    assert by["rr_refiner"].status == "fell_back" and by["rr_refiner"].engine == "plan_b"
    assert rep.summaries["refiner"] == {"pass": "rr_refiner", "engine": "plan_b"}
    assert by["rr_camera"].status == "failed" and "10/100" in by["rr_camera"].reason
    assert by["rr_splat"].status == "skipped" and "camera_solve" in by["rr_splat"].reason
    kinds = {n.kind for n in rep.needs_human}
    assert kinds == {"fallback", "pass_failed", "pass_skipped"}
    assert rep.summary.startswith("rr_v002 done: 1 ran, 1 fell back (rr_refiner -> plan_b)")
    assert "comp" in rep.sidecars
    json.loads(path.read_text())  # plain JSON


def test_a_failed_run_still_writes_its_report(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, ["rr_depth", "rr_broken"])
    with pytest.raises(RuntimeError, match="a real bug"):
        live_action_aov.run(job)
    rep = load_report(tmp_path / "rr" / "v002" / "logs" / "rr_v002.run_report.json")
    assert rep.status == "failed" and "a real bug" in rep.error
    by = {p.name: p.status for p in rep.passes}
    assert by == {"rr_depth": "ran", "rr_broken": "failed"}
    assert rep.needs_human[0].kind == "run_failed"
