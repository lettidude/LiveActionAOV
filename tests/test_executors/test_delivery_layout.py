"""Output layout: multi-layer vs one EXR per pass, plus the comp script."""

from __future__ import annotations

from pathlib import Path

import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.io.oiio_io import HAS_OIIO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed"),
]


def _job(folder: Path, out: Path, layout: str) -> Job:
    shot = Shot(
        name="layout_test",
        folder=folder,
        output_dir=out,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, 3),
        resolution=(640, 360),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=["noop"],
        delivery_layout=layout,
    )
    return Job(shot=shot, passes=[PassConfig(name="noop")])


def test_multilayer_writes_one_sequence_and_comp(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, "multilayer")
    live_action_aov.run(job)
    assert sorted(p.name for p in tmp_path.glob("*.exr")) == [
        f"test_plate.utility.{f:04d}.exr" for f in (1, 2, 3)
    ]
    comp = tmp_path / "layout_test.comp.nk"
    assert job.shot.sidecars["comp"] == comp
    text = comp.read_text()
    assert "test_plate.utility.####.exr" in text
    assert "test_plate.####.exr" in text  # the plate


def test_per_pass_writes_a_folder_per_pass(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, "per_pass")
    live_action_aov.run(job)
    assert not list(tmp_path.glob("*.exr"))
    assert sorted(p.name for p in (tmp_path / "depth").glob("*.exr")) == [
        f"test_plate.depth.{f:04d}.exr" for f in (1, 2, 3)
    ]
    text = (tmp_path / "layout_test.comp.nk").read_text()
    assert "depth/test_plate.depth.####.exr" in text


def test_layout_is_validated() -> None:
    with pytest.raises(ValueError):
        Shot(
            name="x",
            folder=Path("."),
            sequence_pattern="a.####.exr",
            frame_range=(1, 1),
            resolution=(8, 8),
            delivery_layout="zip",
        )
