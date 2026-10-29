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
    run = tmp_path / "layout_test" / "v001"
    job = _job(test_plate_1080p, run, "multilayer")
    live_action_aov.run(job)
    assert not list(run.glob("*.exr"))  # EXRs live in passes/, the .nk on top
    assert sorted(p.name for p in (run / "passes").glob("*.exr")) == [
        f"layout_test_v001.utility.{f:04d}.exr" for f in (1, 2, 3)
    ]
    comp = run / "layout_test_v001.comp.nk"
    assert job.shot.sidecars["comp"] == comp
    text = comp.read_text()
    assert "passes/layout_test_v001.utility.####.exr" in text
    assert "test_plate.####.exr" in text  # the plate
    assert list((run / "logs").glob("submit_*.log"))


def test_per_pass_writes_a_folder_per_pass(test_plate_1080p: Path, tmp_path: Path) -> None:
    job = _job(test_plate_1080p, tmp_path, "per_pass")
    live_action_aov.run(job)
    assert not list((tmp_path / "passes").glob("*.exr"))
    assert sorted(p.name for p in (tmp_path / "passes" / "depth").glob("*.exr")) == [
        f"layout_test.depth.{f:04d}.exr" for f in (1, 2, 3)
    ]
    text = (tmp_path / "layout_test.comp.nk").read_text()
    assert "passes/depth/layout_test.depth.####.exr" in text


def test_version_folders(tmp_path: Path) -> None:
    from live_action_aov.core.output_layout import latest_version_dir, next_version_dir, run_stem

    root = tmp_path / "sh010"
    assert latest_version_dir(root) is None
    assert next_version_dir(root) == root / "v001"
    (root / "v001").mkdir(parents=True)
    (root / "v007").mkdir()
    (root / "notes").mkdir()
    assert next_version_dir(root) == root / "v008"
    assert latest_version_dir(root) == root / "v007"
    shot = Shot(
        name="sh010",
        folder=tmp_path,
        output_dir=root / "v007",
        sequence_pattern="sh010_#####.exr",
        frame_range=(1, 1),
        resolution=(8, 8),
    )
    assert run_stem(shot) == "sh010_v007"
    from live_action_aov.core.output_layout import sidecar_template

    # Plate `name_#####` no longer leaks its underscore: shot-named, plate padding.
    assert sidecar_template(shot) == "passes/sh010_v007.utility.{frame:05d}.exr"


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
