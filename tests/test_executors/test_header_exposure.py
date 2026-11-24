"""The utility header's `input/exposure_offset` is the EV the reader really
applied to the plate before the passes saw it (downstream tools rebuild the
model's display image from the plate with it)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.executors.local import METADATA_NAMESPACE
from live_action_aov.io.oiio_io import HAS_OIIO, read_exr, write_exr

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")

KEY = f"{METADATA_NAMESPACE}/input/exposure_offset"


def _plate(folder: Path, level: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    ys = np.linspace(0.5, 1.5, 36, dtype=np.float32)[:, None, None]
    for f in (1, 2, 3):
        px = np.broadcast_to(ys * level, (36, 64, 3)).astype(np.float32).copy()
        write_exr(folder / f"plate.{f:04d}.exr", px, ["R", "G", "B"])
    return folder


def _run(
    plate: Path, out: Path, transform: DisplayTransformParams, **shot_kw: object
) -> tuple[dict, Job]:
    shot = Shot(
        name="ev",
        folder=plate,
        output_dir=out / "v001",
        sequence_pattern="plate.####.exr",
        frame_range=(1, 3),
        resolution=(64, 36),
        colorspace="lin_rec709",
        transform=transform,
        apply_display_transform=True,
        passes_enabled=["noop"],
        **shot_kw,  # type: ignore[arg-type]
    )
    job = Job(shot=shot, passes=[PassConfig(name="noop")])
    live_action_aov.run(job)
    exr = sorted((out / "v001").rglob("*.0001.exr"))[0]
    return read_exr(exr)[1], job


def test_header_carries_auto_ev_of_a_dark_plate(tmp_path: Path) -> None:
    plate = _plate(tmp_path / "plate", 0.04)
    header, job = _run(plate, tmp_path / "out", DisplayTransformParams())
    ev = job.shot.transform.computed_exposure_ev
    assert ev is not None and ev > 1.0  # ~2 stops under 0.18
    assert header[KEY] == pytest.approx(ev, abs=1e-6)


def test_header_carries_manual_ev(tmp_path: Path) -> None:
    plate = _plate(tmp_path / "plate", 0.04)
    header, _ = _run(plate, tmp_path / "out", DisplayTransformParams(manual_exposure_ev=1.5))
    assert header[KEY] == pytest.approx(1.5)


def test_header_writes_zero_when_ev_is_zero(tmp_path: Path) -> None:
    plate = _plate(tmp_path / "plate", 0.04)
    header, _ = _run(plate, tmp_path / "out", DisplayTransformParams(manual_exposure_ev=0.0))
    assert header[KEY] == 0.0


def test_header_carries_the_ev_in_a_run_in_windows(tmp_path: Path) -> None:
    plate = _plate(tmp_path / "plate", 0.04)
    header, job = _run(plate, tmp_path / "out", DisplayTransformParams(), window_frames=2)
    ev = job.shot.transform.computed_exposure_ev
    assert ev is not None and ev > 1.0
    assert header[KEY] == pytest.approx(ev, abs=1e-6)


def test_run_log_states_the_ev_and_its_source(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    plate = _plate(tmp_path / "plate", 0.04)
    with caplog.at_level(logging.INFO, logger="live_action_aov.executors.local"):
        _run(plate, tmp_path / "out", DisplayTransformParams(manual_exposure_ev=1.5))
    assert "Plate exposure: +1.50 EV (manual)" in caplog.text
