"""An object drawn on a frame the run will not process is refused before the run."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.shot_state import ClickInstance, ShotState
from live_action_aov.gui.submit_worker import (
    _build_pass_configs,
    objects_outside_range,
    outside_range_message,
)


def _shot() -> ShotState:
    s = ShotState(
        name="SH010",
        folder=Path("."),
        sequence_pattern="SH010.%04d.exr",
        frame_range=(1001, 1100),
        resolution=(1920, 1080),
    )
    s.enabled_models = ["sam3_matte_on"]
    s.click_instances.append(ClickInstance(name="hero", seed_frame=1010, points=[(5.0, 5.0, 1)]))
    return s


def test_object_inside_the_range_passes() -> None:
    s = _shot()
    assert objects_outside_range(s) == [] and outside_range_message(s) == ""
    s.work_range = (1005, 1050)
    assert objects_outside_range(s) == []
    _build_pass_configs(s)  # does not raise


def test_object_outside_the_range_is_named_and_refused() -> None:
    s = _shot()
    s.work_range = (1040, 1080)
    assert objects_outside_range(s) == [("hero", 1010)]
    msg = outside_range_message(s)
    assert "'hero' (frame 1010)" in msg and "1040-1080" in msg and "This shot" in msg
    with pytest.raises(ValueError, match="outside the frames to process"):
        _build_pass_configs(s)


def test_object_without_input_or_with_matte_off_is_ignored() -> None:
    s = _shot()
    s.work_range = (1040, 1080)
    s.click_instances[0].points.clear()  # nothing to seed with: it is dropped anyway
    assert objects_outside_range(s) == []
    s.click_instances[0].points.append((5.0, 5.0, 1))
    s.enabled_models = ["moge3"]  # no matte in this run: the objects are not used
    assert objects_outside_range(s) == []
