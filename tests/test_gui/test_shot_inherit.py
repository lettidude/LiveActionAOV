"""A shot added to a session inherits its output routing and colourspace
override instead of resetting to defaults (field report 2026-10-28)."""

from __future__ import annotations

from pathlib import Path

from live_action_aov.gui.shot_list import _inherit_session_settings
from live_action_aov.gui.shot_state import ShotState
from live_action_aov.io.colorspace_detect import DetectedColorspace


def _shot(name: str, confident: bool = False) -> ShotState:
    return ShotState(
        name=name,
        folder=Path(f"A:/plates/{name}"),
        sequence_pattern=f"{name}.####.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
        detected=DetectedColorspace("lin_rec709", "test", confident),
    )


def test_new_shot_takes_output_and_override() -> None:
    old = _shot("a")
    old.output_mode = "root"
    old.output_external_root = Path("A:/out")
    old.output_external_name = "a_custom"
    old.proxy_long_edge = 1920
    old.override = "acescg"
    new = _shot("b")
    _inherit_session_settings(new, old)
    assert new.output_external_root == Path("A:/out")
    # A custom shot-folder name is per shot: inheriting it would merge two shots.
    assert new.output_external_name == ""
    assert new.resolve_output_dir() == Path("A:/out/b")
    assert new.proxy_long_edge == 1920
    assert new.override == "acescg"


def test_confident_header_keeps_its_own_colourspace() -> None:
    old = _shot("a")
    old.override = "acescg"
    new = _shot("b", confident=True)  # e.g. ARRI LogC4 camera metadata
    _inherit_session_settings(new, old)
    assert new.override is None


def test_default_output_is_a_sibling_of_the_plate_folder() -> None:
    shot = _shot("PF0052_v001")
    assert shot.output_mode == "plate"
    assert shot.resolve_output_dir() == Path("A:/plates/PF0052_v001_utility")
    shot.output_external_name = "aov"
    assert shot.resolve_output_dir() == Path("A:/plates/aov")
    # The root only counts in root mode.
    shot.output_external_root = Path("D:/out")
    assert shot.resolve_output_dir() == Path("A:/plates/aov")
    shot.output_mode = "root"
    assert shot.resolve_output_dir() == Path("D:/out/aov")
