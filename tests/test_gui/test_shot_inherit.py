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
    old.output_mode = "external"
    old.output_external_root = Path("A:/out")
    old.proxy_long_edge = 1920
    old.override = "acescg"
    new = _shot("b")
    _inherit_session_settings(new, old)
    assert new.output_mode == "external" and new.output_external_root == Path("A:/out")
    assert new.proxy_long_edge == 1920
    assert new.override == "acescg"


def test_confident_header_keeps_its_own_colourspace() -> None:
    old = _shot("a")
    old.override = "acescg"
    new = _shot("b", confident=True)  # e.g. ARRI LogC4 camera metadata
    _inherit_session_settings(new, old)
    assert new.override is None
