"""GUI plumbing of the anamorphic de-squeeze and the tracker proxy."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.session_io import shot_from_dict, shot_to_dict
from live_action_aov.gui.shot_state import ShotState
from live_action_aov.gui.submit_worker import _build_pass_configs, _shot_state_to_core_shot


def _shot(**kw: object) -> ShotState:
    base: dict[str, object] = dict(
        name="PF0071",
        folder=Path("."),
        sequence_pattern="PF0071.%04d.exr",
        frame_range=(1, 10),
        resolution=(1920, 1080),
        pixel_aspect=2.0,
    )
    base.update(kw)
    return ShotState(**base)  # type: ignore[arg-type]


def test_defaults() -> None:
    s = _shot()
    assert s.desqueeze is True and s.tracker_proxy is False


def test_session_round_trip() -> None:
    s = shot_from_dict(shot_to_dict(_shot(desqueeze=True, tracker_proxy=True)))
    assert s.desqueeze is True and s.tracker_proxy is True
    old = shot_to_dict(_shot())
    old.pop("desqueeze")
    old.pop("tracker_proxy")
    s = shot_from_dict(old)  # a session saved before these existed
    assert s.desqueeze is True and s.tracker_proxy is False
    s = shot_from_dict(shot_to_dict(_shot(desqueeze=False)))
    assert s.desqueeze is False


def test_reaches_the_core_shot() -> None:
    state = _shot(desqueeze=True, tracker_proxy=True)
    state.output_mode = "root"
    state.output_external_root = Path(".") / "_out"
    shot = _shot_state_to_core_shot(state)
    assert shot.desqueeze is True and shot.tracker_proxy is True
    assert shot.pixel_aspect == 2.0


def test_tracker_proxy_turns_on_the_flow_error() -> None:
    on = _build_pass_configs(_shot(enabled_models=["flow"], tracker_proxy=True))
    off = _build_pass_configs(_shot(enabled_models=["flow"]))
    assert [p.params for p in on if p.name == "flow"] == [{"fb_residual": True}]
    assert [p.params for p in off if p.name == "flow"] == [{}]
