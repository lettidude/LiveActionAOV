"""GUI -> job wiring for the SAM 3.1 detector, the whole-frame LTX engine and
the PnP camera fallback; the batch picking up shots ticked while it runs."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.engine_menu import RUN_ENGINES
from live_action_aov.gui.main_window import late_queued_shots
from live_action_aov.gui.session_io import shot_from_dict, shot_to_dict
from live_action_aov.gui.shot_state import ShotState
from live_action_aov.gui.submit_worker import _build_pass_configs


def _shot(**kw: object) -> ShotState:
    base: dict[str, object] = dict(
        name="SH010",
        folder=Path("."),
        sequence_pattern="SH010.%04d.exr",
        frame_range=(1, 10),
        resolution=(1920, 1080),
        enabled_models=["sam3_matte_on"],
    )
    base.update(kw)
    return ShotState(**base)  # type: ignore[arg-type]


def _params(state: ShotState) -> dict[str, dict]:
    return {c.name: dict(c.params or {}) for c in _build_pass_configs(state)}


def test_sam31_detector_reaches_the_matte_pass() -> None:
    assert "detector" not in _params(_shot()).get("sam3_matte", {})
    assert _params(_shot(detector="sam3.1"))["sam3_matte"]["detector"] == "sam3.1"


def test_whole_frame_ltx_is_a_run_engine_with_the_comfy_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from live_action_aov.gui import comfy_dialog

    monkeypatch.setattr(comfy_dialog, "comfy_url", lambda: "http://ai-box:8188")
    assert ("ltx_whole", "") in [c.data for c in RUN_ENGINES]
    p = _params(_shot(default_engine="ltx_whole"))
    assert p["ltx_alpha_refiner"] == {"comfy_url": "http://ai-box:8188", "mode": "whole_frame"}
    assert "vitmatte_refiner" not in p


def test_pnp_fallback_toggle_reaches_the_camera_pass() -> None:
    on = _shot(enabled_models=["camera_pycolmap"])
    assert _params(on)["camera_pycolmap"] == {}  # pass default: on
    off = _shot(enabled_models=["camera_pycolmap"], camera_pnp_fallback=False)
    assert _params(off)["camera_pycolmap"] == {"pnp_fallback": False}


def test_session_keeps_detector_and_pnp_choice() -> None:
    s = _shot(detector="sam3.1", camera_pnp_fallback=False, default_engine="ltx_whole")
    back = shot_from_dict(shot_to_dict(s))
    assert (back.detector, back.camera_pnp_fallback, back.default_engine) == (
        "sam3.1",
        False,
        "ltx_whole",
    )
    old = shot_to_dict(_shot())
    del old["detector"], old["camera_pnp_fallback"]  # a session saved before this change
    back = shot_from_dict(old)
    assert back.detector == "sam3" and back.camera_pnp_fallback is True


def test_batch_picks_up_shots_ticked_while_it_ran() -> None:
    """A batch: some shots submitted, more ticked while it ran (Submit
    is disabled meanwhile). They used to be ignored, and a failure on the
    last submitted shot showed a blocking single-shot dialog."""
    first = [_shot(name=f"A{i}") for i in range(3)]
    later = [_shot(name=f"B{i}") for i in range(2)]
    unticked = _shot(name="C0", queued=False)
    no_passes = _shot(name="D0", enabled_models=[])
    running = _shot(name="E0", status="running")
    seen = {id(s) for s in first}
    shots = first + later + [unticked, no_passes, running]
    assert late_queued_shots(shots, seen) == later
    seen |= {id(s) for s in later}
    assert late_queued_shots(shots, seen) == []
