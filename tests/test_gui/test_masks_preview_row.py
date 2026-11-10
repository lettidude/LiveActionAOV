"""Masks tab: the preview row (Preview, Auto, "with" engine) and the lit names preset."""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from live_action_aov.gui.inspector import InspectorPanel
from live_action_aov.gui.shot_state import ClickInstance, ShotRegistry, ShotState
from live_action_aov.gui.viewport import ViewportPanel
from live_action_aov.passes.matte import ltx_alpha


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


def _shot() -> ShotState:
    return ShotState(
        name="SH010",
        folder=Path("."),
        sequence_pattern="SH010.%04d.exr",
        frame_range=(1001, 1040),
        resolution=(1920, 1080),
    )


def test_names_preset_in_use_stays_lit(qapp: QApplication) -> None:
    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    reg.add(_shot())
    people, vehicles, everything = panel._concept_preset_btns
    # Empty field = the default list = "Everything".
    assert [b.isChecked() for b in (people, vehicles, everything)] == [False, False, True]
    people.click()
    assert [b.isChecked() for b in (people, vehicles, everything)] == [True, False, False]
    # Typing the same list by hand lights the preset; a custom list lights none.
    panel._sam3_concepts_edit.setText("Person,  vehicle")
    panel._on_sam3_concepts_edited("Person,  vehicle")
    assert [b.isChecked() for b in (people, vehicles, everything)] == [False, True, False]
    panel._sam3_concepts_edit.setText("red car")
    panel._on_sam3_concepts_edited("red car")
    assert not any(b.isChecked() for b in (people, vehicles, everything))


def test_preview_with_switches_engine_and_previews_again(qapp: QApplication) -> None:
    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    shot = _shot()
    reg.add(shot)
    asked: list[int] = []
    panel.mask_preview_requested.connect(lambda: asked.append(1))
    assert panel._preview_with_btns["auto"].isChecked()
    # No object with input yet: the choice is stored, nothing to preview.
    panel._preview_with_btns["vitmatte"].click()
    assert shot.preview_refiner == "vitmatte" and asked == []
    panel._on_mask_new()
    shot.click_instances[0].points.append((10.0, 10.0, 1))
    panel._preview_with_btns["none"].click()
    assert shot.preview_refiner == "none" and asked == [1]
    assert [k for k, b in panel._preview_with_btns.items() if b.isChecked()] == ["none"]
    # A session saved with a BiRefNet weight lights the BiRefNet button.
    shot.preview_refiner = "birefnet:ZhengPeng7/BiRefNet"
    panel._sync_preview_with()
    assert [k for k, b in panel._preview_with_btns.items() if b.isChecked()] == ["birefnet:"]


def test_auto_preview_runs_after_a_click_and_reruns_when_busy(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    reg = ShotRegistry()
    view = ViewportPanel(reg)
    shot = _shot()
    reg.add(shot)
    shot.current_frame = 1003
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    view.set_click_mode(True)
    view.set_active_instance(inst)
    calls: list[int] = []
    monkeypatch.setattr(view, "preview_active_mask", lambda: calls.append(1))
    view._on_canvas_pressed(0.5, 0.5, 1)
    qapp.processEvents()
    assert calls == [1]
    # Off: a click no longer previews.
    view.set_auto_preview(False)
    view._on_canvas_pressed(0.4, 0.4, 1)
    qapp.processEvents()
    assert calls == [1]
    # A result that lands after the points changed is dropped and run again.
    view.set_auto_preview(True)
    view._preview_again = True
    view._on_mask_preview_ready(object())
    qapp.processEvents()
    assert view._preview_mask is None and calls == [1, 1] and view._preview_again is False


def test_comfy_not_running_is_said_in_plain_words(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_a: object, **_k: object) -> object:
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(ltx_alpha.urllib.request, "urlopen", refuse)
    ok, msg = ltx_alpha.comfy_check({"comfy_url": "http://127.0.0.1:8187"})
    assert not ok
    assert "nothing is running there" in msg and "8187" in msg and "ViTMatte" in msg
    assert "WinError" not in msg and "urlopen" not in msg
