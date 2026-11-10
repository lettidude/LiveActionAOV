"""Output tab: "This machine ... Suggested ..." and the button that takes it."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from live_action_aov.core.hardware import Hardware
from live_action_aov.gui import inspector as inspector_mod
from live_action_aov.gui.inspector import InspectorPanel
from live_action_aov.gui.shot_state import ShotRegistry, ShotState


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


def _shot(**kw: object) -> ShotState:
    base: dict[str, object] = dict(
        name="SH010",
        folder=Path("."),
        sequence_pattern="SH010.%04d.exr",
        frame_range=(1001, 1040),
        resolution=(4096, 2160),
    )
    base.update(kw)
    return ShotState(**base)  # type: ignore[arg-type]


def test_suggestion_is_shown_and_applied(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        inspector_mod, "_machine", lambda: (Hardware("RTX 4090", 24.0, 64.0, 50.0), True)
    )
    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    shot = _shot(detector="sam3")
    reg.add(shot)
    text = panel._hw_label.text()
    assert "RTX 4090 (24 GB), 64 GB RAM" in text and "proxy 1920" in text and "SAM 3.1" in text
    assert panel._hw_apply_btn.isEnabled() and shot.proxy_long_edge is None
    panel._hw_apply_btn.click()
    assert shot.proxy_long_edge == 1920 and shot.detector == "auto"
    assert panel._proxy_combo.currentData() == 1920
    assert "(in use)" in panel._hw_label.text() and not panel._hw_apply_btn.isEnabled()


def test_small_card_gets_a_small_proxy_and_sam3(
    qapp: QApplication, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        inspector_mod, "_machine", lambda: (Hardware("RTX 4070", 12.0, 32.0, 24.0), True)
    )
    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    reg.add(_shot())
    text = panel._hw_label.text()
    suggested = text.split("Suggested")[1]
    assert "Small GPU" in suggested and "proxy 1280" in suggested
    assert "SAM 3" in suggested and "SAM 3.1" not in suggested  # a 12 GB card: SAM 3


def test_masks_tab_names_presets_and_object_slot(qapp: QApplication) -> None:
    from live_action_aov.gui.session_io import shot_from_dict, shot_to_dict
    from live_action_aov.gui.submit_worker import _build_pass_configs

    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    shot = _shot(enabled_models=["sam3_matte_on"])
    reg.add(shot)
    # No object yet: the per-object controls are off, the names presets work.
    assert not panel._mask_slot_combo.isEnabled() and not panel._mask_goto_btn.isEnabled()
    panel._concept_preset_btns[1].click()  # People + vehicles
    assert shot.sam3_concepts == "person, vehicle"
    assert panel._sam3_concepts_edit.text() == "person, vehicle"
    panel._concept_preset_btns[2].click()  # Everything = the default list
    assert shot.sam3_concepts == ""

    panel._on_mask_new()
    panel._on_mask_new()
    a, b = shot.click_instances
    a.points.append((10.0, 10.0, 1))
    b.box = (20.0, 20.0, 60.0, 80.0)
    panel._mask_list.setCurrentRow(0)
    assert panel._mask_slot_combo.isEnabled()
    panel._mask_slot_combo.setCurrentIndex(panel._mask_slot_combo.findData("r"))
    assert a.slot == "r" and "matte.r" in panel._mask_list.item(0).text()
    panel._mask_list.setCurrentRow(1)
    panel._mask_slot_combo.setCurrentIndex(panel._mask_slot_combo.findData("r"))
    assert (a.slot, b.slot) == ("", "r")  # one object per slot

    asked: list[int] = []
    panel.frame_requested.connect(asked.append)
    panel._mask_goto_btn.click()
    assert asked == [b.seed_frame]

    sam = next(c for c in _build_pass_configs(shot) if c.name == "sam3_matte")
    specs = {s["name"]: s for s in sam.params["prompt_instances"]}
    assert specs[b.name]["slot"] == "r" and "slot" not in specs[a.name]
    back = shot_from_dict(shot_to_dict(shot))
    assert [c.slot for c in back.click_instances] == ["", "r"]
