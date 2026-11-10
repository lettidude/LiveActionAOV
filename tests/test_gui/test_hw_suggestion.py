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
