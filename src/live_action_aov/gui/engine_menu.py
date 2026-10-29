# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""One engine picker for the whole GUI (field feedback: too many engine
dropdowns, then an "Apply" step on top).

The menu shows the recommended choice first and puts every other engine
under "Advanced". Non-commercial engines, when added, carry
NOT FOR PRODUCTION USE in their label (same rule as Depth Large).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QMenu, QPushButton, QSizePolicy, QWidget


@dataclass(frozen=True)
class EngineChoice:
    data: tuple[str, str]  # (engine, BiRefNet weights or "")
    label: str
    advanced: bool = False
    noncommercial: bool = False

    @property
    def menu_label(self) -> str:
        return f"{self.label}  [NC - NOT FOR PRODUCTION USE]" if self.noncommercial else self.label


#: The run's default engine (Passes tab) -> ShotState.default_engine / refiner_model.
RUN_ENGINES: tuple[EngineChoice, ...] = (
    EngineChoice(("vitmatte", ""), "Auto (recommended) - ViTMatte + BiRefNet guide"),
    EngineChoice(("chromakey", ""), "Green / blue screen - Chroma key"),
    EngineChoice(("birefnet", ""), "BiRefNet Portrait - people, hair", advanced=True),
    EngineChoice(
        ("birefnet", "ZhengPeng7/BiRefNet-matting"), "BiRefNet Matting - general soft", True
    ),
    EngineChoice(("birefnet", "ZhengPeng7/BiRefNet"), "BiRefNet General - cleanest licence", True),
    EngineChoice(("rvm", ""), "RVM - video matting, people", advanced=True),
)

#: A click object's own engine (Masks tab) -> ClickInstance.refiner.
OBJECT_ENGINES: tuple[EngineChoice, ...] = (
    EngineChoice(("", ""), "Default (run engine)"),
    EngineChoice(("chromakey", ""), "Green / blue screen - Chroma key"),
    EngineChoice(("vitmatte", ""), "ViTMatte + BiRefNet guide", advanced=True),
    EngineChoice(("birefnet", ""), "BiRefNet Portrait - people, hair", advanced=True),
    EngineChoice(("rvm", ""), "RVM - video matting, people", advanced=True),
)


class EngineButton(QPushButton):
    """A button whose menu lists the engines; shows the current one."""

    picked = Signal(object)  # the chosen EngineChoice.data

    def __init__(self, choices: Sequence[EngineChoice], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._choices = tuple(choices)
        self._current = self._choices[0]
        self.setStyleSheet("text-align: left; padding-left: 8px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        menu = QMenu(self)
        advanced: QMenu | None = None
        for c in self._choices:
            target = menu
            if c.advanced:
                if advanced is None:
                    advanced = menu.addMenu("Advanced")
                target = advanced
            act = target.addAction(c.menu_label)
            act.triggered.connect(lambda _=False, c=c: self._pick(c))
        self.setMenu(menu)
        self._show()

    def current(self) -> tuple[str, str]:
        return self._current.data

    def set_current(self, data: tuple[str, str]) -> None:
        """Show `data` without emitting (unknown -> the first choice)."""
        self._current = next((c for c in self._choices if c.data == tuple(data)), self._choices[0])
        self._show()

    def short_label(self) -> str:
        return self._current.label.split(" - ")[0]

    def _pick(self, c: EngineChoice) -> None:
        self._current = c
        self._show()
        self.picked.emit(c.data)

    def _show(self) -> None:
        # The button shows the short name; the menu and tooltip the full one.
        nc = "  [NC]" if self._current.noncommercial else ""
        self.setText(self.short_label() + nc)
        self.setToolTip(self._current.menu_label)


__all__ = ["OBJECT_ENGINES", "RUN_ENGINES", "EngineButton", "EngineChoice"]
