# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Models -> ComfyUI (LTX Alpha): where the user's ComfyUI runs, and a Test
button that checks the server, the nodes and the LTX-2.5 model files.

The URL is app-level (QSettings), injected into every `ltx_alpha_refiner`
pass at submit. The last test result is kept for the Masks tab status line.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

DEFAULT_URL = "http://127.0.0.1:8188"

#: Last check: (url, ok, message), or None before any check this session.
_last: tuple[str, bool, str] | None = None


def comfy_url() -> str:
    v = str(QSettings("LiveActionAOV", "LiveActionAOV").value("comfy_url", DEFAULT_URL) or "")
    return v.strip() or DEFAULT_URL


def set_comfy_url(url: str) -> None:
    QSettings("LiveActionAOV", "LiveActionAOV").setValue("comfy_url", url.strip() or DEFAULT_URL)


def check(url: str | None = None) -> tuple[bool, str]:
    """Run the LTX readiness check against `url` (default: the saved one)
    and remember the result."""
    global _last
    from live_action_aov.passes.matte.ltx_alpha import comfy_check

    url = (url or comfy_url()).strip()
    ok, msg = comfy_check({"comfy_url": url})
    _last = (url, ok, msg)
    return ok, msg


def status() -> tuple[bool, str]:
    """Readiness of the saved URL: the last result for it, else a fresh check."""
    url = comfy_url()
    if _last is not None and _last[0] == url:
        return _last[1], _last[2]
    return check(url)


class ComfyDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ComfyUI for LTX Alpha")
        self.setMinimumWidth(560)
        intro = QLabel(
            "LTX-2.5 Alpha runs in YOUR ComfyUI (the weights are not bundled). "
            "It needs the LTX-2.5 22B distilled model, its VAEs and text encoder, "
            "and the alpha-gen IC-LoRA, plus ComfyUI-VideoHelperSuite.\n"
            "Licence: LTX-2 Community - free under USD 10M annual revenue."
        )
        intro.setWordWrap(True)
        self._url = QLineEdit(comfy_url())
        self._url.setPlaceholderText(DEFAULT_URL)
        test = QPushButton("Test")
        test.clicked.connect(self._on_test)
        row = QHBoxLayout()
        row.addWidget(QLabel("ComfyUI URL:"))
        row.addWidget(self._url, stretch=1)
        row.addWidget(test)
        self._result = QLabel("")
        self._result.setWordWrap(True)
        self._result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        lay.addLayout(row)
        lay.addWidget(self._result)
        lay.addWidget(buttons)
        if _last is not None and _last[0] == comfy_url():
            self._show(_last[1], _last[2])

    def _show(self, ok: bool, msg: str) -> None:
        self._result.setText(("✓ " if ok else "✗ ") + msg)
        self._result.setStyleSheet(f"color: {'#6c6' if ok else '#e77'};")

    def _on_test(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._show(*check(self._url.text()))
        finally:
            QApplication.restoreOverrideCursor()

    def _on_save(self) -> None:
        set_comfy_url(self._url.text())
        self.accept()


__all__ = ["ComfyDialog", "check", "comfy_url", "set_comfy_url", "status"]
