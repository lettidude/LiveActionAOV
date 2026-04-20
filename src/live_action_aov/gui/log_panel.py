"""Collapsible log panel — bottom of the main window.

Threads three sources of information together:

1. Progress milestones from `SubmitWorker.progress(fraction, label)`.
2. Python `logging` output from every module (diffusers, transformers,
   huggingface_hub, our own code). A single handler forwards every
   record that meets the level threshold.
3. Submit-lifecycle bookends (start / done / failed) logged explicitly
   from the main window's submit handlers.

Users can:
    - Show/hide via the bottom-bar "Show log" toggle.
    - Clear the view (doesn't affect real stdout / log files).
    - Copy-all the visible text to paste into a bug report.

Kept deliberately read-only and monospace. A proper scrollable table
with filters / timestamps / level filters is M4 polish.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class LogLine:
    timestamp: float
    level: str  # DEBUG | INFO | WARNING | ERROR | PROGRESS | LIFECYCLE
    text: str


# Palette per log level. Keeps the panel readable in dark mode without
# relying on Qt theme hooks.
_LEVEL_COLOUR: dict[str, str] = {
    "DEBUG": "#777",
    "INFO": "#cfcfcf",
    "PROGRESS": "#4aa8ff",
    "WARNING": "#e0a040",
    "ERROR": "#e65c5c",
    "LIFECYCLE": "#5ec864",
}


class _QtSignalLogHandler(logging.Handler):
    """Routes `logging` records to a Qt signal.

    Can't subclass QObject directly (metaclass clash with `Handler`),
    so it holds a reference to a dedicated emitter object.
    """

    def __init__(self, emitter: LogEmitter, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._emitter = emitter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        self._emitter.emit_line(LogLine(record.created, record.levelname, msg))


class LogEmitter(QObject):
    """Thread-safe line sink. Qt signals marshal across threads, so
    the executor's worker thread can safely call `emit_line` and the
    `LogPanel` receives it on the UI thread."""

    line = Signal(object)  # LogLine

    def emit_line(self, line: LogLine) -> None:
        self.line.emit(line)


class LogPanel(QWidget):
    """Bottom-of-window log viewer. See module docstring."""

    def __init__(self) -> None:
        super().__init__()
        self._emitter = LogEmitter()
        self._emitter.line.connect(self._on_line)

        # Hook Python logging once — handler stays alive for the
        # lifetime of the GUI. Level INFO: captures diffusers /
        # transformers / huggingface_hub progress bars + our own
        # logging.info calls. DEBUG is usually noise from tokenisers.
        handler = _QtSignalLogHandler(self._emitter, level=logging.INFO)
        handler.setFormatter(
            logging.Formatter("%(name)s: %(message)s")
        )
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

        # --- Layout ---
        self._view = QTextEdit()
        self._view.setReadOnly(True)
        mono = QFont("Consolas, 'Courier New', monospace")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(9)
        self._view.setFont(mono)
        self._view.setStyleSheet(
            "QTextEdit {"
            " background: #121212; color: #cfcfcf;"
            " border: 1px solid #2a2a2a;"
            "}"
        )
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        title = QLabel("Log")
        title.setStyleSheet(
            "font-weight: 600; color: #cfcfcf; padding: 2px 6px;"
        )
        clear_btn = QPushButton("Clear")
        clear_btn.setFlat(True)
        clear_btn.clicked.connect(self._view.clear)
        copy_btn = QPushButton("Copy all")
        copy_btn.setFlat(True)
        copy_btn.clicked.connect(self._on_copy_all)

        header = QHBoxLayout()
        header.setContentsMargins(4, 2, 4, 2)
        header.addWidget(title)
        header.addStretch()
        header.addWidget(copy_btn)
        header.addWidget(clear_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)
        header_frame = QFrame()
        header_frame.setLayout(header)
        header_frame.setStyleSheet(
            "background: #1d1d1d; border-bottom: 1px solid #2a2a2a;"
        )
        root.addWidget(header_frame)
        root.addWidget(self._view, stretch=1)

        self.setMinimumHeight(120)

    # --- Public appenders ---

    def append_progress(self, fraction: float, label: str) -> None:
        """Called from the main window on SubmitWorker.progress."""
        pct = int(round(fraction * 100))
        self._emitter.emit_line(
            LogLine(time.time(), "PROGRESS", f"{pct:3d}% — {label}")
        )

    def append_lifecycle(self, text: str) -> None:
        """Submit bookends and similar high-level marks."""
        self._emitter.emit_line(LogLine(time.time(), "LIFECYCLE", text))

    def append_error(self, text: str) -> None:
        self._emitter.emit_line(LogLine(time.time(), "ERROR", text))

    # --- Internals ---

    def _on_line(self, line: LogLine) -> None:
        colour = _LEVEL_COLOUR.get(line.level, "#cfcfcf")
        ts = time.strftime("%H:%M:%S", time.localtime(line.timestamp))
        # HTML escape the message so stack traces with < > don't bork
        # the render. Newlines in multi-line messages (tracebacks) stay.
        from html import escape

        body = escape(line.text).replace("\n", "<br/>")
        html = (
            f'<span style="color: #666;">[{ts}]</span> '
            f'<span style="color: {colour};"><b>{line.level}</b></span> '
            f'<span style="color: #cfcfcf;">{body}</span>'
        )
        self._view.append(html)
        # Auto-scroll to the bottom. `moveCursor` works regardless of
        # user scroll position; if they scrolled up intentionally they
        # can use the scrollbar while the view keeps appending below.
        self._view.moveCursor(QTextCursor.MoveOperation.End)

    def _on_copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self._view.toPlainText())


# Silence ruff about the Qt constant we use only via the metaclass.
_ = Qt


__all__ = ["LogPanel"]
