"""MainWindow — three-panel QSplitter host for the prep GUI.

Left panel   shot list (ShotListPanel)
Centre       viewport + view-mode radios + frame scrub (ViewportPanel)
Right        inspector: colorspace + exposure + pass toggles (InspectorPanel)

The three panels share a single `ShotRegistry` and communicate via its
signals — no back-references between panels, no inheritance from a
common base. That keeps each panel independently testable and makes
adding a fourth panel (histogram, pixel inspector in M3) a matter of
wiring one more signal handler.

Phase 5 GUI is deliberately prep-only in M1. "Submit local" and
"Submit Deadline" buttons live on the future bottom bar; wiring them
to `LocalExecutor` lands in M2.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from live_action_aov.gui.inspector import InspectorPanel
from live_action_aov.gui.shot_list import ShotListPanel
from live_action_aov.gui.shot_state import ShotRegistry
from live_action_aov.gui.viewport import ViewportPanel

_WINDOW_TITLE = "Live Action AOV — Shot Prep"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_WINDOW_TITLE)
        self.resize(1400, 820)

        self._registry = ShotRegistry()

        self._shot_list = ShotListPanel(self._registry)
        self._viewport = ViewportPanel(self._registry)
        self._inspector = InspectorPanel(self._registry)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._shot_list)
        splitter.addWidget(self._viewport)
        splitter.addWidget(self._inspector)
        # Sensible initial proportions: 220 / 800 / 380 px.
        splitter.setSizes([220, 800, 380])
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)

        self.setCentralWidget(splitter)
        self._build_menus()

        status = QStatusBar()
        status.showMessage(
            "Prep UX — drop a plate folder on the left panel or use File → Add shot."
        )
        self.setStatusBar(status)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        add_action = QAction("&Add shot…", self)
        add_action.setShortcut(QKeySequence.StandardKey.Open)
        add_action.triggered.connect(self._shot_list._on_add_clicked)
        file_menu.addAction(add_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About Live Action AOV",
            "Live Action AOV — Shot Prep GUI (Phase 5, Milestone 1).\n\n"
            "Prep UX only. Execution stays on the CLI until M2 wires\n"
            "`liveaov run-shot` into the Submit Local button.",
        )


# Silence ruff's unused-import noise when we prune widgets in a future iteration.
_ = QWidget

__all__ = ["MainWindow"]
