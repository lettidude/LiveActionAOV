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
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.inspector import InspectorPanel
from live_action_aov.gui.shot_list import ShotListPanel
from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.gui.submit_worker import SubmitResult, SubmitWorker
from live_action_aov.gui.viewport import ViewportPanel

# Pass backends that need the non-commercial gate. Mirrors the set in
# the inspector; declared here so the submit precheck is self-contained.
_NC_BACKENDS: set[str] = {
    "depthcrafter",
    "depthpro",
    "normalcrafter",
    "matanyone2",
}

_WINDOW_TITLE = "Live Action AOV — Shot Prep"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(_WINDOW_TITLE)
        # Default to something that fits a 1080p laptop comfortably.
        # 1400×820 was eating the taskbar on the dev machine and hiding
        # the scrub slider off-screen; keep the footprint smaller.
        self.resize(1200, 720)
        # Anything smaller than this clips the scrub slider or chops the
        # inspector in half — users can still resize below this but we
        # won't let Qt auto-size into an unusable layout.
        self.setMinimumSize(900, 560)

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

        # --- Bottom bar: Submit + progress + reveal-output ---
        # Sits under the splitter so it stays visible regardless of how
        # the three panels are sized. M2 scope is local submit only;
        # Deadline lands when the executor side does.
        self._submit_btn = QPushButton("Submit local")
        self._submit_btn.setEnabled(False)
        self._submit_btn.clicked.connect(self._on_submit_clicked)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)

        self._reveal_btn = QPushButton("Reveal output")
        self._reveal_btn.setEnabled(False)
        self._reveal_btn.setToolTip("Open the sidecar output folder.")
        self._reveal_btn.clicked.connect(self._on_reveal_clicked)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._submit_btn)
        bottom_row.addWidget(self._progress, stretch=1)
        bottom_row.addWidget(self._reveal_btn)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.addWidget(splitter, stretch=1)
        central_layout.addLayout(bottom_row)
        self.setCentralWidget(central)
        self._build_menus()

        status = QStatusBar()
        status.showMessage(
            "Prep UX — drop a plate folder on the left panel or use File → Add shot."
        )
        self.setStatusBar(status)

        # Async executor driver — re-used across submits (one-shot runner
        # per job; owns a QRunnable submitted to the global thread pool).
        self._submit_worker = SubmitWorker()
        self._submit_worker.finished.connect(self._on_submit_finished)

        # Submit gate: button enables when any shot with at least one
        # enabled pass is selected.
        self._registry.current_changed.connect(self._refresh_submit_button)
        self._registry.shot_updated.connect(lambda s: self._refresh_submit_button(s))

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
            "Live Action AOV — Shot Prep GUI (Phase 5, Milestone 2).\n\n"
            "Submit runs the pipeline locally via LocalExecutor.\n"
            "Deadline submit and session autosave land in later milestones.",
        )

    # --- Submit lifecycle ---

    def _refresh_submit_button(self, shot: ShotState | None) -> None:
        cur = self._registry.current()
        has_passes = bool(cur and cur.enabled_passes)
        is_running = cur is not None and cur.status == "running"
        self._submit_btn.setEnabled(has_passes and not is_running)
        self._reveal_btn.setEnabled(cur is not None and cur.last_sidecar_dir is not None)
        del shot  # parameter only present to satisfy the shot_updated signal

    def _on_submit_clicked(self) -> None:
        shot = self._registry.current()
        if shot is None:
            return
        gate = _check_license_gate(shot)
        if gate is not None:
            QMessageBox.warning(self, "Non-commercial backend selected", gate)
            return
        shot.status = "running"
        shot.last_error = ""
        self._registry.notify_updated(shot)
        self._submit_btn.setEnabled(False)
        self._progress.setVisible(True)
        self.statusBar().showMessage(f"Running {shot.name} …")
        self._submit_worker.submit(shot)

    def _on_submit_finished(self, result: SubmitResult) -> None:
        self._progress.setVisible(False)

        # Locate the shot that produced the result. id()-matching is
        # fine because we only ever submit one shot at a time in M2.
        target: ShotState | None = None
        for s in self._registry.shots():
            if id(s) == result.shot_state_id:
                target = s
                break
        if target is None:
            return

        if result.success:
            target.status = "done"
            target.last_sidecar_dir = result.sidecar_dir
            self.statusBar().showMessage(
                f"Done: {target.name} → {result.sidecar_dir}", 10_000
            )
        else:
            target.status = "failed"
            target.last_error = result.error or "unknown error"
            self.statusBar().showMessage(f"Failed: {target.name}", 10_000)
            QMessageBox.critical(
                self,
                f"Submit failed — {target.name}",
                result.error or "unknown error",
            )
        self._registry.notify_updated(target)
        self._refresh_submit_button(target)

    def _on_reveal_clicked(self) -> None:
        cur = self._registry.current()
        if cur is None or cur.last_sidecar_dir is None:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cur.last_sidecar_dir)))


def _check_license_gate(shot: ShotState) -> str | None:
    """Return a warning message if the selected backends need
    `allow_noncommercial` but it's off. Returns None if OK to submit.

    Mirrors the CLI's refusal path so the GUI can't produce a job the
    CLI would reject — keeps the two surfaces behaviourally aligned.
    """
    if shot.allow_noncommercial:
        return None
    blocked: list[str] = []
    for family, backend in shot.pass_backends.items():
        if family.startswith("matte") and "matte" not in shot.enabled_passes:
            continue
        if family == "depth" and "depth" not in shot.enabled_passes:
            continue
        if family == "normals" and "normals" not in shot.enabled_passes:
            continue
        if backend in _NC_BACKENDS:
            blocked.append(f"{family}: {backend}")
    if not blocked:
        return None
    return (
        "The following backends are non-commercial and require the "
        "'Allow non-commercial backends' toggle:\n\n  - "
        + "\n  - ".join(blocked)
        + "\n\nEnable the toggle or pick a commercial-safe backend."
    )


# Silence ruff's unused-import noise when we prune widgets in a future iteration.
_ = QWidget

__all__ = ["MainWindow"]
