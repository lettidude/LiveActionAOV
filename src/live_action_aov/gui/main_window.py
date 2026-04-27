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
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.cuda_check import CudaState, cuda_state
from live_action_aov.gui.inspector import InspectorPanel
from live_action_aov.gui.log_panel import LogPanel
from live_action_aov.gui.pass_catalog import has_noncommercial
from live_action_aov.gui.shot_list import ShotListPanel
from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.gui.submit_worker import SubmitResult, SubmitWorker
from live_action_aov.gui.viewport import ViewportPanel

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

        # Probe CUDA once at startup. The result drives both the
        # header banner (shown only when CUDA is missing) and the
        # submit-time gate that refuses to run neural passes on CPU.
        self._cuda = cuda_state()

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
        # Deadline lands when the executor side does. Submit iterates
        # every queued shot (the per-row checkbox), so users can
        # process a whole list in one click.
        self._submit_btn = QPushButton("Submit local")
        self._submit_btn.setEnabled(False)
        self._submit_btn.clicked.connect(self._on_submit_clicked)
        # Queue state — populated by _on_submit_clicked, drained by
        # _on_submit_finished. The currently-running shot is the head
        # of the deque.
        self._queue: list[ShotState] = []
        # Set when the user clicks Cancel — the next pop in
        # _start_next_in_queue() honours this and short-circuits the
        # remaining batch instead of starting the next shot.
        self._batch_cancelled: bool = False

        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setFormat("%p% — %v")  # replaced with label text at run time
        self._progress.setTextVisible(True)
        self._progress.setVisible(False)

        # Cancel sits next to the progress bar, hidden until a submit
        # starts. We deliberately keep it OUT of the layout slot the
        # Submit button occupies so the user can't accidentally click
        # the wrong action — Submit is "start", Cancel is "stop", and
        # they're mutually exclusive. Cancel only signals at the
        # executor's next checkpoint (between passes / before each
        # sidecar write); a torch inference mid-call still has to
        # complete. Tooltip spells that out so the user isn't
        # confused by the lag between click and abort.
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setToolTip(
            "Stop the current run. Takes effect at the next safe "
            "checkpoint (between passes / before each sidecar write). "
            "Heavy passes already mid-inference must finish first."
        )
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

        self._reveal_btn = QPushButton("Reveal output")
        self._reveal_btn.setEnabled(False)
        self._reveal_btn.setToolTip("Open the sidecar output folder.")
        self._reveal_btn.clicked.connect(self._on_reveal_clicked)

        # Log-panel toggle — the panel stays hidden until the user
        # opts in, so the prep-UX surface area doesn't shrink for
        # people who just want to submit and walk away.
        self._log_toggle_btn = QPushButton("Show log")
        self._log_toggle_btn.setCheckable(True)
        self._log_toggle_btn.setChecked(False)
        self._log_toggle_btn.toggled.connect(self._on_log_toggle)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._submit_btn)
        bottom_row.addWidget(self._progress, stretch=1)
        bottom_row.addWidget(self._cancel_btn)
        bottom_row.addWidget(self._log_toggle_btn)
        bottom_row.addWidget(self._reveal_btn)

        # Log panel — stays at the bottom of the window, wired to
        # SubmitWorker.progress + Python logging. Hidden by default.
        self._log_panel = LogPanel()
        self._log_panel.setVisible(False)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        # Environment banner — only visible when the CUDA preflight
        # failed. Sits at the very top so it's impossible to miss.
        # "Details" opens a dialog with the exact reinstall command.
        banner = _build_cuda_banner(self._cuda, self)
        if banner is not None:
            central_layout.addWidget(banner)
        central_layout.addWidget(splitter, stretch=1)
        central_layout.addLayout(bottom_row)
        central_layout.addWidget(self._log_panel)
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
        self._submit_worker.progress.connect(self._on_submit_progress)

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
        open_logs = QAction("Open &log folder", self)
        open_logs.setToolTip(
            "Open the central directory where every submit writes a "
            "run log (plus a rolling warnings digest)."
        )
        open_logs.triggered.connect(self._open_log_folder)
        help_menu.addAction(open_logs)
        help_menu.addSeparator()
        about = QAction("&About", self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    def _open_log_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from live_action_aov.core.logging_setup import get_log_dir

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_log_dir())))

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
        # Submit iterates every queued shot with passes. The button
        # enables when there's at least one such shot AND nothing is
        # currently running.
        queued_with_passes = [s for s in self._registry.shots() if s.queued and s.enabled_passes]
        any_running = any(s.status == "running" for s in self._registry.shots())
        self._submit_btn.setEnabled(bool(queued_with_passes) and not any_running)
        n = len(queued_with_passes)
        self._submit_btn.setText("Submit local" if n <= 1 else f"Submit local  ({n} shots)")
        cur = self._registry.current()
        self._reveal_btn.setEnabled(cur is not None and cur.last_sidecar_dir is not None)
        del shot

    def _on_submit_clicked(self) -> None:
        queue = [s for s in self._registry.shots() if s.queued and s.enabled_passes]
        if not queue:
            return
        # Environment precheck — every pass in the catalog is a neural
        # model that needs CUDA. Don't let the user wait 20 minutes for
        # a CPU-fp16 failure they could have been warned about in 2s.
        if not self._cuda.available:
            QMessageBox.critical(
                self,
                "CUDA GPU not available",
                self._cuda.advisory,
            )
            return
        # Validate every shot up-front so we fail loud before any
        # expensive work starts.
        for shot in queue:
            if shot.output_mode == "external" and shot.output_external_root is None:
                QMessageBox.warning(
                    self,
                    "External output root missing",
                    f"'{shot.name}': pick an external output root "
                    "(Inspector → Output → Choose root…) or switch back "
                    "to 'Next to plate'.",
                )
                return

        # Non-commercial consent — one dialog for the whole batch,
        # listing every NC model the user has enabled across all
        # queued shots. We can't enforce the license; we confirm the
        # user knows what they've selected and takes responsibility.
        nc_summary = _collect_nc_entries(queue)
        if nc_summary:
            if not _confirm_nc_consent(self, nc_summary):
                return
        # Passed validation — mark queued shots as 'queued' state, then
        # kick the first one.
        for shot in queue:
            shot.status = "queued"
            shot.last_error = ""
            self._registry.notify_updated(shot)
        self._queue = queue
        self._batch_cancelled = False
        self._submit_btn.setEnabled(False)
        # Cancel button activates while the batch runs. Hidden again
        # in _start_next_in_queue() once the queue drains.
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Cancel")
        self._start_next_in_queue()

    def _start_next_in_queue(self) -> None:
        # Honor a mid-batch cancel: if the user clicked Cancel while
        # one shot was running, every remaining shot is dropped. We
        # mark them back to queued (not failed) so re-clicking Submit
        # picks them up next time — a cancelled run is reversible
        # state, not a failure mode.
        if self._batch_cancelled and self._queue:
            for shot in self._queue:
                shot.status = "queued"
                self._registry.notify_updated(shot)
            self._queue.clear()
        if not self._queue:
            # Done with the whole batch (or batch was cancelled).
            self._progress.setVisible(False)
            self._cancel_btn.setVisible(False)
            if self._batch_cancelled:
                self.statusBar().showMessage("Batch cancelled.", 8_000)
                self._log_panel.append_lifecycle("===== Batch cancelled =====")
                self._batch_cancelled = False
            else:
                self.statusBar().showMessage("Batch complete.", 8_000)
                self._log_panel.append_lifecycle("===== Batch complete =====")
            self._refresh_submit_button(None)
            return
        shot = self._queue[0]
        shot.status = "running"
        self._registry.notify_updated(shot)
        self._progress.setValue(0)
        self._progress.setFormat("Starting…")
        self._progress.setVisible(True)
        remaining = len(self._queue)
        msg = f"Running {shot.name} …"
        if remaining > 1:
            msg += f"   ({remaining} shots left in batch)"
        self.statusBar().showMessage(msg)
        self._log_panel.append_lifecycle(
            f"===== Submit start: {shot.name} "
            f"({len(shot.enabled_models)} model{'s' if len(shot.enabled_models) != 1 else ''}) ====="
        )
        self._submit_worker.submit(shot)

    def _on_cancel_clicked(self) -> None:
        # First click flips both the worker's CancelToken AND a
        # batch-level flag so any remaining queued shots are dropped
        # in _start_next_in_queue() once the current shot's worker
        # callback fires. Disable the button immediately so the user
        # gets visual feedback that the click registered, even though
        # the executor won't observe it until its next checkpoint.
        if not self._cancel_btn.isEnabled():
            return
        self._batch_cancelled = True
        self._submit_worker.cancel()
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")
        self.statusBar().showMessage("Cancelling — finishing current step…", 0)
        self._log_panel.append_lifecycle("===== User cancel requested =====")

    def _on_submit_progress(self, fraction: float, label: str) -> None:
        self._progress.setValue(int(fraction * 1000))
        # Keep the percentage visible AND the stage label — compositors
        # want to see "Pass 2/3: depth_anything_v2" not just 67%.
        self._progress.setFormat(f"{int(fraction * 100)}%  —  {label}")
        # Mirror to the log panel so a hidden progress bar still leaves
        # a paper trail.
        self._log_panel.append_progress(fraction, label)

    def _on_log_toggle(self, checked: bool) -> None:
        self._log_panel.setVisible(checked)
        self._log_toggle_btn.setText("Hide log" if checked else "Show log")

    def _on_submit_finished(self, result: SubmitResult) -> None:
        # Locate the shot that produced the result by identity.
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
            self._log_panel.append_lifecycle(
                f"===== Submit done: {target.name} → {result.sidecar_dir} ====="
            )
        elif result.cancelled:
            # User-initiated abort. Mark the shot as cancelled (not
            # failed) — partial sidecars on disk are expected and the
            # user will commonly retry without a code change. No
            # error dialog: the click was the user's intent.
            target.status = "cancelled"
            target.last_error = ""
            self._log_panel.append_lifecycle(f"===== Submit cancelled: {target.name} =====")
        else:
            target.status = "failed"
            target.last_error = result.error or "unknown error"
            self._log_panel.append_error(f"Submit failed — {target.name}: {target.last_error}")
            self._log_panel.append_lifecycle(f"===== Submit FAILED: {target.name} =====")
            # In a batch, a failure on one shot shouldn't nuke the
            # whole queue — warn the user and keep going. For a single
            # shot, surface the error as a blocking dialog.
            if len(self._queue) <= 1:
                QMessageBox.critical(
                    self,
                    f"Submit failed — {target.name}",
                    result.error or "unknown error",
                )
            else:
                self.statusBar().showMessage(f"'{target.name}' failed — continuing batch", 8_000)
        self._registry.notify_updated(target)

        # Pop the head of the queue (whichever shot just finished) and
        # move on. This is resilient to a shot dropping out mid-batch
        # (user un-queues it, or we add cancellation later).
        if self._queue and self._queue[0] is target:
            self._queue.pop(0)
        self._start_next_in_queue()

    def _on_reveal_clicked(self) -> None:
        cur = self._registry.current()
        if cur is None or cur.last_sidecar_dir is None:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cur.last_sidecar_dir)))


def _build_cuda_banner(state: CudaState, parent: QMainWindow) -> QWidget | None:
    """Return an amber warning strip when CUDA isn't available, or
    None when everything's OK (banner becomes invisible by not being
    added to the layout at all)."""
    if state.available:
        return None
    banner = QFrame(parent)
    banner.setFrameShape(QFrame.Shape.NoFrame)
    banner.setStyleSheet("background: #3a2a15; border-bottom: 1px solid #6a4820;")
    banner.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    layout = QHBoxLayout(banner)
    layout.setContentsMargins(12, 6, 12, 6)

    icon = QLabel("⚠")
    icon.setStyleSheet("color: #ffb94a; font-size: 14pt;")
    layout.addWidget(icon)

    msg = QLabel(
        f"<b>No CUDA GPU detected</b> &nbsp;—&nbsp; neural passes will fail. "
        f"<i>{state.torch_version}</i>"
    )
    msg.setStyleSheet("color: #ffdca0;")
    msg.setTextFormat(Qt.TextFormat.RichText)
    layout.addWidget(msg)
    layout.addStretch()

    details_btn = QPushButton("Details / How to fix")
    details_btn.clicked.connect(
        lambda: QMessageBox.information(parent, "CUDA preflight", state.advisory)
    )
    layout.addWidget(details_btn)
    return banner


def _collect_nc_entries(shots: list[ShotState]) -> list[tuple[str, str, str]]:
    """Return `(shot_name, model_label, license_tag)` tuples for every
    enabled non-commercial model across the batch. Stable order so the
    confirmation dialog reads deterministically."""
    entries: list[tuple[str, str, str]] = []
    for shot in shots:
        for entry in has_noncommercial(shot.enabled_models):
            entries.append((shot.name, entry.label, entry.license_tag))
    return entries


def _confirm_nc_consent(parent: QMainWindow, entries: list[tuple[str, str, str]]) -> bool:
    """Single per-submit confirmation dialog for non-commercial use.

    We can't enforce the CC-BY-NC-4.0 terms technically — they govern
    the *outputs* of the model. The dialog makes the legal implication
    visible so the user explicitly takes responsibility before the
    pipeline starts producing deliverables.
    """
    lines = [f"  • {shot} — {label}  ({tag})" for shot, label, tag in entries]
    msg = (
        "The following model(s) you've enabled are distributed under "
        "non-commercial licenses (e.g. CC-BY-NC-4.0). Outputs from "
        "these models cannot be used for commercial work regardless "
        "of what licence this tool itself ships under.\n\n"
        + "\n".join(lines)
        + "\n\nProceed with non-commercial use only?"
    )
    reply = QMessageBox.question(
        parent,
        "Non-commercial models — confirm",
        msg,
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        QMessageBox.StandardButton.Cancel,
    )
    return reply == QMessageBox.StandardButton.Ok


# Silence ruff's unused-import noise when we prune widgets in a future iteration.
_ = QWidget

__all__ = ["MainWindow"]
