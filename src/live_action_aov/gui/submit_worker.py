"""Background submit worker — runs `LocalExecutor` off the UI thread.

The executor's `submit()` reads EXRs, loads torch models, and
iterates frames — blocking the UI for multiple seconds or minutes.
This worker hosts that call in a `QRunnable` and emits Qt signals for
start / finish / fail so the main window can update status + progress
without freezing.

M2 progress is indeterminate (a busy spinner while `submit` runs).
Per-frame progress wants the executor to gain a `progress_callback`
hook; that lands with M3's polish pass alongside the histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.executors.local import LocalExecutor
from live_action_aov.gui.shot_state import ShotState


@dataclass(frozen=True)
class SubmitResult:
    shot_state_id: int
    success: bool
    sidecar_dir: Path | None
    error: str | None


class SubmitWorker(QObject):
    """One-shot async submit driver.

    Usage:
        worker = SubmitWorker()
        worker.finished.connect(main_window.on_submit_finished)
        worker.submit(shot_state)   # returns immediately
    """

    finished = Signal(object)  # SubmitResult
    progress = Signal(float, str)  # fraction 0..1, label

    def __init__(self) -> None:
        super().__init__()
        self._pool = QThreadPool.globalInstance()

    def submit(self, shot_state: ShotState) -> None:
        passes = _build_pass_configs(shot_state)
        job = Job(
            shot=_shot_state_to_core_shot(shot_state),
            passes=passes,
        )
        task = _SubmitTask(
            job=job,
            shot_state_id=id(shot_state),
            emit_result=self._emit_result,
            emit_progress=self._emit_progress,
        )
        self._pool.start(task)

    def _emit_result(self, result: SubmitResult) -> None:
        self.finished.emit(result)

    def _emit_progress(self, fraction: float, label: str) -> None:
        # Signal emit is thread-safe; Qt auto-queues across threads.
        self.progress.emit(float(fraction), str(label))


class _SubmitTask(QRunnable):
    def __init__(
        self,
        *,
        job: Job,
        shot_state_id: int,
        emit_result: Any,
        emit_progress: Any,
    ) -> None:
        super().__init__()
        self.job = job
        self.shot_state_id = shot_state_id
        self._emit = emit_result
        self._emit_progress = emit_progress

    def run(self) -> None:
        try:
            executor = LocalExecutor()
            executor.submit(self.job, progress_callback=self._emit_progress)
            # After submit, the shot's sidecar path is populated. Surface
            # the containing folder so the UI can offer "Reveal in
            # Explorer" without inspecting Shot internals.
            sidecar = self.job.shot.sidecars.get("utility")
            sidecar_dir = sidecar.parent if sidecar else self.job.shot.folder
            self._emit(
                SubmitResult(
                    shot_state_id=self.shot_state_id,
                    success=True,
                    sidecar_dir=sidecar_dir,
                    error=None,
                )
            )
        except Exception as e:
            self._emit(
                SubmitResult(
                    shot_state_id=self.shot_state_id,
                    success=False,
                    sidecar_dir=None,
                    error=f"{type(e).__name__}: {e}",
                )
            )


def _shot_state_to_core_shot(state: ShotState) -> Shot:
    """Project the GUI's `ShotState` into the pydantic `Shot` the
    executor expects. GUI-only fields (view mode, detected provenance,
    status) are dropped — they're prep-time only."""
    transform = DisplayTransformParams(
        input_colorspace=state.effective_colorspace(),
        # Seed analyze_clip with the user's current exposure rather
        # than re-sampling — the GUI's slider value IS the user's
        # declared intent, and surprising them by overriding it is
        # worse than a slightly-off auto at submit time.
        manual_exposure_ev=float(state.exposure_ev),
        tonemap="agx",
        output_eotf="srgb",
        clamp=True,
    )
    # Only pin `output_dir` when the user picked something other than
    # inplace. Passing the plate folder explicitly would lose the "None
    # means default" semantic the executor already honours.
    resolved_out = state.resolve_output_dir()
    output_dir = resolved_out if resolved_out != state.folder else None
    return Shot(
        name=state.name,
        folder=state.folder,
        sequence_pattern=state.sequence_pattern,
        frame_range=state.frame_range,
        resolution=state.resolution,
        pixel_aspect=state.pixel_aspect,
        colorspace=state.effective_colorspace(),
        transform=transform,
        apply_display_transform=True,
        passes_enabled=list(state.enabled_passes),
        output_dir=output_dir,
    )


def _build_pass_configs(state: ShotState) -> list[PassConfig]:
    """Resolve the ShotState's enabled semantic names into concrete
    PassConfig list for the Job.

    Mirrors CLI's `_resolve_semantic_passes` — keep the two logics in
    sync when new pass families land. The matte family expands to two
    configs: detector + refiner.
    """
    resolved: list[str] = []
    seen: set[str] = set()
    backends = state.pass_backends
    for name in state.enabled_passes:
        if name == "flow":
            targets = ["flow"]
        elif name == "depth":
            targets = [backends.get("depth", "depth_anything_v2")]
        elif name == "normals":
            targets = [backends.get("normals", "dsine")]
        elif name == "matte":
            targets = [
                backends.get("matte_detector", "sam3_matte"),
                backends.get("matte_refiner", "rvm_refiner"),
            ]
        else:
            # Already a concrete backend name — flow through.
            targets = [name]
        for t in targets:
            if t not in seen:
                resolved.append(t)
                seen.add(t)
    return [PassConfig(name=n) for n in resolved]


__all__ = ["SubmitResult", "SubmitWorker"]
