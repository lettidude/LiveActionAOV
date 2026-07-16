"""Background submit worker — runs `LocalExecutor` off the UI thread.

The executor's `submit()` reads EXRs, loads torch models, and
iterates frames — blocking the UI for multiple seconds or minutes.
This worker hosts that call in a `QRunnable` and emits Qt signals for
start / finish / fail so the main window can update status + progress
without freezing.

Cancellation: each `submit()` call constructs a fresh `CancelToken`
and stashes it on the worker. The main window's Cancel button calls
`worker.cancel()`, which flips the token; the executor then raises
`CancelledError` at its next safe checkpoint (between passes / before
each sidecar write). The result comes back with `cancelled=True` so
the UI can render "Cancelled" rather than a generic failure dialog.

M2 progress is indeterminate (a busy spinner while `submit` runs).
Per-frame progress wants the executor to gain a `progress_callback`
hook; that lands with M3's polish pass alongside the histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.executors.local import LocalExecutor
from live_action_aov.gui.pass_catalog import expand_models
from live_action_aov.gui.shot_state import ClickInstance, ShotState


@dataclass(frozen=True)
class SubmitResult:
    shot_state_id: int
    success: bool
    sidecar_dir: Path | None
    error: str | None
    # True iff the run ended because the user clicked Cancel (or the
    # token was flipped externally). The MainWindow uses this to
    # branch on cancellation vs a real failure — partial sidecars on
    # disk after a cancel are expected, not an error condition.
    cancelled: bool = False


class SubmitWorker(QObject):
    """One-shot async submit driver.

    Usage:
        worker = SubmitWorker()
        worker.finished.connect(main_window.on_submit_finished)
        worker.submit(shot_state)   # returns immediately
        worker.cancel()             # request abort at next checkpoint
    """

    finished = Signal(object)  # SubmitResult
    progress = Signal(float, str)  # fraction 0..1, label

    def __init__(self) -> None:
        super().__init__()
        self._pool = QThreadPool.globalInstance()
        # The token in flight, or None when nothing is running. Set
        # at submit() entry, cleared by the task on finish so a stale
        # cancel() call after a clean finish is a harmless no-op.
        self._cancel: CancelToken | None = None

    def submit(self, shot_state: ShotState) -> None:
        passes = _build_pass_configs(shot_state)
        job = Job(
            shot=_shot_state_to_core_shot(shot_state),
            passes=passes,
        )
        cancel = CancelToken()
        self._cancel = cancel
        task = _SubmitTask(
            job=job,
            shot_state_id=id(shot_state),
            cancel=cancel,
            emit_result=self._emit_result,
            emit_progress=self._emit_progress,
            on_finished=self._on_task_finished,
        )
        self._pool.start(task)

    def cancel(self) -> None:
        """Request cancellation of the in-flight submit, if any.

        Safe to call multiple times — `CancelToken.cancel()` is
        idempotent. Safe to call when nothing is running — no-op.
        Cancellation takes effect at the executor's next safe
        checkpoint (between passes, before each sidecar write); a
        torch inference call that's mid-flight has to complete first.
        """
        if self._cancel is not None:
            self._cancel.cancel("Cancelled by user")

    def _emit_result(self, result: SubmitResult) -> None:
        self.finished.emit(result)

    def _emit_progress(self, fraction: float, label: str) -> None:
        # Signal emit is thread-safe; Qt auto-queues across threads.
        self.progress.emit(float(fraction), str(label))

    def _on_task_finished(self) -> None:
        # Worker is one-shot per submit; clearing the token here means
        # a Cancel click that arrives after the executor naturally
        # finished does nothing (rather than poisoning the next submit).
        self._cancel = None


class _SubmitTask(QRunnable):
    def __init__(
        self,
        *,
        job: Job,
        shot_state_id: int,
        cancel: CancelToken,
        emit_result: Any,
        emit_progress: Any,
        on_finished: Any,
    ) -> None:
        super().__init__()
        self.job = job
        self.shot_state_id = shot_state_id
        self._cancel = cancel
        self._emit = emit_result
        self._emit_progress = emit_progress
        self._on_finished = on_finished

    def run(self) -> None:
        try:
            executor = LocalExecutor()
            executor.submit(
                self.job,
                progress_callback=self._emit_progress,
                cancel=self._cancel,
            )
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
        except CancelledError as e:
            # User-initiated abort — surface as a non-success result
            # but flag it so the MainWindow doesn't pop a "Submit
            # failed" error dialog.
            self._emit(
                SubmitResult(
                    shot_state_id=self.shot_state_id,
                    success=False,
                    sidecar_dir=None,
                    error=str(e) or "Cancelled",
                    cancelled=True,
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
        finally:
            # Clear the worker's token reference even if emit_result
            # blew up for some reason — we never want a stale token
            # carried across to the next submit.
            try:
                self._on_finished()
            except Exception:
                pass


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
        passes_enabled=list(state.enabled_models),
        output_dir=output_dir,
        proxy_long_edge=state.proxy_long_edge,
        delivery_compression=state.delivery_compression,
        delivery_dtype=state.delivery_dtype,
    )


def _parse_concepts(raw: str) -> list[str]:
    """Split the GUI's comma-separated concepts field into a clean list.

    Comma (not whitespace) is the separator so multi-word concepts like
    "red car" survive. Blanks are dropped; "" -> [] (use pass defaults).
    """
    return [c.strip() for c in raw.split(",") if c.strip()]


def _serialize_click_instances(
    instances: list[ClickInstance],
    ref_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Project the GUI's ClickInstance objects into plain JSON-friendly dicts
    for the `sam3_matte` pass's `prompt_instances` param:
    {name, seed_frame, points: [[x, y, label], ...], box: [x1, y1, x2, y2] |
    None, ref_size: [w, h]}.

    `ref_size` is the plate resolution the clicks were captured at — the pass
    rescales the coordinates onto whatever frames it actually reads, so proxy
    mode can't shift the clicks. Instances with neither points nor a box are
    dropped — there is nothing to seed the tracker with.
    """
    out: list[dict[str, Any]] = []
    for inst in instances:
        pts = [[float(x), float(y), int(lbl)] for (x, y, lbl) in inst.points]
        box = [float(v) for v in inst.box] if inst.box is not None else None
        if not pts and box is None:
            continue
        out.append(
            {
                "name": inst.name,
                "seed_frame": int(inst.seed_frame),
                "points": pts,
                "box": box,
                "ref_size": [int(ref_size[0]), int(ref_size[1])],
            }
        )
    return out


def _sam3_matte_params(state: ShotState) -> dict[str, Any]:
    """Assemble the `sam3_matte` param dict from the GUI: text concepts and/or
    interactive click prompts. Both may be present (named clicks alongside
    concept categories). An empty dict means the pass uses its own defaults."""
    params: dict[str, Any] = {}
    concepts = _parse_concepts(state.sam3_concepts)
    if concepts:
        params["concepts"] = concepts
    clicks = _serialize_click_instances(state.click_instances, state.resolution)
    if clicks:
        params["prompt_instances"] = clicks
    return params


def _build_pass_configs(state: ShotState) -> list[PassConfig]:
    """Resolve the ShotState's enabled model keys into concrete
    PassConfig list for the Job.

    The GUI speaks in catalog keys (one key per user-visible
    checkbox); the executor speaks in plugin names. `expand_models`
    bridges the two — trivially for single-plugin models, via a
    preset pair for matte combos (sam3 + refiner).

    SAM 3 `concepts` (text) and `prompt_instances` (interactive clicks) are
    injected onto the `sam3_matte` pass when set; empty falls back to defaults.
    """
    plugin_names = expand_models(state.enabled_models)
    sam3_params = _sam3_matte_params(state)
    out: list[PassConfig] = []
    for n in plugin_names:
        if n == "sam3_matte" and sam3_params:
            out.append(PassConfig(name=n, params=sam3_params))
        elif n in _REFINER_PLUGINS:
            # Per-shot refiner options: all-masks soft mode + weight choice.
            rp: dict[str, object] = {}
            if state.refine_all_masks:
                rp["refine_all_masks"] = True
            # Weight choice only applies to the BiRefNet-family pass — RVM /
            # MatAnyone have their own model_id defaults that must not be
            # overwritten with a BiRefNet HF id.
            if state.refiner_model and n == "birefnet_refiner":
                rp["model_id"] = state.refiner_model
            # Compare mode: several refiners run side by side — each writes
            # its own layer so they don't overwrite each other in the EXR.
            if "sam3_all_refiners" in state.enabled_models:
                rp["channel_suffix"] = _COMPARE_SUFFIX.get(n, "")
            out.append(PassConfig(name=n, params=rp) if rp else PassConfig(name=n))
        else:
            out.append(PassConfig(name=n))
    return out


#: Soft-matte refiner plugins that accept the `refine_all_masks` param.
_REFINER_PLUGINS = frozenset(
    {"rvm_refiner", "birefnet_refiner", "vitmatte_refiner", "matanyone2"}
)

#: Layer suffix per refiner in compare mode (sam3_all_refiners) — each
#: engine's output lands in matte_<engine>.* / mask_<engine>.<name>.
_COMPARE_SUFFIX = {
    "rvm_refiner": "_rvm",
    "birefnet_refiner": "_birefnet",
    "vitmatte_refiner": "_vitmatte",
    "matanyone2": "_matanyone",
}


__all__ = ["SubmitResult", "SubmitWorker"]
