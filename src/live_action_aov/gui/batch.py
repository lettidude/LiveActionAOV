"""Batch submission — iterate shots, delegate each to the local executor."""

from __future__ import annotations

import logging
from typing import Any

from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.job import Job, Shot
from live_action_aov.executors.local import LocalExecutor
from live_action_aov.executors.report import SubmitReport

_log = logging.getLogger("live_action_aov.batch")


class BatchRunner:
    """Run a list of shots sequentially through the local executor.

    The GUI builds a `Job` per shot (passes + post + colourspace are
    shared; only the shot changes) and hands the list here. We loop,
    delegating each to `LocalExecutor.submit`, and surface progress
    through the same `report(frac, label)` callback the executor uses.

    Error policy (``on_error``)
    ---------------------------
    A single bad shot — a missing frame, an RGBA plate a model chokes
    on, a transient CUDA OOM — must not take down an unattended 40-shot
    weekend batch. With the default ``on_error="continue"`` a shot that
    fails is logged, recorded as a ``status="failed"`` `SubmitReport`,
    and the batch moves on to the next shot. Pass ``on_error="stop"`` to
    restore the old fail-fast behaviour.

    Cancellation is never an "error": a `CancelledError` (the user hit
    Cancel) always aborts the whole batch regardless of ``on_error``.

    Returns one `SubmitReport` per shot, in submission order, so the
    caller can render an end-of-batch summary ("38 ok, 2 failed")
    instead of the failures vanishing into the log. Previously this
    returned ``None`` and the first failure aborted the run — see the
    IT bug report (2026-05-12) where an RGBA plate on shot N would have
    killed every shot after it.
    """

    def __init__(self, executor: LocalExecutor | None = None) -> None:
        self._executor = executor or LocalExecutor()

    def run(
        self,
        jobs: list[tuple[Job, Shot]],
        report: Any,
        cancel: CancelToken | None = None,
        on_error: str = "continue",
    ) -> list[SubmitReport]:
        reports: list[SubmitReport] = []
        for job, shot in jobs:
            try:
                result = self._executor.submit(job, report, shot=shot)
                # `submit` returns a SubmitReport; tolerate a None return
                # (test stubs / older callers) without aborting the batch.
                if isinstance(result, SubmitReport):
                    reports.append(result)
                else:
                    reports.append(SubmitReport(shot_name=shot.name, status="ok"))
            except CancelledError:
                # Deliberate user action — propagate and abort the batch.
                raise
            except Exception as exc:  # noqa: BLE001 — batch must survive any pass-level failure
                _log.exception("Shot %s failed in batch", shot.name)
                reports.append(
                    SubmitReport(
                        shot_name=shot.name,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                if on_error == "stop":
                    raise

            # Cancel checkpoint between shots (success path — `submit`
            # itself only checks mid-shot).
            if cancel is not None:
                cancel.raise_if_cancelled()
        return reports


__all__ = ["BatchRunner"]
