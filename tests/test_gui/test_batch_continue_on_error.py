# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""BatchRunner error policy — a single bad shot must not kill the batch.

Regression guard for the IT report (2026-05-12): an RGBA plate failed
one shot and, under the old fail-fast loop, would have aborted every
shot after it in an unattended batch.
"""

from __future__ import annotations

import pytest

from live_action_aov.core.cancel import CancelledError
from live_action_aov.executors.report import SubmitReport
from live_action_aov.gui.batch import BatchRunner


class _StubShot:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubJob:
    pass


class _FlakyExecutor:
    """Stub executor that raises for any shot whose name contains 'bad'."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def submit(self, job, report, shot=None):  # noqa: ANN001 - test stub
        self.seen.append(shot.name)
        if "bad" in shot.name:
            raise ValueError(f"boom on {shot.name}")
        return SubmitReport(shot_name=shot.name, status="ok")


def _jobs(names: list[str]) -> list[tuple[_StubJob, _StubShot]]:
    return [(_StubJob(), _StubShot(n)) for n in names]


def _noop_report(frac: float, label: str) -> None:
    pass


def test_batch_continues_past_failed_shot() -> None:
    ex = _FlakyExecutor()
    reports = BatchRunner(executor=ex).run(_jobs(["s1", "bad2", "s3"]), _noop_report)
    # All three shots attempted despite the middle one failing.
    assert ex.seen == ["s1", "bad2", "s3"]
    statuses = {r.shot_name: r.status for r in reports}
    assert statuses == {"s1": "ok", "bad2": "failed", "s3": "ok"}
    bad = next(r for r in reports if r.shot_name == "bad2")
    assert "ValueError" in (bad.error or "")


def test_on_error_stop_restores_fail_fast() -> None:
    ex = _FlakyExecutor()
    with pytest.raises(ValueError):
        BatchRunner(executor=ex).run(_jobs(["s1", "bad2", "s3"]), _noop_report, on_error="stop")
    # s3 is never attempted once bad2 raises.
    assert ex.seen == ["s1", "bad2"]


def test_cancellation_always_aborts() -> None:
    class _Cancel:
        def raise_if_cancelled(self) -> None:
            raise CancelledError()

    ex = _FlakyExecutor()
    with pytest.raises(CancelledError):
        BatchRunner(executor=ex).run(_jobs(["s1", "s2"]), _noop_report, cancel=_Cancel())
    # Cancel fires at the post-submit checkpoint after the first shot.
    assert ex.seen == ["s1"]
