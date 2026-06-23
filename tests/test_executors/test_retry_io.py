"""Sidecar-write retry — survive a transient NAS/share blip mid-job."""

from __future__ import annotations

import pytest

from live_action_aov.executors.local import _retry_io


def test_recovers_on_transient_oserror() -> None:
    calls: list[int] = []

    def fn() -> str:
        calls.append(1)
        if len(calls) < 2:
            raise OSError("NAS connection blip")
        return "ok"

    assert _retry_io(fn, what="write", attempts=3, base_delay=0.0) == "ok"
    assert len(calls) == 2  # failed once, succeeded on the retry


def test_raises_after_exhausting_attempts() -> None:
    def fn() -> None:
        raise OSError("share gone")

    with pytest.raises(OSError, match="share gone"):
        _retry_io(fn, what="write", attempts=3, base_delay=0.0)


def test_does_not_retry_non_io_errors() -> None:
    # A real bug (not transient I/O) must surface immediately, not be masked.
    calls: list[int] = []

    def fn() -> None:
        calls.append(1)
        raise ValueError("logic error")

    with pytest.raises(ValueError):
        _retry_io(fn, what="write", attempts=3, base_delay=0.0)
    assert len(calls) == 1
