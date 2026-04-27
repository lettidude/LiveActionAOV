# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Cooperative cancellation token shared across executor + UI threads.

The token is a thin `threading.Event` wrapper. Producers (the GUI's
Cancel button, the CLI's SIGINT handler) flip the event; consumers
(the executor between passes, the sidecar-write loop, future per-pass
inner loops) call `raise_if_cancelled()` at safe checkpoints.

Why an explicit token instead of `KeyboardInterrupt` everywhere:

- The executor runs off the UI thread in the GUI, so a Qt signal
  needs *something* threadsafe to flip. `threading.Event` is exactly
  that.
- `KeyboardInterrupt` is a process-wide signal; cancelling one shot
  in a multi-shot batch is not a process-wide concern.
- Tests can construct a token and trigger it deterministically
  without poking at signal handlers.

What this token cannot do: interrupt a torch inference call mid-step.
A `DepthCrafter.infer()` that's halfway through a 50-frame window has
the GIL in C++ land — the executor can only re-check the token *after*
the call returns. That's why we check between passes and between
sidecar writes; mid-pass cancel would need cooperation from each
pass's inner loop and is out of scope for the first cancel pass.
"""

from __future__ import annotations

import threading
from typing import Final


class CancelledError(RuntimeError):
    """Raised by `CancelToken.raise_if_cancelled()` when the token is set.

    Distinct from `concurrent.futures.CancelledError` /
    `asyncio.CancelledError` so the executor's broad `except Exception`
    can still observe it without needing to import either of those
    modules. We re-raise it from the executor's outer try/except so
    callers (GUI worker, CLI handler) can branch on it explicitly.
    """


_DEFAULT_REASON: Final[str] = "Cancelled"


class CancelToken:
    """Threadsafe one-shot cancellation flag.

    Construct one per submit() call. Producers flip it via `cancel()`;
    consumers poll via `is_cancelled()` or `raise_if_cancelled()`.

    A token cannot be reset — a single submit either runs to completion
    or gets cancelled; cancelled tokens stay cancelled. Construct a new
    token for a new run.
    """

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str = _DEFAULT_REASON

    def cancel(self, reason: str = _DEFAULT_REASON) -> None:
        """Flip the token. Idempotent; second call keeps the first reason."""
        if not self._event.is_set():
            self._reason = reason
            self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        """Raise `CancelledError` iff the token is set.

        Call this at the safe checkpoints the executor walks through
        (between passes, before each sidecar write). Cheap — a single
        `Event.is_set()` is a memory-barrier-free read.
        """
        if self._event.is_set():
            raise CancelledError(self._reason)


__all__ = ["CancelToken", "CancelledError"]
