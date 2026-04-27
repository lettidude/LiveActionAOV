"""CancelToken — unit tests for the core cancellation primitive.

Covers the contract every consumer relies on:

- `is_cancelled()` flips False → True after `cancel()`.
- `raise_if_cancelled()` is a no-op before, raises after.
- `cancel()` is idempotent and the FIRST reason wins (so a second
  cancel from a different code path doesn't overwrite the user's
  context).
- The exception type is the project's `CancelledError`, not the
  stdlib `concurrent.futures.CancelledError` — important so the
  executor's `except CancelledError:` arm catches it without an
  ambiguous import.
"""

from __future__ import annotations

import pytest

from live_action_aov.core.cancel import CancelledError, CancelToken


def test_token_starts_uncancelled() -> None:
    token = CancelToken()
    assert token.is_cancelled() is False
    # raise_if_cancelled is the safe no-op the executor checkpoints rely on.
    token.raise_if_cancelled()


def test_cancel_flips_state_and_raises() -> None:
    token = CancelToken()
    token.cancel("user clicked Cancel")
    assert token.is_cancelled() is True
    assert token.reason() == "user clicked Cancel"

    with pytest.raises(CancelledError) as excinfo:
        token.raise_if_cancelled()
    assert "user clicked Cancel" in str(excinfo.value)


def test_cancel_is_idempotent_and_first_reason_wins() -> None:
    """Two concurrent producers (e.g. SIGINT + GUI button) shouldn't
    fight for the reason field — whichever lands first describes the
    cancel."""
    token = CancelToken()
    token.cancel("SIGINT")
    token.cancel("GUI button")
    assert token.reason() == "SIGINT"


def test_cancelled_error_is_runtime_error() -> None:
    """`CancelledError` deliberately subclasses `RuntimeError` so the
    executor's general `except Exception` arms still observe it before
    the dedicated `except CancelledError` arm catches it. Guard the
    inheritance chain so a future refactor doesn't accidentally break
    the catch order."""
    assert issubclass(CancelledError, RuntimeError)
