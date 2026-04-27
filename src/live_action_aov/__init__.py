# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""LiveActionAOV — AI-driven VFX plate preprocessor.

Public surface (stable across the Phase 0 → Phase 6 evolution):

- `Job`, `Shot`, `PassConfig`, `Task` — core data model
- `run` — one-call entry point for executing a job in-process
- `__version__`

GUI and CLI are thin consumers of this library; the CLI/library is the source
of truth (design §2, principle 5).
"""

from __future__ import annotations

__version__ = "0.1.0"

from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.job import Job, PassConfig, Shot, Task


def run(job: Job, cancel: CancelToken | None = None) -> Job:
    """Execute a Job with the default (local) executor.

    Returns the updated Job (status + sidecar paths filled in).

    `cancel` is an optional cooperative cancellation token (see
    `core.cancel`). When set externally, the executor raises
    `CancelledError` at the next safe checkpoint instead of running
    to completion. Default `None` keeps the historic non-cancellable
    behaviour.
    """
    # Lazy import so `import live_action_aov` stays light.
    from live_action_aov.executors.local import LocalExecutor

    return LocalExecutor().submit(job, cancel=cancel)


__all__ = [
    "CancelToken",
    "CancelledError",
    "Job",
    "PassConfig",
    "Shot",
    "Task",
    "__version__",
    "run",
]
