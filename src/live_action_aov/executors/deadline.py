# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""DeadlineExecutor — stub for v2.

Declared in `pyproject.toml` and discovered via entry points so the CLI can
list the backend (and emit a clear error) without dragging the Deadline
Python API into v1 dependencies.
"""

from __future__ import annotations

from live_action_aov.core.job import Job
from live_action_aov.executors.base import Executor


class DeadlineExecutorStub(Executor):
    name = "deadline"

    def submit(self, job: Job) -> Job:  # pragma: no cover — stub
        raise NotImplementedError(
            "The Deadline executor lands in v2. In v1, use the default local "
            "executor (omit --executor or pass --executor local)."
        )


__all__ = ["DeadlineExecutorStub"]
