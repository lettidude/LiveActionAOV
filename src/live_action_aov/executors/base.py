# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Executor ABC.

An Executor takes a Job and runs it. Subclasses differ in *where* the work
happens (in-process, subprocess, farm). They share the Job shape; the
farm-shaped fields on Job (`priority`, `pool`, `chunk_size`, `gpu_affinity`,
`max_retries`, `timeout_minutes`) are a hint to the executor — LocalExecutor
ignores most of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from live_action_aov.core.job import Job


class Executor(ABC):
    """Run a Job and return it updated with status + sidecar paths."""

    name: str = "executor"

    @abstractmethod
    def submit(self, job: Job) -> Job:
        """Execute the job (synchronously for Local, asynchronously for farm
        variants). Must return the job with `shot.status` updated and
        `shot.sidecars` populated on success."""


__all__ = ["Executor"]
