# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""PipelineAdapter ABC.

A pipeline adapter translates between our `Shot` / `Job` model and an
external pipeline manager's concepts (Prism task, ShotGrid version,
OpenPype product). v1 ships the no-op `StandaloneAdapter`; Prism / ShotGrid
/ OpenPype adapters are stubs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from live_action_aov.core.job import Shot


class PipelineAdapter(ABC):
    """External pipeline adapter."""

    name: str = "base"

    @abstractmethod
    def attach_ids(self, shot: Shot) -> Shot:
        """Populate `prism_task_id` / `shotgrid_version_id` / etc. on `shot`."""

    @abstractmethod
    def publish(self, shot: Shot) -> None:
        """Push the completed shot's sidecars to the external pipeline."""


__all__ = ["PipelineAdapter"]
