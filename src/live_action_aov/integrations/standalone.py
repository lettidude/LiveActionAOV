"""StandaloneAdapter — no external pipeline, default for v1."""

from __future__ import annotations

from live_action_aov.core.job import Shot
from live_action_aov.integrations.base import PipelineAdapter


class StandaloneAdapter(PipelineAdapter):
    name = "standalone"

    def attach_ids(self, shot: Shot) -> Shot:
        return shot

    def publish(self, shot: Shot) -> None:
        return


__all__ = ["StandaloneAdapter"]
