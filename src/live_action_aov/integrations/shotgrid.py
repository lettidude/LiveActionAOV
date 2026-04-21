# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""ShotGrid pipeline adapter — stub for v2."""

from __future__ import annotations

from live_action_aov.core.job import Shot
from live_action_aov.integrations.base import PipelineAdapter


class ShotGridAdapterStub(PipelineAdapter):
    name = "shotgrid"

    def attach_ids(self, shot: Shot) -> Shot:  # pragma: no cover — stub
        raise NotImplementedError("ShotGrid integration lands in v2.")

    def publish(self, shot: Shot) -> None:  # pragma: no cover — stub
        raise NotImplementedError("ShotGrid integration lands in v2.")


__all__ = ["ShotGridAdapterStub"]
