# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Plugin registry — entry-point discovery.

Core has *zero* hardcoded pass imports. At startup we walk the
`live_action_aov.*` entry-point groups and register whatever we find. Built-in
and third-party plugins are identical from the registry's point of view
(design §24).

Tests register a `NoOpPass` via `PluginRegistry.register_pass()` so Phase 0
can run end-to-end without any real inference plugin installed.
"""

from __future__ import annotations

import importlib.metadata as im
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from live_action_aov.core.pass_base import PassType, UtilityPass
    from live_action_aov.executors.base import Executor
    from live_action_aov.integrations.base import PipelineAdapter
    from live_action_aov.io.readers.base import ImageSequenceReader
    from live_action_aov.io.writers.base import SidecarWriter


# Entry point group names (centralized so we have one source of truth).
GROUP_PASSES = "live_action_aov.passes"
GROUP_POST = "live_action_aov.post"
GROUP_EXECUTORS = "live_action_aov.executors"
GROUP_READERS = "live_action_aov.io.readers"
GROUP_WRITERS = "live_action_aov.io.writers"
GROUP_INTEGRATIONS = "live_action_aov.integrations"


class PluginRegistry:
    """Central registry of all plugins discovered via entry points."""

    def __init__(self) -> None:
        self._passes: dict[str, type[UtilityPass]] = {}
        self._post: dict[str, type[Any]] = {}
        self._executors: dict[str, type[Executor]] = {}
        self._readers: dict[str, type[ImageSequenceReader]] = {}
        self._writers: dict[str, type[SidecarWriter]] = {}
        self._integrations: dict[str, type[PipelineAdapter]] = {}
        self._loaded = False

    # --- Discovery ---

    def load_all(self) -> None:
        """Walk all entry points and register everything. Idempotent."""
        if self._loaded:
            return
        self._load_group(GROUP_PASSES, self._passes)
        self._load_group(GROUP_POST, self._post)
        self._load_group(GROUP_EXECUTORS, self._executors)
        self._load_group(GROUP_READERS, self._readers)
        self._load_group(GROUP_WRITERS, self._writers)
        self._load_group(GROUP_INTEGRATIONS, self._integrations)
        self._loaded = True

    @staticmethod
    def _load_group(group: str, target: dict[str, type[Any]]) -> None:
        try:
            eps = im.entry_points(group=group)
        except TypeError:
            # Older importlib.metadata API (pre-3.10-style); shouldn't hit
            # on Python 3.11+, but stay defensive for third-party envs.
            eps = im.entry_points().get(group, [])  # type: ignore[arg-type]
        for ep in eps:
            try:
                cls = ep.load()
            except Exception as e:
                # A broken third-party plugin must not take down the whole
                # registry — log and continue.
                import logging

                logging.getLogger("live_action_aov.registry").warning(
                    "Failed to load entry point %s from group %s: %s", ep.name, group, e
                )
                continue
            target[ep.name] = cls

    # --- Runtime registration (used by tests to inject NoOpPass) ---

    def register_pass(self, name: str, cls: type[UtilityPass]) -> None:
        self._passes[name] = cls

    def register_executor(self, name: str, cls: type[Executor]) -> None:
        self._executors[name] = cls

    def register_reader(self, name: str, cls: type[ImageSequenceReader]) -> None:
        self._readers[name] = cls

    def register_writer(self, name: str, cls: type[SidecarWriter]) -> None:
        self._writers[name] = cls

    def register_integration(self, name: str, cls: type[PipelineAdapter]) -> None:
        self._integrations[name] = cls

    # --- Queries ---

    def list_passes(self) -> list[str]:
        self.load_all()
        return sorted(self._passes)

    def list_by_type(self, pass_type: PassType | str) -> list[str]:
        from live_action_aov.core.pass_base import PassType as _PT

        self.load_all()
        target = pass_type.value if isinstance(pass_type, _PT) else pass_type
        return sorted(
            name
            for name, cls in self._passes.items()
            if getattr(cls, "pass_type", None)
            and getattr(cls.pass_type, "value", cls.pass_type) == target
        )

    def get_pass(self, name: str) -> type[UtilityPass]:
        self.load_all()
        try:
            return self._passes[name]
        except KeyError as e:
            raise KeyError(f"No pass named '{name}'. Available: {self.list_passes()}") from e

    def list_executors(self) -> list[str]:
        self.load_all()
        return sorted(self._executors)

    def get_executor(self, name: str) -> type[Executor]:
        self.load_all()
        return self._executors[name]

    def list_readers(self) -> list[str]:
        self.load_all()
        return sorted(self._readers)

    def get_reader(self, name: str) -> type[ImageSequenceReader]:
        self.load_all()
        return self._readers[name]

    def list_writers(self) -> list[str]:
        self.load_all()
        return sorted(self._writers)

    def get_writer(self, name: str) -> type[SidecarWriter]:
        self.load_all()
        return self._writers[name]

    def list_integrations(self) -> list[str]:
        self.load_all()
        return sorted(self._integrations)

    def get_integration(self, name: str) -> type[PipelineAdapter]:
        self.load_all()
        return self._integrations[name]


# Module-level singleton. Callers do `from ... import registry; registry.list_passes()`.
_registry: PluginRegistry | None = None


def get_registry() -> PluginRegistry:
    """Return the process-wide PluginRegistry (lazy-initialized)."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


__all__ = [
    "GROUP_EXECUTORS",
    "GROUP_INTEGRATIONS",
    "GROUP_PASSES",
    "GROUP_POST",
    "GROUP_READERS",
    "GROUP_WRITERS",
    "PluginRegistry",
    "get_registry",
]
