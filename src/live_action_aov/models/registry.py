"""ModelRegistry — lazy load, reference count, VRAM aware.

Phase 0 ships a minimal implementation: registration, lazy `get()`,
reference counting, explicit `release()`. Actual checkpoint downloading via
`huggingface_hub` is wired in Phase 1 (RAFT) once we have a concrete model
to exercise.

Thread-safety is not a v1 concern — the LocalExecutor is single-threaded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from platformdirs import user_cache_dir


@dataclass
class _Entry:
    loader: Callable[[], Any]
    instance: Any = None
    refcount: int = 0
    bytes_resident: int = 0


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, model_id: str, loader: Callable[[], Any]) -> None:
        """Register a loader for `model_id`. Loader is called on first `get()`."""
        self._entries[model_id] = _Entry(loader=loader)

    def get(self, model_id: str) -> Any:
        """Return the loaded model, loading it if necessary."""
        entry = self._entries.get(model_id)
        if entry is None:
            raise KeyError(
                f"No loader registered for model '{model_id}'. Known: {sorted(self._entries)}"
            )
        if entry.instance is None:
            entry.instance = entry.loader()
        entry.refcount += 1
        return entry.instance

    def release(self, model_id: str) -> None:
        """Drop a reference; if count hits zero, the registry *may* unload.

        In v1 we keep the instance cached. v2 wires VRAM pressure → unload.
        """
        entry = self._entries.get(model_id)
        if entry is None:
            return
        entry.refcount = max(0, entry.refcount - 1)

    def unload(self, model_id: str) -> None:
        """Force-unload a model regardless of refcount."""
        entry = self._entries.get(model_id)
        if entry is None:
            return
        entry.instance = None
        entry.refcount = 0

    def list(self) -> list[str]:
        return sorted(self._entries)


def cache_dir() -> str:
    """Return the user-level model cache directory for LiveActionAOV."""
    return user_cache_dir("live-action-aov", "LiveActionAOV")


_model_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


__all__ = ["ModelRegistry", "cache_dir", "get_model_registry"]
