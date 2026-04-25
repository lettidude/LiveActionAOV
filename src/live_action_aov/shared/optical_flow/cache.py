# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Flow cache shared between the flow pass and the temporal smoother.

Design goal (spec §1.2 + design §9): flow is the keystone intermediate that
every temporally-aware downstream consumer reads from. It should be computed
once per shot and read many times. The cache is keyed by
`(shot_id, frame, direction)` and stores plain numpy arrays of shape
`(2, H, W)` (x, y components in pixels at plate resolution).

Phase 1 keeps everything in memory. A future phase can add on-disk spill
using `FlowCache(max_frames_in_memory=N, spill_dir=...)` without changing
the public API — callers still just call `get()` / `put()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Direction = Literal["forward", "backward"]


@dataclass(frozen=True)
class FlowKey:
    shot_id: str
    frame: int
    direction: Direction


class FlowCache:
    """In-memory store of per-frame flow arrays.

    Keys: `(shot_id, frame, direction)`. Values: `np.ndarray(2, H, W) float32`
    in pixel units at plate resolution.

    Conventions:
    - `direction="forward"` at frame `f` = flow from frame `f` to `f+1`
    - `direction="backward"` at frame `f` = flow from frame `f` to `f-1`

    The smoother reads `forward` at `f-1` to warp frame `f-1` into frame `f`.
    """

    def __init__(self) -> None:
        self._store: dict[FlowKey, np.ndarray] = {}

    def put(self, shot_id: str, frame: int, direction: Direction, flow: np.ndarray) -> None:
        if flow.ndim != 3 or flow.shape[0] != 2:
            raise ValueError(f"FlowCache expects (2, H, W), got shape {flow.shape}")
        self._store[FlowKey(shot_id, frame, direction)] = flow.astype(np.float32, copy=False)

    def get(self, shot_id: str, frame: int, direction: Direction) -> np.ndarray | None:
        """Return the cached flow, or None if missing (e.g. clip endpoint)."""
        return self._store.get(FlowKey(shot_id, frame, direction))

    def has(self, shot_id: str, frame: int, direction: Direction) -> bool:
        return FlowKey(shot_id, frame, direction) in self._store

    def frames(self, shot_id: str, direction: Direction) -> list[int]:
        return sorted(
            k.frame for k in self._store if k.shot_id == shot_id and k.direction == direction
        )

    def clear(self, shot_id: str | None = None) -> None:
        if shot_id is None:
            self._store.clear()
            return
        for k in [k for k in self._store if k.shot_id == shot_id]:
            del self._store[k]

    def __len__(self) -> int:
        return len(self._store)


__all__ = ["Direction", "FlowCache", "FlowKey"]
