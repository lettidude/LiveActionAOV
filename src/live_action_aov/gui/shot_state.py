"""Shot-level state model for the prep GUI.

One `ShotState` per plate folder the user has loaded into the session.
Holds everything the three panels need to render consistently:

- identity    (name, folder, pattern, frame range, resolution)
- detected    (colorspace auto-detect result + provenance)
- override    (user's explicit colorspace override; `None` = use auto)
- transform   (current display-transform knobs — exposure EV etc.)
- ui state    (current frame index being previewed, view mode radio)

Panels mutate this via the `update()` convenience and connect to the
`changed` signal to refresh. The MainWindow owns the list of states
and the "currently selected" pointer.

This is GUI-local — deliberately NOT the core `Shot` pydantic model
from `core.job`, because the GUI's state is a superset (UI radios,
scrub position, auto-detect provenance, …) and the `Shot` must stay
serialization-focused for the CLI handoff. When the user clicks
Submit in a future milestone, `ShotState.to_shot()` will project the
relevant bits into a core `Shot` for the executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal

from live_action_aov.io.colorspace_detect import DetectedColorspace

ViewMode = Literal["original", "transformed", "compare"]


@dataclass
class ShotState:
    """Per-shot GUI state. See module docstring."""

    name: str
    folder: Path
    sequence_pattern: str
    frame_range: tuple[int, int]
    resolution: tuple[int, int]
    pixel_aspect: float = 1.0

    # Auto-detect result; `override` takes precedence when set.
    detected: DetectedColorspace | None = None
    override: str | None = None

    # Preview state
    current_frame: int = 0
    view_mode: ViewMode = "transformed"
    exposure_ev: float = 0.0

    # Passes the user has toggled on (read-only surface in M1).
    enabled_passes: list[str] = field(default_factory=list)

    def effective_colorspace(self) -> str:
        """Return the colorspace the preview should use."""
        if self.override is not None and self.override != "auto":
            return self.override
        if self.detected is not None:
            return self.detected.detected
        return "lin_rec709"

    def colorspace_label(self) -> str:
        """One-line UI string for the inspector's current-colorspace
        status. Surfaces the provenance verbatim — that's the whole
        point of this work."""
        if self.override is not None and self.override != "auto":
            return f"override: {self.override}"
        if self.detected is None:
            return "auto: (not yet probed)"
        return f"auto: {self.detected.detected} ({self.detected.reason})"


class ShotRegistry(QObject):
    """Session-level list of ShotStates + current selection.

    Emitted signals let the three panels stay in sync without
    back-references: a panel listens for `shot_added` or
    `current_changed`, pulls the state, refreshes itself.
    """

    shot_added = Signal(object)  # ShotState
    shot_removed = Signal(object)  # ShotState
    current_changed = Signal(object)  # ShotState | None
    shot_updated = Signal(object)  # ShotState — colorspace override / view / etc.

    def __init__(self) -> None:
        super().__init__()
        self._shots: list[ShotState] = []
        self._current: ShotState | None = None

    def shots(self) -> list[ShotState]:
        return list(self._shots)

    def current(self) -> ShotState | None:
        return self._current

    def add(self, shot: ShotState) -> None:
        self._shots.append(shot)
        self.shot_added.emit(shot)
        if self._current is None:
            self.set_current(shot)

    def remove(self, shot: ShotState) -> None:
        if shot not in self._shots:
            return
        self._shots.remove(shot)
        if self._current is shot:
            self._current = self._shots[0] if self._shots else None
            self.current_changed.emit(self._current)
        self.shot_removed.emit(shot)

    def set_current(self, shot: ShotState | None) -> None:
        if shot is not None and shot not in self._shots:
            raise ValueError("Shot is not registered.")
        if shot is self._current:
            return
        self._current = shot
        self.current_changed.emit(shot)

    def notify_updated(self, shot: ShotState) -> None:
        """Panels that mutated a shot call this so others refresh."""
        if shot in self._shots:
            self.shot_updated.emit(shot)


__all__ = ["ShotRegistry", "ShotState", "ViewMode"]
