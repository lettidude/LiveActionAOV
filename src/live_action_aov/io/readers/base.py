"""ImageSequenceReader ABC (design §14, decision 3).

Discovery goes through the `live_action_aov.io.readers` entry-point group so
additional codecs in v2+ (DPX, MOV, R3D) slot in without core changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class ImageSequenceReader(ABC):
    """Abstract reader for frame-based image sequences."""

    #: Glob-style filename extension this reader claims. Registry uses this
    #: to route `for_path` lookups; overlapping claims are resolved by
    #: registration order.
    extensions: tuple[str, ...] = ()

    def __init__(self, folder: Path, sequence_pattern: str) -> None:
        self.folder = Path(folder)
        self.sequence_pattern = sequence_pattern

    @abstractmethod
    def frame_range(self) -> tuple[int, int]:
        """Return (first_frame, last_frame) inclusive."""

    @abstractmethod
    def resolution(self) -> tuple[int, int]:
        """Return (width, height) of the first frame."""

    @abstractmethod
    def pixel_aspect(self) -> float:
        """Return pixel aspect ratio (1.0 spherical, 2.0 anamorphic, etc.)."""

    @abstractmethod
    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Read one frame. Returns (pixels, attrs).

        `pixels`: (H, W, C) float32 array.
        `attrs`: dict of EXR header attributes.
        """

    def read_range(self, start: int, end: int) -> list[np.ndarray]:
        """Read a contiguous frame range. Default implementation loops."""
        return [self.read_frame(f)[0] for f in range(start, end + 1)]


__all__ = ["ImageSequenceReader"]
