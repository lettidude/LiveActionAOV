"""EXR reader via OpenImageIO.

Handles the common frame-number expansion patterns:

- `shot.####.exr` (four hashes = zero-padded frame number)
- `shot.%04d.exr` (printf-style)
- `shot.0001.exr` (literal, matches one frame)

Enumeration scans the folder once on first access and caches the frame
list. Pixel aspect is preserved from the first frame's header.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.oiio_io import read_exr
from live_action_aov.io.readers.base import ImageSequenceReader


class OIIOExrReader(ImageSequenceReader):
    extensions = (".exr",)

    def __init__(self, folder: Path, sequence_pattern: str) -> None:
        super().__init__(folder, sequence_pattern)
        self._frames: dict[int, Path] | None = None
        self._first_attrs: dict[str, Any] | None = None
        self._first_pixels_shape: tuple[int, ...] | None = None

    # --- Enumeration ---

    def _enumerate(self) -> dict[int, Path]:
        if self._frames is not None:
            return self._frames
        regex = _pattern_to_regex(self.sequence_pattern)
        frames: dict[int, Path] = {}
        for entry in sorted(self.folder.iterdir()):
            if not entry.is_file() or entry.suffix.lower() != ".exr":
                continue
            m = regex.match(entry.name)
            if not m:
                continue
            try:
                frames[int(m.group("frame"))] = entry
            except (KeyError, ValueError):
                continue
        if not frames:
            raise FileNotFoundError(
                f"No frames matched pattern {self.sequence_pattern!r} in {self.folder}"
            )
        self._frames = frames
        return frames

    def _probe_first(self) -> None:
        if self._first_attrs is not None:
            return
        frames = self._enumerate()
        first_idx = min(frames)
        pixels, attrs = read_exr(frames[first_idx])
        self._first_attrs = attrs
        self._first_pixels_shape = pixels.shape

    # --- ImageSequenceReader API ---

    def frame_range(self) -> tuple[int, int]:
        fs = self._enumerate()
        return min(fs), max(fs)

    def resolution(self) -> tuple[int, int]:
        self._probe_first()
        assert self._first_pixels_shape is not None
        h, w = self._first_pixels_shape[:2]
        return (w, h)

    def pixel_aspect(self) -> float:
        self._probe_first()
        assert self._first_attrs is not None
        return float(self._first_attrs.get("pixelAspectRatio", 1.0))

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        frames = self._enumerate()
        if frame not in frames:
            raise FileNotFoundError(
                f"Frame {frame} not present in sequence; available: {min(frames)}..{max(frames)}"
            )
        return read_exr(frames[frame])


def _pattern_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a sequence pattern like `shot.####.exr` to a regex.

    Supports `#` padding, `%04d`-style, and escaped literals. The captured
    group is always named `frame`.
    """
    # Handle #### style
    if "#" in pattern:
        rx = re.escape(pattern)
        # re.escape turns '#' into '\\#'; replace runs back to a digits group.
        rx = re.sub(r"(?:\\#)+", lambda m: f"(?P<frame>\\d{{{(len(m.group(0)) // 2)}}})", rx)
        return re.compile("^" + rx + "$")
    # printf style %0Nd
    m = re.search(r"%0?(\d*)d", pattern)
    if m:
        width = m.group(1)
        width_rx = f"\\d{{{int(width)}}}" if width else r"\d+"
        prefix = re.escape(pattern[: m.start()])
        suffix = re.escape(pattern[m.end() :])
        return re.compile(f"^{prefix}(?P<frame>{width_rx}){suffix}$")
    # Literal — only the one frame would match
    return re.compile("^" + re.escape(pattern) + "$")


__all__ = ["OIIOExrReader"]
