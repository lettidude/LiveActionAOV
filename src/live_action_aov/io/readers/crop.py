# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Crop-on-read wrapper: the passes see one box of the plate.

A comper often needs the utility of one corner of the frame only (the hero
in a wide plate, a small insert) and does not want to pay for the rest.
`CropReader` wraps another reader and returns `pixels[y:y+h, x:x+w]`, so
every pass runs on the crop as if it were the plate; the executor then
writes the EXRs with the crop as data window inside the full plate's
display window, and Nuke places them where they belong.

The crop is in plate pixels, origin top-left, `(x, y, w, h)`, the same
frame the GUI clicks use. It goes on BEFORE the proxy wrapper, so a proxy
scales the crop (not the plate) to its long edge.

Passes that need the whole frame (the camera solve and the Gaussian splat)
declare `needs_full_frame = True` and the executor refuses them with a
crop set, rather than solving a camera from a quarter of the image.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.readers.base import ImageSequenceReader

Crop = tuple[int, int, int, int]


def clamp_crop(crop: Any, resolution: tuple[int, int]) -> Crop:
    """`(x, y, w, h)` as ints inside `resolution`; ValueError when empty."""
    x, y, w, h = (int(round(float(v))) for v in crop)
    pw, ph = int(resolution[0]), int(resolution[1])
    x0, y0 = min(max(x, 0), pw), min(max(y, 0), ph)
    x1, y1 = min(max(x + w, 0), pw), min(max(y + h, 0), ph)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f"crop {tuple(crop)} lies outside the {pw}x{ph} plate")
    return (x0, y0, x1 - x0, y1 - y0)


class CropReader(ImageSequenceReader):
    """Return one box of every frame of the wrapped reader."""

    def __init__(self, inner: ImageSequenceReader, crop: Any) -> None:
        # Like ProxyReader: no `super().__init__`, the inner owns the state.
        self._inner = inner
        self.folder = inner.folder
        self.sequence_pattern = inner.sequence_pattern
        self._crop = clamp_crop(crop, inner.resolution())

    @property
    def crop(self) -> Crop:
        return self._crop

    def frame_range(self) -> tuple[int, int]:
        return self._inner.frame_range()

    def resolution(self) -> tuple[int, int]:
        return (self._crop[2], self._crop[3])

    def pixel_aspect(self) -> float:
        return self._inner.pixel_aspect()

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        pixels, attrs = self._inner.read_frame(frame)
        x, y, w, h = self._crop
        return np.ascontiguousarray(pixels[y : y + h, x : x + w]), attrs


def wrap_if_crop(reader: ImageSequenceReader, crop: Any | None) -> ImageSequenceReader:
    """`reader` cropped when `crop` is set, unchanged otherwise."""
    if crop is None:
        return reader
    return CropReader(reader, crop)


__all__ = ["Crop", "CropReader", "clamp_crop", "wrap_if_crop"]
