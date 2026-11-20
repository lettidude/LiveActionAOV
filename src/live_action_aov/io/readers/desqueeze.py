# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""De-squeeze reader for anamorphic plates (non-square pixels).

The models behind the passes were trained on square pixels. An anamorphic
plate (pixel aspect 2.0: every pixel is twice as wide as it is tall) shows
them a picture squashed sideways. With `Shot.desqueeze` on, the executor
wraps the plate reader in `DesqueezeReader`: every frame is resampled to
square pixels before any pass sees it, and the executor scales each pass
back to the plate's own squeeze when it writes the EXRs (with the right
filter per channel type, see `executors.local`).

The de-squeeze never grows an axis (no invented pixels, no extra cost):
pixel aspect > 1 scales the height (`h / par`), pixel aspect < 1 scales the
width (`w * par`). A proxy, when set, applies on top of the de-squeezed frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.readers.base import ImageSequenceReader

#: Pixel aspects closer to 1.0 than this are treated as square.
SQUARE_TOLERANCE = 1e-3


def is_squeezed(pixel_aspect: float | None) -> bool:
    """True when `pixel_aspect` is a real anamorphic squeeze."""
    try:
        par = float(pixel_aspect or 1.0)
    except (TypeError, ValueError):
        return False
    return par > 0 and abs(par - 1.0) > SQUARE_TOLERANCE


def desqueezed_size(width: int, height: int, pixel_aspect: float) -> tuple[int, int]:
    """Square-pixel size of a `width` x `height` plate with this pixel aspect.

    Neither axis grows: PAR > 1 -> `(w, h / par)`, PAR < 1 -> `(w * par, h)`.
    """
    par = float(pixel_aspect)
    if not is_squeezed(par):
        return int(width), int(height)
    if par > 1.0:
        return int(width), max(1, int(round(height / par)))
    return max(1, int(round(width * par))), int(height)


class DesqueezeReader(ImageSequenceReader):
    """Resample every frame of the wrapped reader to square pixels.

    Reports the de-squeezed `resolution()` and a pixel aspect of 1.0 (what
    the passes see). `plate_resolution` / `plate_pixel_aspect` keep the
    plate's own values for the write-back.
    """

    def __init__(self, inner: ImageSequenceReader, pixel_aspect: float) -> None:
        # Like CropReader / ProxyReader: the inner reader owns the state.
        self._inner = inner
        self.folder = inner.folder
        self.sequence_pattern = inner.sequence_pattern
        self._par = float(pixel_aspect)
        self._plate_w, self._plate_h = inner.resolution()
        self._w, self._h = desqueezed_size(self._plate_w, self._plate_h, self._par)

    @property
    def plate_resolution(self) -> tuple[int, int]:
        return (self._plate_w, self._plate_h)

    @property
    def plate_pixel_aspect(self) -> float:
        return self._par

    def frame_range(self) -> tuple[int, int]:
        return self._inner.frame_range()

    def resolution(self) -> tuple[int, int]:
        return (self._w, self._h)

    def pixel_aspect(self) -> float:
        return 1.0

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        pixels, attrs = self._inner.read_frame(frame)
        if (self._w, self._h) == (pixels.shape[1], pixels.shape[0]):
            return pixels, attrs
        import cv2

        # Area filter: both cases shrink one axis only, so this is a clean
        # box average along that axis (no ringing, no values above the plate's).
        out = cv2.resize(pixels, (self._w, self._h), interpolation=cv2.INTER_AREA)
        if pixels.ndim == 3 and out.ndim == 2:
            out = out[..., None]
        return out.astype(np.float32, copy=False), attrs


def wrap_if_desqueeze(
    reader: ImageSequenceReader, pixel_aspect: float | None, enabled: bool
) -> ImageSequenceReader:
    """`reader` de-squeezed when `enabled` and the plate is anamorphic,
    unchanged otherwise."""
    if not enabled or not is_squeezed(pixel_aspect):
        return reader
    return DesqueezeReader(reader, float(pixel_aspect or 1.0))


__all__ = [
    "SQUARE_TOLERANCE",
    "DesqueezeReader",
    "desqueezed_size",
    "is_squeezed",
    "wrap_if_desqueeze",
]
