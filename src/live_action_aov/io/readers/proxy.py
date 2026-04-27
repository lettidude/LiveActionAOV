"""Proxy-resolution reader wrapper.

Wraps another `ImageSequenceReader` (usually `OIIOExrReader`) and
downscales every frame on read to a target long-edge resolution.
The underlying reader still loads the full plate from disk — cv2
bilinear resize is quick enough that it doesn't become the bottleneck
— but downstream passes see smaller arrays, and sidecar writes land
at the proxy resolution, which is the real disk savings.

Use case: fast iteration on pass / colourspace / exposure decisions
before committing to a full-resolution overnight run. A 6K plate
(4608 × 2592) produces 200 MB sidecars × N frames; at 540p it's
~25 MB each. For a 180-frame plate that's 36 GB → 4.5 GB, a 8× disk
save on top of the compute savings on plate-size-scaling passes
(SAM3 / RVM / SSAO).

`proxy_long_edge=None` bypasses the wrapper entirely — callers just
use the underlying reader. The executor opts in based on
`Shot.proxy_long_edge`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.readers.base import ImageSequenceReader


class ProxyReader(ImageSequenceReader):
    """Resize-on-read shim around another reader.

    Behaves identically to the wrapped reader for metadata queries
    (`frame_range`, `pixel_aspect`) but reports the proxy resolution
    from `resolution()` and returns downscaled frames from
    `read_frame()`. Preserves pixel aspect (so anamorphic plates don't
    squish) by scaling height and width by the same factor.
    """

    def __init__(self, inner: ImageSequenceReader, long_edge: int) -> None:
        # Deliberately skip `super().__init__` — the wrapped reader
        # already has `folder` / `sequence_pattern` set, and we don't
        # want to own its state. Forwarding the attributes keeps
        # existing isinstance-based code paths happy.
        self._inner = inner
        self._long_edge = int(long_edge)
        self.folder = inner.folder
        self.sequence_pattern = inner.sequence_pattern
        # Resolution is computed once — all frames share a shape.
        src_w, src_h = inner.resolution()
        src_long = max(src_w, src_h, 1)
        if src_long <= self._long_edge:
            # Already ≤ target — identity scale.
            self._scale = 1.0
            self._proxy_w, self._proxy_h = src_w, src_h
        else:
            self._scale = self._long_edge / src_long
            self._proxy_w = max(1, int(round(src_w * self._scale)))
            self._proxy_h = max(1, int(round(src_h * self._scale)))

    def frame_range(self) -> tuple[int, int]:
        return self._inner.frame_range()

    def resolution(self) -> tuple[int, int]:
        return (self._proxy_w, self._proxy_h)

    def pixel_aspect(self) -> float:
        return self._inner.pixel_aspect()

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        pixels, attrs = self._inner.read_frame(frame)
        if self._scale >= 1.0:
            return pixels, attrs
        # cv2 for speed + bilinear quality is indistinguishable from
        # area-average at this scale factor. Plates are float32 —
        # cv2.resize handles that natively. Preserves last-dim channel
        # count (3, 4, or greyscale 1).
        import cv2

        if pixels.ndim == 2:
            resized = cv2.resize(
                pixels, (self._proxy_w, self._proxy_h), interpolation=cv2.INTER_AREA
            )
        else:
            resized = cv2.resize(
                pixels, (self._proxy_w, self._proxy_h), interpolation=cv2.INTER_AREA
            )
        return resized.astype(np.float32, copy=False), attrs


def wrap_if_proxy(reader: ImageSequenceReader, proxy_long_edge: int | None) -> ImageSequenceReader:
    """Return `reader` wrapped in a `ProxyReader` when `proxy_long_edge`
    is set, otherwise return the input reader unchanged. Single entry
    point the executor uses so the opt-in flows through one hook."""
    if proxy_long_edge is None or proxy_long_edge <= 0:
        return reader
    return ProxyReader(reader, proxy_long_edge)


# Silence ruff about the unused Path import (we expose it in case
# future subclasses want to take a path directly).
_ = Path

__all__ = ["ProxyReader", "wrap_if_proxy"]
