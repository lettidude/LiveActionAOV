"""Async frame loader for the GUI viewport.

Reads EXRs on a worker thread, linearises according to the shot's
effective colorspace, optionally runs the display transform
(AgX + sRGB EOTF) so the Transformed / Compare modes can show "what
the model will see", resizes to a proxy long-edge, and emits a
`QImage` ready for the viewport to blit.

Why a worker at all: a 2K EXR takes ~100-300 ms to read + resize on a
fast machine, a 4K one 300-800 ms. Scrubbing 150 frames with that on
the UI thread freezes the app for 15+ seconds. With `QThreadPool` the
scrub stays snappy and out-of-order completions get discarded by the
monotonic `request_id`.

This module owns no GUI state — callers pass a `ShotState` in, get a
`QImage` out via signal. The viewport's cache + request bookkeeping
lives in the viewport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage

from live_action_aov.gui.shot_state import ShotState, ViewMode
from live_action_aov.io.ocio_color import from_linear, to_linear
from live_action_aov.io.oiio_io import read_exr


@dataclass(frozen=True)
class PreviewResult:
    """Two images per request: raw (colorspace-converted to sRGB for
    display) and transformed (after display_transform). Compare mode
    renders both side-by-side; the other modes use one."""

    request_id: int
    frame: int
    original_qimage: QImage | None
    transformed_qimage: QImage | None
    view_mode: ViewMode
    error: str | None = None


class PreviewLoader(QObject):
    """Queues frame-preview requests on the Qt global thread pool.

    Usage:
        loader = PreviewLoader(long_edge=1024)
        loader.result.connect(viewport.on_preview_ready)
        req_id = loader.request(shot_state, frame=1001, mode="compare")
    """

    result = Signal(object)  # PreviewResult

    def __init__(self, long_edge: int = 1024) -> None:
        super().__init__()
        self._long_edge = int(long_edge)
        self._next_id = 0
        self._pool = QThreadPool.globalInstance()

    def request(self, shot: ShotState, frame: int, mode: ViewMode) -> int:
        self._next_id += 1
        req_id = self._next_id
        task = _PreviewTask(
            request_id=req_id,
            shot=shot,
            frame=frame,
            mode=mode,
            long_edge=self._long_edge,
            emit_result=self._emit,
        )
        self._pool.start(task)
        return req_id

    # Called on the worker thread; emits the signal back on the GUI thread.
    def _emit(self, result: PreviewResult) -> None:
        self.result.emit(result)


class _PreviewTask(QRunnable):
    def __init__(
        self,
        *,
        request_id: int,
        shot: ShotState,
        frame: int,
        mode: ViewMode,
        long_edge: int,
        emit_result: Any,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.shot = shot
        self.frame = frame
        self.mode = mode
        self.long_edge = long_edge
        self._emit = emit_result

    def run(self) -> None:
        try:
            result = self._build_result()
        except Exception as e:  # pragma: no cover — reported via result.error
            result = PreviewResult(
                request_id=self.request_id,
                frame=self.frame,
                original_qimage=None,
                transformed_qimage=None,
                view_mode=self.mode,
                error=f"{type(e).__name__}: {e}",
            )
        self._emit(result)

    def _build_result(self) -> PreviewResult:
        path = _resolve_frame_path(self.shot, self.frame)
        pixels, _attrs = read_exr(path)  # (H, W, C) float32
        pixels = _to_rgb3(pixels)
        pixels = _proxy_resize(pixels, self.long_edge)

        colorspace = self.shot.effective_colorspace()
        original_q = None
        transformed_q = None

        if self.mode in ("original", "compare"):
            original_q = _to_qimage_sRGB_via_colorspace(pixels, colorspace)
        if self.mode in ("transformed", "compare"):
            transformed_q = _to_qimage_display_transformed(
                pixels, colorspace, self.shot.exposure_ev
            )

        return PreviewResult(
            request_id=self.request_id,
            frame=self.frame,
            original_qimage=original_q,
            transformed_qimage=transformed_q,
            view_mode=self.mode,
            error=None,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_frame_path(shot: ShotState, frame: int):
    """Expand the shot's sequence pattern for `frame` and join with folder.
    Supports `####` and `%04d`-style tokens; same conventions as the
    OIIO reader."""
    pattern = shot.sequence_pattern
    if "#" in pattern:
        hashes = pattern.count("#")
        return shot.folder / pattern.replace("#" * hashes, f"{frame:0{hashes}d}")
    if "%" in pattern:
        import re

        m = re.search(r"%0?(\d*)d", pattern)
        width = int(m.group(1)) if m and m.group(1) else 4
        return shot.folder / re.sub(r"%0?\d*d", f"{frame:0{width}d}", pattern)
    return shot.folder / pattern


def _to_rgb3(pixels: np.ndarray) -> np.ndarray:
    """Collapse multi-channel EXR reads to RGB float32 for display."""
    if pixels.ndim == 2:
        return np.stack([pixels] * 3, axis=-1).astype(np.float32, copy=False)
    if pixels.shape[-1] == 1:
        return np.repeat(pixels, 3, axis=-1).astype(np.float32, copy=False)
    return pixels[..., :3].astype(np.float32, copy=False)


def _proxy_resize(pixels: np.ndarray, long_edge: int) -> np.ndarray:
    """Nearest-ish resize to `long_edge` on the long axis; keeps latency
    low (200-500ms budget) and dodges the scipy/PIL dependency fork."""
    h, w = pixels.shape[:2]
    current_long = max(h, w)
    if current_long <= long_edge:
        return pixels
    scale = long_edge / current_long
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Simple stride-subsample — good enough for a preview; a proper
    # Lanczos downsample is M3 polish.
    row_stride = max(1, h // new_h)
    col_stride = max(1, w // new_w)
    return pixels[::row_stride, ::col_stride][:new_h, :new_w]


def _to_qimage_sRGB_via_colorspace(pixels: np.ndarray, colorspace: str) -> QImage:
    """Colorspace→scene_linear→sRGB→uint8 for display.

    Uses the existing OCIO wrapper (with numpy fallback). Clamps to
    [0, 1] — exposure headroom is a display-time concern for the
    Transformed mode; Original mode is a faithful decode.
    """
    linear = to_linear(pixels, colorspace)
    srgb = from_linear(linear, "srgb_display")
    clipped = np.clip(srgb, 0.0, 1.0)
    return _qimage_from_rgb_float32(clipped)


def _to_qimage_display_transformed(
    pixels: np.ndarray, colorspace: str, exposure_ev: float
) -> QImage:
    """Apply the "what the model will see" display transform:
    colorspace → linear → exposure scale → approximate AgX → sRGB.

    The AgX curve here is a simplified reinhard-style rational that
    matches the PASS display_transform's qualitative look well enough
    for the preview. The executor uses the real AgX LUT at submit time
    — this is just a fast approximation so the preview stays snappy.
    """
    linear = to_linear(pixels, colorspace)
    exposed = linear * float(2.0**exposure_ev)
    tonemapped = exposed / (1.0 + exposed)  # simple reinhard stand-in
    srgb = from_linear(tonemapped, "srgb_display")
    clipped = np.clip(srgb, 0.0, 1.0)
    return _qimage_from_rgb_float32(clipped)


def _qimage_from_rgb_float32(rgb: np.ndarray) -> QImage:
    """Convert (H, W, 3) float32 in [0, 1] to QImage without aliasing
    the numpy buffer (QImage would dangle if we did). The copy() call
    is necessary — QImage docs warn about the no-copy constructor."""
    u8 = (rgb * 255.0 + 0.5).astype(np.uint8, copy=False)
    h, w, _ = u8.shape
    # PySide6's QImage.Format.Format_RGB888 expects contiguous uint8.
    contig = np.ascontiguousarray(u8)
    qimg = QImage(contig.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    # Copy so the backing numpy array can be freed.
    return qimg.copy()


# Silence ruff's unused-import note when Qt is mocked for headless tests.
_ = Qt


__all__ = ["PreviewLoader", "PreviewResult"]
