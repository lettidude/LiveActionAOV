"""Viewport panel — centre of the main window.

Shows the current shot's current frame, with:

- Radio group to pick Original / Transformed / Compare.
- Frame scrub slider below the image.
- Loading placeholder while the async preview worker fetches.

"Compare" mode stitches the raw-decoded and display-transformed
frames side-by-side with a thin divider — the "what the model will
see" preview flagged in the preflight memory. A future M3 polish adds
a wipe slider, but for M1 the side-by-side is explicit and easy to
read.

Requests are fire-and-forget: when the user scrubs fast we enqueue a
preview request per frame, then on each arrival we discard results
whose `request_id` is older than the latest outstanding request. That
way the viewport always shows the user's latest intent even if the
worker pool is backed up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.mask_preview import MaskPreviewWorker
from live_action_aov.gui.preview_loader import PreviewLoader, PreviewResult
from live_action_aov.gui.shot_state import ClickInstance, ShotRegistry, ShotState, ViewMode


def _pixmap_to_rgb_float(pm: QPixmap) -> np.ndarray:
    """QPixmap → (H, W, 3) float32 in [0, 1] — the SAM preview input."""
    img = pm.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    buf = np.frombuffer(img.constBits(), np.uint8, count=h * bpl).reshape(h, bpl)
    return buf[:, : w * 3].reshape(h, w, 3).astype(np.float32) / 255.0


def _mask_to_overlay(mask: np.ndarray) -> QImage:
    """(H, W) float mask → semi-transparent green RGBA image for overlay.

    Alpha is proportional to the mask value (not thresholded) so a SOFT
    refined matte shows its fading edges here, not a hard cutout — that's
    the whole point of previewing the refined result.
    """
    h, w = mask.shape
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 80
    rgba[..., 1] = 220
    rgba[..., 2] = 100
    # Gamma-lifted alpha: proportional to coverage so soft edges read as
    # soft, but lifted (m^0.45) so semi-transparent regions (motion-blurred
    # hands, hair) stay clearly visible instead of looking "eroded".
    rgba[..., 3] = (np.power(m, 0.45) * 140.0).astype(np.uint8)
    img = QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return img.copy()  # detach from the numpy buffer before it goes away


def _canvas_pos_to_norm(
    pos_x: float,
    pos_y: float,
    canvas_w: int,
    canvas_h: int,
    pix_w: int,
    pix_h: int,
) -> tuple[float, float] | None:
    """Map a click on the canvas to normalized [0,1] image coordinates.

    The canvas centres its (aspect-preserving) scaled pixmap, so the image
    sits at a letterbox offset. Returns None when the click lands outside
    the image (in the letterbox bars) or no image is shown. Normalized
    coords are resolution-independent: the caller multiplies by the plate
    resolution, which is what `ClickInstance` stores.
    """
    if pix_w <= 0 or pix_h <= 0:
        return None
    off_x = (canvas_w - pix_w) / 2.0
    off_y = (canvas_h - pix_h) / 2.0
    nx = (pos_x - off_x) / float(pix_w)
    ny = (pos_y - off_y) / float(pix_h)
    if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
        return None
    return nx, ny


class _ClickCanvas(QLabel):
    """The viewport image label, emitting normalized click / box positions.

    Mouse semantics (all in [0,1] image coords; the panel decides whether
    they mean anything — click mode armed, active instance, view mode):
    - Right-click → `pressed(nx, ny, 0)` — an exclude point, immediately.
    - Left-click (no drag) → `pressed(nx, ny, 1)` — an include point, on
      release once it's clear it wasn't a drag.
    - Left-drag → `boxed(nx0, ny0, nx1, ny1)` — a box prompt (normalized,
      top-left → bottom-right). `dragging((…)|None)` fires live for the
      rubber-band overlay.
    """

    pressed = Signal(float, float, int)
    boxed = Signal(float, float, float, float)
    dragging = Signal(object)  # (nx0, ny0, nx1, ny1) live, or None on end

    _DRAG_PX = 5  # movement below this on release = a click, not a box

    def __init__(self) -> None:
        super().__init__()
        self._press_pos: Any = None  # QPointF in widget coords, or None
        self._press_norm: tuple[float, float] | None = None

    def _norm_at(self, pos: Any) -> tuple[float, float] | None:
        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        return _canvas_pos_to_norm(
            pos.x(), pos.y(), self.width(), self.height(), pm.width(), pm.height()
        )

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.RightButton:
            norm = self._norm_at(ev.position())
            if norm is not None:
                self.pressed.emit(norm[0], norm[1], 0)
        elif ev.button() == Qt.MouseButton.LeftButton:
            self._press_pos = ev.position()
            self._press_norm = self._norm_at(ev.position())
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._press_pos is not None and self._press_norm is not None:
            cur = self._norm_at(ev.position())
            if cur is not None:
                self.dragging.emit((self._press_norm[0], self._press_norm[1], cur[0], cur[1]))
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton and self._press_pos is not None:
            start, start_norm = self._press_pos, self._press_norm
            self._press_pos, self._press_norm = None, None
            self.dragging.emit(None)
            end = ev.position()
            end_norm = self._norm_at(end)
            moved = abs(end.x() - start.x()) >= self._DRAG_PX or abs(end.y() - start.y()) >= self._DRAG_PX
            if start_norm is not None:
                if moved and end_norm is not None:
                    x0, x1 = sorted((start_norm[0], end_norm[0]))
                    y0, y1 = sorted((start_norm[1], end_norm[1]))
                    self.boxed.emit(x0, y0, x1, y1)
                else:
                    self.pressed.emit(start_norm[0], start_norm[1], 1)
        super().mouseReleaseEvent(ev)


class ViewportPanel(QWidget):
    def __init__(self, registry: ShotRegistry) -> None:
        super().__init__()
        self._registry = registry
        self._loader = PreviewLoader(long_edge=1024)
        self._loader.result.connect(self._on_preview_ready)

        self._current: ShotState | None = None
        self._latest_request_id = 0

        # Click-to-mask state (driven by the inspector's Masks tab).
        # `_click_mode` arms point placement; `_active_instance` is the
        # ClickInstance receiving clicks. `_last_composed` keeps the last
        # unscaled preview so markers repaint without re-decoding the EXR.
        self._click_mode = False
        self._active_instance: ClickInstance | None = None
        self._last_composed: QPixmap | None = None
        # Single-frame SAM 3 preview of the active instance's mask. The
        # ndarray lives at `_last_composed` resolution; it's invalidated on
        # any point/frame/instance change so a stale mask never lies.
        self._preview_mask: np.ndarray | None = None
        # Live rubber-band rect while the user drags a box, in normalized
        # [0,1] image coords — drawn as a dashed rectangle, cleared on release.
        self._drag_rect: tuple[float, float, float, float] | None = None
        self._mask_worker = MaskPreviewWorker()
        self._mask_worker.ready.connect(self._on_mask_preview_ready)
        self._mask_worker.failed.connect(self._on_mask_preview_failed)
        self._mask_worker.status.connect(self._frame_status)

        # --- View mode radios ---
        # "Transformed" is the default because that's the mode compers
        # live in — "what the model will see". "Compare" is where the
        # raw-vs-transformed side-by-side check lives. "Raw" (renamed
        # from Original) is kept but de-prioritised — it's occasionally
        # useful to confirm a pure colorspace decode without exposure
        # or tonemap confusing things, but it's not a primary mode.
        self._radio_transformed = QRadioButton("Transformed")
        self._radio_compare = QRadioButton("Compare")
        self._radio_original = QRadioButton("Raw")
        self._radio_original.setToolTip(
            "Raw colorspace decode — no exposure, no tonemap. "
            "Useful when sanity-checking a suspicious colorspace tag."
        )
        self._radio_transformed.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_transformed, 1)
        self._mode_group.addButton(self._radio_compare, 2)
        self._mode_group.addButton(self._radio_original, 0)
        self._mode_group.idToggled.connect(self._on_mode_toggled)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._radio_transformed)
        mode_row.addWidget(self._radio_compare)
        mode_row.addWidget(self._radio_original)
        mode_row.addStretch()

        # --- Image canvas ---
        # Min width is a suggestion; the canvas grows to fill the
        # splitter's centre column. Wrapping inside a QScrollArea is a
        # future move when we add high-res overlays — for now the
        # QLabel's scaled pixmap fits any viewport size.
        self._canvas = _ClickCanvas()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(320, 200)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet("background: #1a1a1a; color: #888;")
        self._canvas.setWordWrap(True)
        self._canvas.setText("(no shot loaded)")
        self._canvas.pressed.connect(self._on_canvas_pressed)
        self._canvas.boxed.connect(self._on_canvas_boxed)
        self._canvas.dragging.connect(self._on_canvas_dragging)

        # --- Frame scrub slider + label ---
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self._frame_label = QLabel("Frame: —")
        self._frame_label.setFixedWidth(110)

        scrub_row = QHBoxLayout()
        scrub_row.addWidget(self._frame_label)
        scrub_row.addWidget(self._frame_slider, stretch=1)

        # --- Assemble ---
        root = QVBoxLayout(self)
        root.addLayout(mode_row)
        root.addWidget(self._canvas, stretch=1)
        root.addLayout(scrub_row)

        # Registry wiring last.
        self._registry.current_changed.connect(self._on_current_changed)
        self._registry.shot_updated.connect(self._on_shot_updated)

    # --- Registry → UI ---

    def _on_current_changed(self, shot: ShotState | None) -> None:
        self._current = shot
        self._refresh_for_current()

    def _on_shot_updated(self, shot: ShotState) -> None:
        if shot is self._current:
            self._request_preview()

    def _refresh_for_current(self) -> None:
        shot = self._current
        if shot is None:
            self._frame_slider.setEnabled(False)
            self._frame_slider.setRange(0, 0)
            self._frame_label.setText("Frame: —")
            self._last_composed = None
            self._canvas.setPixmap(QPixmap())
            self._canvas.setText("(no shot loaded)")
            return

        start, end = shot.frame_range
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(start, end)
        self._frame_slider.setValue(shot.current_frame or start)
        self._frame_slider.setEnabled(end > start)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"Frame: {shot.current_frame or start}")

        # Reflect the stored view_mode back into the radios.
        mode_to_button = {
            "original": self._radio_original,
            "transformed": self._radio_transformed,
            "compare": self._radio_compare,
        }
        self._mode_group.blockSignals(True)
        mode_to_button[shot.view_mode].setChecked(True)
        self._mode_group.blockSignals(False)

        self._canvas.setText("loading…")
        self._request_preview()

    # --- UI → state ---

    def _on_frame_slider_changed(self, value: int) -> None:
        if self._current is None:
            return
        self._current.current_frame = value
        self._frame_label.setText(f"Frame: {value}")
        # Mask preview belongs to the seed frame — scrubbing away drops it.
        self._preview_mask = None
        self._request_preview()

    def _on_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked or self._current is None:
            return
        modes: dict[int, ViewMode] = {0: "original", 1: "transformed", 2: "compare"}
        self._current.view_mode = modes[button_id]
        self._request_preview()

    # --- Worker plumbing ---

    def _request_preview(self) -> None:
        shot = self._current
        if shot is None:
            return
        self._latest_request_id = self._loader.request(
            shot=shot, frame=shot.current_frame, mode=shot.view_mode
        )

    def _on_preview_ready(self, result: PreviewResult) -> None:
        # Drop stale results from earlier scrubs.
        if result.request_id != self._latest_request_id:
            return
        if result.error is not None:
            self._canvas.setText(f"Preview failed:\n{result.error}")
            return
        pixmap = _compose_pixmap(result)
        if pixmap is None:
            self._canvas.setText("(nothing to show)")
            return
        self._last_composed = pixmap
        self._repaint_canvas()

    # --- Click-to-mask (driven by the inspector's Masks tab) ---

    def set_click_mode(self, on: bool) -> None:
        """Arm/disarm viewport point placement (inspector Masks tab toggle)."""
        self._click_mode = bool(on)
        self._canvas.setCursor(
            Qt.CursorShape.CrossCursor if self._click_mode else Qt.CursorShape.ArrowCursor
        )
        self._repaint_canvas()

    def set_active_instance(self, inst: object) -> None:
        """Select which ClickInstance receives clicks (or None)."""
        new = inst if isinstance(inst, ClickInstance) else None
        if new is not self._active_instance:
            self._preview_mask = None
        self._active_instance = new
        self._repaint_canvas()

    def preview_active_mask(self) -> None:
        """Run SAM 3 on the seed frame only and overlay the resulting mask —
        the 'what will my points produce?' check before a full submit."""
        shot = self._current
        inst = self._active_instance
        if shot is None or inst is None or (not inst.points and inst.box is None):
            self._frame_status("Preview: add at least one point first.")
            return
        frame = shot.current_frame or shot.frame_range[0]
        if frame != inst.seed_frame:
            self._frame_status(f"Preview: scrub to the seed frame (f{inst.seed_frame}).")
            return
        if shot.view_mode == "compare":
            self._frame_status("Preview: exit Compare mode first.")
            return
        if self._last_composed is None or self._last_composed.isNull():
            self._frame_status("Preview: no image loaded yet.")
            return
        if self._mask_worker.is_busy():
            return
        image = _pixmap_to_rgb_float(self._last_composed)
        ih, iw = image.shape[0], image.shape[1]
        w, h = shot.resolution
        pts = [[px / float(w) * iw, py / float(h) * ih] for (px, py, _lbl) in inst.points]
        lbls = [int(lbl) for (_px, _py, lbl) in inst.points]
        box = (
            [
                inst.box[0] / float(w) * iw,
                inst.box[1] / float(h) * ih,
                inst.box[2] / float(w) * iw,
                inst.box[3] / float(h) * ih,
            ]
            if inst.box is not None
            else None
        )
        # The preview ALWAYS shows the refined (soft) mask with the shot's
        # chosen refiner weights — one extra ~0.3s pass, and what you see is
        # what the submit produces. Model comparison = change the Refiner
        # dropdown and re-preview.
        # Preview engine: the Masks-tab "Preview with" override wins; "auto"
        # mirrors the run choice from the Passes tab (the tabs communicate).
        override = str(getattr(shot, "preview_refiner", "auto") or "auto")
        if override == "auto":
            enabled = shot.enabled_models or []
            if "sam3_vitmatte" in enabled:
                kind, model_id = "vitmatte", ""
            elif "sam3_rvm" in enabled:
                kind, model_id = "rvm", ""
            else:
                kind, model_id = "birefnet", str(shot.refiner_model or "")
        elif override.startswith("birefnet:"):
            kind, model_id = "birefnet", override.split(":", 1)[1]
        else:
            kind, model_id = override, ""
        self._mask_worker.request(
            image,
            pts,
            lbls,
            box,
            refine=True,
            model_id=model_id,
            refiner_kind=kind,
        )

    def release_mask_preview(self) -> None:
        """Drop the overlay and free the resident preview model — called
        before Submit so the batch executor gets the whole GPU."""
        self._preview_mask = None
        self._mask_worker.unload()
        self._repaint_canvas()

    def _frame_status(self, text: str) -> None:
        self._frame_label.setText(text)

    def _on_mask_preview_ready(self, mask: object) -> None:
        self._preview_mask = mask if isinstance(mask, np.ndarray) else None
        shot = self._current
        frame = (shot.current_frame or shot.frame_range[0]) if shot else "—"
        self._frame_label.setText(f"Frame: {frame}")
        self._repaint_canvas()

    def _on_mask_preview_failed(self, error: str) -> None:
        self._preview_mask = None
        self._frame_status(f"Preview failed: {error[:60]}")

    def _can_edit_active(self) -> bool:
        """Shared guard for point/box input. True if the armed active
        instance can take input on the current frame — anchoring the seed
        frame on the very first input — else False with a status hint."""
        shot = self._current
        inst = self._active_instance
        if not self._click_mode or shot is None or inst is None:
            return False
        if shot.view_mode == "compare":
            self._frame_label.setText("Frame: — (exit Compare to place points)")
            return False
        frame = shot.current_frame or shot.frame_range[0]
        if not inst.points and inst.box is None:
            inst.seed_frame = frame  # first input anchors the seed frame
            return True
        if frame != inst.seed_frame:
            self._frame_label.setText(f"Frame: {frame} (object lives on f{inst.seed_frame})")
            return False
        return True

    def _on_canvas_pressed(self, nx: float, ny: float, label: int) -> None:
        if not self._can_edit_active():
            return
        shot, inst = self._current, self._active_instance
        assert shot is not None and inst is not None
        w, h = shot.resolution
        inst.points.append((nx * float(w), ny * float(h), int(label)))
        self._preview_mask = None  # point set changed → previewed mask stale
        self._repaint_canvas()
        self._registry.notify_updated(shot)

    def _on_canvas_boxed(self, nx0: float, ny0: float, nx1: float, ny1: float) -> None:
        if not self._can_edit_active():
            return
        shot, inst = self._current, self._active_instance
        assert shot is not None and inst is not None
        w, h = shot.resolution
        inst.box = (nx0 * float(w), ny0 * float(h), nx1 * float(w), ny1 * float(h))
        self._preview_mask = None
        self._drag_rect = None
        self._repaint_canvas()
        self._registry.notify_updated(shot)

    def _on_canvas_dragging(self, rect: object) -> None:
        """Live rubber-band overlay while the user drags a box."""
        if not self._click_mode or self._active_instance is None:
            return
        self._drag_rect = rect if isinstance(rect, tuple) else None
        self._repaint_canvas()

    def _repaint_canvas(self) -> None:
        """Scale the last composed preview to the canvas and overlay the
        active instance's points (seed frame only). Pure-paint — never
        re-decodes the EXR."""
        base = self._last_composed
        if base is None or base.isNull():
            return
        scaled = base.scaled(
            self._canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        shot = self._current
        inst = self._active_instance
        on_seed = (
            shot is not None
            and inst is not None
            and self._click_mode
            and shot.view_mode != "compare"
            and (shot.current_frame or shot.frame_range[0]) == inst.seed_frame
        )
        has_marks = inst is not None and (
            inst.points or inst.box is not None or self._preview_mask is not None
        )
        if shot is not None and inst is not None and on_seed and (has_marks or self._drag_rect):
            w, h = shot.resolution
            sw, sh = scaled.width(), scaled.height()
            painter = QPainter(scaled)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            # Mask preview first (under markers): SAM 3 result at composed-
            # image resolution, scaled onto the canvas pixmap.
            if self._preview_mask is not None:
                overlay = _mask_to_overlay(self._preview_mask)
                painter.drawImage(
                    scaled.rect(),
                    overlay.scaled(
                        scaled.size(),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    ),
                )
            # Committed box (solid cyan).
            if inst.box is not None:
                bx0, by0, bx1, by1 = inst.box
                painter.setPen(QPen(QColor(90, 200, 255), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(
                    int(bx0 / w * sw),
                    int(by0 / h * sh),
                    int((bx1 - bx0) / w * sw),
                    int((by1 - by0) / h * sh),
                )
            # Live rubber-band while dragging (dashed white).
            if self._drag_rect is not None:
                dx0, dy0, dx1, dy1 = self._drag_rect
                pen = QPen(QColor(255, 255, 255), 1.0, Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(
                    int(min(dx0, dx1) * sw),
                    int(min(dy0, dy1) * sh),
                    int(abs(dx1 - dx0) * sw),
                    int(abs(dy1 - dy0) * sh),
                )
            for px, py, lbl in inst.points:
                cx = (px / float(w)) * sw
                cy = (py / float(h)) * sh
                colour = QColor(80, 220, 100) if lbl == 1 else QColor(230, 80, 80)
                painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                painter.setBrush(colour)
                painter.drawEllipse(int(cx) - 4, int(cy) - 4, 8, 8)
            painter.end()
        self._canvas.setPixmap(scaled)


def _compose_pixmap(result: PreviewResult) -> QPixmap | None:
    """Side-by-side for Compare, single-image otherwise."""
    mode = result.view_mode
    if mode == "original" and result.original_qimage is not None:
        return QPixmap.fromImage(result.original_qimage)
    if mode == "transformed" and result.transformed_qimage is not None:
        return QPixmap.fromImage(result.transformed_qimage)
    if (
        mode == "compare"
        and result.original_qimage is not None
        and result.transformed_qimage is not None
    ):
        return _stitch_side_by_side(result.original_qimage, result.transformed_qimage)
    return None


def _stitch_side_by_side(left: QImage, right: QImage) -> QPixmap:
    """2-wide composite with a 2px divider. Both sides are letterboxed
    to the taller image's height so different aspect ratios align."""
    height = max(left.height(), right.height())
    divider = 2
    width = left.width() + right.width() + divider
    combined = QPixmap(width, height)
    combined.fill(Qt.GlobalColor.black)
    painter = QPainter(combined)
    painter.drawImage(0, (height - left.height()) // 2, left)
    painter.fillRect(left.width(), 0, divider, height, Qt.GlobalColor.darkGray)
    painter.drawImage(left.width() + divider, (height - right.height()) // 2, right)
    painter.end()
    return combined


__all__ = ["ViewportPanel"]
