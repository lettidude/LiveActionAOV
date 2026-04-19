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

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.preview_loader import PreviewLoader, PreviewResult
from live_action_aov.gui.shot_state import ShotRegistry, ShotState, ViewMode


class ViewportPanel(QWidget):
    def __init__(self, registry: ShotRegistry) -> None:
        super().__init__()
        self._registry = registry
        self._loader = PreviewLoader(long_edge=1024)
        self._loader.result.connect(self._on_preview_ready)

        self._current: ShotState | None = None
        self._latest_request_id = 0

        # --- View mode radios ---
        self._radio_original = QRadioButton("Original")
        self._radio_transformed = QRadioButton("Transformed")
        self._radio_compare = QRadioButton("Compare")
        self._radio_transformed.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._radio_original, 0)
        self._mode_group.addButton(self._radio_transformed, 1)
        self._mode_group.addButton(self._radio_compare, 2)
        self._mode_group.idToggled.connect(self._on_mode_toggled)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self._radio_original)
        mode_row.addWidget(self._radio_transformed)
        mode_row.addWidget(self._radio_compare)
        mode_row.addStretch()

        # --- Image canvas ---
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setMinimumSize(320, 180)
        self._canvas.setStyleSheet("background: #1a1a1a; color: #888;")
        self._canvas.setText("(no shot loaded)")

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
        self._canvas.setPixmap(
            pixmap.scaled(
                self._canvas.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


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
    painter.drawImage(
        left.width() + divider, (height - right.height()) // 2, right
    )
    painter.end()
    return combined


__all__ = ["ViewportPanel"]
