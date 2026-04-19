"""Shot list panel — left of the three-panel main window.

A `QListWidget` with an "Add shot…" button at the top. Each list entry
shows the shot name + frame count; selection emits through the shared
`ShotRegistry` so the viewport and inspector refresh together.

"Add shot" opens a folder picker, auto-discovers an EXR sequence
(skipping `.utility.` / `.hero.` / `.mask.` sidecars the way the CLI
does), probes the first frame's header for colorspace, and registers
a new `ShotState`. Drag-and-drop of folders onto the list is also
wired — that's the muscle-memory VFX compers have.

Pass the discovery errors through to a `QMessageBox` so a bad folder
gives a loud, recoverable failure rather than a silent no-op.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import detect_colorspace
from live_action_aov.io.display_transform import DisplayTransform
from live_action_aov.io.oiio_io import read_exr

_SIDECAR_TOKENS = (".utility.", ".hero.", ".mask.")


class ShotListPanel(QWidget):
    def __init__(self, registry: ShotRegistry) -> None:
        super().__init__()
        self._registry = registry

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_selection_changed)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        # Pin a minimum width so the "Add shot…" button + shortest shot
        # name don't clip when the splitter is squashed.
        self.setMinimumWidth(180)

        add_btn = QPushButton("Add shot…")
        add_btn.clicked.connect(self._on_add_clicked)

        top = QHBoxLayout()
        top.addWidget(add_btn)
        top.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._list)

        # Drag-and-drop: accept folder drops.
        self.setAcceptDrops(True)

        # Stay in sync when shots are added/removed elsewhere.
        self._registry.shot_added.connect(self._on_shot_added)
        self._registry.shot_removed.connect(self._on_shot_removed)

    # --- DnD ---

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self._add_shot_from_folder(path)
        event.acceptProposedAction()

    # --- Actions ---

    def _on_add_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select plate folder")
        if folder:
            self._add_shot_from_folder(Path(folder))

    def _add_shot_from_folder(self, folder: Path) -> None:
        try:
            pattern, frame_range, resolution, pixel_aspect, first_frame_path = _sniff_sequence(
                folder
            )
        except FileNotFoundError as e:
            QMessageBox.warning(self, "No EXR sequence found", str(e))
            return

        # Probe the first frame's header + pixels for colorspace detection.
        # Pixels enable the lying-tag heuristic; the header gives the
        # authoritative answer when it exists.
        try:
            pixels, attrs = read_exr(first_frame_path)
        except Exception as e:
            QMessageBox.warning(self, "Could not read first frame", str(e))
            return

        detected = detect_colorspace(attrs, sample_pixels=pixels)

        # Seed auto-exposure from the first frame so the preview lands
        # on a sensible look without the user touching the slider. This
        # mirrors what the executor's display transform computes at
        # submit time — sampling more frames would be more accurate
        # but also slow the add step; one-frame seed is good enough for
        # first-view and the user can refine live via the slider.
        auto = _seed_auto_exposure(pixels, detected.detected)

        shot = ShotState(
            name=folder.name,
            folder=folder,
            sequence_pattern=pattern,
            frame_range=frame_range,
            resolution=resolution,
            pixel_aspect=pixel_aspect,
            detected=detected,
            current_frame=frame_range[0],
            auto_ev=auto.get("ev"),
            auto_ev_source=str(auto.get("source", "")),
            sampled_luma=auto.get("sampled_luma"),
            # Seed the slider with the auto value so the first render is
            # already in the right exposure neighbourhood.
            exposure_ev=float(auto.get("ev") or 0.0),
        )
        self._registry.add(shot)

    # --- Registry → widget sync ---

    def _on_shot_added(self, shot: ShotState) -> None:
        item = QListWidgetItem(_format_item_label(shot))
        item.setData(Qt.ItemDataRole.UserRole, shot)
        # Full info in a tooltip so the label can truncate without losing
        # data — compers hover to confirm frame range + resolution when
        # the panel is sized narrow.
        item.setToolTip(_format_item_tooltip(shot))
        self._list.addItem(item)
        self._list.setCurrentItem(item)

    def _on_shot_removed(self, shot: ShotState) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is shot:
                self._list.takeItem(i)
                return

    def _on_selection_changed(
        self, current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        shot = current.data(Qt.ItemDataRole.UserRole) if current else None
        self._registry.set_current(shot)


def _seed_auto_exposure(pixels: np.ndarray, colorspace: str) -> dict:
    """Run the real `DisplayTransform.analyze_clip` against a single
    sampled frame — returns the `{ev, source, sampled_luma}` dict the
    executor would produce at submit time.

    Previewing with this EV makes the GUI's Transformed mode look like
    what the pipeline will feed the model. The executor re-runs the
    same analysis across N frames at submit; the GUI's one-frame seed
    is a fast approximation of that result.
    """
    # Linearize via the same preview path the viewport uses so the
    # sampled luma is in a consistent scale.
    from live_action_aov.gui.preview_loader import _preview_to_linear

    linear = _preview_to_linear(pixels, colorspace)
    # Trim border pixels (frequently pure black from codec padding)
    # that would drag the median down.
    params = DisplayTransformParams(
        auto_exposure_enabled=True, exposure_anchor="median", exposure_target=0.18
    )
    return DisplayTransform().analyze_clip(
        [linear],
        params,
        working_space="lin_rec709" if colorspace != "acescg" else "acescg",
    )


def _format_item_label(shot: ShotState) -> str:
    """One-line label for the shot list row.

    Kept short so a narrow panel doesn't truncate the shot name itself.
    Full info lives in the tooltip (`_format_item_tooltip`).
    """
    start, end = shot.frame_range
    n = end - start + 1
    return f"{shot.name}   ({n} fr)"


def _format_item_tooltip(shot: ShotState) -> str:
    start, end = shot.frame_range
    n = end - start + 1
    w, h = shot.resolution
    lines = [
        f"<b>{shot.name}</b>",
        f"Folder: {shot.folder}",
        f"Pattern: {shot.sequence_pattern}",
        f"Frames: {start}–{end}  ({n} total)",
        f"Resolution: {w} × {h}",
    ]
    if shot.detected is not None:
        lines.append(f"Colorspace: {shot.colorspace_label()}")
    return "<br/>".join(lines)


def _sniff_sequence(folder: Path):
    """Discover an EXR sequence in `folder` and return `(pattern,
    frame_range, resolution, pixel_aspect, first_frame_path)`.

    Handles messy real-world folders:
      - Multiple sequences (picks the one with the most frames).
      - Extra files (`.lut`, `.txt`, screenshots, single-file refs) —
        silently ignored, they don't contribute to any pattern.
      - Sidecars from prior runs (`.utility.`, `.hero.`, `.mask.`) —
        filtered out before sniffing so a re-added shot doesn't latch
        onto its own output.

    Algorithm: for each EXR, derive a template by replacing its final
    digit-run before `.exr` with `#` of the same width. Group files by
    template; pick the template with the most files. The frame range
    comes from the min/max of the frame numbers in that group.
    """
    candidates = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".exr"
        and not any(tok in p.name for tok in _SIDECAR_TOKENS)
    ]
    if not candidates:
        raise FileNotFoundError(f"No .exr plate files found in {folder}")

    # Each file contributes a (pattern, frame_number) if the last
    # digit-run before `.exr` is parseable; otherwise the file is
    # silently skipped (single-frame reference, etc.).
    groups: dict[str, list[tuple[int, Path]]] = {}
    tail_digits_re = re.compile(r"(\d+)(?=\.exr$)", re.IGNORECASE)
    for p in candidates:
        m = tail_digits_re.search(p.name)
        if not m:
            continue
        digits = m.group(1)
        width = len(digits)
        pattern = p.name[: m.start()] + ("#" * width) + p.name[m.end() :]
        groups.setdefault(pattern, []).append((int(digits), p))

    if not groups:
        raise FileNotFoundError(
            f"No sequenced .exr files found in {folder} "
            "(single-file references and non-sequenced names were skipped)."
        )

    # Biggest sequence wins. Break ties by lexicographic pattern so the
    # choice is deterministic across runs.
    best_pattern = max(groups.keys(), key=lambda k: (len(groups[k]), -ord(k[0]) if k else 0))
    entries = sorted(groups[best_pattern], key=lambda t: t[0])
    frame_numbers = [f for f, _ in entries]
    frame_range = (min(frame_numbers), max(frame_numbers))
    first_frame_path = entries[0][1]

    pixels, attrs = read_exr(first_frame_path)
    h, w = pixels.shape[:2]
    par = float(attrs.get("pixelAspectRatio", 1.0))
    return best_pattern, frame_range, (w, h), par, first_frame_path


__all__ = ["ShotListPanel"]
