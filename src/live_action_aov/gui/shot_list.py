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

from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import detect_colorspace
from live_action_aov.io.oiio_io import read_exr

_SIDECAR_TOKENS = (".utility.", ".hero.", ".mask.")


class ShotListPanel(QWidget):
    def __init__(self, registry: ShotRegistry) -> None:
        super().__init__()
        self._registry = registry

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.currentItemChanged.connect(self._on_selection_changed)

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

        shot = ShotState(
            name=folder.name,
            folder=folder,
            sequence_pattern=pattern,
            frame_range=frame_range,
            resolution=resolution,
            pixel_aspect=pixel_aspect,
            detected=detected,
            current_frame=frame_range[0],
        )
        self._registry.add(shot)

    # --- Registry → widget sync ---

    def _on_shot_added(self, shot: ShotState) -> None:
        item = QListWidgetItem(_format_item_label(shot))
        item.setData(Qt.ItemDataRole.UserRole, shot)
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


def _format_item_label(shot: ShotState) -> str:
    start, end = shot.frame_range
    n = end - start + 1
    return f"{shot.name}    [{n} frames @ {shot.resolution[0]}×{shot.resolution[1]}]"


def _sniff_sequence(folder: Path):
    """Mirror the CLI's `_sniff_sequence` but also return the first-frame
    path so the colorspace detector can probe pixels.

    Skips `.utility.` / `.hero.` / `.mask.` sidecars that the tool itself
    writes — otherwise they'd get picked up as the "plate" on a
    previously-run shot.
    """
    exrs = sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".exr"
        and not any(tok in p.name for tok in _SIDECAR_TOKENS)
    )
    if not exrs:
        raise FileNotFoundError(f"No .exr plate files found in {folder}")

    sample = exrs[0].name
    m = re.search(r"(\d{3,})(?=[^\d]*\.exr$)", sample)
    if not m:
        pattern = sample
    else:
        width = len(m.group(1))
        pattern = sample[: m.start()] + ("#" * width) + sample[m.end() :]

    # Frame range — scan filenames matching the pattern.
    if "#" in pattern:
        hashes = pattern.count("#")
        frame_regex = re.compile(
            re.escape(pattern).replace("#" * hashes, r"(\d{" + str(hashes) + r"})")
        )
        frames: list[int] = []
        for p in exrs:
            m2 = frame_regex.fullmatch(p.name)
            if m2:
                frames.append(int(m2.group(1)))
        frame_range = (min(frames), max(frames)) if frames else (0, 0)
    else:
        frame_range = (0, 0)

    pixels, attrs = read_exr(exrs[0])
    h, w = pixels.shape[:2]
    par = float(attrs.get("pixelAspectRatio", 1.0))
    return pattern, frame_range, (w, h), par, exrs[0]


__all__ = ["ShotListPanel"]
