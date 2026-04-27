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
from PySide6.QtGui import QBrush, QColor, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut
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
        # When the user toggles the per-shot checkbox we need to mirror
        # it onto ShotState.queued. `itemChanged` fires for any item
        # data change (including colour updates from rebuild), so we
        # guard reentrancy with a flag.
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._suppress_item_change = False
        # Pin a minimum width so the "Add shot…" button + shortest shot
        # name don't clip when the splitter is squashed.
        self.setMinimumWidth(180)

        add_btn = QPushButton("Add shot…")
        add_btn.clicked.connect(self._on_add_clicked)

        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove the selected shot (Delete / Backspace).")
        remove_btn.clicked.connect(self._on_remove_clicked)

        top = QHBoxLayout()
        top.addWidget(add_btn)
        top.addWidget(remove_btn)
        top.addStretch()

        # Keyboard shortcut: Delete / Backspace on a focused list row.
        # Scope to the list widget so global typing elsewhere isn't
        # swallowed — Qt treats "WidgetShortcut" as "only when this
        # widget has focus".
        for key in (QKeySequence.StandardKey.Delete, QKeySequence("Backspace")):
            sc = QShortcut(key, self._list)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self._on_remove_clicked)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._list)

        # Drag-and-drop: accept folder drops.
        self.setAcceptDrops(True)

        # Stay in sync when shots are added/removed elsewhere.
        self._registry.shot_added.connect(self._on_shot_added)
        self._registry.shot_removed.connect(self._on_shot_removed)
        # Status changes (running / done / failed) need to refresh the
        # row label so the list is the at-a-glance submit dashboard.
        self._registry.shot_updated.connect(self._on_shot_updated)

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

    def _on_remove_clicked(self) -> None:
        item = self._list.currentItem()
        # PySide6 stubs annotate `currentItem()` as never-None; runtime it
        # absolutely can be None when the list is empty. Keep the runtime
        # guard, suppress mypy's stub-driven "unreachable" complaint.
        # `unused-ignore` covers stub updates that fix it upstream.
        if item is None:
            return  # type: ignore[unreachable, unused-ignore]
        shot = item.data(Qt.ItemDataRole.UserRole)
        if shot is not None:
            self._registry.remove(shot)

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
        # Per-row checkbox drives ShotState.queued — determines which
        # shots Submit local will iterate.
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if shot.queued else Qt.CheckState.Unchecked)
        # Full info in a tooltip so the label can truncate without losing
        # data — compers hover to confirm frame range + resolution when
        # the panel is sized narrow.
        item.setToolTip(_format_item_tooltip(shot))
        _apply_status_colour(item, shot)
        self._list.addItem(item)
        self._list.setCurrentItem(item)

    def _on_shot_removed(self, shot: ShotState) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is shot:
                self._list.takeItem(i)
                return

    def _on_shot_updated(self, shot: ShotState) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is shot:
                self._suppress_item_change = True
                try:
                    item.setText(_format_item_label(shot))
                    item.setToolTip(_format_item_tooltip(shot))
                    # Keep the checkbox state in sync when another
                    # surface (e.g. a "queue all" button) toggles it.
                    desired = Qt.CheckState.Checked if shot.queued else Qt.CheckState.Unchecked
                    if item.checkState() != desired:
                        item.setCheckState(desired)
                    _apply_status_colour(item, shot)
                finally:
                    self._suppress_item_change = False
                return

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        """Sync the per-row checkbox → ShotState.queued. Called by the
        list whenever item data changes; guarded against reentrant
        refreshes from `_on_shot_updated`."""
        if self._suppress_item_change:
            return
        shot = item.data(Qt.ItemDataRole.UserRole)
        if shot is None:
            return
        new_queued = item.checkState() == Qt.CheckState.Checked
        if shot.queued != new_queued:
            shot.queued = new_queued
            self._registry.notify_updated(shot)

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


_STATUS_MARKER = {
    "new": "",
    "queued": "• queued",
    "running": "⟳ running",
    "done": "✓ done",
    "failed": "✗ failed",
}

# Foreground colours per status — keeps the default row background so
# selection highlight still reads. "done" lifts to a saturated green,
# "failed" drops to red, "running" glows cyan.
_STATUS_COLOURS: dict[str, QColor] = {
    "running": QColor("#4aa8ff"),
    "done": QColor("#5ec864"),
    "failed": QColor("#e65c5c"),
}


def _apply_status_colour(item: QListWidgetItem, shot: ShotState) -> None:
    colour = _STATUS_COLOURS.get(shot.status)
    brush = QBrush(colour) if colour is not None else QBrush()
    item.setForeground(brush)


def _format_item_label(shot: ShotState) -> str:
    """One-line label for the shot list row.

    Kept short so a narrow panel doesn't truncate the shot name itself.
    Status marker appended when the shot has been submitted so the list
    doubles as a submit dashboard. Full info lives in the tooltip.
    """
    start, end = shot.frame_range
    n = end - start + 1
    marker = _STATUS_MARKER.get(shot.status, "")
    suffix = f"   {marker}" if marker else ""
    return f"{shot.name}   ({n} fr){suffix}"


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


def _sniff_sequence(
    folder: Path,
) -> tuple[str, tuple[int, int], tuple[int, int], float, Path]:
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
