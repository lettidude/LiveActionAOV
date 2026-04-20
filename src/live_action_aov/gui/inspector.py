"""Inspector panel — right of the main window.

The heart of the M1 prep UX: the colorspace dropdown + provenance
label. The memory file
`project_gui_preflight_scenarios.md` documents exactly why this exists
— the ACTIONVFX pack routinely mis-tags EXRs, and the compositor
needs to see what the tool thinks the plate is (and why) so they can
override it with evidence rather than guess.

Layout:

    Shot:  <name>
    Frames: <start>–<end>  @  <W>×<H>

    Colorspace
    ┌──────────────────────────────┐
    │ [ auto: lin_rec709         ▾]│
    └──────────────────────────────┘
    auto: lin_rec709 (from `oiio:ColorSpace` header = 'lin_rec709')
    ⚠ low confidence (heuristic fallback)           (only when relevant)

    Exposure (EV)
    ┌────────────■──────────────────┐   [  0.0  ]
    -5                               +5

    Passes (read-only in M1)
    ☐ flow   ☐ depth   ☐ normals   ☐ matte

When the user changes either control, we mutate the `ShotState` and
emit `shot_updated` so the viewport re-renders with the new settings.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.pass_catalog import PASS_CATALOG
from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import SUPPORTED_COLORSPACES


def _category_of(model_key: str) -> str | None:
    """Which category section does this model belong to?

    Central lookup so the selection handler can drop the sibling keys
    in a model's category when a new radio is chosen. Returns None for
    keys the catalog doesn't know about (shouldn't happen, but we
    guard rather than KeyError on a stale shot).
    """
    for category, entries in PASS_CATALOG.items():
        for entry in entries:
            if entry.key == model_key:
                return category
    return None


class InspectorPanel(QWidget):
    def __init__(self, registry: ShotRegistry) -> None:
        super().__init__()
        self._registry = registry
        self._current: ShotState | None = None
        self._building = False  # guard against recursive signal fires during rebuilds

        # --- Shot identity labels ---
        self._name_label = QLabel("—")
        self._frames_label = QLabel("—")
        self._name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # --- Colorspace dropdown + provenance ---
        self._colorspace_box = QComboBox()
        self._colorspace_box.addItems(list(SUPPORTED_COLORSPACES))
        self._colorspace_box.currentTextChanged.connect(self._on_colorspace_changed)

        self._provenance_label = QLabel("(no shot selected)")
        self._provenance_label.setWordWrap(True)
        self._provenance_label.setStyleSheet("color: #888; font-size: 10pt;")

        self._low_conf_warning = QLabel("")
        self._low_conf_warning.setStyleSheet("color: #c08040; font-size: 10pt;")
        self._low_conf_warning.setWordWrap(True)

        # --- Exposure slider + spinbox ---
        self._exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self._exposure_slider.setRange(-50, 50)  # 0.1 EV granularity
        self._exposure_slider.setValue(0)
        self._exposure_slider.valueChanged.connect(self._on_exposure_slider_changed)

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(-5.0, 5.0)
        self._exposure_spin.setSingleStep(0.1)
        self._exposure_spin.setDecimals(2)
        self._exposure_spin.valueChanged.connect(self._on_exposure_spin_changed)

        exposure_row = QHBoxLayout()
        exposure_row.addWidget(self._exposure_slider, stretch=1)
        exposure_row.addWidget(self._exposure_spin)

        # Auto-exposure readout — computed when the shot is added.
        # Shows the user what the pipeline's analyze_clip picked and
        # why (median luma sample), so they can trust the default or
        # tweak with evidence.
        self._auto_ev_label = QLabel("")
        self._auto_ev_label.setStyleSheet("color: #888; font-size: 10pt;")
        self._auto_ev_label.setWordWrap(True)

        # --- Output location ---
        # Three-way radio: write next to the plate (default), drop into
        # a `utility/` subfolder, or route to an external render root
        # where each shot gets its own named subfolder. Last is what
        # most studios will want on shared drives.
        self._out_inplace = QRadioButton("Next to plate (default)")
        self._out_subfolder = QRadioButton("Subfolder of plate")
        self._out_external = QRadioButton("External root  (<root>/<name>/)")
        self._out_group = QButtonGroup(self)
        self._out_group.setExclusive(True)
        self._out_group.addButton(self._out_inplace, 0)
        self._out_group.addButton(self._out_subfolder, 1)
        self._out_group.addButton(self._out_external, 2)
        self._out_inplace.setChecked(True)
        # Per-radio `toggled` instead of QButtonGroup.idToggled — avoids
        # the double-fire (deselect + select) pattern that was causing
        # the Choose-root button to sometimes miss its enable event.
        self._out_inplace.toggled.connect(
            lambda c: c and self._set_output_mode("inplace")
        )
        self._out_subfolder.toggled.connect(
            lambda c: c and self._set_output_mode("subfolder")
        )
        self._out_external.toggled.connect(
            lambda c: c and self._set_output_mode("external")
        )

        # Subfolder name field — lets the user pick a name other than
        # "utility". Only enabled in subfolder mode.
        self._subfolder_name_edit = QLineEdit()
        self._subfolder_name_edit.setPlaceholderText("utility")
        self._subfolder_name_edit.setToolTip(
            "Folder name created inside the plate folder. Default: utility."
        )
        self._subfolder_name_edit.setEnabled(False)
        self._subfolder_name_edit.textEdited.connect(self._on_subfolder_name_edited)

        subfolder_row = QHBoxLayout()
        subfolder_row.addSpacing(24)
        subfolder_row.addWidget(QLabel("Name:"))
        subfolder_row.addWidget(self._subfolder_name_edit, stretch=1)

        # External-root controls — picker for the root, name field for
        # the per-shot subfolder within it.
        self._out_root_label = QLabel("<no root chosen>")
        self._out_root_label.setStyleSheet("color: #888; font-size: 10pt;")
        self._out_root_label.setWordWrap(True)
        self._out_root_btn = QPushButton("Choose root…")
        self._out_root_btn.setEnabled(False)
        self._out_root_btn.clicked.connect(self._on_pick_external_root)

        out_root_row = QHBoxLayout()
        out_root_row.addSpacing(24)
        out_root_row.addWidget(self._out_root_btn)
        out_root_row.addWidget(self._out_root_label, stretch=1)

        self._external_name_edit = QLineEdit()
        self._external_name_edit.setPlaceholderText("<shot name>")
        self._external_name_edit.setToolTip(
            "Subfolder name created inside the external root. "
            "Default: the shot's name."
        )
        self._external_name_edit.setEnabled(False)
        self._external_name_edit.textEdited.connect(self._on_external_name_edited)

        external_name_row = QHBoxLayout()
        external_name_row.addSpacing(24)
        external_name_row.addWidget(QLabel("Name:"))
        external_name_row.addWidget(self._external_name_edit, stretch=1)

        self._resolved_out_label = QLabel("")
        self._resolved_out_label.setStyleSheet("color: #aaa; font-size: 10pt;")
        self._resolved_out_label.setWordWrap(True)

        # --- Reset button ---
        # One-click rollback to the auto-detected colorspace + auto-
        # seeded exposure. Users experimenting with overrides want a
        # cheap "undo my exploration" escape hatch; saves them
        # reading the provenance label and re-typing the auto values.
        self._reset_btn = QPushButton("Reset to auto-detected defaults")
        self._reset_btn.setToolTip(
            "Revert colorspace override → auto and exposure → auto-seeded EV."
        )
        self._reset_btn.clicked.connect(self._on_reset_clicked)

        # --- Passes — THE main feature ---
        # One radio per model, grouped exclusively per category. The
        # sidecar has a single `Z` / `N.*` / matte slot per channel —
        # picking two depth models just has the second overwrite the
        # first, wasting compute. Radios make the single-pick nature
        # visually obvious; an explicit "Off" per category covers the
        # "I don't want depth at all" case.
        #
        # License marker sits next to each label (green = commercial,
        # amber = non-commercial). Flow has one entry + Off; Matte is
        # a single combo + Off; Depth and Normals are where the
        # single-select constraint really matters.
        self._model_radios: dict[str, QRadioButton] = {}
        self._off_radios: dict[str, QRadioButton] = {}
        self._category_groups: dict[str, QButtonGroup] = {}
        passes_block = QVBoxLayout()
        passes_block.setSpacing(4)
        for category, entries in PASS_CATALOG.items():
            header = QLabel(category)
            header.setStyleSheet(
                "font-weight: 600; color: #dcdcdc; padding: 8px 0 2px 0;"
            )
            header.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            passes_block.addWidget(header)

            group = QButtonGroup(self)
            group.setExclusive(True)
            self._category_groups[category] = group

            # "Off" is the default — users opt in per category. Keeping
            # it visible rather than hidden-by-default makes the "I
            # don't want depth right now" intent explicit.
            off_row = QWidget()
            off_row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            off_layout = QHBoxLayout(off_row)
            off_layout.setContentsMargins(16, 0, 0, 0)
            off_layout.setSpacing(6)
            off_radio = QRadioButton("Off")
            off_radio.setStyleSheet("color: #888;")
            off_radio.setChecked(True)
            off_radio.toggled.connect(
                lambda checked, cat=category: checked and self._on_category_off(cat)
            )
            self._off_radios[category] = off_radio
            group.addButton(off_radio)
            off_layout.addWidget(off_radio)
            off_layout.addStretch()
            passes_block.addWidget(off_row)

            for entry in entries:
                row = QWidget()
                row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(16, 0, 0, 0)
                row_layout.setSpacing(6)
                radio = QRadioButton(entry.label)
                radio.toggled.connect(
                    lambda checked, key=entry.key: checked and self._on_model_selected(key)
                )
                self._model_radios[entry.key] = radio
                group.addButton(radio)
                row_layout.addWidget(radio)
                row_layout.addStretch()
                colour = "#5ec864" if entry.commercial else "#e0a040"
                prefix = "" if entry.commercial else "⚠ "
                license_marker = QLabel(f"{prefix}{entry.license_tag}")
                license_marker.setStyleSheet(
                    f"color: {colour}; font-size: 10pt; padding-right: 2px;"
                )
                row_layout.addWidget(license_marker)
                passes_block.addWidget(row)

        # --- Assemble ---
        form = QFormLayout()
        form.addRow("Shot:", self._name_label)
        form.addRow("Frames:", self._frames_label)

        cs_block = QVBoxLayout()
        cs_block.addWidget(self._colorspace_box)
        cs_block.addWidget(self._provenance_label)
        cs_block.addWidget(self._low_conf_warning)

        # Passes section is the tool's main feature — we put it right
        # after the identity labels, above colourspace / exposure /
        # output so it reads as the primary control. Wrap in a scroll
        # area because the catalog can grow and we don't want to push
        # the rest of the inspector off-screen.
        passes_header = QLabel("PASSES")
        passes_header.setStyleSheet(
            "font-weight: 700; color: #fff; font-size: 11pt; "
            "padding: 4px 0; letter-spacing: 0.5px;"
        )
        passes_hint = QLabel(
            "Pick one or more models per category. New models "
            "appear here as we add them."
        )
        passes_hint.setStyleSheet("color: #999; font-size: 9pt;")
        passes_hint.setWordWrap(True)

        # --- Tabbed layout ---
        # Shot identity stays above the tabs so the user always knows
        # which shot they're editing regardless of which tab they're
        # on. Passes is the first/leftmost tab because it's the main
        # feature; Preview groups the viewport-relevant knobs
        # (colourspace, exposure, reset); Output isolates the
        # write-destination controls which are only touched once per
        # batch.

        # Passes tab.
        passes_tab = QWidget()
        passes_layout = QVBoxLayout(passes_tab)
        passes_layout.setContentsMargins(8, 8, 8, 8)
        passes_layout.addWidget(passes_header)
        passes_layout.addWidget(passes_hint)
        passes_layout.addLayout(passes_block)
        passes_layout.addStretch()

        # Preview tab — colourspace + exposure controls live together
        # because they both affect the viewport render.
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.addWidget(_section_label("Colorspace"))
        preview_layout.addLayout(cs_block)
        preview_layout.addSpacing(12)
        preview_layout.addWidget(_section_label("Exposure (EV)"))
        preview_layout.addLayout(exposure_row)
        preview_layout.addWidget(self._auto_ev_label)
        preview_layout.addSpacing(12)
        preview_layout.addWidget(self._reset_btn)
        preview_layout.addStretch()

        # Output tab.
        output_tab = QWidget()
        output_layout = QVBoxLayout(output_tab)
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.addWidget(_section_label("Output"))
        output_layout.addWidget(self._out_inplace)
        output_layout.addWidget(self._out_subfolder)
        output_layout.addLayout(subfolder_row)
        output_layout.addWidget(self._out_external)
        output_layout.addLayout(out_root_row)
        output_layout.addLayout(external_name_row)
        output_layout.addWidget(self._resolved_out_label)
        output_layout.addStretch()

        tabs = QTabWidget()
        tabs.addTab(passes_tab, "Passes")
        tabs.addTab(preview_tab, "Preview")
        tabs.addTab(output_tab, "Output")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.addLayout(form)  # Shot / Frames, always visible
        outer.addSpacing(4)
        outer.addWidget(tabs, stretch=1)

        # Wire registry signals last (after all widgets exist so the
        # refresh never lands against a half-built layout).
        self._registry.current_changed.connect(self._on_current_changed)
        self._registry.shot_updated.connect(self._on_shot_updated)

    # --- Registry → UI ---

    def _on_current_changed(self, shot: ShotState | None) -> None:
        self._current = shot
        self._rebuild_from_shot()

    def _on_shot_updated(self, shot: ShotState) -> None:
        # Only refresh if the updated shot is the one on screen.
        if shot is self._current:
            self._rebuild_from_shot()

    def _rebuild_from_shot(self) -> None:
        self._building = True
        try:
            shot = self._current
            if shot is None:
                self._name_label.setText("—")
                self._frames_label.setText("—")
                self._provenance_label.setText("(no shot selected)")
                self._low_conf_warning.setText("")
                self._colorspace_box.setCurrentText("auto")
                self._exposure_slider.setValue(0)
                self._exposure_spin.setValue(0.0)
                return

            self._name_label.setText(shot.name)
            start, end = shot.frame_range
            w, h = shot.resolution
            self._frames_label.setText(f"{start}–{end}   @   {w}×{h}")

            # Populate dropdown from the supported list plus any exotic
            # tag the detector surfaced verbatim (e.g. studio-internal
            # colorspace names we don't recognise).
            self._refresh_colorspace_options(shot)

            # Pick the selection: override wins, else "auto".
            self._colorspace_box.setCurrentText(shot.override or "auto")
            self._provenance_label.setText(shot.colorspace_label())

            if shot.detected is not None and not shot.detected.confident:
                self._low_conf_warning.setText(
                    "⚠ low confidence — double-check the preview against a known-good display."
                )
            else:
                self._low_conf_warning.setText("")

            ev = float(shot.exposure_ev)
            self._exposure_slider.setValue(int(round(ev * 10)))
            self._exposure_spin.setValue(ev)

            # Output location radios.
            mode_to_button = {
                "inplace": self._out_inplace,
                "subfolder": self._out_subfolder,
                "external": self._out_external,
            }
            # Per-radio blockSignals — QButtonGroup's blockSignals is
            # unreliable in exclusive mode.
            for rb in (self._out_inplace, self._out_subfolder, self._out_external):
                rb.blockSignals(True)
            mode_to_button.get(shot.output_mode, self._out_inplace).setChecked(True)
            for rb in (self._out_inplace, self._out_subfolder, self._out_external):
                rb.blockSignals(False)
            self._subfolder_name_edit.setEnabled(shot.output_mode == "subfolder")
            self._out_root_btn.setEnabled(shot.output_mode == "external")
            self._external_name_edit.setEnabled(shot.output_mode == "external")
            self._subfolder_name_edit.setText(shot.output_subfolder_name)
            self._external_name_edit.setText(shot.output_external_name)
            self._rebuild_resolved_out_label()

            # Pass radios reflect the stored enabled_models. The single-
            # select constraint is enforced in the selection handler;
            # here we just surface whatever's in the state — if more
            # than one model from a category slipped in (legacy YAMLs,
            # CLI handoff), we pick the first one to display and leave
            # the state alone until the user edits.
            enabled = set(shot.enabled_models)
            for category, entries in PASS_CATALOG.items():
                # Block signals on the whole group so programmatic
                # setChecked doesn't fire the handler and clobber state.
                group = self._category_groups[category]
                for btn in group.buttons():
                    btn.blockSignals(True)
                picked: str | None = None
                for entry in entries:
                    if entry.key in enabled:
                        picked = entry.key
                        break
                if picked is not None:
                    self._model_radios[picked].setChecked(True)
                else:
                    self._off_radios[category].setChecked(True)
                for btn in group.buttons():
                    btn.blockSignals(False)

            # Render the auto-EV provenance line. Matches the format
            # of the colorspace line: value + source + a small evidence
            # number so the user can sanity-check the pipeline's guess.
            if shot.auto_ev is not None:
                luma_str = (
                    f" (sampled luma p50 = {shot.sampled_luma:.3f})"
                    if shot.sampled_luma is not None
                    else ""
                )
                delta = ev - float(shot.auto_ev)
                if abs(delta) < 0.05:
                    self._auto_ev_label.setText(
                        f"auto: {shot.auto_ev:+.2f} EV{luma_str}"
                    )
                else:
                    self._auto_ev_label.setText(
                        f"manual: {ev:+.2f} EV   (auto was {shot.auto_ev:+.2f}{luma_str})"
                    )
            else:
                self._auto_ev_label.setText("")
        finally:
            self._building = False

    def _refresh_colorspace_options(self, shot: ShotState) -> None:
        """Make sure any exotic detected colorspace name (studio
        internal, etc.) also appears in the dropdown as an option."""
        current = self._colorspace_box.currentText()
        desired = list(SUPPORTED_COLORSPACES)
        if shot.detected is not None and shot.detected.detected not in desired:
            desired.append(shot.detected.detected)
        # Only rebuild if the set changed, to avoid flicker.
        actual = [self._colorspace_box.itemText(i) for i in range(self._colorspace_box.count())]
        if actual != desired:
            self._colorspace_box.blockSignals(True)
            self._colorspace_box.clear()
            self._colorspace_box.addItems(desired)
            if current in desired:
                self._colorspace_box.setCurrentText(current)
            self._colorspace_box.blockSignals(False)

    # --- UI → Shot state ---

    def _on_colorspace_changed(self, choice: str) -> None:
        if self._building or self._current is None:
            return
        self._current.override = None if choice == "auto" else choice
        # Update the provenance label immediately — don't wait for the
        # round-trip through the registry.
        self._provenance_label.setText(self._current.colorspace_label())
        self._registry.notify_updated(self._current)

    def _on_exposure_slider_changed(self, value: int) -> None:
        if self._building or self._current is None:
            return
        ev = value / 10.0
        self._current.exposure_ev = ev
        self._exposure_spin.blockSignals(True)
        self._exposure_spin.setValue(ev)
        self._exposure_spin.blockSignals(False)
        self._registry.notify_updated(self._current)

    def _on_exposure_spin_changed(self, ev: float) -> None:
        if self._building or self._current is None:
            return
        self._current.exposure_ev = float(ev)
        self._exposure_slider.blockSignals(True)
        self._exposure_slider.setValue(int(round(ev * 10)))
        self._exposure_slider.blockSignals(False)
        self._registry.notify_updated(self._current)

    def _set_output_mode(self, mode: str) -> None:
        """Radio handler — runs once per new selection (not for the
        deselected button), so it's safe to trust `mode` directly."""
        if self._building or self._current is None:
            return
        self._current.output_mode = mode
        self._subfolder_name_edit.setEnabled(mode == "subfolder")
        self._out_root_btn.setEnabled(mode == "external")
        self._external_name_edit.setEnabled(mode == "external")
        self._rebuild_resolved_out_label()
        self._registry.notify_updated(self._current)

    def _on_subfolder_name_edited(self, text: str) -> None:
        if self._building or self._current is None:
            return
        self._current.output_subfolder_name = text.strip() or "utility"
        self._rebuild_resolved_out_label()
        self._registry.notify_updated(self._current)

    def _on_external_name_edited(self, text: str) -> None:
        if self._building or self._current is None:
            return
        self._current.output_external_name = text.strip()  # empty → default to shot.name
        self._rebuild_resolved_out_label()
        self._registry.notify_updated(self._current)

    def _on_pick_external_root(self) -> None:
        if self._current is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose output root")
        if not folder:
            return
        self._current.output_external_root = Path(folder)
        self._rebuild_resolved_out_label()
        self._registry.notify_updated(self._current)

    def _rebuild_resolved_out_label(self) -> None:
        if self._current is None:
            self._resolved_out_label.setText("")
            return
        root = self._current.output_external_root
        if self._current.output_mode == "external" and root is None:
            self._out_root_label.setText("<no root chosen>")
            self._resolved_out_label.setText(
                "⚠ Pick an external root before submitting."
            )
            return
        if root is not None:
            self._out_root_label.setText(str(root))
        resolved = self._current.resolve_output_dir()
        self._resolved_out_label.setText(f"→ {resolved}")

    def _on_model_selected(self, key: str) -> None:
        """Radio turned on — enforce single-select per category by
        dropping any other model from this key's category from
        `enabled_models` and adding `key` itself."""
        if self._building or self._current is None:
            return
        category = _category_of(key)
        if category is None:
            return
        # Collect the keys that belong to this category from the catalog
        # so we can prune them all in one pass.
        category_keys = {e.key for e in PASS_CATALOG.get(category, [])}
        enabled = [k for k in self._current.enabled_models if k not in category_keys]
        enabled.append(key)
        self._current.enabled_models = enabled
        self._registry.notify_updated(self._current)

    def _on_category_off(self, category: str) -> None:
        """The category's "Off" radio was turned on — drop every model
        belonging to that category from `enabled_models`."""
        if self._building or self._current is None:
            return
        category_keys = {e.key for e in PASS_CATALOG.get(category, [])}
        self._current.enabled_models = [
            k for k in self._current.enabled_models if k not in category_keys
        ]
        self._registry.notify_updated(self._current)

    def _on_reset_clicked(self) -> None:
        """Revert both knobs to auto-detected values in one shot."""
        if self._current is None:
            return
        self._current.override = None
        self._current.exposure_ev = float(self._current.auto_ev or 0.0)
        # Rebuild flips both controls back and updates the provenance
        # labels; then a single `notify_updated` triggers the viewport
        # re-render with the reset settings.
        self._rebuild_from_shot()
        self._registry.notify_updated(self._current)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: #ccc;")
    return lbl


__all__ = ["InspectorPanel"]
