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

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.pass_catalog import PASS_CATALOG
from live_action_aov.gui.shot_state import ClickInstance, ShotRegistry, ShotState
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
    # Click-to-mask wiring (MainWindow connects these to the ViewportPanel):
    # arm/disarm viewport point placement, and which ClickInstance is active.
    click_mode_changed = Signal(bool)
    active_click_instance_changed = Signal(object)  # ClickInstance | None
    # "Preview mask (this frame)" — MainWindow routes this to the viewport,
    # which owns the preview image + the single-frame SAM 3 engine.
    mask_preview_requested = Signal()

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
        # Conditional `if c else None` form (rather than `c and …`) so
        # mypy sees the lambda as `() -> None` instead of an expression
        # that returns the value of a void function.
        self._out_inplace.toggled.connect(lambda c: self._set_output_mode("inplace") if c else None)
        self._out_subfolder.toggled.connect(
            lambda c: self._set_output_mode("subfolder") if c else None
        )
        self._out_external.toggled.connect(
            lambda c: self._set_output_mode("external") if c else None
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
            "Subfolder name created inside the external root. Default: the shot's name."
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

        # Proxy resolution — fast-iteration dropdown. Plate-native
        # is the default; users drop to 540p / 720p / 1080p for
        # quick passes before committing to a full-res run.
        self._proxy_combo = QComboBox()
        self._proxy_combo.addItem("Full plate (native)", None)
        self._proxy_combo.addItem("1080p  (long edge 1920)", 1920)
        self._proxy_combo.addItem("720p  (long edge 1280)", 1280)
        self._proxy_combo.addItem("540p  (long edge 960)", 960)
        self._proxy_combo.setToolTip(
            "Resize every plate frame to this long edge before passes "
            "see it. Sidecars are written at the same resolution. Use "
            "for fast iteration; switch back to Full plate for delivery."
        )
        self._proxy_combo.currentIndexChanged.connect(self._on_proxy_changed)

        # Delivery: EXR codec + bit depth. Bandwidth-bound jobs (separate
        # machines, limited share) need compact sidecars. Item data is a
        # (compression, dtype) tuple. Cryptomatte is auto-split to a lossless
        # float32 sibling by the writer when a lossy/half delivery is chosen,
        # so its hash IDs always survive.
        self._delivery_combo = QComboBox()
        self._delivery_combo.addItem("Lossless 32-bit (ZIP) - largest", ("zip", "float32"))
        self._delivery_combo.addItem("Lossless 16-bit half (ZIP)", ("zip", "float16"))
        self._delivery_combo.addItem(
            "Compact - DWAB 16-bit (recommended for delivery)", ("dwab:45", "float16")
        )
        self._delivery_combo.addItem("Compact max - DWAB high 16-bit", ("dwab:120", "float16"))
        self._delivery_combo.setToolTip(
            "How sidecar EXRs are compressed. DWAB is lossy but ~5-20x smaller "
            "on depth/normals/flow/mattes (fine for utility passes and temp "
            "roto). Cryptomatte is automatically split into a lossless .crypto "
            "EXR so its IDs are never corrupted."
        )
        self._delivery_combo.currentIndexChanged.connect(self._on_delivery_changed)

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
        # Collapsible section containers so users can hide categories
        # they aren't using — matters more as the catalog grows with
        # future models (fresnel, alt-depth backends, etc.).
        self._category_sections: dict[str, QWidget] = {}
        self._category_headers: dict[str, QPushButton] = {}
        passes_block = QVBoxLayout()
        passes_block.setSpacing(4)
        for category, entries in PASS_CATALOG.items():
            # Header button — flat, label-like, but clickable to toggle
            # the section below. Chevron prefix makes expand/collapse
            # state readable at a glance.
            header_btn = QPushButton(f"▾  {category}")
            header_btn.setCheckable(True)
            header_btn.setChecked(True)
            header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            header_btn.setStyleSheet(
                "QPushButton {"
                " font-weight: 600; color: #dcdcdc; text-align: left;"
                " padding: 8px 0 2px 0; border: none; background: transparent;"
                "}"
                "QPushButton:hover { color: #ffffff; }"
            )
            header_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self._category_headers[category] = header_btn
            passes_block.addWidget(header_btn)

            # One section widget per category wraps every row inside —
            # show/hide toggles visibility as a unit without the rows
            # needing to know about the header.
            section = QWidget()
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(4)
            self._category_sections[category] = section

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
                lambda checked, cat=category: self._on_category_off(cat) if checked else None
            )
            self._off_radios[category] = off_radio
            group.addButton(off_radio)
            off_layout.addWidget(off_radio)
            off_layout.addStretch()
            section_layout.addWidget(off_row)

            for entry in entries:
                row = QWidget()
                row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(16, 0, 0, 0)
                row_layout.setSpacing(6)
                radio = QRadioButton(entry.label)
                radio.toggled.connect(
                    lambda checked, key=entry.key: self._on_model_selected(key) if checked else None
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
                section_layout.addWidget(row)

            passes_block.addWidget(section)
            # Wire the header click — late so the closure captures the
            # right section widget. Toggling hides the section as a
            # unit (no layout churn) and updates the chevron arrow.
            header_btn.toggled.connect(
                lambda checked, h=header_btn, s=section, c=category: (
                    s.setVisible(checked),
                    h.setText(f"{'▾' if checked else '▸'}  {c}"),
                )
            )

        # SAM 3 concepts — free-text list of what SAM 3 should detect.
        # Drives BOTH the Matte and Cryptomatte passes (they share the
        # `sam3_matte` detector). SAM 3 is concept-prompted: it only
        # finds what you ask for, so an empty/black matte usually just
        # means the subject wasn't on the list. Empty here = the pass's
        # built-in default concepts.
        self._sam3_concepts_edit = QLineEdit()
        self._sam3_concepts_edit.setPlaceholderText(
            "person, vehicle, tree, building, sky, water, animal"
        )
        self._sam3_concepts_edit.setToolTip(
            "What SAM 3 should detect, comma-separated (e.g. \"person, red car, dog\").\n"
            "Drives both the Matte and Cryptomatte passes. SAM 3 only finds the\n"
            "concepts you prompt — leave empty to use the built-in defaults."
        )
        self._sam3_concepts_edit.textEdited.connect(self._on_sam3_concepts_edited)

        concepts_label = QLabel("SAM 3 detects")
        concepts_label.setStyleSheet(
            "font-weight: 600; color: #ddd; font-size: 9pt; padding-top: 6px;"
        )
        concepts_hint = QLabel(
            "Comma-separated. Matte + Cryptomatte. Empty = defaults."
        )
        concepts_hint.setStyleSheet("color: #999; font-size: 8pt;")
        concepts_hint.setWordWrap(True)
        passes_block.addSpacing(8)
        passes_block.addWidget(concepts_label)
        passes_block.addWidget(self._sam3_concepts_edit)
        passes_block.addWidget(concepts_hint)

        # Refiner options live HERE with the model choice — what gets
        # created belongs on the Passes tab. The Masks tab only previews
        # (it shows a note with the current refiner — the tabs communicate).
        # BiRefNet weights dropdown: used by the BiRefNet combo; ignored by
        # ViTMatte (trimap, fixed weights) and RVM.
        self._refiner_model_combo = QComboBox()
        self._refiner_model_combo.addItem("BiRefNet Portrait - people/hair (default)", "")
        self._refiner_model_combo.addItem(
            "BiRefNet Matting - general soft", "ZhengPeng7/BiRefNet-matting"
        )
        self._refiner_model_combo.addItem(
            "BiRefNet General - hard, cleanest licence", "ZhengPeng7/BiRefNet"
        )
        self._refiner_model_combo.addItem(
            "RMBG-2.0 (BiRefNet arch) - paid licence for commercial", "briaai/RMBG-2.0"
        )
        self._refiner_model_combo.setToolTip(
            "Soft-edge refiner weights for this shot (BiRefNet combos only). "
            "Used by the Masks-tab preview AND by the submit. Same speed; "
            "they differ in edge softness (Portrait best on hair) + licence."
        )
        self._refiner_model_combo.currentIndexChanged.connect(self._on_refiner_model_changed)

        self._refine_all_check = QCheckBox("Refine ALL mask.<name> channels at submit (slower)")
        self._refine_all_check.setToolTip(
            "Refine every detected/clicked object at submit so each mask.<name> "
            "gets roto-grade soft edges, not just the 4 hero matte slots. "
            "Costs one refinement per object. The preview is always soft."
        )
        self._refine_all_check.toggled.connect(self._on_refine_all_toggled)

        refiner_weights_row = QHBoxLayout()
        self._refiner_weights_label = QLabel("BiRefNet weights:")
        refiner_weights_row.addWidget(self._refiner_weights_label)
        refiner_weights_row.addWidget(self._refiner_model_combo, stretch=1)
        passes_block.addSpacing(8)
        passes_block.addLayout(refiner_weights_row)
        passes_block.addWidget(self._refine_all_check)

        # "Apply to all shots" — one-click broadcast of the active
        # shot's pass selections to every other shot in the queue.
        # Unlike Output (which is always session-wide), Passes stay
        # per-shot by default because users might want different
        # models per shot; the button is the escape hatch when they
        # don't.
        self._apply_passes_btn = QPushButton("Apply passes to all shots")
        self._apply_passes_btn.setToolTip(
            "Copy the currently-selected model in each category "
            "(incl. Off choices) to every other shot in the list."
        )
        self._apply_passes_btn.clicked.connect(self._on_apply_passes_to_all)
        passes_block.addSpacing(8)
        passes_block.addWidget(self._apply_passes_btn)

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
            "font-weight: 700; color: #fff; font-size: 11pt; padding: 4px 0; letter-spacing: 0.5px;"
        )
        passes_hint = QLabel(
            "Pick one or more models per category. New models appear here as we add them."
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

        # Helper — every tab wraps its content in a scroll area so when
        # the log panel at the bottom of the main window eats vertical
        # space, rows stop collapsing on top of each other. User
        # reported the Passes rows overlapping when the log opened;
        # a scroll area keeps natural heights and adds a scrollbar
        # when needed.
        def _scrollable(content_layout: QVBoxLayout) -> QScrollArea:
            content = QWidget()
            content.setLayout(content_layout)
            sa = QScrollArea()
            sa.setWidget(content)
            sa.setWidgetResizable(True)
            # Horizontal scrollbar AS-NEEDED (was AlwaysOff): when the panel
            # is narrow, long model labels + license tags overflowed and were
            # clipped with no way to reach them. With widgetResizable the
            # content still fills the viewport when wide; the bar only appears
            # when the longest row can't fit, so the user can scroll to the
            # truncated license tags.
            sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            sa.setStyleSheet("QScrollArea { border: none; }")
            return sa

        # Passes tab.
        passes_layout = QVBoxLayout()
        passes_layout.setContentsMargins(8, 8, 8, 8)
        passes_layout.addWidget(passes_header)
        passes_layout.addWidget(passes_hint)
        passes_layout.addLayout(passes_block)
        passes_layout.addStretch()
        passes_tab = _scrollable(passes_layout)

        # Preview tab — colourspace + exposure controls live together
        # because they both affect the viewport render.
        preview_layout = QVBoxLayout()
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
        preview_tab = _scrollable(preview_layout)

        # Output tab.
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.addWidget(_section_label("Output"))
        session_wide_note = QLabel("These settings apply to every shot in the session.")
        session_wide_note.setStyleSheet("color: #777; font-size: 8pt; font-style: italic;")
        output_layout.addWidget(session_wide_note)
        output_layout.addWidget(self._out_inplace)
        output_layout.addWidget(self._out_subfolder)
        output_layout.addLayout(subfolder_row)
        output_layout.addWidget(self._out_external)
        output_layout.addLayout(out_root_row)
        output_layout.addLayout(external_name_row)
        output_layout.addWidget(self._resolved_out_label)
        output_layout.addSpacing(16)
        output_layout.addWidget(_section_label("Proxy resolution"))
        proxy_hint = QLabel(
            "Resize plate before passes run; sidecars land at this "
            "resolution. Use Full plate for delivery."
        )
        proxy_hint.setStyleSheet("color: #888; font-size: 9pt;")
        proxy_hint.setWordWrap(True)
        output_layout.addWidget(proxy_hint)
        output_layout.addWidget(self._proxy_combo)
        output_layout.addSpacing(16)
        output_layout.addWidget(_section_label("Delivery (size vs fidelity)"))
        delivery_hint = QLabel(
            "EXR compression + bit depth. DWAB is lossy but far smaller - "
            "use it for bandwidth-bound delivery between machines. "
            "Cryptomatte is auto-split to a lossless sidecar so its IDs survive."
        )
        delivery_hint.setStyleSheet("color: #888; font-size: 9pt;")
        delivery_hint.setWordWrap(True)
        output_layout.addWidget(delivery_hint)
        output_layout.addWidget(self._delivery_combo)
        output_layout.addStretch()
        output_tab = _scrollable(output_layout)

        # Masks tab — interactive click-to-mask (v0.3). Entirely OPTIONAL:
        # an empty object list changes nothing at submit. Each object here
        # is a ClickInstance: the user arms point placement, clicks the
        # element in the viewport (left = include, right = exclude), and at
        # submit the SAM 3 tracker propagates it across the shot into a
        # mask.<name> channel + Cryptomatte ID.
        masks_hint = QLabel(
            "Drag a box or click points (left = include, right = exclude) "
            "to seed a tracked mask.<name> + Cryptomatte ID. "
            "FEW INPUTS WORK BEST: a box or 2-6 points, Preview, then one "
            "corrective click."
        )
        masks_hint.setStyleSheet("color: #999; font-size: 9pt;")
        masks_hint.setWordWrap(True)

        self._mask_mode_check = QCheckBox("Place points in viewport")
        self._mask_mode_check.setToolTip(
            "Arms the viewport: clicks add points to the selected object "
            "instead of doing nothing. Disable to scrub freely."
        )
        self._mask_mode_check.toggled.connect(self._on_mask_mode_toggled)

        self._mask_list = QListWidget()
        self._mask_list.setFixedHeight(140)
        self._mask_list.currentRowChanged.connect(self._on_mask_selected)

        self._mask_name_edit = QLineEdit()
        self._mask_name_edit.setPlaceholderText("object name (→ Cryptomatte)")
        self._mask_name_edit.textEdited.connect(self._on_mask_name_edited)

        self._mask_new_btn = QPushButton("New object")
        self._mask_new_btn.clicked.connect(self._on_mask_new)
        self._mask_del_btn = QPushButton("Delete")
        self._mask_del_btn.clicked.connect(self._on_mask_delete)
        self._mask_undo_btn = QPushButton("Undo point")
        self._mask_undo_btn.setToolTip("Remove the last point added to the selected object.")
        self._mask_undo_btn.clicked.connect(self._on_mask_undo_point)
        self._mask_clear_btn = QPushButton("Clear points")
        self._mask_clear_btn.clicked.connect(self._on_mask_clear_points)

        # On-demand single-frame SAM 3 preview: see what the points produce
        # on the seed frame BEFORE running the whole shot. The heavy lifting
        # lives in the viewport (which owns the preview image); this button
        # just asks for it.
        self._mask_preview_btn = QPushButton("Preview mask (this frame)")
        self._mask_preview_btn.setToolTip(
            "Runs SAM 3 on the seed frame only and overlays the resulting "
            "mask in the viewport. First use loads the model (one-time "
            "wait); after that it's quick. Adjust points and preview again."
        )
        self._mask_preview_btn.clicked.connect(self.mask_preview_requested.emit)

        mask_btn_row = QHBoxLayout()
        mask_btn_row.addWidget(self._mask_new_btn)
        mask_btn_row.addWidget(self._mask_del_btn)
        mask_btn_row.addWidget(self._mask_undo_btn)
        mask_btn_row.addWidget(self._mask_clear_btn)

        masks_layout = QVBoxLayout()
        masks_layout.setContentsMargins(8, 8, 8, 8)
        masks_layout.addWidget(_section_label("Click-to-mask"))
        masks_layout.addWidget(masks_hint)
        masks_layout.addSpacing(6)
        masks_layout.addWidget(self._mask_mode_check)
        masks_layout.addLayout(mask_btn_row)
        # Too-many-points guardrail. Empirically verified on real plates:
        # 3+2 points → full-object mask (34.8% of frame); 50+ points → the
        # mask collapses to nearly nothing (1.6%). SAM's interactive
        # training regime is few-click; warn before the user dots the
        # object to death and concludes the tool is broken.
        self._mask_points_warn = QLabel()
        self._mask_points_warn.setStyleSheet("color: #e0a040; font-size: 9pt;")
        self._mask_points_warn.setWordWrap(True)
        self._mask_points_warn.setVisible(False)

        masks_layout.addSpacing(6)
        # PREVIEW engine override — the run refiner is chosen on the Passes
        # tab; here you can preview with ANY engine to eye-compare per shot
        # before committing. "Same as run" mirrors the Passes choice.
        self._preview_refiner_combo = QComboBox()
        self._preview_refiner_combo.addItem("Same as run (Passes tab)", "auto")
        self._preview_refiner_combo.addItem("ViTMatte (trimap)", "vitmatte")
        self._preview_refiner_combo.addItem("BiRefNet Portrait", "birefnet:")
        self._preview_refiner_combo.addItem(
            "BiRefNet Matting", "birefnet:ZhengPeng7/BiRefNet-matting"
        )
        self._preview_refiner_combo.addItem("BiRefNet General", "birefnet:ZhengPeng7/BiRefNet")
        self._preview_refiner_combo.addItem("RMBG-2.0", "birefnet:briaai/RMBG-2.0")
        self._preview_refiner_combo.addItem("RVM", "rvm")
        self._preview_refiner_combo.setToolTip(
            "Which refiner the 'Preview mask' button applies — preview-only, "
            "does NOT change what the submit runs (that's the Passes tab). "
            "Use it to eye-compare algorithms on this shot, then pick the "
            "final one in Passes."
        )
        self._preview_refiner_combo.currentIndexChanged.connect(self._on_preview_refiner_changed)
        preview_with_row = QHBoxLayout()
        preview_with_row.addWidget(QLabel("Preview with:"))
        preview_with_row.addWidget(self._preview_refiner_combo, stretch=1)
        masks_layout.addLayout(preview_with_row)

        # Note mirroring what the RUN will use — how the two tabs communicate.
        self._preview_refiner_note = QLabel("")
        self._preview_refiner_note.setStyleSheet("color: #888; font-size: 8pt;")
        self._preview_refiner_note.setWordWrap(True)
        masks_layout.addWidget(self._preview_refiner_note)
        masks_layout.addWidget(self._mask_preview_btn)
        masks_layout.addWidget(self._mask_list)
        masks_layout.addWidget(self._mask_points_warn)
        masks_layout.addWidget(QLabel("Name:"))
        masks_layout.addWidget(self._mask_name_edit)
        masks_layout.addStretch()
        masks_tab = _scrollable(masks_layout)

        tabs = QTabWidget()
        tabs.addTab(passes_tab, "Passes")
        tabs.addTab(masks_tab, "Masks")
        tabs.addTab(preview_tab, "Colour")
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
                self._refresh_mask_list()
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
            # Match the proxy combo's current index to the shot value.
            # itemData returns the stored payload; None = Full plate
            # (index 0), 1920/1280/960 = 1/2/3. Any other value falls
            # through to Full plate.
            proxy_index = 0
            for i in range(self._proxy_combo.count()):
                if self._proxy_combo.itemData(i) == shot.proxy_long_edge:
                    proxy_index = i
                    break
            self._proxy_combo.blockSignals(True)
            self._proxy_combo.setCurrentIndex(proxy_index)
            self._proxy_combo.blockSignals(False)
            # Match the delivery combo to the shot's (compression, dtype).
            delivery_index = 0
            want = (shot.delivery_compression, shot.delivery_dtype)
            for i in range(self._delivery_combo.count()):
                if self._delivery_combo.itemData(i) == want:
                    delivery_index = i
                    break
            self._delivery_combo.blockSignals(True)
            self._delivery_combo.setCurrentIndex(delivery_index)
            self._delivery_combo.blockSignals(False)
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

            # SAM 3 concepts free-text (drives Matte + Cryptomatte).
            self._sam3_concepts_edit.setText(shot.sam3_concepts)

            self._refine_all_check.blockSignals(True)
            self._refine_all_check.setChecked(bool(shot.refine_all_masks))
            self._refine_all_check.blockSignals(False)

            rm_index = 0
            for i in range(self._refiner_model_combo.count()):
                if self._refiner_model_combo.itemData(i) == shot.refiner_model:
                    rm_index = i
                    break
            self._refiner_model_combo.blockSignals(True)
            self._refiner_model_combo.setCurrentIndex(rm_index)
            self._refiner_model_combo.blockSignals(False)
            pr_index = 0
            for i in range(self._preview_refiner_combo.count()):
                if self._preview_refiner_combo.itemData(i) == shot.preview_refiner:
                    pr_index = i
                    break
            self._preview_refiner_combo.blockSignals(True)
            self._preview_refiner_combo.setCurrentIndex(pr_index)
            self._preview_refiner_combo.blockSignals(False)
            self._update_preview_refiner_note()

            # Click-to-mask object list (Masks tab).
            self._refresh_mask_list()

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
                    self._auto_ev_label.setText(f"auto: {shot.auto_ev:+.2f} EV{luma_str}")
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
        self._broadcast_output_to_all_shots()

    def _on_subfolder_name_edited(self, text: str) -> None:
        if self._building or self._current is None:
            return
        self._current.output_subfolder_name = text.strip() or "utility"
        self._rebuild_resolved_out_label()
        self._broadcast_output_to_all_shots()

    def _on_external_name_edited(self, text: str) -> None:
        if self._building or self._current is None:
            return
        self._current.output_external_name = text.strip()  # empty → default to shot.name
        self._rebuild_resolved_out_label()
        self._broadcast_output_to_all_shots()

    def _on_sam3_concepts_edited(self, text: str) -> None:
        """SAM 3 concepts edited — stored verbatim (comma-separated).
        Parsing into a list happens at submit time in the worker, so a
        half-typed "person, " mid-edit isn't destroyed. Per-shot like
        the model selection; the "Apply passes to all" button copies it
        alongside enabled_models."""
        if self._building or self._current is None:
            return
        self._current.sam3_concepts = text
        self._registry.notify_updated(self._current)

    # --- Click-to-mask (Masks tab) ---

    def _active_mask_instance(self) -> ClickInstance | None:
        if self._current is None:
            return None
        row = self._mask_list.currentRow()
        if 0 <= row < len(self._current.click_instances):
            return self._current.click_instances[row]
        return None

    @staticmethod
    def _mask_item_text(inst: ClickInstance) -> str:
        n = len(inst.points)
        box = "  ·  box" if inst.box is not None else ""
        return f"{inst.name}  ·  f{inst.seed_frame}  ·  {n} pt{'s' if n != 1 else ''}{box}"

    def _refresh_mask_list(self) -> None:
        """Repopulate the Masks list from the current shot, preserving the
        selected row, then re-announce the active instance to the viewport."""
        shot = self._current
        row = self._mask_list.currentRow()
        self._mask_list.blockSignals(True)
        self._mask_list.clear()
        if shot is not None and shot.click_instances:
            for inst in shot.click_instances:
                self._mask_list.addItem(self._mask_item_text(inst))
            row = min(max(row, 0), len(shot.click_instances) - 1)
            self._mask_list.setCurrentRow(row)
        self._mask_list.blockSignals(False)
        active = self._active_mask_instance()
        self._mask_name_edit.blockSignals(True)
        self._mask_name_edit.setText(active.name if active is not None else "")
        self._mask_name_edit.blockSignals(False)
        self._update_mask_points_warning(active)
        self.active_click_instance_changed.emit(active)

    def _update_mask_points_warning(self, inst: ClickInstance | None) -> None:
        """Amber warning once an object has more points than SAM's
        few-click sweet spot — measured: 5 points = full object, 50+ =
        collapsed mask. Threshold 9 = the upper end of sane refinement."""
        too_many = inst is not None and len(inst.points) > 9
        if too_many and inst is not None:
            self._mask_points_warn.setText(
                f"⚠ {len(inst.points)} points — SAM degrades with many "
                "points and the mask can collapse to nothing. Try Clear "
                "points, then 2–6 clicks + Preview, refining one click "
                "at a time."
            )
        self._mask_points_warn.setVisible(bool(too_many))

    def _on_mask_mode_toggled(self, checked: bool) -> None:
        self.click_mode_changed.emit(bool(checked))

    def _on_refine_all_toggled(self, checked: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.refine_all_masks = bool(checked)

    def _on_refiner_model_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.refiner_model = str(self._refiner_model_combo.itemData(idx) or "")
        self._update_preview_refiner_note()

    def _on_preview_refiner_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.preview_refiner = str(self._preview_refiner_combo.itemData(idx) or "auto")
        self._update_preview_refiner_note()

    def _update_preview_refiner_note(self) -> None:
        """Masks-tab note mirroring the Passes-tab RUN choice, and Passes-tab
        weights-dropdown enablement — how the two tabs communicate without
        duplicating controls."""
        shot = self._current
        if shot is None:
            self._preview_refiner_note.setText("")
            return
        enabled = shot.enabled_models or []
        # The BiRefNet weights dropdown only applies to the BiRefNet combo —
        # grey it out otherwise so it can't mislead.
        is_brn = "sam3_birefnet" in enabled
        self._refiner_model_combo.setEnabled(is_brn)
        self._refiner_weights_label.setEnabled(is_brn)

        if "sam3_vitmatte" in enabled:
            what = "ViTMatte (trimap)"
        elif is_brn:
            what = self._refiner_model_combo.currentText().split(" - ")[0]
        elif "sam3_rvm" in enabled:
            what = "RVM"
        else:
            self._preview_refiner_note.setText("No matte model selected (Passes tab).")
            return
        override = str(shot.preview_refiner or "auto")
        if override == "auto":
            self._preview_refiner_note.setText(f"Run refines with: {what}  (set in Passes tab)")
        else:
            self._preview_refiner_note.setText(
                f"Run refines with: {what}  —  preview overridden above"
            )

    def _on_mask_new(self) -> None:
        if self._current is None:
            return
        shot = self._current
        seed = shot.current_frame or shot.frame_range[0]
        shot.click_instances.append(
            ClickInstance(
                name=f"object_{len(shot.click_instances) + 1}",
                seed_frame=int(seed),
            )
        )
        self._refresh_mask_list()
        self._mask_list.setCurrentRow(len(shot.click_instances) - 1)
        # Creating an object means the user wants to click — arm the viewport.
        if not self._mask_mode_check.isChecked():
            self._mask_mode_check.setChecked(True)  # emits click_mode_changed
        self._registry.notify_updated(shot)

    def _on_mask_delete(self) -> None:
        if self._current is None:
            return
        row = self._mask_list.currentRow()
        if 0 <= row < len(self._current.click_instances):
            del self._current.click_instances[row]
            self._refresh_mask_list()
            self._registry.notify_updated(self._current)

    def _on_mask_undo_point(self) -> None:
        """Undo the last input on the selected object — pop the last point,
        or if there are none, drop the box. The cheap misclick fix between
        nothing and Clear points."""
        inst = self._active_mask_instance()
        if inst is None or self._current is None:
            return
        if inst.points:
            inst.points.pop()
        elif inst.box is not None:
            inst.box = None
        else:
            return
        self._refresh_mask_list()
        self._registry.notify_updated(self._current)

    def _on_mask_clear_points(self) -> None:
        inst = self._active_mask_instance()
        if inst is None or self._current is None:
            return
        inst.points.clear()
        inst.box = None
        self._refresh_mask_list()
        self._registry.notify_updated(self._current)

    def _on_mask_selected(self, row: int) -> None:
        if self._building:
            return
        inst = self._active_mask_instance()
        self._mask_name_edit.blockSignals(True)
        self._mask_name_edit.setText(inst.name if inst is not None else "")
        self._mask_name_edit.blockSignals(False)
        self.active_click_instance_changed.emit(inst)

    def _on_mask_name_edited(self, text: str) -> None:
        """Rename the selected object. The name becomes the mask.<name>
        channel + Cryptomatte manifest entry; empty falls back to a
        click_<n> default at pass level, so storing '' is safe."""
        if self._building or self._current is None:
            return
        inst = self._active_mask_instance()
        if inst is None:
            return
        inst.name = text.strip()
        item = self._mask_list.currentItem()
        if item is not None:
            item.setText(self._mask_item_text(inst))

    def _on_proxy_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.proxy_long_edge = self._proxy_combo.itemData(idx)
        self._broadcast_output_to_all_shots()

    def _on_delivery_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        comp, dtype = self._delivery_combo.itemData(idx)
        self._current.delivery_compression = comp
        self._current.delivery_dtype = dtype
        self._broadcast_output_to_all_shots()

    def _on_pick_external_root(self) -> None:
        if self._current is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose output root")
        if not folder:
            return
        self._current.output_external_root = Path(folder)
        self._rebuild_resolved_out_label()
        self._broadcast_output_to_all_shots()

    def _broadcast_output_to_all_shots(self) -> None:
        """Output is effectively session-wide: any change on the active
        shot propagates to every other shot in the registry. Users
        overwhelmingly want a single destination for a batch; the
        per-shot storage model is kept because `Shot` uses it for
        serialisation, but the GUI never lets it diverge. Proxy
        resolution also broadcasts — it's an output-tab control and
        the batch-wide consistency rule applies."""
        if self._current is None:
            return
        active = self._current
        for shot in self._registry.shots():
            if shot is active:
                self._registry.notify_updated(shot)
                continue
            shot.output_mode = active.output_mode
            shot.output_subfolder_name = active.output_subfolder_name
            shot.output_external_root = active.output_external_root
            shot.output_external_name = active.output_external_name
            shot.proxy_long_edge = active.proxy_long_edge
            shot.delivery_compression = active.delivery_compression
            shot.delivery_dtype = active.delivery_dtype
            self._registry.notify_updated(shot)

    def _rebuild_resolved_out_label(self) -> None:
        if self._current is None:
            self._resolved_out_label.setText("")
            return
        root = self._current.output_external_root
        if self._current.output_mode == "external" and root is None:
            self._out_root_label.setText("<no root chosen>")
            self._resolved_out_label.setText("⚠ Pick an external root before submitting.")
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
        self._update_preview_refiner_note()
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
        self._update_preview_refiner_note()
        self._registry.notify_updated(self._current)

    def _on_apply_passes_to_all(self) -> None:
        """Broadcast the active shot's enabled_models list (and the SAM 3
        concepts that drive Matte + Cryptomatte) to every other shot in
        the registry. Non-destructive wrt Output etc. — only the pass
        selection and its concepts are copied."""
        if self._current is None:
            return
        source = list(self._current.enabled_models)
        source_concepts = self._current.sam3_concepts
        count = 0
        for shot in self._registry.shots():
            if shot is self._current:
                continue
            shot.enabled_models = list(source)
            shot.sam3_concepts = source_concepts
            self._registry.notify_updated(shot)
            count += 1
        # Tiny feedback — tooltip-style text in the button so the user
        # knows the click did something on a repeat press.
        from PySide6.QtCore import QTimer

        original = self._apply_passes_btn.text()
        self._apply_passes_btn.setText(f"Applied to {count} shot{'s' if count != 1 else ''} ✓")
        QTimer.singleShot(1500, lambda: self._apply_passes_btn.setText(original))

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
