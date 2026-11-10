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
    QAbstractButton,
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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.core import hardware
from live_action_aov.core.output_layout import (
    NAMING_PRESETS,
    file_name,
    next_version_dir,
    validate_naming,
)
from live_action_aov.gui.engine_menu import OBJECT_ENGINES, RUN_ENGINES, EngineButton
from live_action_aov.gui.pass_catalog import PASS_CATALOG, ModelEntry, recommended_models
from live_action_aov.gui.shot_list import remember_output_mode, remember_output_root
from live_action_aov.gui.shot_state import ClickInstance, ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import SUPPORTED_COLORSPACES

#: Categories where ANY COMBINATION of entries may be enabled together
#: (checkboxes, no Off radio — all-unchecked means off). When more than one
#: matte refiner runs, the submit worker auto-namespaces each engine's
#: layers (matte_<engine>.*) so they can't overwrite each other.
_MULTI_SELECT_CATEGORIES: set[str] = set()


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
    #: Masks tab "Auto": preview by itself after every click or box.
    mask_auto_preview_toggled = Signal(bool)
    #: Masks tab "Go to" : scrub the viewer to an object's seed frame.
    frame_requested = Signal(int)
    #: Output tab "Draw crop" armed / disarmed: the viewer takes one drag.
    crop_mode_changed = Signal(bool)

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
        # Next to the plate (<plate folder>_utility/v###/, default: in
        # production the utility lives in the plate's own tree) or one
        # chosen root for the session (<root>/<name>/v###/).
        self._out_mode_plate = QRadioButton("Next to the plate  (<plate folder>_utility)")
        self._out_mode_root = QRadioButton("One root for every shot")
        self._out_mode_group = QButtonGroup(self)
        self._out_mode_group.addButton(self._out_mode_plate)
        self._out_mode_group.addButton(self._out_mode_root)
        self._out_mode_plate.toggled.connect(self._on_output_mode_toggled)
        self._out_root_label = QLabel("<no root chosen>")
        self._out_root_label.setStyleSheet("color: #888; font-size: 10pt;")
        self._out_root_label.setWordWrap(True)
        self._out_root_btn = QPushButton("Choose root…")
        self._out_root_btn.clicked.connect(self._on_pick_external_root)

        out_root_row = QHBoxLayout()
        out_root_row.addWidget(self._out_root_btn)
        out_root_row.addWidget(self._out_root_label, stretch=1)

        self._external_name_edit = QLineEdit()
        self._external_name_edit.setPlaceholderText("<shot name>")
        self._external_name_edit.setToolTip(
            "Folder the versions go in: next to the plate folder, or inside the "
            "root. Default: <plate folder>_utility / the shot's name."
        )
        self._external_name_edit.textEdited.connect(self._on_external_name_edited)

        external_name_row = QHBoxLayout()
        external_name_row.addWidget(QLabel("Shot folder:"))
        external_name_row.addWidget(self._external_name_edit, stretch=1)

        self._resolved_out_label = QLabel("")
        self._resolved_out_label.setStyleSheet("color: #aaa; font-size: 10pt;")
        self._resolved_out_label.setWordWrap(True)

        # Proxy resolution — fast-iteration dropdown. Plate-native
        # is the default; users drop to 540p / 720p / 1080p for
        # quick passes before committing to a full-res run.
        self._proxy_combo = QComboBox()
        self._proxy_combo.addItem("Full plate (native)", None)
        self._proxy_combo.addItem("Large  (long edge 2816)", 2816)
        self._proxy_combo.addItem("1080p  (long edge 1920)", 1920)
        self._proxy_combo.addItem("720p  (long edge 1280)", 1280)
        self._proxy_combo.addItem("540p  (long edge 960)", 960)
        self._proxy_combo.setToolTip(
            "Resize every plate frame to this long edge before the passes see it. "
            "The EXRs are scaled back to plate size, so they line up in comp. "
            "Smaller = faster and less memory, slightly softer matte edges."
        )
        self._proxy_combo.currentIndexChanged.connect(self._on_proxy_changed)
        # What this machine (card + RAM) can run for the selected shot, and
        # one click to take it. It proposes; nothing changes until pressed.
        self._hw_label = QLabel("")
        self._hw_label.setStyleSheet("color: #aaa; font-size: 9pt;")
        self._hw_label.setWordWrap(True)
        self._hw_apply_btn = QPushButton("Use suggested")
        self._hw_apply_btn.setToolTip(
            "Set the proxy size and the detector to what this card and the free RAM "
            "can run for this shot. Long clips are cut in windows by themselves."
        )
        self._hw_apply_btn.clicked.connect(self._on_hw_apply)
        self._hw_suggestion: hardware.Suggestion | None = None

        # Delivery: EXR codec + bit depth. Bandwidth-bound jobs (separate
        # machines, limited share) need compact sidecars. Item data is a
        # (compression, dtype) tuple. Cryptomatte is auto-split to a lossless
        # float32 sibling by the writer when a lossy/half delivery is chosen,
        # so its hash IDs always survive.
        self._delivery_combo = QComboBox()
        self._delivery_combo.addItem("Lossless 32-bit (ZIP) - largest", ("zips", "float32"))
        self._delivery_combo.addItem("Lossless 16-bit half (ZIP)", ("zips", "float16"))
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
        self._layout_combo = QComboBox()
        self._layout_combo.addItem("One multi-layer EXR (all passes)", "multilayer")
        self._layout_combo.addItem("One EXR per pass (subfolders)", "per_pass")
        self._layout_combo.setToolTip(
            "Multi-layer: one Read in Nuke, a Shuffle per pass. Per pass: "
            "passes/depth/, passes/matte/ ... each its own EXR sequence. Either way "
            "<shot>.comp.nk opens with everything already wired."
        )
        self._layout_combo.currentIndexChanged.connect(self._on_layout_changed)

        # --- This shot only: work range + crop ---
        # Mark (studio): "most of the time I want to crop and work on a part
        # of the video only". Frames in/out inside the plate, and a crop box
        # drawn in the viewer; both per shot, unlike the rest of this tab.
        self._range_in = QSpinBox()
        self._range_out = QSpinBox()
        for sp in (self._range_in, self._range_out):
            sp.setRange(-9_999_999, 9_999_999)
            sp.setKeyboardTracking(False)
            sp.setMinimumWidth(80)
        self._range_in.valueChanged.connect(self._on_range_changed)
        self._range_out.valueChanged.connect(self._on_range_changed)
        self._range_reset_btn = QPushButton("Whole plate")
        self._range_reset_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._range_reset_btn.clicked.connect(self._on_range_reset)
        self._crop_label = QLabel("whole frame")
        self._crop_label.setStyleSheet("color: #aaa;")
        self._crop_draw_btn = QPushButton("Draw in viewer")
        self._crop_draw_btn.setCheckable(True)
        self._crop_draw_btn.setToolTip(
            "Then drag a box on the image. Passes run on that box only; the "
            "EXRs keep the full plate as display window, so Nuke places them. "
            "Camera and Gaussian splat need the whole frame."
        )
        self._crop_draw_btn.toggled.connect(self._on_crop_draw_toggled)
        self._crop_clear_btn = QPushButton("Clear")
        self._crop_clear_btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._crop_clear_btn.clicked.connect(self._on_crop_clear)

        # --- Naming (session-wide, like the rest of Output) ---
        self._naming_combo = QComboBox()
        for label, template in NAMING_PRESETS:
            self._naming_combo.addItem(label, template)
        self._naming_combo.addItem("Custom…", "")
        self._naming_combo.currentIndexChanged.connect(self._on_naming_preset)
        self._naming_edit = QLineEdit()
        self._naming_edit.setPlaceholderText(NAMING_PRESETS[0][1])
        self._naming_edit.setToolTip(
            "Tokens: {shot} {plate} {version} {pass} {frame} {ext}. {pass} is "
            "utility / crypto / camera / splat / comp (or the pass group in the "
            "per-pass layout); files without a frame drop {frame}."
        )
        self._naming_edit.editingFinished.connect(self._on_naming_edited)
        self._naming_example = QLabel("")
        self._naming_example.setStyleSheet("color: #8f9196; font-size: 8pt;")
        self._naming_example.setWordWrap(True)

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
        self._model_radios: dict[str, QAbstractButton] = {}
        self._off_radios: dict[str, QRadioButton] = {}
        self._category_groups: dict[str, QButtonGroup] = {}
        # Collapsible section containers so users can hide categories
        # they aren't using — matters more as the catalog grows with
        # future models (fresnel, alt-depth backends, etc.).
        self._category_sections: dict[str, QWidget] = {}
        self._category_headers: dict[str, QPushButton] = {}
        passes_block = QVBoxLayout()
        passes_block.setSpacing(4)
        last_block: str | None = None
        for block_title, blurb, category, entries in _iter_pass_blocks():
            if block_title != last_block:
                passes_block.addWidget(_block_header(block_title, blurb))
                last_block = block_title
            # Header button — flat, label-like, but clickable to toggle
            # the section below. Chevron prefix makes expand/collapse
            # state readable at a glance.
            header_btn = QPushButton(f"▾  {category}")
            header_btn.setCheckable(True)
            header_btn.setChecked(True)
            header_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            header_btn.setStyleSheet(_FOLD_HEADER_QSS)
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

            multi = category in _MULTI_SELECT_CATEGORIES
            group: QButtonGroup | None = None
            if not multi:
                group = QButtonGroup(self)
                group.setExclusive(True)
                self._category_groups[category] = group

            # "Off" is the default — users opt in per category. Keeping
            # it visible rather than hidden-by-default makes the "I
            # don't want depth right now" intent explicit.
            if not multi and group is not None:
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
                if entry.hidden:
                    continue
                row = QWidget()
                row.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(16, 0, 0, 0)
                row_layout.setSpacing(6)
                radio: QAbstractButton
                if multi:
                    radio = QCheckBox(entry.label)
                    radio.toggled.connect(
                        lambda checked, key=entry.key: self._on_model_multi_toggled(key, checked)
                    )
                else:
                    radio = QRadioButton(entry.label)
                    radio.toggled.connect(
                        lambda checked, key=entry.key: (
                            self._on_model_selected(key) if checked else None
                        )
                    )
                    assert group is not None
                    group.addButton(radio)
                self._model_radios[entry.key] = radio
                row_layout.addWidget(radio)
                row_layout.addStretch()
                # Short marker (the full licence is the tooltip) so the
                # inspector never needs a horizontal scroll.
                colour = "#5ec864" if entry.commercial else "#e0a040"
                license_marker = QLabel("✓" if entry.commercial else "⚠ NC")
                license_marker.setToolTip(
                    f"{entry.license_tag} - commercial use OK"
                    if entry.commercial
                    else f"{entry.license_tag} - non-commercial: NOT FOR PRODUCTION USE"
                )
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
            'What SAM 3 should detect, comma-separated (e.g. "person, red car, dog").\n'
            "Drives both the Matte and Cryptomatte passes. SAM 3 only finds the\n"
            "concepts you prompt — leave empty to use the built-in defaults."
        )
        self._sam3_concepts_edit.textEdited.connect(self._on_sam3_concepts_edited)

        concepts_hint = QLabel(
            "What gets a matte and a Cryptomatte ID (people, vehicles, your own objects) "
            "is chosen on the Masks tab."
        )
        concepts_hint.setStyleSheet("color: #999; font-size: 8pt;")
        concepts_hint.setWordWrap(True)
        passes_block.addSpacing(8)
        passes_block.addWidget(concepts_hint)

        # DEFAULT ENGINE for the run — one dropdown, covering both the
        # engine and (for BiRefNet) the weight variant. Objects pinned on
        # the Masks tab override this per object.
        self._default_engine_combo = EngineButton(RUN_ENGINES)
        self._default_engine_combo.setToolTip(
            "Engine used for every object that has no engine of its own "
            "(Masks tab) and for the concept-detected unions. Other engines "
            "are under Advanced."
        )
        self._default_engine_combo.picked.connect(self._on_default_engine_changed)

        self._refine_all_check = QCheckBox("Refine ALL mask.<name> channels at submit (slower)")
        self._refine_all_check.setToolTip(
            "Refine every detected/clicked object at submit so each mask.<name> "
            "gets roto-grade soft edges, not just the 4 hero matte slots. "
            "Costs one refinement per object. The preview is always soft."
        )
        self._refine_all_check.toggled.connect(self._on_refine_all_toggled)

        self._outline_check = QCheckBox("BiRefNet decides the outline (SAM only gates leaks)")
        self._outline_check.setToolTip(
            "Off: BiRefNet only softens the edge of SAM 3's shape, so a blobby SAM "
            "mask (filled arm gaps) stays blobby. On: BiRefNet draws the whole "
            "silhouette and SAM only stops it leaking onto other things; a frame "
            "where it keeps less than 75% of SAM's area falls back to off. Best on "
            "small or distant people. BiRefNet engine only."
        )
        self._outline_check.toggled.connect(self._on_outline_toggled)

        # Small subjects: SAM 3 also detects on zoomed tiles of the frame.
        self._small_objects_combo = QComboBox()
        self._small_objects_combo.addItem("Auto - zoomed 2x2 tiles on 2K+ plates", "auto")
        self._small_objects_combo.addItem("Max - zoomed 3x3 tiles (crowds, far people)", "max")
        self._small_objects_combo.addItem("Off - full frame only", "off")
        self._small_objects_combo.setToolTip(
            "SAM 3 sees about 1000 px: on a 4K plate a 40 px person reaches it as 10 px "
            "and is missed. Auto/Max also run detection on overlapping zoomed tiles "
            "(the tracker always follows the full frame), and keep objects down to "
            "~10 px tall. Max finds ~90% of 12-96 px people; costs a few seconds per "
            "re-detection frame. Small objects keep SAM's edge (anti-aliased). "
            "With the SAM 3.1 detector, Max seeds several frames (first, middle, last, "
            "every 40) and keeps objects down to 0.03% of the frame; slower."
        )
        self._small_objects_combo.currentIndexChanged.connect(self._on_small_objects_changed)
        small_row = QHBoxLayout()
        self._small_objects_label = QLabel("Small objects:")
        small_row.addWidget(self._small_objects_label)
        small_row.addWidget(self._small_objects_combo, stretch=1)

        # Detector: SAM 3 (built in) or SAM 3.1 Object Multiplex.
        self._detector_combo = QComboBox()
        self._detector_combo.addItem("Auto - SAM 3.1 when installed, else SAM 3", "auto")
        self._detector_combo.addItem("SAM 3 (built in)", "sam3")
        self._detector_combo.addItem(
            "SAM 3.1 Object Multiplex - many objects, fast motion", "sam3.1"
        )
        self._detector_combo.setToolTip(
            "SAM 3.1 tracks every instance of a concept together and keeps identities "
            "where SAM 3 can swap two similar objects or lose a part in motion blur. "
            "It runs in its own Python 3.12 environment: `liveaov setup-sam31` builds it "
            "(docs/install.md, SAM 3.1). Auto uses it when it is installed and the card "
            "has the memory, else SAM 3. Clicked objects still use SAM 3."
        )
        self._detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        detector_row = QHBoxLayout()
        self._detector_label = QLabel("Detector:")
        detector_row.addWidget(self._detector_label)
        detector_row.addWidget(self._detector_combo, stretch=1)

        self._pnp_check = QCheckBox("Camera: solve from depth when SfM fails (PnP fallback)")
        self._pnp_check.setToolTip(
            "Fast handheld action often gives COLMAP no usable parallax, and the camera "
            "moves too much for the rotation-only fallback. Then the camera is solved "
            "from MoGe-3 depth of the middle frame plus tracked corners. Scale is "
            "MoGe-3's metric estimate. Camera track only."
        )
        self._pnp_check.toggled.connect(self._on_pnp_toggled)

        refiner_weights_row = QHBoxLayout()
        self._refiner_weights_label = QLabel("Default engine:")
        refiner_weights_row.addWidget(self._refiner_weights_label)
        refiner_weights_row.addWidget(self._default_engine_combo, stretch=1)
        passes_block.addSpacing(8)
        passes_block.addLayout(refiner_weights_row)

        # Advanced: the knobs a first run does not need. Folded by default;
        # the Pro playbooks say when to open it.
        self._advanced_btn = QPushButton("▸  Advanced")
        self._advanced_btn.setCheckable(True)
        self._advanced_btn.setChecked(False)
        self._advanced_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._advanced_btn.setStyleSheet(_FOLD_HEADER_QSS)
        self._advanced_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._advanced_box = QWidget()
        adv_layout = QVBoxLayout(self._advanced_box)
        adv_layout.setContentsMargins(16, 0, 0, 0)
        adv_layout.setSpacing(4)
        adv_layout.addLayout(small_row)
        adv_layout.addLayout(detector_row)
        adv_layout.addWidget(self._refine_all_check)
        adv_layout.addWidget(self._outline_check)
        adv_layout.addWidget(self._pnp_check)
        self._advanced_box.setVisible(False)
        self._advanced_btn.toggled.connect(self._on_advanced_toggled)
        passes_block.addSpacing(4)
        passes_block.addWidget(self._advanced_btn)
        passes_block.addWidget(self._advanced_box)

        # "Apply to all shots" — one-click broadcast of the active
        # shot's pass selections to every other shot in the queue.
        # Unlike Output (which is always session-wide), Passes stay
        # per-shot by default because users might want different
        # models per shot; the button is the escape hatch when they
        # don't.
        self._apply_passes_btn = QPushButton("Apply passes to all shots")
        self._apply_passes_btn.setToolTip(
            "Copy the currently-selected model in each category (incl. Off "
            "choices), the SAM 3 concepts, the refiner weights choice and "
            "the 'soft edges on all masks' toggle to every other shot."
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
        passes_header = QLabel("Passes")
        passes_header.setStyleSheet(
            "font-weight: 700; color: #fff; font-size: 11pt; padding: 4px 0; letter-spacing: 0.5px;"
        )
        passes_hint = QLabel(
            "Tick what the comp needs. Recommended = the set we run daily "
            "(mattes, Cryptomatte, metric depth + normals, motion, camera + splat background)."
        )
        self._recommended_btn = QPushButton("Use the recommended set")
        self._recommended_btn.setToolTip(
            "Matte (SAM + default engine), Cryptomatte, MoGe-3 depth + normals, "
            "RAFT motion, camera track + Gaussian splat background "
            "(camera only when gsplat is not installed)."
        )
        self._recommended_btn.clicked.connect(self._on_recommended_clicked)
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
        passes_layout.addWidget(self._recommended_btn)
        passes_layout.addSpacing(4)
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
        grade_note = QLabel(
            "This grade is exactly what the AI models see: matte/key "
            "quality depends on it. A flat or desaturated view weakens "
            "screen pulls and soft edges - grade for a punchy, saturated "
            "image before submitting."
        )
        grade_note.setStyleSheet("color: #8a8; font-size: 8pt;")
        grade_note.setWordWrap(True)
        preview_layout.addWidget(grade_note)
        preview_layout.addSpacing(12)
        preview_layout.addWidget(_section_label("Pixel aspect"))
        pa_hint = QLabel(
            "Anamorphic desqueeze for the VIEWER only (sidecars stay in "
            "pixel space). Auto-filled from the EXR header; override it "
            "here when the header lies (1.0 = square pixels, 2.0 = 2x "
            "anamorphic)."
        )
        pa_hint.setStyleSheet("color: #888; font-size: 9pt;")
        pa_hint.setWordWrap(True)
        preview_layout.addWidget(pa_hint)
        self._pixel_aspect_spin = QDoubleSpinBox()
        self._pixel_aspect_spin.setRange(0.25, 4.0)
        self._pixel_aspect_spin.setSingleStep(0.05)
        self._pixel_aspect_spin.setDecimals(3)
        self._pixel_aspect_spin.setValue(1.0)
        self._pixel_aspect_spin.valueChanged.connect(self._on_pixel_aspect_changed)
        preview_layout.addWidget(self._pixel_aspect_spin)
        preview_layout.addSpacing(12)
        preview_layout.addWidget(self._reset_btn)
        preview_layout.addStretch()
        preview_tab = _scrollable(preview_layout)

        # "This shot" tab: the ad-hoc settings of ONE shot. Everything in
        # the Output tab is for the whole session; a batch must not need a
        # visit to every shot.
        shot_layout = QVBoxLayout()
        shot_layout.setContentsMargins(8, 8, 8, 8)
        shot_layout.addWidget(_section_label("Only this shot"))
        shot_only_note = QLabel(
            "Optional. These apply to the selected shot alone; the Output tab is the "
            "same for every shot of the session."
        )
        shot_only_note.setStyleSheet("color: #888; font-size: 9pt;")
        shot_only_note.setWordWrap(True)
        shot_layout.addWidget(shot_only_note)
        shot_layout.addSpacing(10)
        shot_layout.addWidget(_section_label("Frames to process"))
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Frames"))
        range_row.addWidget(self._range_in)
        range_row.addWidget(QLabel("to"))
        range_row.addWidget(self._range_out)
        range_row.addWidget(self._range_reset_btn)
        range_row.addStretch(1)
        shot_layout.addLayout(range_row)
        shot_layout.addSpacing(10)
        shot_layout.addWidget(_section_label("Part of the frame"))
        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel("Crop:"))
        crop_row.addWidget(self._crop_label, stretch=1)
        crop_row.addWidget(self._crop_draw_btn)
        crop_row.addWidget(self._crop_clear_btn)
        shot_layout.addLayout(crop_row)
        shot_layout.addSpacing(10)
        shot_layout.addWidget(_section_label("Folder name"))
        shot_layout.addLayout(external_name_row)
        shot_layout.addStretch()
        shot_tab = _scrollable(shot_layout)

        # Output tab: the same for every shot of the session.
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(8, 8, 8, 8)
        output_layout.addWidget(_section_label("Output - every shot"))
        session_wide_note = QLabel(
            "One setting for the whole session. A single shot's frames, crop and folder "
            "name are in the This shot tab."
        )
        session_wide_note.setStyleSheet("color: #888; font-size: 9pt;")
        session_wide_note.setWordWrap(True)
        output_layout.addWidget(session_wide_note)
        output_layout.addWidget(self._out_mode_plate)
        output_layout.addWidget(self._out_mode_root)
        output_layout.addLayout(out_root_row)
        output_layout.addWidget(self._resolved_out_label)
        output_layout.addSpacing(16)
        output_layout.addWidget(_section_label("Proxy resolution"))
        proxy_hint = QLabel(
            "The models run on a smaller copy of the plate (faster, less memory); "
            "the EXRs are still written at plate size."
        )
        proxy_hint.setStyleSheet("color: #888; font-size: 9pt;")
        proxy_hint.setWordWrap(True)
        output_layout.addWidget(proxy_hint)
        output_layout.addWidget(self._proxy_combo)
        hw_row = QHBoxLayout()
        hw_row.addWidget(self._hw_label, stretch=1)
        hw_row.addWidget(self._hw_apply_btn)
        output_layout.addLayout(hw_row)
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
        layout_hint = QLabel(
            "Folder layout inside the version folder: <shot>_v###.comp.nk at the "
            "top, EXRs in passes/, camera/, splat/, logs/."
        )
        layout_hint.setStyleSheet("color: #888; font-size: 9pt;")
        layout_hint.setWordWrap(True)
        output_layout.addWidget(layout_hint)
        output_layout.addWidget(self._layout_combo)
        output_layout.addSpacing(16)
        output_layout.addWidget(_section_label("File naming"))
        naming_hint = QLabel(
            "Match the studio's convention. One template for every file of the "
            "run; the version folder stays v###."
        )
        naming_hint.setStyleSheet("color: #888; font-size: 9pt;")
        naming_hint.setWordWrap(True)
        output_layout.addWidget(naming_hint)
        output_layout.addWidget(self._naming_combo)
        output_layout.addWidget(self._naming_edit)
        output_layout.addWidget(self._naming_example)
        output_layout.addStretch()
        output_tab = _scrollable(output_layout)

        # Masks tab — interactive click-to-mask (v0.3). Entirely OPTIONAL:
        # an empty object list changes nothing at submit. Each object here
        # is a ClickInstance: the user arms point placement, clicks the
        # element in the viewport (left = include, right = exclude), and at
        # submit the SAM 3 tracker propagates it across the shot into a
        # mask.<name> channel + Cryptomatte ID.
        masks_hint = QLabel(
            "For anything the names above do not find, or one object you want on its "
            "own: press New object, then drag a box around it in the viewer or click on "
            "it (left = include, right = exclude). A box or 2-6 clicks is enough: "
            "Preview, then one corrective click. Each object gets its own mask.<name> "
            "and Cryptomatte ID."
        )
        masks_hint.setStyleSheet("color: #999; font-size: 9pt;")
        masks_hint.setWordWrap(True)
        found_hint = QLabel(
            "Every one of these found in the shot is tracked and gets a Cryptomatte ID "
            "and a mask. Comma-separated, any words (person, red car, dog). Sky, "
            "buildings and trees get an ID but never one of the four matte slots. "
            "Empty = the default list; with your own objects below, empty = only those."
        )
        found_hint.setStyleSheet("color: #999; font-size: 9pt;")
        found_hint.setWordWrap(True)
        self._concept_preset_btns: list[QPushButton] = []
        preset_row = QHBoxLayout()
        for label, text, tip in (
            ("People", "person", "Only people."),
            ("People + vehicles", "person, vehicle", "People, cars, bikes, trucks."),
            (
                "Everything",
                "",
                "The default list: person, vehicle, tree, building, sky, water, animal.",
            ),
        ):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.setCheckable(True)  # the preset in use stays lit
            btn.setProperty("preset", text)
            btn.clicked.connect(lambda _=False, t=text: self._on_concept_preset(t))
            preset_row.addWidget(btn)
            self._concept_preset_btns.append(btn)
        preset_row.addStretch(1)

        # Point placement is ALWAYS armed while the Masks tab is open (and
        # disarmed on any other tab) — the old "Place points in viewport"
        # checkbox was one more thing to forget. Scrubbing stays free: the
        # frame slider lives outside the canvas.
        self._mask_list = QListWidget()
        self._mask_list.setMinimumHeight(110)
        self._mask_list.setMaximumHeight(220)
        self._mask_list.currentRowChanged.connect(self._on_mask_selected)
        # The selected object's own settings, in one block under the list.
        self._mask_slot_combo = QComboBox()
        for label, slot in (
            ("Auto - the ranking decides", ""),
            ("matte.r (red)", "r"),
            ("matte.g (green)", "g"),
            ("matte.b (blue)", "b"),
            ("matte.a (alpha)", "a"),
        ):
            self._mask_slot_combo.addItem(label, slot)
        self._mask_slot_combo.setToolTip(
            "Which of the four matte.rgba channels carries this object's soft matte. "
            "Auto lets the biggest, most central subjects take them; pick one when the "
            "comp must always find the hero in the same channel."
        )
        self._mask_slot_combo.currentIndexChanged.connect(self._on_mask_slot_changed)
        self._mask_seed_label = QLabel("-")
        self._mask_goto_btn = QPushButton("Go to")
        self._mask_goto_btn.setToolTip(
            "Scrub the viewer to the frame this object was drawn on: its points and "
            "box only show, and only preview, on that frame."
        )
        self._mask_goto_btn.clicked.connect(self._on_mask_goto_seed)

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
        self._mask_preview_btn = QPushButton("Preview")
        self._mask_preview_btn.setToolTip(
            "Runs SAM on the object's frame only and overlays the soft matte "
            "in the viewer. First use loads the model (one-time wait); after "
            "that it is quick. Adjust the points and preview again."
        )
        self._mask_preview_btn.clicked.connect(self.mask_preview_requested.emit)
        self._mask_auto_preview_btn = QPushButton("Auto")
        self._mask_auto_preview_btn.setCheckable(True)
        self._mask_auto_preview_btn.setChecked(True)
        self._mask_auto_preview_btn.setToolTip(
            "On: the preview runs by itself after every click or box, so you "
            "always see what is being matted. Off: only when you press Preview."
        )
        self._mask_auto_preview_btn.toggled.connect(self.mask_auto_preview_toggled.emit)
        # What the preview is made with: the run engine, or one you want to
        # compare against it. Switching re-previews the selected object.
        self._preview_with_btns: dict[str, QPushButton] = {}
        preview_row = QHBoxLayout()
        preview_row.addWidget(self._mask_preview_btn)
        preview_row.addWidget(self._mask_auto_preview_btn)
        preview_row.addSpacing(12)
        preview_with_label = QLabel("with")
        preview_with_label.setStyleSheet("color: #999;")
        preview_row.addWidget(preview_with_label)
        for label, key, tip in (
            (
                "Run engine",
                "auto",
                "The object's own engine, else the run engine. LTX Alpha only runs at "
                "submit: its preview is ViTMatte.",
            ),
            ("ViTMatte", "vitmatte", "Soft edge from ViTMatte (trimap matting)."),
            ("BiRefNet", "birefnet:", "Soft edge from BiRefNet."),
            ("SAM only", "none", "The hard SAM mask, no soft edge: is the object right at all?"),
        ):
            b = QPushButton(label)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._on_preview_with(k))
            preview_row.addWidget(b)
            self._preview_with_btns[key] = b
        self._preview_with_btns["auto"].setChecked(True)
        preview_row.addStretch(1)

        mask_btn_row = QHBoxLayout()
        mask_btn_row.addWidget(self._mask_new_btn)
        mask_btn_row.addWidget(self._mask_del_btn)
        mask_btn_row.addWidget(self._mask_undo_btn)
        mask_btn_row.addWidget(self._mask_clear_btn)

        masks_layout = QVBoxLayout()
        masks_layout.setContentsMargins(8, 8, 8, 8)
        masks_layout.addWidget(_section_label("Found by name"))
        masks_layout.addWidget(found_hint)
        masks_layout.addWidget(self._sam3_concepts_edit)
        masks_layout.addLayout(preset_row)
        masks_layout.addSpacing(14)
        masks_layout.addWidget(_section_label("Your own objects"))
        masks_layout.addWidget(masks_hint)
        masks_layout.addSpacing(6)
        masks_layout.addLayout(mask_btn_row)
        masks_layout.addLayout(preview_row)
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
        # Note mirroring what the RUN will use — how the two tabs communicate.
        self._preview_refiner_note = QLabel("")
        self._preview_refiner_note.setStyleSheet("color: #888; font-size: 8pt;")
        self._preview_refiner_note.setWordWrap(True)
        # Per-object ENGINE, BESIDE the list: the weight belongs to the
        # LAYER, not the shot — portrait on the person, trimap/chroma on
        # the car, in one run. Selecting a row shows its engine here.
        self._mask_engine_combo = EngineButton(OBJECT_ENGINES)
        self._mask_engine_combo.setToolTip(
            "Soft-edge engine for THIS object only (Default = the run engine on "
            "the Passes tab). Picking one re-previews the object with it; the "
            "submit refines its mask.<name> with the same engine."
        )
        self._mask_engine_combo.picked.connect(self._on_mask_engine_changed)
        masks_layout.addWidget(self._mask_list)
        masks_layout.addWidget(self._mask_points_warn)
        masks_layout.addSpacing(8)
        self._mask_detail_title = _section_label("Selected object")
        masks_layout.addWidget(self._mask_detail_title)
        detail = QFormLayout()
        detail.setContentsMargins(0, 2, 0, 0)
        detail.addRow("Name", self._mask_name_edit)
        detail.addRow("Soft edge", self._mask_engine_combo)
        detail.addRow("Matte slot", self._mask_slot_combo)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self._mask_seed_label, stretch=1)
        seed_row.addWidget(self._mask_goto_btn)
        detail.addRow("Drawn on", seed_row)
        masks_layout.addLayout(detail)
        masks_layout.addSpacing(6)
        masks_layout.addWidget(self._preview_refiner_note)
        masks_layout.addStretch()
        masks_tab = _scrollable(masks_layout)

        tabs = QTabWidget()
        tabs.addTab(passes_tab, "Passes")
        tabs.addTab(masks_tab, "Masks")
        tabs.addTab(preview_tab, "Colour")
        tabs.addTab(output_tab, "Output")
        tabs.addTab(shot_tab, "This shot")
        # Auto-arm point placement while the Masks tab is open (replaces
        # the old "Place points in viewport" checkbox).
        self._masks_tab_index = tabs.indexOf(masks_tab)
        tabs.currentChanged.connect(self._on_tab_changed)

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
            self._pixel_aspect_spin.setValue(float(shot.pixel_aspect or 1.0))

            self._rebuild_range_and_crop(shot)
            self._rebuild_naming(shot)
            self._external_name_edit.setText(shot.output_external_name)
            self._out_mode_plate.setChecked(shot.output_mode == "plate")
            self._out_mode_root.setChecked(shot.output_mode != "plate")
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
            self._rebuild_hw_suggestion(shot)
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
            layout_index = max(0, self._layout_combo.findData(shot.delivery_layout))
            self._layout_combo.blockSignals(True)
            self._layout_combo.setCurrentIndex(layout_index)
            self._layout_combo.blockSignals(False)
            self._rebuild_resolved_out_label()

            # Pass radios reflect the stored enabled_models. The single-
            # select constraint is enforced in the selection handler;
            # here we just surface whatever's in the state — if more
            # than one model from a category slipped in (legacy YAMLs,
            # CLI handoff), we pick the first one to display and leave
            # the state alone until the user edits.
            enabled = set(shot.enabled_models)
            for category, entries in PASS_CATALOG.items():
                if category in _MULTI_SELECT_CATEGORIES:
                    for entry in entries:
                        cb = self._model_radios[entry.key]
                        cb.blockSignals(True)
                        cb.setChecked(entry.key in enabled)
                        cb.blockSignals(False)
                    continue
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
            self._sync_concept_presets()

            self._refine_all_check.blockSignals(True)
            self._refine_all_check.setChecked(bool(shot.refine_all_masks))
            self._refine_all_check.blockSignals(False)
            self._outline_check.blockSignals(True)
            self._outline_check.setChecked(bool(shot.refiner_outline))
            self._outline_check.blockSignals(False)
            self._small_objects_combo.blockSignals(True)
            so = self._small_objects_combo.findData(shot.small_objects or "auto")
            self._small_objects_combo.setCurrentIndex(max(so, 0))
            self._small_objects_combo.blockSignals(False)
            self._detector_combo.blockSignals(True)
            di = self._detector_combo.findData(shot.detector or "auto")
            self._detector_combo.setCurrentIndex(max(di, 0))
            self._detector_combo.blockSignals(False)
            self._pnp_check.blockSignals(True)
            self._pnp_check.setChecked(bool(shot.camera_pnp_fallback))
            self._pnp_check.blockSignals(False)

            self._default_engine_combo.set_current((shot.default_engine, shot.refiner_model))
            self._sync_preview_with()
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

    # --- This shot: work range + crop ---

    def _rebuild_range_and_crop(self, shot: ShotState) -> None:
        first, last = shot.frame_range
        a, b = shot.effective_frame_range()
        for sp, v in ((self._range_in, a), (self._range_out, b)):
            sp.blockSignals(True)
            sp.setRange(first, last)
            sp.setValue(v)
            sp.blockSignals(False)
        self._range_reset_btn.setEnabled(shot.work_range is not None)
        if shot.crop is None:
            self._crop_label.setText("whole frame")
        else:
            x, y, w, h = shot.crop
            self._crop_label.setText(f"{w} × {h} px at ({x}, {y})")
        self._crop_clear_btn.setEnabled(shot.crop is not None)

    def _on_range_changed(self, _v: int) -> None:
        if self._building or self._current is None:
            return
        shot = self._current
        first, last = shot.frame_range
        a, b = sorted((self._range_in.value(), self._range_out.value()))
        shot.work_range = None if (a, b) == (first, last) else (a, b)
        self._rebuild_range_and_crop(shot)
        self._registry.notify_updated(shot)

    def _on_range_reset(self) -> None:
        if self._current is None:
            return
        self._current.work_range = None
        self._rebuild_range_and_crop(self._current)
        self._registry.notify_updated(self._current)

    def _on_crop_draw_toggled(self, on: bool) -> None:
        self.crop_mode_changed.emit(bool(on))

    def crop_mode_finished(self) -> None:
        """The viewer took its drag (or dropped it): pop the button out."""
        self._crop_draw_btn.blockSignals(True)
        self._crop_draw_btn.setChecked(False)
        self._crop_draw_btn.blockSignals(False)

    def _on_crop_clear(self) -> None:
        if self._current is None:
            return
        self._current.crop = None
        self._rebuild_range_and_crop(self._current)
        self._registry.notify_updated(self._current)

    # --- Naming ---

    def _rebuild_naming(self, shot: ShotState) -> None:
        template = shot.naming or NAMING_PRESETS[0][1]
        idx = next(
            (
                i
                for i in range(self._naming_combo.count())
                if self._naming_combo.itemData(i) == template
            ),
            self._naming_combo.count() - 1,  # "Custom…"
        )
        self._naming_combo.blockSignals(True)
        self._naming_combo.setCurrentIndex(idx)
        self._naming_combo.blockSignals(False)
        self._naming_edit.blockSignals(True)
        self._naming_edit.setText(template)
        self._naming_edit.blockSignals(False)
        self._naming_example.setStyleSheet("color: #8f9196; font-size: 8pt;")
        self._naming_example.setText(_naming_example(shot))

    def _on_naming_preset(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        template = str(self._naming_combo.itemData(idx) or "")
        if not template:  # Custom…: keep whatever is in the field, let them type
            self._naming_edit.setFocus()
            return
        self._apply_naming(template)

    def _on_naming_edited(self) -> None:
        if self._building or self._current is None:
            return
        self._apply_naming(self._naming_edit.text())

    def _apply_naming(self, template: str) -> None:
        if self._current is None:
            return
        try:
            clean = validate_naming(template)
        except ValueError as e:
            self._naming_example.setStyleSheet("color: #e07a5f; font-size: 8pt;")
            self._naming_example.setText(str(e))
            return
        self._current.naming = "" if clean == NAMING_PRESETS[0][1] else clean
        self._broadcast_output_to_all_shots()

    def _on_pixel_aspect_changed(self, pa: float) -> None:
        if self._building or self._current is None:
            return
        self._current.pixel_aspect = float(pa)
        self._registry.notify_updated(self._current)  # viewer re-desqueezes

    def _on_external_name_edited(self, text: str) -> None:
        if self._building or self._current is None:
            return
        self._current.output_external_name = text.strip()  # empty → default to shot.name
        self._rebuild_resolved_out_label()

    def _on_sam3_concepts_edited(self, text: str) -> None:
        """SAM 3 concepts edited — stored verbatim (comma-separated).
        Parsing into a list happens at submit time in the worker, so a
        half-typed "person, " mid-edit isn't destroyed. Per-shot like
        the model selection; the "Apply passes to all" button copies it
        alongside enabled_models."""
        if self._building or self._current is None:
            return
        self._current.sam3_concepts = text
        self._sync_concept_presets()
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
        # The engine ON the row: every layer's model visible at a glance
        # (field feedback — the whole point of per-layer engines).
        eng_names = {
            "vitmatte": "ViTMatte",
            "birefnet": "BiRefNet",
            "chromakey": "ChromaKey",
            "rvm": "RVM",
            "ltx": "LTX Alpha",
        }
        eng = f"  ·  [{eng_names.get(inst.refiner, inst.refiner)}]" if inst.refiner else ""
        slot = f"  ·  matte.{inst.slot}" if inst.slot else ""
        prompts = f"{n} pt{'s' if n != 1 else ''}{box}" if (n or inst.box is not None) else "empty"
        return f"{inst.name}  ·  f{inst.seed_frame}  ·  {prompts}{eng}{slot}"

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
        self._sync_mask_detail(active)
        self._update_mask_points_warning(active)
        self.active_click_instance_changed.emit(active)

    def _sync_mask_detail(self, active: ClickInstance | None) -> None:
        """The "Selected object" block follows the list selection."""
        self._mask_slot_combo.blockSignals(True)
        self._mask_slot_combo.setCurrentIndex(
            max(self._mask_slot_combo.findData(active.slot if active is not None else ""), 0)
        )
        self._mask_slot_combo.blockSignals(False)
        self._mask_seed_label.setText(
            f"frame {active.seed_frame}" if active is not None else "no object selected"
        )
        for w in (
            self._mask_name_edit,
            self._mask_engine_combo,
            self._mask_slot_combo,
            self._mask_goto_btn,
            self._mask_preview_btn,
            self._mask_del_btn,
            self._mask_undo_btn,
            self._mask_clear_btn,
        ):
            w.setEnabled(active is not None)

    def _on_concept_preset(self, text: str) -> None:
        """People / People + vehicles / Everything: fill the names field."""
        if self._current is None:
            return
        self._sam3_concepts_edit.setText(text)
        self._current.sam3_concepts = text
        self._sync_concept_presets()
        self._registry.notify_updated(self._current)

    def _sync_concept_presets(self) -> None:
        """Light the preset that matches the names field (none for a custom list)."""

        def norm(text: str) -> str:
            return ", ".join(p.strip().lower() for p in text.split(",") if p.strip())

        now = norm(self._sam3_concepts_edit.text())
        for btn in self._concept_preset_btns:
            btn.blockSignals(True)
            btn.setChecked(norm(str(btn.property("preset") or "")) == now)
            btn.blockSignals(False)

    def _on_preview_with(self, key: str) -> None:
        """Masks tab "with": what the preview is made with; re-previews."""
        if self._current is None:
            self._sync_preview_with()
            return
        self._current.preview_refiner = key
        self._sync_preview_with()
        inst = self._active_mask_instance()
        if inst is not None and (inst.points or inst.box is not None):
            self.mask_preview_requested.emit()

    def _sync_preview_with(self) -> None:
        want = str(getattr(self._current, "preview_refiner", "auto") or "auto")
        if want.startswith("birefnet:"):
            want = "birefnet:"
        for key, btn in self._preview_with_btns.items():
            btn.blockSignals(True)
            btn.setChecked(key == want)
            btn.blockSignals(False)

    def _on_mask_slot_changed(self, idx: int) -> None:
        inst = self._active_mask_instance()
        if self._building or self._current is None or inst is None:
            return
        slot = str(self._mask_slot_combo.itemData(idx) or "")
        if slot:
            # One object per slot: taking a slot frees it on the others.
            for other in self._current.click_instances:
                if other is not inst and other.slot == slot:
                    other.slot = ""
        inst.slot = slot
        self._refresh_mask_list()

    def _on_mask_goto_seed(self) -> None:
        inst = self._active_mask_instance()
        if inst is not None:
            self.frame_requested.emit(int(inst.seed_frame))

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

    def _on_tab_changed(self, index: int) -> None:
        self.click_mode_changed.emit(index == self._masks_tab_index)

    def _on_refine_all_toggled(self, checked: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.refine_all_masks = bool(checked)

    def _on_outline_toggled(self, checked: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.refiner_outline = bool(checked)

    def _on_small_objects_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.small_objects = str(self._small_objects_combo.itemData(idx) or "auto")

    def _on_detector_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.detector = str(self._detector_combo.itemData(idx) or "auto")

    def _on_pnp_toggled(self, checked: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.camera_pnp_fallback = bool(checked)

    def _on_default_engine_changed(self, data: tuple[str, str]) -> None:
        if self._building or self._current is None:
            return
        self._current.default_engine, self._current.refiner_model = str(data[0]), str(data[1])
        self._update_preview_refiner_note()

    def _update_preview_refiner_note(self) -> None:
        """Masks-tab note mirroring the Passes-tab run setup."""
        shot = self._current
        if shot is None:
            self._preview_refiner_note.setText("")
            return
        enabled = shot.enabled_models or []
        # BiRefNet-only option: shown only when BiRefNet is the run engine.
        self._outline_check.setVisible(shot.default_engine == "birefnet")
        self._default_engine_combo.setEnabled("sam3_matte_on" in enabled)
        self._small_objects_combo.setEnabled("sam3_matte_on" in enabled)
        self._small_objects_label.setEnabled("sam3_matte_on" in enabled)
        self._detector_combo.setEnabled("sam3_matte_on" in enabled)
        self._detector_label.setEnabled("sam3_matte_on" in enabled)
        from live_action_aov.gui.pass_catalog import expand_models

        self._pnp_check.setEnabled("camera_pycolmap" in expand_models(list(enabled)))
        self._refiner_weights_label.setEnabled("sam3_matte_on" in enabled)
        if "sam3_matte_on" not in enabled:
            self._preview_refiner_note.setText(
                "Matte is OFF (Passes tab) - previews work, but the submit writes no mattes."
            )
            return
        label = self._default_engine_combo.short_label()
        pinned = sum(1 for ci in shot.click_instances if ci.refiner)
        extra = f", {pinned} object(s) with their own engine" if pinned else ""
        text = f"Run engine: {label}{extra} (one object can use another: Soft edge, above)"
        if any(ci.refiner == "ltx" for ci in shot.click_instances):
            from live_action_aov.gui import comfy_dialog

            last = comfy_dialog._last
            if last is None or last[0] != comfy_dialog.comfy_url():
                text += "\nLTX Alpha: ComfyUI not checked yet - Models > ComfyUI for LTX Alpha."
            elif last[1]:
                text += (
                    "\nLTX Alpha: ✓ ComfyUI ready. Preview shows the run engine;"
                    " LTX runs at submit."
                )
            else:
                text += (
                    f"\nLTX Alpha: ✗ {last[2]} - falls back to ViTMatte."
                    " Fix in Models > ComfyUI for LTX Alpha."
                )
        self._preview_refiner_note.setText(text)

    def refresh_engine_note(self) -> None:
        self._update_preview_refiner_note()

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
        want = getattr(inst, "refiner", "") if inst is not None else ""
        self._mask_engine_combo.set_current((want or "", ""))
        self._sync_mask_detail(inst)
        self.active_click_instance_changed.emit(inst)

    def _on_mask_engine_changed(self, data: tuple[str, str]) -> None:
        if self._building or self._current is None:
            return
        inst = self._active_mask_instance()
        if inst is None:
            return
        inst.refiner = str(data[0])
        if inst.refiner == "ltx":
            from PySide6.QtWidgets import QApplication

            from live_action_aov.gui import comfy_dialog

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                comfy_dialog.status()  # check once, the note shows the result
            finally:
                QApplication.restoreOverrideCursor()
        item = self._mask_list.currentItem()
        if item is not None:
            item.setText(self._mask_item_text(inst))
        self._update_preview_refiner_note()
        # Try-and-see in one step: re-preview the object with its new engine.
        if inst.points or inst.box is not None:
            self.mask_preview_requested.emit()

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
        self._rebuild_hw_suggestion(self._current)

    def _rebuild_hw_suggestion(self, shot: ShotState) -> None:
        """ "This machine: ... Suggested: ..." under the proxy menu."""
        hw, sam31 = _machine()
        first, last = shot.effective_frame_range()
        res = (shot.crop[2], shot.crop[3]) if shot.crop else shot.resolution
        s = hardware.suggest(hw, res, last - first + 1, sam31_installed=sam31)
        self._hw_suggestion = s
        applied = shot.proxy_long_edge == s.proxy_long_edge and shot.detector in (
            "auto",
            s.detector,
        )
        self._hw_label.setText(
            f"This machine: {hw.summary()}. Suggested - {s.one_line()}"
            + (" (in use)." if applied else ".")
        )
        self._hw_label.setToolTip("\n".join(s.notes))
        self._hw_apply_btn.setEnabled(s.tier != "none" and not applied)

    def _on_hw_apply(self) -> None:
        s = self._hw_suggestion
        if self._current is None or s is None:
            return
        self._current.proxy_long_edge = s.proxy_long_edge
        self._current.detector = "auto"  # the pass picks SAM 3.1 / SAM 3 the same way
        self._broadcast_output_to_all_shots()
        self._rebuild_from_shot()

    def _on_delivery_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        comp, dtype = self._delivery_combo.itemData(idx)
        self._current.delivery_compression = comp
        self._current.delivery_dtype = dtype
        self._broadcast_output_to_all_shots()

    def _on_layout_changed(self, idx: int) -> None:
        if self._building or self._current is None:
            return
        self._current.delivery_layout = str(self._layout_combo.itemData(idx))
        self._broadcast_output_to_all_shots()

    def _on_output_mode_toggled(self, plate: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.output_mode = "plate" if plate else "root"
        remember_output_mode(self._current.output_mode)
        self._rebuild_resolved_out_label()
        self._broadcast_output_to_all_shots()

    def _on_pick_external_root(self) -> None:
        if self._current is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Choose output root")
        if not folder:
            return
        self._current.output_external_root = Path(folder)
        self._current.output_mode = "root"
        self._out_mode_root.setChecked(True)
        remember_output_root(Path(folder))
        remember_output_mode("root")
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
            shot.output_external_root = active.output_external_root
            shot.proxy_long_edge = active.proxy_long_edge
            shot.delivery_compression = active.delivery_compression
            shot.delivery_dtype = active.delivery_dtype
            shot.delivery_layout = active.delivery_layout
            shot.naming = active.naming
            self._registry.notify_updated(shot)

    def _rebuild_resolved_out_label(self) -> None:
        if self._current is None:
            self._resolved_out_label.setText("")
            return
        plate = self._current.output_mode == "plate"
        # Choosing a root also switches to root mode, so the button stays live.
        self._out_root_label.setEnabled(not plate)
        self._external_name_edit.setPlaceholderText(
            f"{self._current.folder.name}_utility" if plate else "<shot name>"
        )
        root = self._current.output_external_root
        self._out_root_label.setText("<no root chosen>" if root is None else str(root))
        if not plate and root is None:
            self._resolved_out_label.setText("⚠ Pick an output root before submitting.")
            return
        shot_root = self._current.resolve_output_dir()
        nxt = next_version_dir(shot_root) if shot_root is not None else None
        self._resolved_out_label.setText(f"next run → {nxt}")

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

    def _on_model_multi_toggled(self, key: str, checked: bool) -> None:
        """Multi-select category toggle: add/remove `key` independently —
        any combination of matte engines may run together."""
        if self._building or self._current is None:
            return
        enabled = [k for k in self._current.enabled_models if k != key]
        if checked:
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
        """Broadcast the active shot's pass selection AND its refiner
        settings (weights choice, all-masks soft toggle) to every other shot.
        Field feedback: the refiner options are shot-based, so a user who set
        "soft edges on all masks" on one shot expected the propagate button
        to carry it to the batch — it silently didn't. Non-destructive wrt
        Output etc."""
        if self._current is None:
            return
        source = list(self._current.enabled_models)
        source_concepts = self._current.sam3_concepts
        src_refine_all = bool(self._current.refine_all_masks)
        src_outline = bool(self._current.refiner_outline)
        src_small = self._current.small_objects
        src_detector = self._current.detector
        src_pnp = bool(self._current.camera_pnp_fallback)
        src_refiner_model = self._current.refiner_model
        src_default_engine = self._current.default_engine
        count = 0
        for shot in self._registry.shots():
            if shot is self._current:
                continue
            shot.enabled_models = list(source)
            shot.sam3_concepts = source_concepts
            shot.refine_all_masks = src_refine_all
            shot.refiner_outline = src_outline
            shot.small_objects = src_small
            shot.detector = src_detector
            shot.camera_pnp_fallback = src_pnp
            shot.refiner_model = src_refiner_model
            shot.default_engine = src_default_engine
            self._registry.notify_updated(shot)
            count += 1
        # Tiny feedback — tooltip-style text in the button so the user
        # knows the click did something on a repeat press.
        from PySide6.QtCore import QTimer

        original = self._apply_passes_btn.text()
        self._apply_passes_btn.setText(f"Applied to {count} shot{'s' if count != 1 else ''} ✓")
        QTimer.singleShot(1500, lambda: self._apply_passes_btn.setText(original))

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_box.setVisible(checked)
        self._advanced_btn.setText(f"{'▾' if checked else '▸'}  Advanced")

    def _on_recommended_clicked(self) -> None:
        """One click to the set we run daily (the README's "what you get")."""
        if self._current is None:
            return
        self._current.enabled_models = recommended_models()
        self._rebuild_from_shot()
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


def _naming_example(shot: ShotState) -> str:
    """`e.g. passes/SH010_v001.utility.1001.exr`, from the shot's own names."""
    from types import SimpleNamespace

    root = shot.resolve_output_dir()
    fake = SimpleNamespace(
        name=shot.name,
        folder=shot.folder,
        output_dir=(next_version_dir(root) if root is not None else shot.folder / "v001"),
        sequence_pattern=shot.sequence_pattern,
        naming=shot.naming,
    )
    first = shot.effective_frame_range()[0]
    try:
        return (
            f"e.g. passes/{file_name(fake, 'utility', 'exr', first)}   "
            f"camera/{file_name(fake, 'camera', 'nk')}"
        )
    except ValueError as e:
        return str(e)


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: #ccc;")
    return lbl


_FOLD_HEADER_QSS = (
    "QPushButton {"
    " font-weight: 600; color: #cfcfcf; text-align: left;"
    " padding: 6px 0 2px 0; border: none; background: transparent;"
    "}"
    "QPushButton:hover { color: #ffffff; }"
)

#: The Passes tab groups the catalog's categories the way the comp sees
#: them (one block per deliverable), in this order. Categories missing
#: here (new ones) land in a trailing "More" block, so nothing is lost.
_PASS_BLOCKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Mattes",
        "Soft alpha per object, Cryptomatte IDs, screen pull.",
        ("Matte", "Cryptomatte", "Key"),
    ),
    (
        "Geometry",
        "Depth in metres, normals, world position.",
        ("Depth", "Normals", "Albedo"),
    ),
    ("Motion", "Forward and backward motion vectors.", ("Motion",)),
    (
        "Camera & splat",
        "Nuke camera with its verdict in the label, point cloud, Gaussian splat background (the set behind the masked objects).",
        ("Camera",),
    ),
)


def _iter_pass_blocks() -> list[tuple[str, str, str, list[ModelEntry]]]:
    out: list[tuple[str, str, str, list[ModelEntry]]] = []
    seen: set[str] = set()
    for title, blurb, cats in _PASS_BLOCKS:
        for cat in cats:
            if cat in PASS_CATALOG:
                out.append((title, blurb, cat, PASS_CATALOG[cat]))
                seen.add(cat)
    for cat, entries in PASS_CATALOG.items():
        if cat not in seen:
            out.append(("More", "", cat, entries))
    return out


def _block_header(title: str, blurb: str) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 10, 0, 0)
    lay.setSpacing(0)
    t = QLabel(title)
    t.setStyleSheet(
        "font-weight: 700; color: #ffffff; font-size: 10pt; "
        "border-bottom: 1px solid #34363b; padding-bottom: 3px;"
    )
    lay.addWidget(t)
    if blurb:
        b = QLabel(blurb)
        b.setStyleSheet("color: #8f9196; font-size: 8pt; padding-top: 2px;")
        b.setWordWrap(True)
        lay.addWidget(b)
    return box


__all__ = ["InspectorPanel"]


_MACHINE: tuple[hardware.Hardware, bool] | None = None


def _machine() -> tuple[hardware.Hardware, bool]:
    """(the card and RAM, is SAM 3.1 installed), read once per session
    except the free RAM, which is read every time."""
    global _MACHINE
    from live_action_aov.passes.matte import sam31_bridge

    if _MACHINE is None:
        _MACHINE = (hardware.detect(), sam31_bridge.available({})[0])
    hw, sam31 = _MACHINE
    _total, free = hardware.detect_ram()
    return hardware.Hardware(hw.gpu_name, hw.vram_gb, hw.ram_gb, free), sam31
