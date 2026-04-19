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
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import SUPPORTED_COLORSPACES

# Semantic pass families the user enables. Concrete plugin backends
# are chosen per-family via the dropdowns below.
_PASS_NAMES = ("flow", "depth", "normals", "matte")

# Which backends to offer per family. Matches pyproject entry points.
# Keep in sync with the CLI's `_resolve_semantic_passes` defaults —
# users expect the GUI and CLI to produce the same sidecar for the
# same ShotState.
_BACKEND_CHOICES: dict[str, list[str]] = {
    "depth": [
        "depth_anything_v2",       # Apache-2.0, commercial-safe default
        "video_depth_anything",    # Apache-2.0, temporal-aware
        "depthcrafter",            # CC-BY-NC
        "depthpro",                # CC-BY-NC
    ],
    "normals": [
        "dsine",                   # MIT
        "normalcrafter",           # CC-BY-NC
    ],
}
# Backends that need `--allow-noncommercial`. Gated with a toast in
# the UI when the user picks one.
_NONCOMMERCIAL_BACKENDS: set[str] = {
    "depthcrafter",
    "depthpro",
    "normalcrafter",
    "matanyone2",
}


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
        self._out_subfolder = QRadioButton("Subfolder of plate  (plate/utility/)")
        self._out_external = QRadioButton("External root  (<root>/<shot>/)")
        self._out_group = QButtonGroup(self)
        self._out_group.addButton(self._out_inplace, 0)
        self._out_group.addButton(self._out_subfolder, 1)
        self._out_group.addButton(self._out_external, 2)
        self._out_inplace.setChecked(True)
        self._out_group.idToggled.connect(self._on_output_mode_toggled)

        self._out_root_label = QLabel("<no root chosen>")
        self._out_root_label.setStyleSheet("color: #888; font-size: 10pt;")
        self._out_root_label.setWordWrap(True)
        self._out_root_btn = QPushButton("Choose root…")
        self._out_root_btn.setEnabled(False)
        self._out_root_btn.clicked.connect(self._on_pick_external_root)

        out_root_row = QHBoxLayout()
        out_root_row.addWidget(self._out_root_btn)
        out_root_row.addWidget(self._out_root_label, stretch=1)

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

        # --- Pass toggles + per-family backend pickers ---
        # Each enabled family contributes one (or two) PassConfigs at
        # submit time, resolved via ShotState.pass_backends.
        self._pass_checks: dict[str, QCheckBox] = {}
        self._backend_combos: dict[str, QComboBox] = {}
        passes_block = QVBoxLayout()
        for name in _PASS_NAMES:
            row = QHBoxLayout()
            cb = QCheckBox(name)
            cb.toggled.connect(lambda checked, n=name: self._on_pass_toggled(n, checked))
            self._pass_checks[name] = cb
            row.addWidget(cb)

            if name in _BACKEND_CHOICES:
                combo = QComboBox()
                combo.addItems(_BACKEND_CHOICES[name])
                combo.setToolTip(
                    f"Backend for the `{name}` pass. Non-commercial options "
                    "require toggling 'Allow non-commercial' below."
                )
                combo.currentTextChanged.connect(
                    lambda txt, n=name: self._on_backend_changed(n, txt)
                )
                self._backend_combos[name] = combo
                row.addWidget(combo, stretch=1)
            row.addStretch()
            passes_block.addLayout(row)

        # Non-commercial license gate. Users without this flag are
        # prevented from submitting a job that uses an NC backend —
        # same policy as the CLI's `--allow-noncommercial`.
        self._allow_nc_check = QCheckBox("Allow non-commercial backends")
        self._allow_nc_check.setToolTip(
            "Required before Submit will accept a non-commercial pass backend "
            "(DepthCrafter, DepthPro, NormalCrafter, MatAnyone2)."
        )
        self._allow_nc_check.toggled.connect(self._on_allow_nc_toggled)
        passes_block.addWidget(self._allow_nc_check)

        # --- Assemble ---
        form = QFormLayout()
        form.addRow("Shot:", self._name_label)
        form.addRow("Frames:", self._frames_label)

        cs_block = QVBoxLayout()
        cs_block.addWidget(self._colorspace_box)
        cs_block.addWidget(self._provenance_label)
        cs_block.addWidget(self._low_conf_warning)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addSpacing(8)
        root.addWidget(_section_label("Colorspace"))
        root.addLayout(cs_block)
        root.addSpacing(12)
        root.addWidget(_section_label("Exposure (EV)"))
        root.addLayout(exposure_row)
        root.addWidget(self._auto_ev_label)
        root.addSpacing(12)
        root.addWidget(self._reset_btn)
        root.addSpacing(12)
        root.addWidget(_section_label("Output"))
        root.addWidget(self._out_inplace)
        root.addWidget(self._out_subfolder)
        root.addWidget(self._out_external)
        root.addLayout(out_root_row)
        root.addWidget(self._resolved_out_label)
        root.addSpacing(12)
        root.addWidget(_section_label("Passes"))
        root.addLayout(passes_block)
        root.addStretch()

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
            self._out_group.blockSignals(True)
            mode_to_button.get(shot.output_mode, self._out_inplace).setChecked(True)
            self._out_group.blockSignals(False)
            self._out_root_btn.setEnabled(shot.output_mode == "external")
            self._rebuild_resolved_out_label()

            # Pass toggles + backend dropdowns reflect the stored state.
            enabled = set(shot.enabled_passes)
            for name, cb in self._pass_checks.items():
                cb.setChecked(name in enabled)
            for name, combo in self._backend_combos.items():
                current = shot.pass_backends.get(name, combo.currentText())
                if combo.findText(current) >= 0:
                    combo.setCurrentText(current)
            self._allow_nc_check.setChecked(bool(shot.allow_noncommercial))

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

    def _on_output_mode_toggled(self, button_id: int, checked: bool) -> None:
        if not checked or self._building or self._current is None:
            return
        mode = {0: "inplace", 1: "subfolder", 2: "external"}[button_id]
        self._current.output_mode = mode
        self._out_root_btn.setEnabled(mode == "external")
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

    def _on_pass_toggled(self, name: str, checked: bool) -> None:
        if self._building or self._current is None:
            return
        enabled = list(self._current.enabled_passes)
        if checked and name not in enabled:
            enabled.append(name)
        elif not checked and name in enabled:
            enabled.remove(name)
        self._current.enabled_passes = enabled
        self._registry.notify_updated(self._current)

    def _on_backend_changed(self, family: str, choice: str) -> None:
        if self._building or self._current is None:
            return
        self._current.pass_backends[family] = choice
        self._registry.notify_updated(self._current)

    def _on_allow_nc_toggled(self, checked: bool) -> None:
        if self._building or self._current is None:
            return
        self._current.allow_noncommercial = bool(checked)
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


# Quiet ruff about unused import when we rearrange the layout.
_ = QFrame

__all__ = ["InspectorPanel"]
