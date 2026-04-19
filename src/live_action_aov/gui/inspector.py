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

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from live_action_aov.gui.shot_state import ShotRegistry, ShotState
from live_action_aov.io.colorspace_detect import SUPPORTED_COLORSPACES

_PASS_NAMES = ("flow", "depth", "normals", "matte")


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

        # --- Pass toggles (read-only in M1) ---
        self._pass_checks: dict[str, QCheckBox] = {}
        passes_row = QHBoxLayout()
        for name in _PASS_NAMES:
            cb = QCheckBox(name)
            cb.setEnabled(False)  # read-only in M1 until the executor is wired in M2
            cb.setToolTip("Pass toggles wire to the executor in the M2 PR.")
            self._pass_checks[name] = cb
            passes_row.addWidget(cb)
        passes_row.addStretch()

        # --- Assemble ---
        form = QFormLayout()
        form.addRow("Shot:", self._name_label)
        form.addRow("Frames:", self._frames_label)

        cs_block = QVBoxLayout()
        cs_block.addWidget(self._colorspace_box)
        cs_block.addWidget(self._provenance_label)
        cs_block.addWidget(self._low_conf_warning)

        passes_block = QVBoxLayout()
        passes_block.addLayout(passes_row)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addSpacing(8)
        root.addWidget(_section_label("Colorspace"))
        root.addLayout(cs_block)
        root.addSpacing(12)
        root.addWidget(_section_label("Exposure (EV)"))
        root.addLayout(exposure_row)
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


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: #ccc;")
    return lbl


# Quiet ruff about unused import when we rearrange the layout.
_ = QFrame

__all__ = ["InspectorPanel"]
