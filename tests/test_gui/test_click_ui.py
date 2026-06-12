"""Click-to-mask UI plumbing — Masks tab + viewport mapping (offscreen).

Covers the GPU-free interaction layer: canvas→plate coordinate mapping
(letterbox offsets), the Masks tab lifecycle (new/rename/delete/clear,
auto-arm, signals to the viewport), and the viewport's click handling
rules (plate-px append, compare-mode guard, seed-frame anchoring).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from live_action_aov.gui.shot_state import ClickInstance, ShotRegistry, ShotState
from live_action_aov.gui.viewport import ViewportPanel, _canvas_pos_to_norm


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app  # type: ignore[return-value]


def _shot(**kw: object) -> ShotState:
    base: dict[str, object] = dict(
        name="PF0071",
        folder=Path("."),
        sequence_pattern="PF0071.%04d.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
    )
    base.update(kw)
    return ShotState(**base)  # type: ignore[arg-type]


# --- _canvas_pos_to_norm ----------------------------------------------

def test_norm_mapping_centered_letterbox() -> None:
    # Canvas 1000x600, pixmap 800x450 → offsets (100, 75).
    assert _canvas_pos_to_norm(100, 75, 1000, 600, 800, 450) == (0.0, 0.0)
    assert _canvas_pos_to_norm(900, 525, 1000, 600, 800, 450) == (1.0, 1.0)
    nx, ny = _canvas_pos_to_norm(500, 300, 1000, 600, 800, 450)  # type: ignore[misc]
    assert nx == pytest.approx(0.5)
    assert ny == pytest.approx(0.5)


def test_norm_mapping_rejects_letterbox_and_empty() -> None:
    # Click in the left letterbox bar → None.
    assert _canvas_pos_to_norm(50, 300, 1000, 600, 800, 450) is None
    # Degenerate pixmap → None.
    assert _canvas_pos_to_norm(10, 10, 100, 100, 0, 0) is None


# --- ViewportPanel click handling --------------------------------------

def _viewport_with_shot(qapp: QApplication) -> tuple[ViewportPanel, ShotState, ShotRegistry]:
    reg = ShotRegistry()
    panel = ViewportPanel(reg)
    shot = _shot()
    reg.add(shot)
    shot.current_frame = 1003
    return panel, shot, reg


def test_click_appends_plate_px_point(qapp: QApplication) -> None:
    panel, shot, _ = _viewport_with_shot(qapp)
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    panel.set_click_mode(True)
    panel.set_active_instance(inst)
    panel._on_canvas_pressed(0.5, 0.5, 1)
    panel._on_canvas_pressed(0.25, 0.75, 0)
    assert inst.points == [(960.0, 540.0, 1), (480.0, 810.0, 0)]


def test_click_ignored_when_disarmed_or_no_instance(qapp: QApplication) -> None:
    panel, shot, _ = _viewport_with_shot(qapp)
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    # Armed but no active instance.
    panel.set_click_mode(True)
    panel.set_active_instance(None)
    panel._on_canvas_pressed(0.5, 0.5, 1)
    # Active instance but disarmed.
    panel.set_click_mode(False)
    panel.set_active_instance(inst)
    panel._on_canvas_pressed(0.5, 0.5, 1)
    assert inst.points == []


def test_click_ignored_in_compare_mode(qapp: QApplication) -> None:
    panel, shot, _ = _viewport_with_shot(qapp)
    shot.view_mode = "compare"
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    panel.set_click_mode(True)
    panel.set_active_instance(inst)
    panel._on_canvas_pressed(0.5, 0.5, 1)
    assert inst.points == []


def test_first_click_anchors_seed_frame_then_locks(qapp: QApplication) -> None:
    panel, shot, _ = _viewport_with_shot(qapp)
    # Instance created while viewing 1001, but user scrubbed to 1003.
    inst = ClickInstance(name="hero", seed_frame=1001)
    shot.click_instances.append(inst)
    panel.set_click_mode(True)
    panel.set_active_instance(inst)
    panel._on_canvas_pressed(0.5, 0.5, 1)
    assert inst.seed_frame == 1003  # re-anchored to the viewed frame
    # Scrub away; further clicks are rejected (points live on the seed frame).
    shot.current_frame = 1007
    panel._on_canvas_pressed(0.1, 0.1, 1)
    assert len(inst.points) == 1


# --- InspectorPanel Masks tab ------------------------------------------

def test_masks_tab_lifecycle_and_signals(qapp: QApplication) -> None:
    from live_action_aov.gui.inspector import InspectorPanel

    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    armed: list[bool] = []
    actives: list[object] = []
    panel.click_mode_changed.connect(armed.append)
    panel.active_click_instance_changed.connect(actives.append)

    shot = _shot()
    reg.add(shot)
    shot.current_frame = 1005

    # New object: created at the viewed frame, selected, viewport armed.
    panel._on_mask_new()
    assert len(shot.click_instances) == 1
    inst = shot.click_instances[0]
    assert inst.seed_frame == 1005
    assert armed[-1] is True
    assert actives[-1] is inst

    # Rename feeds the state (→ Cryptomatte name).
    panel._mask_name_edit.setText("hero")
    panel._on_mask_name_edited("hero")
    assert inst.name == "hero"

    # Clear points after the viewport added some.
    inst.points.append((10.0, 10.0, 1))
    panel._on_mask_clear_points()
    assert inst.points == []

    # Delete empties the list and announces no active instance.
    panel._on_mask_delete()
    assert shot.click_instances == []
    assert actives[-1] is None
