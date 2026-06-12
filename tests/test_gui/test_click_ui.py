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


def test_undo_removes_only_last_point(qapp: QApplication) -> None:
    from live_action_aov.gui.inspector import InspectorPanel

    reg = ShotRegistry()
    panel = InspectorPanel(reg)
    shot = _shot()
    reg.add(shot)
    panel._on_mask_new()
    inst = shot.click_instances[0]
    inst.points.extend([(1.0, 1.0, 1), (2.0, 2.0, 0), (3.0, 3.0, 1)])
    panel._on_mask_undo_point()
    assert inst.points == [(1.0, 1.0, 1), (2.0, 2.0, 0)]
    panel._on_mask_undo_point()
    panel._on_mask_undo_point()
    panel._on_mask_undo_point()  # extra undo on empty list is a no-op
    assert inst.points == []


# --- Mask preview (GPU-free parts) --------------------------------------

def test_preview_guards_no_points_and_wrong_frame(qapp: QApplication) -> None:
    panel, shot, _ = _viewport_with_shot(qapp)
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    panel.set_click_mode(True)
    panel.set_active_instance(inst)
    # No points yet → hint, no crash, no worker call.
    panel.preview_active_mask()
    assert "point" in panel._frame_label.text().lower()
    # Points exist but user scrubbed off the seed frame → hint with frame no.
    inst.points.append((10.0, 10.0, 1))
    shot.current_frame = 1007
    panel.preview_active_mask()
    assert "f1003" in panel._frame_label.text()


def test_preview_mask_invalidated_on_change(qapp: QApplication) -> None:
    import numpy as np

    panel, shot, _ = _viewport_with_shot(qapp)
    inst = ClickInstance(name="hero", seed_frame=1003)
    shot.click_instances.append(inst)
    panel.set_click_mode(True)
    panel.set_active_instance(inst)
    # Pretend a preview happened.
    panel._preview_mask = np.ones((8, 8), dtype=np.float32)
    # New point → stale → cleared.
    panel._on_canvas_pressed(0.5, 0.5, 1)
    assert panel._preview_mask is None
    # Again, then scrubbing clears it too.
    panel._preview_mask = np.ones((8, 8), dtype=np.float32)
    panel._on_frame_slider_changed(1005)
    assert panel._preview_mask is None
    # Again, then switching instance clears it.
    panel._preview_mask = np.ones((8, 8), dtype=np.float32)
    other = ClickInstance(name="b", seed_frame=1001)
    panel.set_active_instance(other)
    assert panel._preview_mask is None


def test_pixmap_and_overlay_conversions(qapp: QApplication) -> None:
    import numpy as np
    from PySide6.QtGui import QColor, QImage, QPixmap

    from live_action_aov.gui.viewport import _mask_to_overlay, _pixmap_to_rgb_float

    # Solid mid-grey pixmap round-trips to ~0.5 float RGB.
    img = QImage(16, 8, QImage.Format.Format_RGB888)
    img.fill(QColor(128, 128, 128))
    arr = _pixmap_to_rgb_float(QPixmap.fromImage(img))
    assert arr.shape == (8, 16, 3)
    assert abs(float(arr.mean()) - 128 / 255) < 0.01
    # Overlay: masked pixels get alpha, unmasked stay fully transparent.
    mask = np.zeros((8, 16), dtype=np.float32)
    mask[2:6, 4:12] = 1.0
    overlay = _mask_to_overlay(mask)
    assert overlay.size().width() == 16
    assert overlay.pixelColor(5, 3).alpha() > 0
    assert overlay.pixelColor(0, 0).alpha() == 0
