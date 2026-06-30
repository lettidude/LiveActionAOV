"""Session save/load — full-fidelity round-trip of the prep state.

The promise: a 50-shot prep survives a crash. So the round-trip must
preserve EVERYTHING the artist set — colour decisions, models, concepts,
click-to-mask instances, output routing — and loading must be resilient
(missing plate folders are skipped with a warning, not a crash).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.session_io import (
    SESSION_SUFFIX,
    load_session,
    save_session,
    shot_from_dict,
    shot_to_dict,
)
from live_action_aov.gui.shot_state import ClickInstance, ShotState
from live_action_aov.io.colorspace_detect import DetectedColorspace


def _full_shot(folder: Path) -> ShotState:
    """A ShotState with every user-settable field at a non-default value."""
    return ShotState(
        name="PF0052",
        folder=folder,
        sequence_pattern="PF0052.%04d.exr",
        frame_range=(1001, 1150),
        resolution=(2048, 1080),
        pixel_aspect=2.0,
        detected=DetectedColorspace(detected="arri_logc4", reason="camera metadata", confident=True),
        override="acescg",
        current_frame=1006,
        view_mode="compare",
        exposure_ev=1.25,
        auto_ev=0.75,
        auto_ev_source="auto",
        sampled_luma=0.123,
        enabled_models=["sam3_rvm", "depth_anything_v2"],
        sam3_concepts="person, red car",
        refine_all_masks=True,
        click_instances=[
            ClickInstance(
                name="hero",
                seed_frame=1006,
                points=[(100.5, 200.25, 1), (300.0, 400.0, 0)],
                box=(10.0, 20.0, 1000.0, 900.0),
            ),
            ClickInstance(name="prop", seed_frame=1001, points=[(5.0, 6.0, 1)]),
        ],
        output_mode="subfolder",
        output_external_root=Path("X:/renders"),
        output_subfolder_name="CryptoUpdate",
        output_external_name="v002",
        proxy_long_edge=1280,
        queued=False,
        # Runtime fields — must NOT survive the round-trip.
        status="failed",
        last_error="boom",
    )


def test_shot_dict_roundtrip_preserves_everything(tmp_path: Path) -> None:
    original = _full_shot(tmp_path)
    restored = shot_from_dict(json.loads(json.dumps(shot_to_dict(original))))
    assert restored.name == original.name
    assert restored.folder == original.folder
    assert restored.sequence_pattern == original.sequence_pattern
    assert restored.frame_range == original.frame_range
    assert restored.resolution == original.resolution
    assert restored.pixel_aspect == original.pixel_aspect
    assert restored.detected is not None
    assert restored.detected.detected == "arri_logc4"
    assert restored.detected.confident is True
    assert restored.override == "acescg"
    assert restored.current_frame == 1006
    assert restored.view_mode == "compare"
    assert restored.exposure_ev == 1.25
    assert restored.auto_ev == 0.75
    assert restored.sampled_luma == 0.123
    assert restored.enabled_models == ["sam3_rvm", "depth_anything_v2"]
    assert restored.sam3_concepts == "person, red car"
    assert restored.refine_all_masks is True
    assert restored.output_mode == "subfolder"
    assert restored.output_external_root == Path("X:/renders")
    assert restored.output_subfolder_name == "CryptoUpdate"
    assert restored.output_external_name == "v002"
    assert restored.proxy_long_edge == 1280
    assert restored.queued is False
    # Click instances survive with exact coordinates and labels.
    hero = restored.click_instances[0]
    assert hero.name == "hero"
    assert hero.seed_frame == 1006
    assert hero.points == [(100.5, 200.25, 1), (300.0, 400.0, 0)]
    assert hero.box == (10.0, 20.0, 1000.0, 900.0)
    assert restored.click_instances[1].box is None
    # Runtime state resets — a restored session starts idle.
    assert restored.status == "new"
    assert restored.last_error == ""


def test_save_load_session_file(tmp_path: Path) -> None:
    plate_a = tmp_path / "plateA"
    plate_a.mkdir()
    shots = [_full_shot(plate_a)]
    f = tmp_path / f"myprep{SESSION_SUFFIX}"
    save_session(f, shots)
    loaded, warnings = load_session(f)
    assert len(loaded) == 1
    assert warnings == []
    assert loaded[0].name == "PF0052"
    assert loaded[0].click_instances[0].points[0] == (100.5, 200.25, 1)


def test_load_skips_missing_plate_folders(tmp_path: Path) -> None:
    plate_a = tmp_path / "plateA"
    plate_a.mkdir()
    gone = tmp_path / "plateGONE"  # never created
    f = tmp_path / f"s{SESSION_SUFFIX}"
    save_session(f, [_full_shot(plate_a), _full_shot(gone)])
    loaded, warnings = load_session(f)
    assert len(loaded) == 1
    assert len(warnings) == 1
    assert "plateGONE" in warnings[0]


def test_load_rejects_foreign_and_future_files(tmp_path: Path) -> None:
    not_session = tmp_path / "x.json"
    not_session.write_text(json.dumps({"kind": "something_else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Not a LiveActionAOV session"):
        load_session(not_session)
    future = tmp_path / "y.json"
    future.write_text(
        json.dumps({"kind": "session", "version": 999, "shots": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="version 999"):
        load_session(future)


def test_save_is_atomic_no_tmp_left_behind(tmp_path: Path) -> None:
    plate = tmp_path / "p"
    plate.mkdir()
    f = tmp_path / f"a{SESSION_SUFFIX}"
    save_session(f, [_full_shot(plate)])
    save_session(f, [_full_shot(plate)])  # overwrite path also clean
    assert f.is_file()
    assert not list(tmp_path.glob("*.tmp"))
