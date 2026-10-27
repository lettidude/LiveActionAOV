"""Schema-level checks on `CameraSolve`.

We don't need to assert COLMAP numerics here — that's the pass's job.
These tests guard the JSON contract: a freshly-minted `CameraSolve`
must round-trip through `model_dump_json` without losing fields, extra
keys must be rejected (so a drifting schema breaks loudly), and the
defaults must match the Y-up/unitless promises in the docstring.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from live_action_aov.io.camera_schema import (
    BackendInfo,
    CameraSolve,
    Extrinsic,
    Intrinsics,
    Quality,
    ScenePoint,
    ShotRef,
)


def _minimal_solve() -> CameraSolve:
    return CameraSolve(
        shot=ShotRef(
            name="shot_01",
            plate_source="/plates/shot_01",
            frame_range=(1001, 1002),
            resolution=(1920, 1080),
        ),
        backend=BackendInfo(name="pycolmap", version="4.0.3"),
        intrinsics=Intrinsics(
            fx_px=2048.0,
            fy_px=2048.0,
            cx_px=960.0,
            cy_px=540.0,
            width_px=1920,
            height_px=1080,
        ),
        extrinsics=[
            Extrinsic(
                frame=1001,
                translation=(0.0, 0.0, 0.0),
                rotation_euler_zxy_deg=(0.0, 0.0, 0.0),
                rotation_matrix=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            ),
        ],
        points=[
            ScenePoint(xyz=(1.0, 2.0, 3.0), color_rgb=(128, 128, 128)),
        ],
        quality=Quality(
            per_frame_reprojection_error_px={1001: 0.4},
            aggregate_reprojection_error_px=0.4,
            tracked_points_count={1001: 100},
            registered_frame_count=1,
            total_frame_count=2,
            solve_confidence=0.75,
        ),
    )


def test_roundtrip_json_is_lossless():
    solve = _minimal_solve()
    dumped = solve.model_dump_json()
    loaded = CameraSolve.model_validate_json(dumped)
    assert loaded.model_dump() == solve.model_dump()


def test_defaults_declare_y_up_rh_unitless():
    solve = _minimal_solve()
    assert solve.conventions.axis_system == "y_up_rh"
    assert solve.conventions.camera_convention == "opengl"
    assert solve.conventions.rot_order == "ZXY"
    assert solve.conventions.angle_unit == "degrees"
    assert solve.conventions.units == "unitless"


def test_intrinsics_require_four_canonical_fields():
    # Missing required field (fx_px) should raise, not silently default.
    with pytest.raises(ValidationError):
        Intrinsics(
            fy_px=2048.0,
            cx_px=960.0,
            cy_px=540.0,
            width_px=1920,
            height_px=1080,
        )  # type: ignore[call-arg]


def test_extra_fields_rejected():
    # The schema uses `extra="forbid"` so unknown top-level fields break
    # the parse. This ensures a rogue writer can't smuggle fields into
    # the sidecar that downstream readers will silently drop.
    payload = json.loads(_minimal_solve().model_dump_json())
    payload["mystery_field"] = "lurking"
    with pytest.raises(ValidationError):
        CameraSolve.model_validate(payload)


def test_empty_extrinsics_and_points_valid():
    # A pass that registered zero frames should still produce a valid
    # sidecar with an empty-but-structured payload; downstream QC reads
    # quality.registered_frame_count == 0 to decide what to show.
    solve = CameraSolve(
        backend=BackendInfo(name="pycolmap", version="4.0.3"),
        intrinsics=Intrinsics(
            fx_px=2048.0,
            fy_px=2048.0,
            cx_px=960.0,
            cy_px=540.0,
            width_px=1920,
            height_px=1080,
        ),
    )
    assert solve.extrinsics == []
    assert solve.points == []
    assert solve.quality.registered_frame_count == 0
