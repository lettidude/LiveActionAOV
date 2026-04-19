"""Unit + integration tests for `PyColmapCameraPass`.

Strategy: run pycolmap *never* in CI — it's expensive and needs real
image features. Instead:

* The pure helpers (`opencv_to_opengl_world_from_camera`,
  `rotation_matrix_to_euler_zxy_deg`, `build_solve_from_reconstruction`)
  are covered with explicit matrices and a hand-built fake
  `Reconstruction`. These are where the coordinate-system correctness
  lives — if they drift, downstream Nuke import breaks silently, which
  is the worst failure mode for VFX tooling. So we over-test them.
* The end-to-end `run_shot` → `emit_sidecars` path is covered with
  pycolmap's entry points monkeypatched to no-ops, confirming the
  pipeline wires correctly into the schema and writer without actually
  solving anything.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from live_action_aov.core.job import Shot
from live_action_aov.io.camera_schema import CameraSolve
from live_action_aov.passes.camera.pycolmap_track import (
    PyColmapCameraPass,
    build_solve_from_reconstruction,
    opencv_to_opengl_world_from_camera,
    rotation_matrix_to_euler_zxy_deg,
)

# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def test_identity_cam_from_world_yields_origin_camera_with_flipped_local_frame():
    # Camera at origin looking along world +Z in OpenCV (y-down camera).
    R_cw = np.eye(3)
    t_cw = np.zeros(3)
    R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
    # Camera's world position: origin.
    assert np.allclose(t_gl, [0.0, 0.0, 0.0])
    # Local-frame flip means R_gl equals diag(1, -1, -1) — the Y and Z
    # axes of the local camera frame have been negated (OpenCV→OpenGL).
    assert np.allclose(R_gl, np.diag([1.0, -1.0, -1.0]))


def test_translated_camera_recovers_world_position():
    # In COLMAP, `cam_from_world.translation` is NOT the camera position;
    # it's the translation that takes world-origin into camera frame.
    # With identity rotation and t_cw = (0, 0, 5), the world origin lands
    # 5 units in front of the camera, i.e. the camera sits at (0, 0, -5).
    R_cw = np.eye(3)
    t_cw = np.array([0.0, 0.0, 5.0])
    _R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
    assert np.allclose(t_gl, [0.0, 0.0, -5.0])


def test_rotated_camera_world_position_is_inverse_transform():
    # Camera rotated 90° about world Y in OpenCV, with cam_from_world
    # translation (0, 0, 1). world_from_cam.translation = -R_cw.T @ t_cw;
    # R_cw.T (which is Ry(-90°)) applied to (0, 0, 1) gives (-1, 0, 0);
    # negating yields camera world position (1, 0, 0).
    theta = math.pi / 2
    Ry = np.array(
        [
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)],
        ]
    )
    t_cw = np.array([0.0, 0.0, 1.0])
    _, t_gl = opencv_to_opengl_world_from_camera(Ry, t_cw)
    assert np.allclose(t_gl, [1.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Euler decomposition
# ---------------------------------------------------------------------------


def _euler_zxy_to_matrix(alpha_deg: float, beta_deg: float, gamma_deg: float) -> np.ndarray:
    """Reference implementation: R = Ry(β) · Rx(α) · Rz(γ)."""
    a = math.radians(alpha_deg)
    b = math.radians(beta_deg)
    g = math.radians(gamma_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def test_euler_zxy_identity():
    alpha, beta, gamma = rotation_matrix_to_euler_zxy_deg(np.eye(3))
    assert abs(alpha) < 1e-9
    assert abs(beta) < 1e-9
    assert abs(gamma) < 1e-9


@pytest.mark.parametrize(
    "alpha,beta,gamma",
    [
        (0.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        (0.0, 45.0, 0.0),
        (0.0, 0.0, 60.0),
        (10.0, 20.0, 30.0),
        (-15.0, 75.0, -40.0),
    ],
)
def test_euler_zxy_roundtrip(alpha, beta, gamma):
    R = _euler_zxy_to_matrix(alpha, beta, gamma)
    a_out, b_out, g_out = rotation_matrix_to_euler_zxy_deg(R)
    R_round = _euler_zxy_to_matrix(a_out, b_out, g_out)
    # Matrix round-trip is what matters (Euler angles can have multiple
    # equivalent representations, especially near gimbal lock).
    assert np.allclose(R, R_round, atol=1e-9)


def test_euler_zxy_gimbal_lock_pins_gamma_to_zero():
    # α = 90° makes R[1,2] = -1 → cos α = 0 (gimbal lock). Our
    # decomposition pins γ = 0 and pushes the combined roll into β.
    R = _euler_zxy_to_matrix(90.0, 30.0, 0.0)
    alpha, _beta, gamma = rotation_matrix_to_euler_zxy_deg(R)
    assert abs(alpha - 90.0) < 1e-6
    assert abs(gamma) < 1e-6


# ---------------------------------------------------------------------------
# Fake pycolmap Reconstruction for build_solve_from_reconstruction
# ---------------------------------------------------------------------------


class _FakeRotation:
    def __init__(self, R: np.ndarray) -> None:
        self._R = R

    def matrix(self) -> np.ndarray:
        return self._R


class _FakeRigid3d:
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.rotation = _FakeRotation(R)
        self.translation = t


class _FakeCamera:
    def __init__(
        self,
        width: int,
        height: int,
        params: list[float],
        model_name: str = "PINHOLE",
    ) -> None:
        self.width = width
        self.height = height
        self.params = params
        self.model_name = model_name


class _FakePoint2D:
    def __init__(self, point3D_id: int, has_pt3d: bool = True) -> None:
        self.point3D_id = point3D_id
        self._has = has_pt3d

    def has_point3D(self) -> bool:
        return self._has


class _FakeImage:
    def __init__(
        self,
        name: str,
        camera_id: int,
        R_cw: np.ndarray,
        t_cw: np.ndarray,
        points2D: list[_FakePoint2D],
    ) -> None:
        self.name = name
        self.camera_id = camera_id
        self.cam_from_world = _FakeRigid3d(R_cw, t_cw)
        self.points2D = points2D


class _FakeTrack:
    def __init__(self, length: int) -> None:
        self._length = length

    def length(self) -> int:
        return self._length


class _FakePoint3D:
    def __init__(
        self,
        xyz: tuple[float, float, float],
        color: tuple[int, int, int] = (200, 200, 200),
        error: float = 0.5,
        track_length: int = 5,
    ) -> None:
        self.xyz = np.array(xyz, dtype=np.float64)
        self.color = np.array(color, dtype=np.uint8)
        self.error = error
        self.track = _FakeTrack(track_length)


class _FakeReconstruction:
    def __init__(
        self,
        cameras: dict[int, _FakeCamera],
        images: dict[int, _FakeImage],
        points3D: dict[int, _FakePoint3D],
    ) -> None:
        self.cameras = cameras
        self.images = images
        self.points3D = points3D

    def num_reg_images(self) -> int:
        return len(self.images)


def _two_frame_reconstruction() -> _FakeReconstruction:
    cam = _FakeCamera(
        width=1920,
        height=1080,
        params=[2100.0, 2100.0, 960.0, 540.0],
        model_name="PINHOLE",
    )
    R_cw = np.eye(3)
    img1 = _FakeImage(
        "frame_001001.png",
        camera_id=1,
        R_cw=R_cw,
        t_cw=np.array([0.0, 0.0, 0.0]),
        points2D=[_FakePoint2D(101), _FakePoint2D(102)],
    )
    img2 = _FakeImage(
        "frame_001002.png",
        camera_id=1,
        R_cw=R_cw,
        t_cw=np.array([0.0, 0.0, 1.0]),
        points2D=[_FakePoint2D(101), _FakePoint2D(102)],
    )
    pt1 = _FakePoint3D((1.0, 2.0, 3.0), error=0.4, track_length=5)
    pt2 = _FakePoint3D((-1.0, 0.5, 4.5), error=0.6, track_length=3)
    return _FakeReconstruction(
        cameras={1: cam},
        images={1: img1, 2: img2},
        points3D={101: pt1, 102: pt2},
    )


def test_build_solve_extracts_shared_intrinsics():
    rec = _two_frame_reconstruction()
    frame_list = [(1001, "frame_001001.png"), (1002, "frame_001002.png")]
    solve = build_solve_from_reconstruction(
        rec,
        frame_list=frame_list,
        total_frame_count=2,
        params=PyColmapCameraPass.DEFAULT_PARAMS,
    )
    assert solve.intrinsics.fx_px == 2100.0
    assert solve.intrinsics.fy_px == 2100.0
    assert solve.intrinsics.cx_px == 960.0
    assert solve.intrinsics.width_px == 1920
    assert solve.intrinsics.distortion_model == "pinhole"
    # focal_mm_hint derivation: 2100 * 24.892 / 1920 = 27.225625
    assert solve.intrinsics.focal_mm_hint == pytest.approx(27.225625, abs=1e-4)


def test_build_solve_produces_one_extrinsic_per_registered_frame():
    rec = _two_frame_reconstruction()
    frame_list = [(1001, "frame_001001.png"), (1002, "frame_001002.png")]
    solve = build_solve_from_reconstruction(
        rec,
        frame_list=frame_list,
        total_frame_count=2,
        params=PyColmapCameraPass.DEFAULT_PARAMS,
    )
    assert [e.frame for e in solve.extrinsics] == [1001, 1002]
    # All four elements of the rotation matrix + translation must be finite.
    for e in solve.extrinsics:
        for row in e.rotation_matrix:
            for val in row:
                assert math.isfinite(val)
        for val in e.translation:
            assert math.isfinite(val)


def test_build_solve_flags_unregistered_frames():
    rec = _two_frame_reconstruction()
    # Pretend 1003 was submitted but didn't register.
    frame_list = [
        (1001, "frame_001001.png"),
        (1002, "frame_001002.png"),
        (1003, "frame_001003.png"),
    ]
    solve = build_solve_from_reconstruction(
        rec,
        frame_list=frame_list,
        total_frame_count=3,
        params=PyColmapCameraPass.DEFAULT_PARAMS,
    )
    assert solve.unregistered_frames == [1003]
    assert solve.quality.registered_frame_count == 2
    assert solve.quality.total_frame_count == 3
    # Registration ratio below 0.9 triggers the warning.
    assert any("registered" in w for w in solve.quality.warnings)


def test_build_solve_harvests_points_with_color_and_error():
    rec = _two_frame_reconstruction()
    frame_list = [(1001, "frame_001001.png"), (1002, "frame_001002.png")]
    solve = build_solve_from_reconstruction(
        rec,
        frame_list=frame_list,
        total_frame_count=2,
        params=PyColmapCameraPass.DEFAULT_PARAMS,
    )
    assert len(solve.points) == 2
    # Points were hand-seeded with specific errors and track lengths.
    errors = {p.reprojection_error_px for p in solve.points}
    assert errors == {0.4, 0.6}
    lengths = {p.track_length for p in solve.points}
    assert lengths == {5, 3}


# ---------------------------------------------------------------------------
# emit_sidecars — no solve, and happy path
# ---------------------------------------------------------------------------


def _make_shot(tmp_path: Path) -> Shot:
    folder = tmp_path / "plate"
    folder.mkdir()
    return Shot(
        name="shot_01",
        folder=folder,
        sequence_pattern="shot_01.####.exr",
        frame_range=(1001, 1002),
        resolution=(1920, 1080),
    )


def test_emit_sidecars_is_noop_before_run_shot(tmp_path: Path) -> None:
    pass_ = PyColmapCameraPass({})
    shot = _make_shot(tmp_path)
    # run_shot was never called, solve is None → hook returns empty dict
    # and writes nothing, so the executor merges nothing into
    # shot.sidecars. This is the safe default for any pass whose
    # preflight failed.
    assert pass_.emit_sidecars(shot) == {}
    assert not (shot.folder / "camera_track").exists()


def test_emit_sidecars_writes_json_and_nk_in_camera_track_subdir(tmp_path: Path):
    rec = _two_frame_reconstruction()
    frame_list = [(1001, "frame_001001.png"), (1002, "frame_001002.png")]
    pass_ = PyColmapCameraPass({})
    pass_._solve = build_solve_from_reconstruction(  # type: ignore[attr-defined]
        rec,
        frame_list=frame_list,
        total_frame_count=2,
        params=pass_.params,
    )
    shot = _make_shot(tmp_path)
    paths = pass_.emit_sidecars(shot)
    assert set(paths.keys()) == {"camera", "camera_nk"}
    json_path = paths["camera"]
    nk_path = paths["camera_nk"]
    assert json_path == shot.folder / "camera_track" / "shot_01.camera.json"
    assert nk_path == shot.folder / "camera_track" / "shot_01.camera.nk"
    assert json_path.exists()
    assert nk_path.exists()
    # JSON re-parses into a CameraSolve and carries the Shot identity.
    reloaded = CameraSolve.model_validate_json(json_path.read_text(encoding="utf-8"))
    assert reloaded.shot.name == "shot_01"
    assert reloaded.shot.plate_source == str(shot.folder)
    # .nk contains the shot name in the node name slot.
    nk_text = nk_path.read_text(encoding="utf-8")
    assert "LiveActionAOV_Camera_shot_01" in nk_text


# ---------------------------------------------------------------------------
# Mocked end-to-end run_shot
# ---------------------------------------------------------------------------


class _FakeReader:
    """Stands in for DisplayTransformedReader — returns a noisy RGB frame
    so Pillow has something to encode without complaining."""

    def __init__(self, rng: np.random.Generator, size: tuple[int, int] = (64, 48)) -> None:
        self._rng = rng
        self._size = size

    def read_frame(self, f: int) -> tuple[np.ndarray, dict[str, Any]]:
        h, w = self._size[1], self._size[0]
        arr = self._rng.random((h, w, 3), dtype=np.float32)
        return arr, {"frame": f}


def _install_fake_pycolmap(monkeypatch: pytest.MonkeyPatch, rec: _FakeReconstruction) -> None:
    """Replace pycolmap entry points with no-ops and a canned reconstruction."""
    import pycolmap  # type: ignore

    # The four top-level calls the pass makes — make them inert.
    monkeypatch.setattr(pycolmap, "extract_features", lambda **_kw: None)
    monkeypatch.setattr(pycolmap, "match_sequential", lambda *_a, **_kw: None)
    monkeypatch.setattr(pycolmap, "match_exhaustive", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        pycolmap,
        "incremental_mapping",
        lambda **_kw: {0: rec},
    )


def test_run_shot_then_emit_sidecars_writes_both_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pycolmap")
    pytest.importorskip("PIL")
    _install_fake_pycolmap(monkeypatch, _two_frame_reconstruction())

    rng = np.random.default_rng(0)
    reader = _FakeReader(rng)
    pass_ = PyColmapCameraPass({"keep_workdir": False})

    # run_shot should drive the (now mocked) pycolmap pipeline and stash
    # a CameraSolve on the instance without writing anything yet.
    out = pass_.run_shot(reader, (1001, 1002))
    assert out == {}
    assert pass_._solve is not None  # type: ignore[attr-defined]
    assert pass_._solve.quality.registered_frame_count == 2  # type: ignore[attr-defined]

    # emit_sidecars should now write both files into the plate subdir.
    shot = _make_shot(tmp_path)
    paths = pass_.emit_sidecars(shot)
    assert (shot.folder / "camera_track" / "shot_01.camera.json").exists()
    assert (shot.folder / "camera_track" / "shot_01.camera.nk").exists()
    assert paths["camera"].name.endswith(".camera.json")
    assert paths["camera_nk"].name.endswith(".camera.nk")


def test_run_shot_rejects_under_two_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pycolmap")
    pytest.importorskip("PIL")
    _install_fake_pycolmap(monkeypatch, _two_frame_reconstruction())

    pass_ = PyColmapCameraPass({"keep_workdir": False})
    reader = _FakeReader(np.random.default_rng(1))
    with pytest.raises(RuntimeError, match="at least 2 frames"):
        pass_.run_shot(reader, (1001, 1001))


# ---------------------------------------------------------------------------
# Declarative contract surface
# ---------------------------------------------------------------------------


def test_declared_contract_is_commercial_safe_and_video_clip():
    # These attrs are what the CLI license gate + scheduler read *before*
    # constructing the pass, so they must stay accurate even as the
    # implementation evolves.
    assert PyColmapCameraPass.license.commercial_use is True
    assert PyColmapCameraPass.license.spdx == "BSD-3-Clause"
    assert PyColmapCameraPass.temporal_mode.value == "video_clip"
    assert PyColmapCameraPass.pass_type.value == "camera"
    assert PyColmapCameraPass.produces_channels == []
    sidecar_names = {s.name for s in PyColmapCameraPass.produces_sidecars}
    assert sidecar_names == {"camera", "camera_nk"}
