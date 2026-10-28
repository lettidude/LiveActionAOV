"""Unit + integration tests for `PyColmapCameraPass`.

Strategy: run pycolmap *never* in CI — it's expensive and needs real
image features. Instead:

* The pure helpers (`opencv_to_opengl_world_from_camera`,
  `rotation_matrix_to_euler_zxy_deg`, `build_solve_from_reconstruction`)
  are covered with explicit matrices and a hand-built fake
  `Reconstruction`. These are where the coordinate-system correctness
  lives — if they drift, downstream Nuke import breaks silently, which
  is the worst failure mode for VFX tooling. So we over-test them.
* The end-to-end `run_shot` -> `emit_sidecars` path is covered with the
  COLMAP solve monkeypatched to return a fake reconstruction.
* The export itself was validated once against real Nuke 16 (`nuke -t`:
  paste the .nk, project solved points with Nuke's own world_matrix)
  on ACTIONVFX PF0081: 1.01 px mean, identical to the solve.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from live_action_aov.core.job import Shot
from live_action_aov.io.camera_schema import CameraSolve
from live_action_aov.passes.camera import pycolmap_track as mod
from live_action_aov.passes.camera.pycolmap_track import (
    PyColmapCameraPass,
    build_solve_from_reconstruction,
    opencv_to_opengl_world_from_camera,
    rotation_matrix_to_euler_zxy_deg,
    world_point_to_gl,
)

# ---------------------------------------------------------------------------
# Coordinate conversion helpers
# ---------------------------------------------------------------------------


def test_identity_cam_from_world_yields_upright_origin_camera():
    # COLMAP's world IS the first camera's OpenCV frame. Flipping both the
    # camera-local frame and the world makes that camera an identity
    # OpenGL camera: at the origin, Y up, looking down -Z (Nuke default).
    R_gl, t_gl = opencv_to_opengl_world_from_camera(np.eye(3), np.zeros(3))
    assert np.allclose(t_gl, [0.0, 0.0, 0.0])
    assert np.allclose(R_gl, np.eye(3))


def test_translated_camera_recovers_world_position():
    # In COLMAP, `cam_from_world.translation` is NOT the camera position;
    # it's the translation that takes world-origin into camera frame.
    # With identity rotation and t_cw = (0, 0, 5), the world origin lands
    # 5 units in front of the camera, i.e. the camera sits at (0, 0, -5).
    R_cw = np.eye(3)
    t_cw = np.array([0.0, 0.0, 5.0])
    _R_gl, t_gl = opencv_to_opengl_world_from_camera(R_cw, t_cw)
    # ...and (0, 0, 5) once the world is flipped Y-up / Z-back.
    assert np.allclose(t_gl, [0.0, 0.0, 5.0])


def test_rotated_camera_world_position_is_inverse_transform():
    # Camera rotated 90° about world Y in OpenCV, with cam_from_world
    # translation (0, 0, 1). world_from_cam.translation = -R_cw.T @ t_cw;
    # R_cw.T (which is Ry(-90°)) applied to (0, 0, 1) gives (-1, 0, 0);
    # negating yields (1, 0, 0); the world flip leaves X untouched.
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
# Fake pycolmap objects (pycolmap 4.x shapes)
# ---------------------------------------------------------------------------


def _rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


class _Rot:
    def __init__(self, R: np.ndarray) -> None:
        self._R = R

    def matrix(self) -> np.ndarray:
        return self._R


class _Pose:
    def __init__(self, R: np.ndarray, t: np.ndarray) -> None:
        self.rotation = _Rot(R)
        self.translation = t


class _P2D:
    def __init__(self, xy: np.ndarray, pid: int) -> None:
        self.xy = xy
        self.point3D_id = pid

    def has_point3D(self) -> bool:
        return True


class _Camera:
    model_name = "SIMPLE_PINHOLE"

    def __init__(self, f: float, w: int, h: int) -> None:
        self.params = [f, w / 2, h / 2]
        self.width, self.height = w, h


class _Track:
    def __init__(self, n: int) -> None:
        self.n = n

    def length(self) -> int:
        return self.n


class _Point3D:
    def __init__(self, xyz: np.ndarray) -> None:
        self.xyz = xyz
        self.color = (10, 20, 30)
        self.error = 0.25
        self.track = _Track(7)


class _Image:
    def __init__(self, name: str, R: np.ndarray, t: np.ndarray, cam: _Camera) -> None:
        self.name, self.camera_id = name, 1
        self._pose = _Pose(R, t)
        self._cam = cam
        self.points2D: list[_P2D] = []

    def has_pose(self) -> bool:
        return True

    def cam_from_world(self) -> _Pose:  # a METHOD in pycolmap 4.x
        return self._pose

    def project_point(self, xyz: np.ndarray) -> np.ndarray:
        c = self._pose.rotation.matrix() @ xyz + self._pose.translation
        f, cx, cy = self._cam.params
        return np.array([cx + f * c[0] / c[2], cy + f * c[1] / c[2]])


class _Rec:
    """Cameras dollying along +X (OpenCV world), panning a little, looking at a cloud."""

    def __init__(
        self, names: list[str], cam_w: int = 1024, cam_h: int = 540, noise_px: float = 0.5
    ) -> None:
        self.cameras = {1: _Camera(980.0, cam_w, cam_h)}
        rng = np.random.default_rng(0)
        self.points3D = {i + 1: _Point3D(rng.uniform([-2, -1, 4], [2, 1, 8])) for i in range(40)}
        self.images = {}
        for k, n in enumerate(names):
            R = _rot_y(2.0 * k)
            t = -R @ np.array([0.3 * k, 0.0, 0.0])
            im = _Image(n, R, t, self.cameras[1])
            for pid, pt in self.points3D.items():
                im.points2D.append(_P2D(im.project_point(pt.xyz) + np.array([noise_px, 0.0]), pid))
            self.images[k + 1] = im


_NAMES = ["frame_001001.png", "frame_001002.png", "frame_001003.png"]
_FRAMES = [(1001, _NAMES[0]), (1002, _NAMES[1]), (1003, _NAMES[2]), (1004, "frame_001004.png")]


def _build(**kw: Any) -> CameraSolve:
    return build_solve_from_reconstruction(
        _Rec(_NAMES, **kw),
        frame_list=_FRAMES,
        plate_wh=(2048, 1080),
        params=dict(PyColmapCameraPass.DEFAULT_PARAMS),
    )


def test_build_solve_rescales_intrinsics_and_errors_to_plate_resolution():
    s = _build()
    # Solved at 1024 wide, plate is 2048: focal and principal point double.
    assert s.intrinsics.width_px == 2048 and s.intrinsics.height_px == 1080
    assert s.intrinsics.fx_px == pytest.approx(1960.0)
    assert s.intrinsics.cx_px == pytest.approx(1024.0)
    assert s.intrinsics.vaperture_mm_hint == pytest.approx(24.892 * 1080 / 2048)
    assert s.intrinsics.focal_mm_hint == pytest.approx(1960.0 * 24.892 / 2048)
    # 0.5 px residual at solve res = 1.0 px at plate res.
    assert s.quality.aggregate_reprojection_error_px == pytest.approx(1.0, abs=1e-6)
    assert all(
        v == pytest.approx(1.0, abs=1e-6)
        for v in s.quality.per_frame_reprojection_error_px.values()
    )


def test_build_solve_flags_unregistered_frames_and_counts():
    s = _build()
    assert [e.frame for e in s.extrinsics] == [1001, 1002, 1003]
    assert s.unregistered_frames == [1004]
    assert s.quality.registered_frame_count == 3 and s.quality.total_frame_count == 4
    assert any("3/4 frames registered" in w for w in s.quality.warnings)


def test_gl_export_projects_exactly_like_colmap():
    """The contract that matters for Nuke: projecting a GL-world point
    through the GL camera (looking down -Z, Y up) lands on the same pixel
    COLMAP observed. The same check passed inside Nuke 16 on a real solve."""
    rec = _Rec(_NAMES, noise_px=0.0)
    s = build_solve_from_reconstruction(
        rec, frame_list=_FRAMES[:3], plate_wh=(1024, 540), params={}
    )
    f, cx, cy = s.intrinsics.fx_px, s.intrinsics.cx_px, s.intrinsics.cy_px
    for e, im in zip(s.extrinsics, rec.images.values(), strict=True):
        R, t = np.array(e.rotation_matrix), np.array(e.translation)
        for p2d in im.points2D[:10]:
            X = np.array(world_point_to_gl(tuple(rec.points3D[p2d.point3D_id].xyz)))
            c = R.T @ (X - t)
            xy = np.array([cx + f * c[0] / -c[2], cy - f * c[1] / -c[2]])
            assert np.allclose(xy, p2d.xy, atol=1e-6)


def test_points_are_in_the_same_flipped_world():
    s = _build()
    assert len(s.points) == 40
    p0 = s.points[0]
    assert p0.color_rgb == (10, 20, 30) and p0.track_length == 7
    assert p0.reprojection_error_px == pytest.approx(0.5)  # 0.25 px at half res
    assert p0.xyz[2] < 0  # in front of the first camera = -Z in GL


def test_first_camera_is_upright_identity():
    s = _build()
    assert np.allclose(np.array(s.extrinsics[0].rotation_matrix), np.eye(3))
    assert np.allclose(s.extrinsics[0].rotation_euler_zxy_deg, (0.0, 0.0, 0.0))


# ---------------------------------------------------------------------------
# SAM 3 exclusion masks
# ---------------------------------------------------------------------------


def _pass_with_masks(tmp_path: Path, **params: Any) -> PyColmapCameraPass:
    p = PyColmapCameraPass(params)
    stack = np.zeros((2, 54, 102), np.uint8)
    stack[0, 20:30, 40:50] = 1  # frame 1001: a person in the middle
    car = np.zeros((1, 54, 102), np.uint8)
    car[0, 0:5, 0:5] = 1
    p.ingest_artifacts(
        {
            "sam3_hard_masks": {
                1001: {
                    7: {"label": "person", "frames": [1001, 1002], "stack": stack},
                    9: {"label": "car", "frames": [1002], "stack": car},
                }
            }
        }
    )
    p._workdir = tmp_path
    return p


def test_masks_exclude_dilated_objects_with_colmap_naming(tmp_path: Path):
    import cv2

    p = _pass_with_masks(tmp_path)
    d = p._export_masks(_FRAMES[:3], (204, 108))  # 2x the SAM resolution
    assert d is not None and p._masked_labels == ["car", "person"]
    m1 = cv2.imread(str(d / "frame_001001.png.png"), cv2.IMREAD_GRAYSCALE)
    assert m1.shape == (108, 204)
    assert m1[50, 90] == 0  # inside the (upscaled) person -> ignored
    assert m1[39, 79] == 0  # just outside it: dilation (1% of 204 = 2 px)
    assert m1[5, 200] == 255  # background kept
    m2 = cv2.imread(str(d / "frame_001002.png.png"), cv2.IMREAD_GRAYSCALE)
    assert m2[2, 2] == 0  # the car, only present on 1002
    m3 = cv2.imread(str(d / "frame_001003.png.png"), cv2.IMREAD_GRAYSCALE)
    assert (m3 == 255).all()  # no object tracked on 1003


def test_mask_labels_filter_and_switch(tmp_path: Path):
    p = _pass_with_masks(tmp_path, mask_labels=["Car"])
    p._export_masks(_FRAMES[:3], (102, 54))
    assert p._masked_labels == ["car"]
    off = _pass_with_masks(tmp_path, mask_objects=False)
    assert off._export_masks(_FRAMES[:3], (102, 54)) is None


def test_no_sam3_in_job_means_no_masks(tmp_path: Path):
    p = PyColmapCameraPass()
    p.ingest_artifacts({})
    p._workdir = tmp_path
    assert p._export_masks(_FRAMES[:3], (102, 54)) is None and p._masked_labels == []


# ---------------------------------------------------------------------------
# run_shot -> emit_sidecars wiring
# ---------------------------------------------------------------------------


class _Reader:
    def __init__(self, n: int, h: int = 108, w: int = 192) -> None:
        self.n, self.h, self.w = n, h, w

    def read_frame(self, f: int):
        rng = np.random.default_rng(f)
        return rng.random((self.h, self.w, 3), dtype=np.float32), {}


def test_run_shot_then_emit_sidecars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("pycolmap")
    seen: dict[str, Any] = {}

    def fake_solve(self, workdir, image_dir, mask_dir, frame_list):
        seen["pngs"] = sorted(x.name for x in image_dir.glob("*.png"))
        seen["mask_dir"] = mask_dir
        self._sparse_dir = workdir / "sparse" / "0"
        self._sparse_dir.mkdir(parents=True)
        (self._sparse_dir / "cameras.bin").write_bytes(b"x")
        return _Rec([n for _, n in frame_list[:3]], cam_w=192, cam_h=108)

    monkeypatch.setattr(PyColmapCameraPass, "_solve_colmap", fake_solve)
    p = PyColmapCameraPass()
    assert p.run_shot(_Reader(4), (1001, 1004)) == {}
    assert seen["pngs"] == [n for _, n in _FRAMES] and seen["mask_dir"] is None
    shot = Shot(
        name="PF0081-03",
        folder=tmp_path / "plate",
        output_dir=tmp_path / "out",
        sequence_pattern="PF0081-03.####.exr",
        frame_range=(1001, 1004),
        resolution=(192, 108),
    )
    workdir = p._workdir
    paths = p.emit_sidecars(shot)
    cam_dir = tmp_path / "out" / "camera_track"
    assert paths["camera"] == cam_dir / "PF0081-03.camera.json"
    assert paths["camera_nk"].read_text(encoding="utf-8").count("Camera2 {") == 1
    assert "element vertex 40" in paths["camera_points"].read_text(encoding="ascii")
    # OBJ point cloud for Nuke's ReadGeo2: 40 vertices, every one in a `p` element.
    obj = paths["camera_points_obj"].read_text(encoding="ascii").splitlines()
    assert sum(ln.startswith("v ") for ln in obj) == 40
    idx = [int(t) for ln in obj if ln.startswith("p ") for t in ln.split()[1:]]
    assert sorted(idx) == list(range(1, 41))
    assert (paths["camera_colmap"] / "cameras.bin").exists()
    solve = CameraSolve.model_validate_json(paths["camera"].read_text(encoding="utf-8"))
    assert solve.shot.name == "PF0081-03" and solve.unregistered_frames == [1004]
    assert workdir is not None and not workdir.exists()  # temp cleaned (keep_workdir False)


def test_run_shot_rejects_under_three_frames():
    pytest.importorskip("pycolmap")
    with pytest.raises(RuntimeError, match="at least 3 frames"):
        PyColmapCameraPass().run_shot(_Reader(2), (1, 2))


def test_unsupported_camera_model_is_rejected():
    pytest.importorskip("pycolmap")
    with pytest.raises(ValueError, match="camera_model"):
        PyColmapCameraPass({"camera_model": "OPENCV_FISHEYE"}).run_shot(_Reader(3), (1, 3))


def test_declared_contract():
    assert PyColmapCameraPass.license.commercial_use is True
    assert PyColmapCameraPass.license.spdx == "BSD-3-Clause"
    assert PyColmapCameraPass.temporal_mode.value == "video_clip"
    assert PyColmapCameraPass.pass_type.value == "camera"
    assert PyColmapCameraPass.produces_channels == []
    assert {s.name for s in PyColmapCameraPass.produces_sidecars} == {
        "camera",
        "camera_nk",
        "camera_points",
    }
    # SAM 3 is a SOFT dependency: the camera runs with or without it.
    assert PyColmapCameraPass.requires_artifacts == []
    assert PyColmapCameraPass.optional_artifacts == ["sam3_hard_masks"]
    assert mod._SUPPORTED_MODELS == ("SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL")
