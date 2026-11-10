"""The point cloud made for looking at, and the USD that carries the camera.

Synthetic scene: a wall 5 units in front of a camera that slides sideways.
The depth pass says 2.5 "metres" for it, so the solve is 2 units per metre:
the dense cloud has to land on the wall, not at 2.5."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from live_action_aov.io.camera_schema import (
    BackendInfo,
    CameraSolve,
    Extrinsic,
    Intrinsics,
    ScenePoint,
    ShotRef,
)
from live_action_aov.io.writers.display_cloud import build_display_cloud, write_cloud_obj
from live_action_aov.io.writers.nuke_comp import CompRead, render_comp_script
from live_action_aov.io.writers.usd import render_usd

W, H, F = 320, 180, 300.0
WALL = -5.0  # world z of the wall; the cameras look down -Z


def _solve(n_frames: int = 12, n_points: int = 400) -> CameraSolve:
    rng = np.random.default_rng(1)
    pts = [
        ScenePoint(
            xyz=(float(x), float(y), WALL),
            color_rgb=(200, 100, 50),
            reprojection_error_px=0.5,
            track_length=6,
        )
        for x, y in zip(
            rng.uniform(-2.2, 3.2, n_points), rng.uniform(-1.3, 1.3, n_points), strict=True
        )
    ]
    # One wild point far behind the wall with a bad error: an outlier.
    pts.append(
        ScenePoint(
            xyz=(0.0, 0.0, -400.0), color_rgb=(255, 0, 0), reprojection_error_px=9.0, track_length=2
        )
    )
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ext = [
        Extrinsic(
            frame=1001 + i,
            translation=(i * 0.1, 0.0, 0.0),
            rotation_euler_zxy_deg=(0.0, 0.0, 0.0),
            rotation_matrix=eye,
        )
        for i in range(n_frames)
    ]
    return CameraSolve(
        shot=ShotRef(name="SH-010", frame_range=(1001, 1000 + n_frames), resolution=(W, H)),
        backend=BackendInfo(name="test", version="0"),
        intrinsics=Intrinsics(fx_px=F, fy_px=F, cx_px=W / 2, cy_px=H / 2, width_px=W, height_px=H),
        extrinsics=ext,
        points=pts,
    )


def _plate(_frame: int) -> np.ndarray:
    img = np.zeros((H // 2, W // 2, 3), np.float32)  # a proxy: half the plate size
    img[..., 0] = 0.10
    img[..., 1] = 0.05
    return img


def _passes(with_person: bool = True):  # type: ignore[no-untyped-def]
    def read(_frame: int) -> dict[str, np.ndarray]:
        out = {"Z": np.full((H, W), 2.5, np.float32)}
        if with_person:
            person = np.zeros((H, W), np.float32)
            person[:, : W // 4] = 1.0
            out["mask.person"] = person
        return out

    return read


def test_dense_cloud_lands_on_the_wall_in_solve_units() -> None:
    cloud = build_display_cloud(_solve(), _plate, _passes(), max_points=4000)
    assert cloud is not None and cloud.dense and "dense" in cloud.note
    assert 1500 < len(cloud) <= 4000
    # 2.5 "metres" of depth x 2 units per metre = the wall at z = -5.
    assert np.allclose(cloud.xyz[:, 2], WALL, atol=1e-3)
    # The outlier far behind the wall did not pull the scale or enter the cloud.
    assert cloud.xyz[:, 2].min() > WALL - 0.01
    # Colours are the plate's, lifted because the plate is dark (0.10 -> brighter, hue kept).
    assert cloud.rgb[:, 0].min() > 0.3 and np.allclose(
        cloud.rgb[:, 0] / cloud.rgb[:, 1], 2.0, atol=0.01
    )
    assert cloud.rgb.max() <= 1.0 and (cloud.size > 0).all()


def test_tracked_objects_are_left_out_and_a_surface_is_drawn_once() -> None:
    solve = _solve()
    with_person = build_display_cloud(solve, _plate, _passes(True), max_points=100000)
    without = build_display_cloud(solve, _plate, _passes(False), max_points=100000)
    one_key = build_display_cloud(solve, _plate, _passes(False), max_points=100000, keyframes=1)
    assert with_person is not None and without is not None and one_key is not None
    # The person covers the left quarter of every frame: fewer points with it.
    assert len(with_person) < 0.9 * len(without)
    # Six keyframes add the strip the slide reveals, not six copies of the wall.
    assert len(one_key) < len(without) < 1.5 * len(one_key)


def test_without_depth_the_cloud_is_the_feature_points_without_outliers() -> None:
    solve = _solve()
    cloud = build_display_cloud(solve, _plate, None)
    assert cloud is not None and not cloud.dense
    # The outlier is gone, with at most the far 2% of the real points.
    assert 0.97 * 400 <= len(cloud) <= 400
    assert np.allclose(cloud.xyz[:, 2], WALL)
    # No readers at all: the solve's own colours.
    bare = build_display_cloud(solve)
    assert bare is not None and np.allclose(
        bare.rgb[0] / bare.rgb[0, 0], [1.0, 0.5, 0.25], atol=0.01
    )


def test_obj_is_one_quad_per_point_with_an_atlas(tmp_path: Path) -> None:
    import OpenImageIO as oiio

    cloud = build_display_cloud(_solve(), _plate, _passes(), max_points=900)
    assert cloud is not None
    obj, png = write_cloud_obj(cloud, tmp_path / "sh.cloud.obj", np.zeros(3))
    lines = obj.read_text().splitlines()
    n = len(cloud)
    assert sum(ln.startswith("v ") for ln in lines) == 4 * n
    assert sum(ln.startswith("vt ") for ln in lines) == n
    faces = [ln for ln in lines if ln.startswith("f ")]
    assert len(faces) == n and faces[0] == "f 1/1 2/1 3/1 4/1"
    # Every UV sits inside the atlas, and the atlas holds at least one texel per point.
    uv = np.array([[float(v) for v in ln.split()[1:]] for ln in lines if ln.startswith("vt ")])
    assert uv.min() > 0.0 and uv.max() < 1.0
    spec = oiio.ImageInput.open(str(png)).spec()
    assert spec.width == spec.height and spec.width**2 >= n and png.name == "sh.cloud.png"
    # The quads are small: a few thousandths of the distance to the wall.
    v = np.array([[float(x) for x in ln.split()[1:]] for ln in lines if ln.startswith("v ")])
    assert 0 < np.ptp(v[:4], axis=0).max() < 0.5


def test_usd_carries_the_camera_and_the_cloud() -> None:
    solve = _solve(n_frames=3)
    cloud = build_display_cloud(solve, _plate, _passes(), max_points=500)
    text = render_usd(solve, cloud, fps=25)
    assert text.startswith("#usda 1.0") and 'upAxis = "Y"' in text
    assert "startTimeCode = 1001" in text and "endTimeCode = 1003" in text
    assert "timeCodesPerSecond = 25" in text
    assert 'def Camera "SH_010_camera"' in text and 'def Points "SH_010_points"' in text
    # One matrix per solved frame; row vectors: the translation is the last row.
    samples = re.findall(r"^\s+(\d+): \( (.*) \),$", text, re.M)
    assert [int(f) for f, _ in samples] == [1001, 1002, 1003]
    last_row = [float(v) for v in re.findall(r"\(([^()]*)\)", samples[2][1])[3].split(",")]
    assert last_row == [0.2, 0.0, 0.0, 1.0]
    # The lens is the Nuke camera's: fx * film back / width.
    assert f"float focalLength = {F * 24.892 / W:.6f}" in text
    assert "float horizontalAperture = 24.892000" in text
    assert cloud is not None
    points = re.search(r"point3f\[\] points = \[(.*)\]", text)
    assert points is not None and points.group(1).count("(") == len(cloud)
    assert 'interpolation = "vertex"' in text
    # Without a cloud: the solve's own points.
    assert render_usd(solve).count("def Points") == 1


def test_comp_script_reads_the_cloud_with_its_colours(tmp_path: Path) -> None:
    cam = tmp_path / "sh.camera.nk"
    cam.write_text("# header\n\nCamera2 {\n inputs 0\n rot_order ZXY\n name Cam\n}\n")
    s = render_comp_script(
        shot_name="sh",
        plate_pattern="p.####.exr",
        frame_range=(1, 2),
        resolution=(64, 32),
        pixel_aspect=1.0,
        reads=[
            CompRead(group="all passes", rel_pattern="passes/sh.utility.####.exr", channels=["Z"])
        ],
        camera_nk=cam,
        points_obj_rel="camera/sh.cloud.obj",
        points_tex_rel="camera/sh.cloud.png",
    )
    assert "camera/sh.cloud.png" in s and "name PointCloud_colours" in s
    geo = s[s.index("ReadGeo2 {") :]
    assert "inputs 0" not in geo[: geo.index("}")]  # the atlas feeds the geometry
    assert "look at it in the 3D viewer" in s
    assert s.count("{") == s.count("}")
