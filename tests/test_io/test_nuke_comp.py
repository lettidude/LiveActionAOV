"""Comp starter script (`<shot>.comp.nk`) generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_action_aov.io.writers.nuke_comp import (
    CompRead,
    nuke_channel,
    pass_group,
    render_comp_script,
)


@pytest.mark.parametrize(
    ("ours", "nuke"),
    [
        # Verified against Nuke 16 reading our own EXR.
        ("Z", "depth.Z"),
        ("Z_raw", "other.Z_raw"),
        ("depth.confidence", "depth_extra.confidence"),
        ("mask.person", "mask_extra.person"),
        ("mask_vitmatte.person", "mask_vitmatte.person"),
        ("matte.r", "matte.red"),
        ("splat.A", "splat.alpha"),
        ("splat.Z", "splat.Z"),
        ("motion.x", "motion_extra.x"),
        ("forward.u", "forward.u"),
        ("N.x", "N.x"),
        ("ao.a", "ao.alpha"),
    ],
)
def test_nuke_channel_names(ours: str, nuke: str) -> None:
    assert nuke_channel(ours) == nuke


def test_pass_groups() -> None:
    assert pass_group("Z") == "depth"
    assert pass_group("mask.car") == "matte"
    assert pass_group("matte_vitmatte.r") == "matte"
    assert pass_group("forward.u") == "flow"
    assert pass_group("splat.R") == "splat"
    assert pass_group("CryptoObject00.r") == "cryptomatte"


def _script(tmp_path: Path, **kw: object) -> str:
    chans = ["mask.person", "mask.car", "splat.R", "splat.G", "splat.B", "splat.A", "splat.Z", "Z"]
    return render_comp_script(
        shot_name="sh-010",
        plate_pattern="A:/plates/sh-010/sh-010.####.exr",
        frame_range=(1001, 1010),
        resolution=(2048, 1080),
        pixel_aspect=1.0,
        reads=[CompRead(group="all passes", rel_pattern="sh-010.utility.####.exr", channels=chans)],
        **kw,  # type: ignore[arg-type]
    )


def test_script_declares_layers_and_wires_every_pass(tmp_path: Path) -> None:
    s = _script(tmp_path)
    # Custom layers declared up front (else Shuffle2 mappings load black).
    assert "add_layer {mask_extra mask_extra.person mask_extra.car}" in s
    assert "add_layer {splat splat.red splat.green splat.blue splat.alpha splat.Z}" in s
    assert "depth" not in [ln.split()[1] for ln in s.splitlines() if ln.startswith("add_layer")]
    # Relative to the script, data read raw.
    assert 'file "[file dirname [value root.name]]/sh-010.utility.####.exr"' in s
    assert "raw true" in s
    # One Shuffle per mask, the splat clean plate, depth.
    for name in ("mask_extra_person", "mask_extra_car", "splat_splat", "depth_depth"):
        assert f"name {name}" in s, name
    # Union of masks + the clean-plate example.
    assert 'expr3 "max(mask_extra.person, mask_extra.car)"' in s
    assert "Keymix {" in s and "inputs 3" in s
    assert s.count("{") == s.count("}")


def test_camera_and_points_embedded(tmp_path: Path) -> None:
    cam = tmp_path / "sh.camera.nk"
    cam.write_text("# header\n\nCamera2 {\n inputs 0\n rot_order ZXY\n name Cam\n}\n")
    s = _script(tmp_path, camera_nk=cam, points_obj_rel="camera_track/sh.points.obj")
    assert "Camera2 {" in s and "# header" not in s
    assert "ReadGeo2 {" in s and "camera_track/sh.points.obj" in s
    assert "ScanlineRender {" in s
    assert s.count("{") == s.count("}")


def test_no_example_without_splat(tmp_path: Path) -> None:
    s = render_comp_script(
        shot_name="sh",
        plate_pattern="p.####.exr",
        frame_range=(1, 2),
        resolution=(64, 32),
        pixel_aspect=1.0,
        reads=[
            CompRead(group="matte", rel_pattern="matte/sh.matte.####.exr", channels=["mask.a1"])
        ],
    )
    assert "Keymix" not in s
    assert "matte/sh.matte.####.exr" in s
