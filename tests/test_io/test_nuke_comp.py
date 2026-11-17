"""Comp starter script (`<shot>.comp.nk`) generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_action_aov.io.writers.nuke_comp import (
    CompRead,
    MatteInfo,
    matte_engine_label,
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
    # One Shuffle per mask, the splat background, depth.
    for name in ("mask_extra_person", "mask_extra_car", "splat_splat", "depth_depth"):
        assert f"name {name}" in s, name
    # Union of masks + the splat-fill example.
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


# --- which engine made each matte -------------------------------------------

_MATTE_CHANS = ["mask.person", "mask.car", "matte.r", "matte.g", "matte.b", "matte.a"]


def _matte_script(info: MatteInfo | None) -> str:
    return render_comp_script(
        shot_name="sh",
        plate_pattern="p.####.exr",
        frame_range=(1, 2),
        resolution=(64, 32),
        pixel_aspect=1.0,
        reads=[
            CompRead(
                group="all passes",
                rel_pattern="sh.utility.####.exr",
                channels=[*_MATTE_CHANS, "foreground.a"],
            )
        ],
        matte_info=info,
    )


def _label_of(script: str, name: str) -> str:
    lines = script.splitlines()
    i = lines.index(f" name {name}")
    return lines[i + 1].removeprefix(" label ")


@pytest.mark.parametrize(
    ("args", "label"),
    [
        (("ltx_alpha_refiner", "ltx", "whole_frame"), "LTX Alpha (whole frame)"),
        (("ltx_alpha_refiner", "ltx", "per_object"), "LTX Alpha (per object)"),
        (("ltx_alpha_refiner", "vitmatte", "whole_frame"), "ViTMatte (LTX Alpha did not run)"),
        (("vitmatte_refiner", "", ""), "ViTMatte"),
        (("matanyone2", "", ""), "MatAnyone 2"),
        (("some_future_refiner", "", ""), "some_future_refiner"),
    ],
)
def test_matte_engine_label(args: tuple[str, str, str], label: str) -> None:
    assert matte_engine_label(*args) == label


def test_matte_engine_label_partial_fallback() -> None:
    label = matte_engine_label("ltx_alpha_refiner", "ltx", "whole_frame", partial=True)
    assert label.splitlines() == [
        "LTX Alpha (whole frame)",
        "some objects: fallback, see run report",
    ]


def test_labels_say_which_engine_made_each_matte() -> None:
    s = _matte_script(MatteInfo(matte="LTX Alpha (whole frame)", detector="SAM 3.1"))
    assert 'label "<center><b>matte: LTX Alpha (whole frame)</b>"' in s
    assert " name Backdrop_0_matte" in s  # node names do not change
    assert _label_of(s, "matte_matte") == (
        r'"matte (hero RGBA)\nLTX Alpha (whole frame)\nr g b a = heroes 1-4"'
    )
    assert _label_of(s, "foreground_foreground") == (
        r'"foreground: all LTX foreground\n(owned or not)"'
    )
    assert _label_of(s, "mask_extra_person") == r'"person\nSAM 3.1 hard mask"'
    assert _label_of(s, "All_objects_matte") == r'"every mask (max)\nSAM 3.1 hard mask"'
    assert s.count("{") == s.count("}")


def test_soft_masks_name_the_matte_engine() -> None:
    info = MatteInfo(matte="LTX Alpha (whole frame)", detector="SAM 3.1", soft_masks=True)
    s = _matte_script(info)
    assert _label_of(s, "mask_extra_car") == r'"car\nsoft edge: LTX Alpha (whole frame)"'


def test_no_matte_info_keeps_the_old_labels() -> None:
    s = _matte_script(None)
    assert 'label "<center><b>matte</b>"' in s
    assert _label_of(s, "matte_matte") == '"matte (hero RGBA)"'
    assert _label_of(s, "mask_extra_person") == '"person"'
    assert _label_of(s, "All_objects_matte") == '"every mask (max)"'


def test_alpha_only_layer_fills_rgb_too() -> None:
    # foreground.a alone: RGB black in the Viewer made the matte look missing.
    s = _matte_script(None)
    lines = s.splitlines()
    i = lines.index(" name foreground_foreground")
    mappings = lines[i - 1]
    for d in ("red", "green", "blue", "alpha"):
        assert f"foreground.alpha 0 0 rgba.{d}" in mappings, d
