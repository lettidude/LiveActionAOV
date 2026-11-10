"""File naming template (`core.output_layout`): the studio's convention
for every output of a run, from one template."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_action_aov.core.job import Shot
from live_action_aov.core.output_layout import (
    DEFAULT_NAMING,
    NAMING_PRESETS,
    file_name,
    frame_template,
    sidecar_template,
    validate_naming,
)


def _shot(naming: str = "", out: str | None = "X:/r/SH010/v003") -> Shot:
    return Shot(
        name="SH010",
        folder=Path("X:/plates/SH010_plate_v002"),
        output_dir=None if out is None else Path(out),
        sequence_pattern="SH010_plate_v002.####.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
        naming=naming,
    )


def test_default_names_are_the_historic_layout() -> None:
    shot = _shot()
    assert file_name(shot, "utility", "exr", 1001) == "SH010_v003.utility.1001.exr"
    assert file_name(shot, "camera", "nk") == "SH010_v003.camera.nk"
    assert file_name(shot, "comp", "nk") == "SH010_v003.comp.nk"
    assert sidecar_template(shot) == "passes/SH010_v003.utility.{frame:04d}.exr"
    assert (
        frame_template(shot, "depth", sub="depth")
        == "passes/depth/SH010_v003.depth.{frame:04d}.exr"
    )
    assert frame_template(shot, "crypto") == "passes/SH010_v003.crypto.{frame:04d}.exr"


def test_plain_run_dir_drops_the_version() -> None:
    shot = _shot(out=None)
    assert file_name(shot, "utility", "exr", 7) == "SH010.utility.0007.exr"
    assert file_name(shot, "splat", "ply") == "SH010.splat.ply"
    shot2 = _shot("{shot}_{pass}_{version}.{frame}.{ext}", out=None)
    assert file_name(shot2, "utility", "exr", 7) == "SH010_utility.0007.exr"


@pytest.mark.parametrize(
    ("template", "frame_file", "camera_file"),
    [
        (
            "{shot}_{pass}_{version}.{frame}.{ext}",
            "SH010_utility_v003.1001.exr",
            "SH010_camera_v003.nk",
        ),
        (
            "{shot}_{version}_{pass}.{frame}.{ext}",
            "SH010_v003_utility.1001.exr",
            "SH010_v003_camera.nk",
        ),
        (
            "{plate}_{pass}_{version}.{frame}.{ext}",
            "SH010_plate_v002_utility_v003.1001.exr",
            "SH010_plate_v002_camera_v003.nk",
        ),
        ("{shot}-{frame}-{pass}.{ext}", "SH010-1001-utility.exr", "SH010-camera.nk"),
    ],
)
def test_custom_templates(template: str, frame_file: str, camera_file: str) -> None:
    shot = _shot(template)
    assert file_name(shot, "utility", "exr", 1001) == frame_file
    assert file_name(shot, "camera", "nk") == camera_file


def test_presets_are_valid_and_default_normalises_to_empty() -> None:
    for _, template in NAMING_PRESETS:
        assert validate_naming(template) == template
    assert validate_naming("") == DEFAULT_NAMING
    assert _shot(DEFAULT_NAMING).naming == ""  # stored as "the default"
    assert _shot(" {shot}_{pass}_{version}.{frame}.{ext} ").naming == (
        "{shot}_{pass}_{version}.{frame}.{ext}"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "{shot}_{version}.{frame}.{ext}",  # no {pass}
        "{shot}_{pass}.{ext}",  # no {frame}
        "{shot}_{pass}.{frame}.exr",  # no {ext}
        "{shot}/{pass}.{frame}.{ext}",  # path separator
        "{shot}_{take}.{pass}.{frame}.{ext}",  # unknown token
        "{shot_{pass}.{frame}.{ext}",  # unbalanced
    ],
)
def test_bad_templates_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        validate_naming(bad)
    with pytest.raises(ValueError):
        _shot(bad)
