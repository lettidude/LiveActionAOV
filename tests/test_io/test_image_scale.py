"""Scale sidecar for outputs that are not at the passes' size."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from live_action_aov.io.image_scale import (
    SCHEMA,
    read_scale_sidecar,
    scale_to_passes,
    sidecar_path,
    write_scale_sidecar,
)


def test_round_trip_folder_and_file(tmp_path: Path) -> None:
    for target in (tmp_path / "model", tmp_path / "frames" / "f.####.jpg"):
        target.parent.mkdir(parents=True, exist_ok=True)
        path = write_scale_sidecar(target, (1024, 540), (2048, 1080), what="frames")
        assert path == str(target) + ".scale.json"
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        assert doc["schema"] == SCHEMA and doc["scale_to_passes"] == [2.0, 2.0]
        assert scale_to_passes(target) == (2.0, 2.0)


def test_trailing_slash_is_the_same_folder(tmp_path: Path) -> None:
    write_scale_sidecar(tmp_path / "m", (10, 10), (30, 20))
    assert scale_to_passes(str(tmp_path / "m") + "/") == (3.0, 2.0)
    assert sidecar_path(str(tmp_path / "m") + "\\") == str(tmp_path / "m") + ".scale.json"


def test_sidecar_beats_the_sizes_and_foreign_json_is_ignored(tmp_path: Path) -> None:
    t = tmp_path / "x"
    write_scale_sidecar(t, (100, 50), (200, 100))
    assert scale_to_passes(t, image_size=(1, 1), passes_size=(1, 1)) == (2.0, 2.0)
    Path(sidecar_path(t)).write_text('{"schema": "other/1"}', encoding="utf-8")
    assert read_scale_sidecar(t) is None
    assert scale_to_passes(t) == (1.0, 1.0)


def test_bad_sizes_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_scale_sidecar(tmp_path / "x", (0, 10), (10, 10))
