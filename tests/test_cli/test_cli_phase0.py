"""CLI exit criterion: --version, plugins list, run-shot."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import live_action_aov
from live_action_aov.cli.app import app
from live_action_aov.io.oiio_io import HAS_OIIO

runner = CliRunner()


def test_version_prints() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert live_action_aov.__version__ in result.stdout


def test_plugins_list_includes_noop() -> None:
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "noop" in result.stdout


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_run_shot_produces_sidecar(test_plate_1080p: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot",
            str(test_plate_1080p),
            "--out",
            str(test_plate_1080p.parent / "out"),
            "--passes",
            "noop",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Sidecar file should have been written in the same folder.
    written = sorted(
        (
            sorted(test_plate_1080p.parent.joinpath("out", test_plate_1080p.name).glob("v*"))[-1]
            / "passes"
        ).glob("*.utility.*.exr")
    )
    assert len(written) == 5


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_run_shot_without_out_writes_next_to_the_plate(test_plate_1080p: Path) -> None:
    result = runner.invoke(app, ["run-shot", str(test_plate_1080p), "--passes", "noop"])
    assert result.exit_code == 0, result.stdout
    sibling = test_plate_1080p.parent / f"{test_plate_1080p.name}_utility"
    written = sorted((sibling / "v001" / "passes").glob("*.utility.*.exr"))
    assert len(written) == 5
    assert not list(test_plate_1080p.glob("*.utility.*"))  # plate folder stays clean
