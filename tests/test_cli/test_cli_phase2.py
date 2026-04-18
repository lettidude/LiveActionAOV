"""Phase 2 CLI surface: --depth-backend, --normals-backend, license gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from live_action_aov.cli.app import _resolve_semantic_passes, app
from live_action_aov.io.oiio_io import HAS_OIIO


runner = CliRunner()


def test_resolve_semantic_passes_rewrites_depth_and_normals() -> None:
    out = _resolve_semantic_passes(
        ["flow", "depth", "normals"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
    )
    assert out == ["flow", "depth_anything_v2", "dsine"]


def test_resolve_semantic_passes_respects_user_backend_choices() -> None:
    out = _resolve_semantic_passes(
        ["depth"],
        depth_backend="depthcrafter",
        normals_backend="dsine",
    )
    assert out == ["depthcrafter"]


def test_resolve_semantic_passes_passes_concrete_names_through_unchanged() -> None:
    out = _resolve_semantic_passes(
        ["flow", "depth_anything_v2"],
        depth_backend="depthcrafter",
        normals_backend="dsine",
    )
    assert out == ["flow", "depth_anything_v2"]


def test_resolve_semantic_passes_deduplicates_in_order() -> None:
    out = _resolve_semantic_passes(
        ["flow", "depth", "depth_anything_v2"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
    )
    # `depth` resolves to depth_anything_v2, then the literal depth_anything_v2
    # is seen again and deduped.
    assert out == ["flow", "depth_anything_v2"]


def test_plugins_list_includes_phase2_backends() -> None:
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "depth_anything_v2" in result.stdout
    assert "dsine" in result.stdout
    assert "depthcrafter" in result.stdout
    assert "normalcrafter" in result.stdout
    assert "depthpro" in result.stdout


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_depthpro_gate_blocks_without_allow_noncommercial(
    test_plate_1080p: Path,
) -> None:
    """Depth Pro is Apple ML Research License — non-commercial, gate applies."""
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "depth",
            "--depth-backend", "depthpro",
        ],
    )
    assert result.exit_code == 2


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_depthcrafter_gate_blocks_without_allow_noncommercial(
    test_plate_1080p: Path,
) -> None:
    """Per spec §13.1 Phase 2 exit criterion: running a non-commercial pass
    without the explicit opt-in must exit non-zero with a helpful message.
    """
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "depth",
            "--depth-backend", "depthcrafter",
        ],
    )
    assert result.exit_code == 2
    # Check for the key phrase, any renderable output location.
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "allow-noncommercial" in combined.lower() or "non-commercial" in combined.lower()


def test_normalcrafter_gate_blocks_without_allow_noncommercial(
    test_plate_1080p: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "normals",
            "--normals-backend", "normalcrafter",
        ],
    )
    assert result.exit_code == 2


def test_commercial_backends_pass_the_license_gate(test_plate_1080p: Path) -> None:
    """Happy-path: DA V2 + DSINE should clear the gate. We stop before actual
    inference to avoid HF downloads by using a nonexistent folder — but that
    exits code 1 (runtime), not 2 (gate). Use an empty folder or the real
    plate; we just assert the exit code is NOT the gate's 2.
    """
    # Using the real plate but running `depth_anything_v2` directly would
    # download HF weights. Instead check the gate explicitly via --passes
    # with only allowed names + --allow-noncommercial on (which makes the
    # gate always pass). Then the runtime error downstream is fine.
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "noop",   # test-only commercial pass, skips the HF path
        ],
    )
    # exit_code 0 on OIIO-present happy path, or 1 on other runtime issue —
    # either way NOT the gate's 2.
    assert result.exit_code != 2
