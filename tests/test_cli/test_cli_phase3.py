"""Phase 3 CLI surface: `matte` compound alias, `--matte-detector`,
`--refiner`, license gate (SAM3 + RVM commercial-safe, MatAnyone 2 NC)."""

from __future__ import annotations

from typer.testing import CliRunner

from live_action_aov.cli.app import _resolve_semantic_passes, app

runner = CliRunner()


def test_matte_alias_expands_to_detector_plus_refiner() -> None:
    """`matte` is a compound alias — expands to two real passes in one go."""
    out = _resolve_semantic_passes(
        ["flow", "matte"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
    )
    # Detector first (deterministic expansion order), refiner second. The
    # DAG re-sorts, but the resolver's output order is stable here.
    assert out == ["flow", "sam3_matte", "rvm_refiner"]


def test_matte_alias_respects_detector_and_refiner_choice() -> None:
    out = _resolve_semantic_passes(
        ["matte"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
        matte_detector="sam3_matte",
        refiner="matanyone2",
    )
    assert out == ["sam3_matte", "matanyone2"]


def test_matte_alias_deduplicates_with_explicit_passes() -> None:
    """User types `matte,sam3_matte` — the explicit one must not duplicate."""
    out = _resolve_semantic_passes(
        ["matte", "sam3_matte"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
    )
    assert out == ["sam3_matte", "rvm_refiner"]


def test_plugins_list_includes_matte_backends() -> None:
    """sam3_matte + rvm_refiner discovered via entry points."""
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    assert "sam3_matte" in result.stdout
    assert "rvm_refiner" in result.stdout


def test_rvm_is_marked_commercial_safe_in_plugins_list() -> None:
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    # Find the line with rvm_refiner and assert "yes" (commercial_use=True).
    rvm_line = next(
        (line for line in result.stdout.splitlines() if "rvm_refiner" in line),
        "",
    )
    assert rvm_line, "rvm_refiner not listed"
    assert "MIT" in rvm_line
    assert "yes" in rvm_line


def test_sam3_is_marked_commercial_safe_in_plugins_list() -> None:
    """SAM 3's license has a military/ITAR carve-out but is commercial=true."""
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    sam3_line = next(
        (line for line in result.stdout.splitlines() if "sam3_matte" in line),
        "",
    )
    assert sam3_line, "sam3_matte not listed"
    assert "SAM" in sam3_line.upper()
    assert "yes" in sam3_line


# ---------------------------------------------------------------------------
# Round 2: MatAnyone 2 — NC license gate + plugins-list visibility
# ---------------------------------------------------------------------------


def test_matanyone2_listed_as_noncommercial() -> None:
    result = runner.invoke(app, ["plugins", "list"])
    assert result.exit_code == 0
    line = next(
        (ln for ln in result.stdout.splitlines() if "matanyone2" in ln),
        "",
    )
    assert line, "matanyone2 not listed"
    assert "NTU" in line.upper()
    # commercial column is "no"
    assert " no " in line or line.rstrip().endswith(" no")


def test_matte_alias_with_matanyone2_refiner_respected() -> None:
    """--refiner matanyone2 flows through the `matte` alias expansion."""
    out = _resolve_semantic_passes(
        ["matte"],
        depth_backend="depth_anything_v2",
        normals_backend="dsine",
        refiner="matanyone2",
    )
    assert out == ["sam3_matte", "matanyone2"]


def test_matanyone2_license_gate_blocks_without_opt_in(test_plate_1080p) -> None:
    """Phase 3 round 2 exit criterion: NC refiner must not run without
    --allow-noncommercial, even when the detector is commercial-safe."""
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "matte",
            "--refiner", "matanyone2",
        ],
    )
    assert result.exit_code == 2
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "matanyone2" in combined.lower()


def test_matanyone2_gate_clears_with_opt_in(test_plate_1080p) -> None:
    """With --allow-noncommercial the gate opens; runtime error comes from
    the stub NotImplementedError (exit 1), NOT the gate (exit 2)."""
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "matte",
            "--refiner", "matanyone2",
            "--allow-noncommercial",
        ],
    )
    assert result.exit_code != 2
