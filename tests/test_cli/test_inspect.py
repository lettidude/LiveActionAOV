"""`liveaov inspect <sidecar.exr>` — structural QC command.

Three tests matching the plan (plan file: dynamic-honking-emerson):

1. **Happy path** — synthesize a sidecar via `ExrSidecarWriter` with
   known channels + matte metadata, invoke the CLI, assert the text
   report mentions each expected channel name, the commercial flag,
   and the hero line.

2. **`--json` schema** — same sidecar, `--json` mode, parse the output
   with `json.loads`, assert the pinned top-level key set. Batch
   runners will bind against this contract.

3. **Not-an-EXR input** — point the command at a plain text file,
   confirm exit 1 and a clean one-line error on stderr (no traceback).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from live_action_aov.cli.app import app
from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.io.oiio_io import HAS_OIIO
from live_action_aov.io.writers.exr import ExrSidecarWriter

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")

# `CliRunner(mix_stderr=False)` — keep stdout/stderr separated so the
# "error goes to stderr, not a traceback on stdout" assertion in test 3
# is checkable without parsing terminal colour codes.
runner = CliRunner()


def _write_fake_sidecar(path: Path) -> None:
    """Write a sidecar whose shape mirrors a real `flow,matte` run: one
    dynamic `mask.person` channel, all four matte slots (r populated, b/g/a
    zero), plus the full `liveActionAOV/matte/*` metadata block."""
    h, w = 8, 12
    # matte.r has real content; the other three slots are zero (mirrors
    # what RVM emits when there's one hero in slot r).
    matte_r = np.full((h, w), 0.4, dtype=np.float32)
    zeros = np.zeros((h, w), dtype=np.float32)
    mask_person = np.full((h, w), 0.6, dtype=np.float32)

    channels = {
        "mask.person": mask_person,
        CH_MATTE_R: matte_r,
        CH_MATTE_G: zeros,
        CH_MATTE_B: zeros,
        CH_MATTE_A: zeros,
    }
    attrs = {
        "liveActionAOV/matte/commercial": "true",
        "liveActionAOV/matte/detector": "sam3_matte",
        "liveActionAOV/matte/refiner": "rvm_refiner",
        "liveActionAOV/matte/concepts": "person",
        "liveActionAOV/matte/hero_r/label": "person",
        "liveActionAOV/matte/hero_r/track_id": 3,
        "liveActionAOV/matte/hero_r/score": 0.91,
    }
    ExrSidecarWriter().write_frame(path, channels, attrs=attrs)


# ---------------------------------------------------------------------------
# Test 1 — happy path text output
# ---------------------------------------------------------------------------


def test_inspect_text_output_contains_expected_lines(tmp_path: Path) -> None:
    sidecar = tmp_path / "plate.utility.0010.exr"
    _write_fake_sidecar(sidecar)

    result = runner.invoke(app, ["inspect", str(sidecar)])
    assert result.exit_code == 0, result.stdout

    stdout = result.stdout

    # Header — resolution + channel count.
    assert "Resolution: 12 x 8" in stdout
    assert "Channels (5)" in stdout

    # Every channel we wrote must appear in the stat block.
    for expected_ch in ("mask.person", "matte.r", "matte.g", "matte.b", "matte.a"):
        assert expected_ch in stdout, (
            f"Channel {expected_ch!r} missing from inspect output:\n{stdout}"
        )

    # Bucket labels are rendered.
    assert "matte:" in stdout
    assert "mask:" in stdout

    # Metadata block — the commercial flag is the highest-value attribute
    # to confirm the writer stamped correctly. Quoted because strings are
    # shown quoted, ints/floats are not. We regex past the column-alignment
    # whitespace so formatting tweaks don't break this assertion.
    import re

    assert re.search(r'matte/commercial\s+= "true"', stdout), (
        f"commercial flag not rendered correctly:\n{stdout}"
    )
    assert "matte/detector" in stdout
    assert "matte/refiner" in stdout

    # Heroes block — slot r is populated, the others explicitly (empty).
    assert "Heroes (1 of 4 slots filled)" in stdout
    assert "matte.r = person (track 3, score 0.91)" in stdout
    assert "matte.g = (empty)" in stdout
    assert "matte.a = (empty)" in stdout


# ---------------------------------------------------------------------------
# Test 2 — JSON schema lock
# ---------------------------------------------------------------------------


def test_inspect_json_has_pinned_top_level_schema(tmp_path: Path) -> None:
    """Mirrors the CorridorKey schema-lock pattern from Phase 3 Round 2:
    any silent rename/drop in `format_json` breaks this test before it
    breaks a downstream batch runner. Additive changes (new keys) are
    fine and shouldn't require updating this test."""
    sidecar = tmp_path / "plate.utility.0010.exr"
    _write_fake_sidecar(sidecar)

    result = runner.invoke(app, ["inspect", str(sidecar), "--json"])
    assert result.exit_code == 0, result.stdout

    doc = json.loads(result.stdout)
    # Top-level pinned keys.
    assert set(doc.keys()) >= {"file", "resolution", "channels", "metadata", "heroes"}
    assert doc["resolution"] == {"width": 12, "height": 8}

    # channels: list of dicts with per-channel stats + bucket label.
    assert isinstance(doc["channels"], list)
    assert len(doc["channels"]) == 5
    sample = doc["channels"][0]
    for k in ("name", "bucket", "min", "max", "mean", "out_of_range"):
        assert k in sample, f"channels[0] missing key {k!r}"
    # Buckets fall in the expected set.
    buckets = {c["bucket"] for c in doc["channels"]}
    assert buckets <= {"canonical", "matte", "mask", "other"}

    # metadata: the full liveActionAOV/* slice, flat dict keyed by attr name.
    assert doc["metadata"]["liveActionAOV/matte/commercial"] == "true"
    assert int(doc["metadata"]["liveActionAOV/matte/hero_r/track_id"]) == 3

    # heroes: four-slot list, r populated and the others empty.
    heroes_by_slot = {h["slot"]: h for h in doc["heroes"]}
    assert set(heroes_by_slot) == {"r", "g", "b", "a"}
    assert heroes_by_slot["r"]["label"] == "person"
    assert heroes_by_slot["r"]["track_id"] == 3
    assert heroes_by_slot["r"]["score"] == pytest.approx(0.91)
    assert heroes_by_slot["r"]["empty"] is False
    for slot in ("g", "b", "a"):
        assert heroes_by_slot[slot]["empty"] is True
        assert heroes_by_slot[slot]["label"] is None


# ---------------------------------------------------------------------------
# Test 3 — unreadable input produces a clean error
# ---------------------------------------------------------------------------


def test_inspect_non_exr_input_exits_one_no_traceback(tmp_path: Path) -> None:
    bogus = tmp_path / "not_an_exr.exr"
    bogus.write_text("this is not an EXR file at all", encoding="utf-8")

    result = runner.invoke(app, ["inspect", str(bogus)])
    assert result.exit_code == 1, (
        f"expected exit 1 for non-EXR input, got {result.exit_code}\nstdout: {result.stdout!r}"
    )

    # Message should mention the file and be a single-ish line, not a
    # multiline Python traceback dumped from `read_exr`.
    combined = (result.stdout or "") + (getattr(result, "stderr", "") or "")
    assert "inspect:" in combined
    assert str(bogus) in combined or bogus.name in combined
    # A bare `Traceback` string would indicate the exception leaked
    # through — exactly the regression we're guarding against.
    assert "Traceback" not in combined


# ---------------------------------------------------------------------------
# Test 4 — non-matte sidecar skips the Heroes block
# ---------------------------------------------------------------------------


def test_inspect_flow_only_sidecar_omits_heroes_block(tmp_path: Path) -> None:
    """A flow-only sidecar has no matte metadata. The Heroes block
    shouldn't appear at all (vs. four `(empty)` rows) — keeps the output
    focused on what the pipeline actually produced."""
    h, w = 8, 12
    from live_action_aov.io.channels import (
        CH_FLOW_CONFIDENCE,
        CH_MOTION_X,
        CH_MOTION_Y,
    )

    channels = {
        CH_MOTION_X: np.zeros((h, w), dtype=np.float32),
        CH_MOTION_Y: np.zeros((h, w), dtype=np.float32),
        CH_FLOW_CONFIDENCE: np.ones((h, w), dtype=np.float32),
    }
    sidecar = tmp_path / "flow_only.utility.0001.exr"
    ExrSidecarWriter().write_frame(sidecar, channels)

    result = runner.invoke(app, ["inspect", str(sidecar)])
    assert result.exit_code == 0, result.stdout

    # No matte metadata → no Heroes block and no matte channels listed.
    assert "Heroes" not in result.stdout
    assert "matte.r" not in result.stdout
    # But the canonical flow channels must show up.
    assert "motion.x" in result.stdout
    assert "flow.confidence" in result.stdout
