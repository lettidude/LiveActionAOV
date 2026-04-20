"""End-to-end matte integration with real SAM 3 + RVM weights.

**@slow @gpu** — marked so it's deselected by default. **Do not skip this
test at a release cut** (Phase 3 brainstorm decision #7). It's the only
coverage that confirms the real model downloads + HF/torch-hub loaders
actually work end-to-end on a GPU; fake-model unit tests shield the
contract but cannot catch upstream API drift.

What it does:
1. Generates a small synthetic plate (10 frames @ 256×256).
2. Runs the `flow,matte` CLI pipeline (→ sam3_matte + rvm_refiner) with
   the commercial-safe default combination. No `--allow-noncommercial`
   needed.
3. Reads back one sidecar EXR and asserts the *structural* invariants:
   - at least one `mask.<concept>` channel
   - all four `matte.r/g/b/a` channels present
   - `liveaov/matte/commercial = "true"`
   - detector + refiner identity stamped
4. The actual mask *pixel quality* is unverified — we trust upstream
   model QC for that; this test only proves the plumbing is intact.

Why not also run MatAnyone 2? It's non-commercial and requires the
`--allow-noncommercial` opt-in. A release cut is presumed to be for a
commercial build; running the gated refiner here would muddy the
invariant "default pipeline is commercial-safe end-to-end."
A dedicated MatAnyone integration test can be added later behind a
separate marker if/when a commercial mirror ships.

Running locally::

    pytest tests/test_passes/test_matte_integration_slow.py -m "slow and gpu" -v

Expect: ~1-2 GB of weights downloaded on first run, 30-120 s on a
modern GPU, much longer on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from live_action_aov.cli.app import app  # noqa: E402
from live_action_aov.io.channels import (  # noqa: E402
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
    MASK_PREFIX,
)
from live_action_aov.io.oiio_io import HAS_OIIO  # noqa: E402

pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Real-weights smoke needs a GPU"),
    pytest.mark.skipif(
        os.environ.get("LIVEAOV_SKIP_MATTE_INTEGRATION") == "1",
        reason="LIVEAOV_SKIP_MATTE_INTEGRATION=1 set (CI opt-out); release cuts MUST NOT set this",
    ),
]


runner = CliRunner()


@pytest.fixture(scope="module")
def _integration_plate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate a tiny plate with a salient subject — a moving white disk
    on a mid-grey background. SAM 3's concept detector won't semantically
    label it as `person`, but it should pick it up under `object` /
    `animal` / any catch-all concept; the test only cares that *some*
    mask layer lands. We use the shared fixture generator but at a
    reduced resolution for speed."""
    from tests.fixtures.generate_test_plate import generate

    folder = tmp_path_factory.mktemp("matte_integration_plate")
    generate(folder, frame_count=10, width=256, height=256)
    return folder


def test_real_sam3_plus_rvm_end_to_end(_integration_plate: Path) -> None:
    """The full commercial-safe matte pipeline runs and produces valid
    sidecar EXRs. Pixel-level correctness is upstream's job; we verify
    plumbing only."""
    # `matte` alias expands to sam3_matte + rvm_refiner; both commercial-safe.
    result = runner.invoke(
        app,
        [
            "run-shot",
            str(_integration_plate),
            "--passes",
            "flow,matte",
        ],
    )
    # Any non-zero exit is a plumbing failure we want to catch at release cut.
    assert result.exit_code == 0, f"Pipeline failed:\nstdout: {result.stdout}"

    from live_action_aov.io.oiio_io import read_exr

    sidecars = sorted(_integration_plate.glob("*.utility.*.exr"))
    assert sidecars, "No sidecars written"

    # Read every sidecar up front — RVM's recurrent state is cold on the
    # first frame (alpha ~= 0 until r1..r4 warm up), so any single-frame
    # assertion on matte content would be misleading. The checks below
    # that care about *structure* (channel names, metadata, [0,1] bounds)
    # use the first sidecar; the "refiner produced something" check looks
    # across the whole clip.
    all_reads = [read_exr(s) for s in sidecars]
    pixels, attrs = all_reads[0]

    # Matte channels must all land — RVM produces all four slots (zeros for
    # slots without a hero).
    names = list(attrs["channelnames"])
    for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
        assert ch in names, f"{ch} missing — refiner didn't produce all slots"

    # At least one dynamic `mask.<concept>` channel must appear. (If SAM 3
    # genuinely finds nothing in the test plate, widen the concept list
    # via `params` rather than relaxing this assertion — a silent no-ops
    # detector is exactly the bug this test guards against.)
    dynamic_masks = [n for n in names if n.startswith(MASK_PREFIX)]
    assert dynamic_masks, (
        f"No mask.<concept> channels found. Channels: {names}. "
        f"Either SAM 3 regressed or the plate has no salient subjects."
    )

    # Metadata: commercial-safe default, both identities stamped.
    assert attrs.get("liveaov/matte/commercial") == "true"
    assert attrs.get("liveaov/matte/detector") == "sam3_matte"
    assert attrs.get("liveaov/matte/refiner") == "rvm_refiner"
    # Concept list is non-empty (mirrors the mask.* channel assertion).
    concepts = attrs.get("liveaov/matte/concepts", "")
    assert concepts, "matte/concepts attr is empty but mask.* channels exist"

    # Sanity: matte channels are within [0, 1] on every frame.
    for frame_pixels, frame_attrs in all_reads:
        frame_names = list(frame_attrs["channelnames"])
        for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
            idx = frame_names.index(ch)
            band = frame_pixels[..., idx]
            assert band.min() >= -1e-6 and band.max() <= 1.0 + 1e-6, (
                f"{ch} outside [0,1]: min={band.min()} max={band.max()}"
            )

    # Plate is non-trivial: at least one matte slot has nonzero pixels
    # *somewhere in the clip*. Checking only frame 1 would false-fail on
    # the RVM recurrent-state cold-start (alpha ~= 0 until r1..r4 settle).
    clip_matte_max = 0.0
    for frame_pixels, frame_attrs in all_reads:
        frame_names = list(frame_attrs["channelnames"])
        for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
            idx = frame_names.index(ch)
            clip_matte_max = max(clip_matte_max, float(frame_pixels[..., idx].max()))
    assert clip_matte_max > 0.0, (
        "All matte channels are zero across the whole clip — detector "
        "found nothing OR refiner swallowed everything. Inspect the "
        "sam3_hard_masks artifact."
    )
