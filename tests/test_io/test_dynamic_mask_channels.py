"""Dynamic `mask.<concept>` + `matte.{r,g,b,a}` channel writer round-trip.

Spec §13.1 Phase 3 brainstorm decision #6 makes explicit that the writer
must accept any `{channel_name: array}` dict *without* a pre-declared
allowlist, as long as the name follows the `mask.*` convention. The
canonical ordering (matte.r/g/b/a in position) still applies, and
dynamic channels follow in insertion order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.io.oiio_io import HAS_OIIO
from live_action_aov.io.writers.exr import ExrSidecarWriter, _order_channels

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")


def test_order_channels_places_matte_in_canonical_order() -> None:
    """matte.r/g/b/a sit in the canonical-ordered head; dynamics follow."""
    got = _order_channels(
        [
            f"{'mask.'}person",
            CH_MATTE_A,
            f"{'mask.'}vehicle",
            CH_MATTE_R,
        ]
    )
    # matte.r and matte.a both appear in CANONICAL_CHANNEL_ORDER, so they come
    # first in canonical positions; mask.* are unknowns appended in input order.
    assert got[0] == CH_MATTE_R
    assert got[1] == CH_MATTE_A
    assert got[2:] == ["mask.person", "mask.vehicle"]


def test_writer_accepts_arbitrary_mask_concept_channels(tmp_path: Path) -> None:
    """Round-trip: write three mask.<concept> + all four matte slots, read
    back, verify every channel name and pixel value survives."""
    h, w = 16, 24
    writer = ExrSidecarWriter()
    channels = {
        "mask.person": np.full((h, w), 0.10, dtype=np.float32),
        "mask.vehicle": np.full((h, w), 0.20, dtype=np.float32),
        "mask.animal": np.full((h, w), 0.30, dtype=np.float32),
        CH_MATTE_R: np.full((h, w), 0.40, dtype=np.float32),
        CH_MATTE_G: np.full((h, w), 0.50, dtype=np.float32),
        CH_MATTE_B: np.full((h, w), 0.60, dtype=np.float32),
        CH_MATTE_A: np.full((h, w), 0.70, dtype=np.float32),
    }
    out_path = tmp_path / "dynamic.exr"
    writer.write_frame(
        out_path, channels, attrs={"liveaov/matte/concepts": "person,vehicle,animal"}
    )

    from live_action_aov.io.oiio_io import read_exr

    back, attrs = read_exr(out_path)

    # Every channel round-trips by value. OIIO may reorder internally, so
    # look up each channel by name via `attrs["channelnames"]` rather than
    # trusting position.
    names_in_file = list(attrs["channelnames"])
    # All requested channels must have survived (no silent drops).
    for requested in channels:
        assert requested in names_in_file, (
            f"Dynamic channel {requested!r} was stripped by the writer"
        )
    assert back.shape == (h, w, len(channels))
    for name, expected in channels.items():
        idx = names_in_file.index(name)
        assert np.allclose(back[..., idx], expected, atol=1e-5), (
            f"Channel {name} did not survive round-trip"
        )
    # Concept-list attribute survives.
    assert attrs.get("liveaov/matte/concepts") == "person,vehicle,animal"
