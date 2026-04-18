"""End-to-end integration test: NoOpPass produces a valid sidecar EXR.

This is Phase 0's exit-criterion verification: the full pipeline runs
start-to-finish (read EXR → pass → write sidecar) without any AI inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.io.channels import CH_Z
from live_action_aov.io.oiio_io import HAS_OIIO, read_exr

pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    not HAS_OIIO, reason="OpenImageIO not installed"
)]


def test_noop_pass_produces_sidecar_with_Z_channel(test_plate_1080p: Path) -> None:
    shot = Shot(
        name="test_plate",
        folder=test_plate_1080p,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, 5),
        resolution=(640, 360),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=["noop"],
    )
    job = Job(shot=shot, passes=[PassConfig(name="noop")])
    live_action_aov.run(job)

    assert shot.status == "done"
    sidecar = shot.sidecars.get("utility")
    assert sidecar is not None and sidecar.exists()

    # Verify channel content.
    pixels, attrs = read_exr(sidecar)
    assert attrs["nchannels"] == 1
    assert attrs["channelnames"] == [CH_Z]
    # NoOpPass writes zeros.
    assert pixels.max() == 0.0
    # Metadata namespace landed in the header.
    assert any(k.startswith("liveActionAOV/") for k in attrs)
    assert attrs["pixelAspectRatio"] == 1.0
