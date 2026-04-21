"""EXR read/write round-trip: pixel fidelity, channel names, pixel aspect,
custom attributes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.oiio_io import HAS_OIIO, read_exr, write_exr

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")


def test_write_read_multichannel_roundtrip(tmp_path: Path) -> None:
    h, w = 32, 48
    pixels = np.stack(
        [
            np.full((h, w), 0.25, dtype=np.float32),
            np.full((h, w), 0.5, dtype=np.float32),
            np.full((h, w), 0.75, dtype=np.float32),
        ],
        axis=-1,
    )
    out = tmp_path / "r.exr"
    write_exr(
        out,
        pixels,
        channel_names=["Z", "N.x", "motion.x"],
        attrs={"liveaov/test": "value", "liveaov/count": 3},
        pixel_aspect=2.0,
    )
    back, attrs = read_exr(out)
    assert back.shape == (h, w, 3)
    assert np.allclose(back, pixels, atol=1e-5)
    assert attrs["pixelAspectRatio"] == pytest.approx(2.0)
    # Custom attrs survive the round-trip.
    assert attrs.get("liveaov/test") == "value"
