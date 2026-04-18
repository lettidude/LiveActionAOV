"""Executor writes metric-depth metadata when a metric backend runs.

DepthPro emits a `depth_metric` artifact so the executor stamps
`depth/space=metric` + `depth/unit=meters`. This is the counterpart to the
relative-depth case (`depth/space=relative`, `depth/unit=normalized_per_clip`)
covered elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from live_action_aov.cli.app import app  # noqa: E402
from live_action_aov.core.pass_base import (  # noqa: E402
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.registry import get_registry  # noqa: E402
from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_Z, CH_Z_RAW  # noqa: E402
from live_action_aov.io.oiio_io import HAS_OIIO  # noqa: E402


runner = CliRunner()


class _FakeMetricDepthPass(UtilityPass):
    """Fake metric-depth backend that mimics DepthPro's artifact contract:
    emits `depth_metric` to flag the output as metric-in-meters.
    """

    name = "fake_metric_depth"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [
        ChannelSpec(name=CH_Z),
        ChannelSpec(name=CH_Z_RAW),
        ChannelSpec(name=CH_DEPTH_CONFIDENCE),
    ]
    smoothable_channels = [CH_Z]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._frame_keys: list[int] = []

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        h, w = tensor.shape[1:3] if tensor.ndim == 4 else tensor.shape[:2]
        # Flat 5-meter depth — metric, not normalized.
        z = np.full((h, w), 5.0, dtype=np.float32)
        c = np.full((h, w), 1.0, dtype=np.float32)
        return {CH_Z: z, CH_Z_RAW: z.copy(), CH_DEPTH_CONFIDENCE: c}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        first, last = frame_range
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            frame, _ = reader.read_frame(f)
            out[f] = self.postprocess(frame[None, ...])
        self._frame_keys = list(range(first, last + 1))
        return out

    def emit_artifacts(self):  # type: ignore[override]
        if not self._frame_keys:
            return {}
        return {
            "depth_metric": {self._frame_keys[0]: np.asarray([1.0], dtype=np.float32)},
        }


@pytest.fixture(autouse=True)
def _register_fake() -> None:
    get_registry().register_pass("fake_metric_depth", _FakeMetricDepthPass)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_sidecar_has_metric_space_and_meters_unit(test_plate_1080p: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "fake_metric_depth",
        ],
    )
    assert result.exit_code == 0, result.stdout

    from live_action_aov.io.oiio_io import read_exr

    sidecar = sorted(test_plate_1080p.glob("test_plate.utility.*.exr"))[0]
    _, attrs = read_exr(sidecar)
    assert attrs.get("liveActionAOV/depth/space") == "metric"
    assert attrs.get("liveActionAOV/depth/unit") == "meters"
    # Relative-depth wiring must NOT have fired.
    assert "liveActionAOV/depth/normalization/min" not in attrs
    assert "liveActionAOV/depth/normalization/max" not in attrs
