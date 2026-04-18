"""LocalExecutor auto-wires TemporalSmoother for PER_FRAME passes with
`smooth: auto` — but only when a flow pass actually ran.

These tests use fake passes (no HF downloads) to assert the wiring logic in
isolation: which pass opts in, the `_auto_for` disambiguator, and the
"no flow → no smoother" guard.
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
from live_action_aov.io.channels import CH_Z  # noqa: E402
from live_action_aov.io.oiio_io import HAS_OIIO  # noqa: E402


runner = CliRunner()


class _FakeDepthPass(UtilityPass):
    """PER_FRAME pass whose postprocess emits a flat depth field. Drives the
    auto-smoother wiring without needing HF/torch.hub downloads."""

    name = "fake_depth_for_smooth_test"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name=CH_Z)]
    smoothable_channels = [CH_Z]
    DEFAULT_PARAMS = {"smooth": "auto"}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        h, w = tensor.shape[1:3] if tensor.ndim == 4 else tensor.shape[:2]
        return {CH_Z: np.full((h, w), 0.5, dtype=np.float32)}


class _FakeFlowPass(UtilityPass):
    """Pair pass that mimics RAFT's artifact surface: emits forward/backward
    flow = zeros for each frame, so the executor's auto-smoother sees
    `forward_flow` as present."""

    name = "fake_flow_for_smooth_test"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.MOTION
    temporal_mode = TemporalMode.PAIR
    produces_channels: list[ChannelSpec] = []
    provides_artifacts = ["forward_flow", "backward_flow"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._fwd: dict[int, np.ndarray] = {}
        self._bwd: dict[int, np.ndarray] = {}

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        first, last = frame_range
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            frame, _ = reader.read_frame(f)
            h, w = frame.shape[:2]
            zero = np.zeros((2, h, w), dtype=np.float32)
            self._fwd[f] = zero
            self._bwd[f] = zero
            out[f] = {}
        return out

    def emit_artifacts(self):  # type: ignore[override]
        return {
            "forward_flow": dict(self._fwd),
            "backward_flow": dict(self._bwd),
        }


@pytest.fixture(autouse=True)
def _register_fakes() -> None:
    reg = get_registry()
    reg.register_pass("fake_depth_for_smooth_test", _FakeDepthPass)
    reg.register_pass("fake_flow_for_smooth_test", _FakeFlowPass)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_smoother_auto_wires_when_flow_and_per_frame_pass_present(
    test_plate_1080p: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "fake_flow_for_smooth_test,fake_depth_for_smooth_test",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # Sidecar metadata should record an auto-wired smoother.
    from live_action_aov.io.oiio_io import read_exr

    sidecar = sorted(test_plate_1080p.glob("test_plate.utility.*.exr"))[0]
    _, attrs = read_exr(sidecar)
    names = " ".join(f"{k}={v}" for k, v in attrs.items())
    assert "liveActionAOV/smooth/post_processors" in attrs
    # Auto-wired entry is tagged with `::<pass_name>`.
    assert "temporal_smooth::fake_depth_for_smooth_test" in attrs.get(
        "liveActionAOV/smooth/post_processors", ""
    ), names


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_smoother_NOT_wired_when_no_flow_pass(test_plate_1080p: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run-shot", str(test_plate_1080p),
            "--passes", "fake_depth_for_smooth_test",
        ],
    )
    assert result.exit_code == 0, result.stdout
    from live_action_aov.io.oiio_io import read_exr

    sidecar = sorted(test_plate_1080p.glob("test_plate.utility.*.exr"))[0]
    _, attrs = read_exr(sidecar)
    assert "liveActionAOV/smooth/post_processors" not in attrs
