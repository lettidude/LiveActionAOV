"""Opt-in window stabiliser wiring (`stabilise` pass param / `--stabilise`).

Fake passes, no model downloads. The key guarantee: with the stabiliser
off (the default), the sidecars are bit-identical to a run that never heard
of it - pixels and header.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from live_action_aov.cli.app import app
from live_action_aov.core.pass_base import ChannelSpec, License, PassType, TemporalMode, UtilityPass
from live_action_aov.core.registry import get_registry
from live_action_aov.io.channels import CH_Z
from live_action_aov.io.oiio_io import HAS_OIIO

runner = CliRunner()
DEPTH = "fake_noisy_depth_for_stab_test"
FLOW = "fake_flow_for_stab_test"


class _NoisyDepth(UtilityPass):
    name = DEPTH
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
        return {}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        out = {}
        for f in range(frame_range[0], frame_range[1] + 1):
            frame, _ = reader.read_frame(f)
            h, w = frame.shape[:2]
            rng = np.random.default_rng(f)
            out[f] = {CH_Z: (3.0 + 0.1 * rng.standard_normal((h, w))).astype(np.float32)}
        return out


class _ZeroFlow(UtilityPass):
    name = FLOW
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.MOTION
    temporal_mode = TemporalMode.PAIR
    produces_channels: list[ChannelSpec] = []
    provides_artifacts = ["forward_flow", "backward_flow"]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._f: dict[int, np.ndarray] = {}

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]):  # type: ignore[override]
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(frame_range[0], frame_range[1] + 1):
            frame, _ = reader.read_frame(f)
            self._f[f] = np.zeros((2, *frame.shape[:2]), np.float32)
            out[f] = {}
        return out

    def emit_artifacts(self):  # type: ignore[override]
        return {"forward_flow": dict(self._f), "backward_flow": dict(self._f)}


@pytest.fixture(autouse=True)
def _register_fakes() -> None:
    reg = get_registry()
    reg.register_pass(DEPTH, _NoisyDepth)
    reg.register_pass(FLOW, _ZeroFlow)


_VOLATILE = (
    "created",
    "job_id",
    "time",
    "date",
    "version",
    "run_id",
    "log",
    "output",
    "path",
    "elapsed",
    "host",
)


def _run(plate: Path, out: Path, *extra: str) -> list[tuple[np.ndarray, dict[str, Any]]]:
    from live_action_aov.io.oiio_io import read_exr

    res = runner.invoke(
        app,
        ["run-shot", str(plate), "--out", str(out), "--passes", f"{FLOW},{DEPTH}", *extra],
    )
    assert res.exit_code == 0, res.stdout
    vdir = sorted((out / plate.name).glob("v*"))[-1] / "passes"
    return [read_exr(p) for p in sorted(vdir.glob("*.utility.*.exr"))]


def _stable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in attrs.items() if not any(s in k.lower() for s in _VOLATILE)}


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_off_is_bit_identical_and_on_changes_pixels(test_plate_1080p: Path, tmp_path: Path) -> None:
    base = _run(test_plate_1080p, tmp_path / "a")
    on = _run(test_plate_1080p, tmp_path / "b", "--stabilise", f"{DEPTH}=median:5")
    # `--stabilise` absent vs present: different pixels...
    assert any(not np.array_equal(x[0], y[0]) for x, y in zip(base, on, strict=True))
    # ...and the header says which.
    assert "liveaov/smooth/temporal_stabilise::" + DEPTH + "/mode" in on[0][1]
    assert on[0][1]["liveaov/smooth/temporal_stabilise::" + DEPTH + "/mode"] == "median"
    # The EMA of that pass is replaced, not stacked.
    assert "temporal_smooth::" + DEPTH not in on[0][1].get("liveaov/smooth/post_processors", "")
    # A second default run is bit-identical to the first (pixels and header).
    again = _run(test_plate_1080p, tmp_path / "c")
    assert len(base) == len(again)
    for (pa, aa), (pb, ab) in zip(base, again, strict=True):
        assert np.array_equal(pa, pb)
        assert _stable_attrs(aa) == _stable_attrs(ab)
        assert not any("temporal_stabilise" in k for k in aa)


def test_explicit_off_values_mean_off() -> None:
    from live_action_aov.executors.local import _stabilise_mode

    for v in (None, False, "off", "", "no", "OFF"):
        assert _stabilise_mode(v) is None
    assert _stabilise_mode(True) == "median"
    assert _stabilise_mode("blend") == "blend"
    with pytest.raises(ValueError):
        _stabilise_mode("max")


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_cli_rejects_bad_specs(test_plate_1080p: Path, tmp_path: Path) -> None:
    def code(*args: str) -> int:
        return runner.invoke(
            app, ["run-shot", str(test_plate_1080p), "--out", str(tmp_path), *args]
        ).exit_code

    assert code("--passes", f"{FLOW},{DEPTH}", "--stabilise", f"{DEPTH}=max") == 2
    assert code("--passes", f"{FLOW},{DEPTH}", "--stabilise", f"{DEPTH}=mean:4") == 2
    assert code("--passes", f"{FLOW},{DEPTH}", "--stabilise", "nope") == 2
    assert code("--passes", DEPTH, "--stabilise", DEPTH) == 2  # no motion pass
