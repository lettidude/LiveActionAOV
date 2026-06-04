"""Executor frees each pass's model after run_shot (VRAM hygiene).

Without this, a multi-pass / multi-shot session accumulates resident models
until it OOMs — a 5-pass stack crashed loading SAM 3 on the 2nd shot. These
assert the `unload()` default nulls model attrs and that the executor calls
it after every pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("torch")

from live_action_aov.cli.app import app
from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.registry import get_registry
from live_action_aov.io.channels import CH_Z
from live_action_aov.io.oiio_io import HAS_OIIO

runner = CliRunner()


class _MinimalPass(UtilityPass):
    name = "min_unload_test"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}


class _FakeUnloadPass(UtilityPass):
    """PER_FRAME pass that records executor unload() calls at class level
    (the executor instantiates internally, so an instance flag is unreachable)."""

    name = "fake_unload_pass"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name=CH_Z)]
    unload_calls = 0

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._model = object()  # stand-in for a loaded GPU model

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        h, w = tensor.shape[1:3] if tensor.ndim == 4 else tensor.shape[:2]
        return {CH_Z: np.full((h, w), 0.5, dtype=np.float32)}

    def unload(self) -> None:
        type(self).unload_calls += 1
        super().unload()  # exercise the default nulling too


def test_unload_default_nulls_model_attrs() -> None:
    p = _MinimalPass()
    p._model = object()  # type: ignore[attr-defined]
    p._pipe = object()  # type: ignore[attr-defined]
    p._det_model = object()  # type: ignore[attr-defined]
    p.unload()
    assert p._model is None  # type: ignore[attr-defined]
    assert p._pipe is None  # type: ignore[attr-defined]
    assert p._det_model is None  # type: ignore[attr-defined]


def test_free_gpu_memory_never_raises() -> None:
    from live_action_aov.executors.local import _free_gpu_memory

    _free_gpu_memory()  # no torch.cuda or no torch -> graceful no-op


@pytest.fixture
def _register_fake() -> None:
    get_registry().register_pass("fake_unload_pass", _FakeUnloadPass)


@pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")
def test_executor_unloads_each_pass(test_plate_1080p: Path, _register_fake: None) -> None:
    _FakeUnloadPass.unload_calls = 0
    result = runner.invoke(app, ["run-shot", str(test_plate_1080p), "--passes", "fake_unload_pass"])
    assert result.exit_code == 0, result.stdout
    assert _FakeUnloadPass.unload_calls >= 1  # executor freed the pass after run_shot
