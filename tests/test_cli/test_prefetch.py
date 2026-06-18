"""`liveaov prefetch` (dummy preload) + `--offline` global flag.

prefetch loads each pass's model once to trigger downloads, then unloads.
We register fake passes whose `_load_model` is a no-op (or raises) so the
command wiring is tested without any network or GPU.
"""

from __future__ import annotations

import os

import numpy as np
from typer.testing import CliRunner

from live_action_aov.cli.app import app
from live_action_aov.core.pass_base import License, PassType, UtilityPass
from live_action_aov.core.registry import get_registry

runner = CliRunner()


class _OkPass(UtilityPass):
    name = "fake_prefetch_ok"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="test")
    pass_type = PassType.GEOMETRIC
    loaded = 0
    unloaded = 0

    def _load_model(self) -> None:
        type(self).loaded += 1
        self._model = object()  # type: ignore[attr-defined]

    def unload(self) -> None:
        type(self).unloaded += 1
        super().unload()

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}


class _FailPass(UtilityPass):
    name = "fake_prefetch_fail"
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="test")
    pass_type = PassType.GEOMETRIC

    def _load_model(self) -> None:
        raise RuntimeError("simulated download failure")

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}


def test_prefetch_loads_and_unloads_each_pass() -> None:
    get_registry().register_pass("fake_prefetch_ok", _OkPass)
    _OkPass.loaded = 0
    _OkPass.unloaded = 0
    result = runner.invoke(app, ["prefetch", "--passes", "fake_prefetch_ok"])
    assert result.exit_code == 0, result.stdout
    assert _OkPass.loaded == 1
    assert _OkPass.unloaded == 1
    assert "cached" in result.stdout
    # Prefetch forces CPU so it never fights for VRAM.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def test_prefetch_reports_failures_and_exits_nonzero() -> None:
    get_registry().register_pass("fake_prefetch_fail", _FailPass)
    result = runner.invoke(app, ["prefetch", "--passes", "fake_prefetch_fail"])
    assert result.exit_code == 1
    assert "skipped" in result.stdout
    assert "simulated download failure" in result.stdout


def test_offline_flag_sets_env(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    result = runner.invoke(app, ["--offline", "plugins", "list"])
    assert result.exit_code == 0
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
