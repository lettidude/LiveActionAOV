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


def test_default_prefetch_covers_all_commercial_weight_passes() -> None:
    """The invariant behind FlamingFrames' BUGREPORT_prefetch: every
    registered commercial-safe pass that loads weights must be in the
    default prefetch set, or a plain `liveaov prefetch` prints "all
    cached" while an --offline run using the missing pass dies — the
    exact failure prefetch exists to prevent. (vitmatte_refiner and
    video_depth_anything were both missing when this was reported.)"""
    from live_action_aov.cli.app import _DEFAULT_PREFETCH, _PREFETCH_EXEMPT
    from live_action_aov.core.pass_base import UtilityPass
    from live_action_aov.core.registry import get_registry

    registry = get_registry()
    uncovered = []
    for name in registry.list_passes():
        if name in _DEFAULT_PREFETCH or name in _PREFETCH_EXEMPT:
            continue
        if name.startswith("fake_"):  # test fixtures polluting the singleton
            continue
        cls = registry.get_pass(name)
        has_weights = any(
            "_load_model" in c.__dict__ for c in cls.__mro__ if c is not UtilityPass
        )
        if has_weights and cls.declared_license().commercial_use:
            uncovered.append(name)
    assert not uncovered, (
        f"Commercial-safe weight-loading passes missing from _DEFAULT_PREFETCH: "
        f"{uncovered} — add them (or to _PREFETCH_EXEMPT if truly weight-free)."
    )


def test_warn_uncovered_passes_fires_on_gap(capsys) -> None:
    from live_action_aov.cli.app import _DEFAULT_PREFETCH, _warn_uncovered_passes

    _warn_uncovered_passes([n for n in _DEFAULT_PREFETCH if n != "vitmatte_refiner"])
    out = capsys.readouterr().out
    assert "vitmatte_refiner" in out and "--passes" in out


def test_warn_uncovered_passes_silent_when_covered(capsys) -> None:
    from live_action_aov.cli.app import _DEFAULT_PREFETCH, _warn_uncovered_passes
    from live_action_aov.core.registry import get_registry

    # Treat this module's fake fixture passes as covered — they pollute the
    # singleton registry but don't exist in real runs.
    fakes = [n for n in get_registry().list_passes() if n.startswith("fake_")]
    _warn_uncovered_passes(list(_DEFAULT_PREFETCH) + fakes)
    assert capsys.readouterr().out.strip() == ""
