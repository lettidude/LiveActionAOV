"""Device choice (`core.device`) and the per-pass Mac declaration
(`DeviceSupport`), with CUDA / MPS availability mocked: runs the same on a
CUDA PC, a CPU-only CI runner and a Mac."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from live_action_aov.core import device
from live_action_aov.core.device import resolve_device_name
from live_action_aov.core.pass_base import DeviceSupport, MacSupport

GPU = ("cuda", "mps", "cpu")


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(device.ENV_VAR, raising=False)
    device._warned.clear()


def _machine(monkeypatch: pytest.MonkeyPatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


# --- the order: cuda, then mps, then cpu --------------------------------------


@pytest.mark.parametrize(
    ("cuda", "mps", "want"),
    [(True, False, "cuda"), (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_order(cuda: bool, mps: bool, want: str) -> None:
    assert resolve_device_name(GPU, override="", cuda_ok=cuda, mps_ok=mps) == want


def test_a_pass_that_does_not_list_mps_never_gets_it() -> None:
    # The default (today's behaviour) is CUDA, else the CPU.
    assert DeviceSupport().devices == ("cuda", "cpu")
    assert resolve_device_name(override="", cuda_ok=False, mps_ok=True) == "cpu"
    assert resolve_device_name(("cuda",), override="", cuda_ok=False, mps_ok=True) == "cpu"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(device.ENV_VAR, "cpu")
    assert resolve_device_name(GPU, cuda_ok=True, mps_ok=False) == "cpu"
    monkeypatch.setenv(device.ENV_VAR, " MPS ")
    assert resolve_device_name(GPU, cuda_ok=True, mps_ok=True) == "mps"
    monkeypatch.setenv(device.ENV_VAR, "cuda:1")
    assert resolve_device_name(GPU, cuda_ok=True, mps_ok=False) == "cuda:1"
    monkeypatch.setenv(device.ENV_VAR, "auto")
    assert resolve_device_name(GPU, cuda_ok=False, mps_ok=True) == "mps"


@pytest.mark.parametrize(
    ("value", "allowed", "want"),
    [
        ("mps", GPU, "cuda"),  # asked for a device the machine does not have
        ("tpu", GPU, "cuda"),  # not a device
        ("mps", ("cuda", "cpu"), "cuda"),  # the pass does not support it
    ],
)
def test_a_bad_override_falls_back_to_auto(
    value: str, allowed: tuple[str, ...], want: str, caplog: pytest.LogCaptureFixture
) -> None:
    got = resolve_device_name(allowed, override=value, cuda_ok=True, mps_ok=False)
    assert got == want
    assert device.ENV_VAR in caplog.text


def test_cuda_choice_is_the_same_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CUDA and no override: exactly the device every pass used to pick."""
    for cuda, mps in [(True, False), (True, True), (False, False)]:
        _machine(monkeypatch, cuda=cuda, mps=mps)
        before = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for allowed in (GPU, ("cuda", "cpu"), DeviceSupport(devices=GPU)):
            assert device.pick_device(allowed) == before


def test_pick_device_on_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    _machine(monkeypatch, cuda=False, mps=True)
    assert device.mps_available()
    assert device.pick_device(DeviceSupport(devices=GPU)) == torch.device("mps")
    assert device.pick_device(DeviceSupport()) == torch.device("cpu")


def test_availability_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> bool:
        raise RuntimeError("driver")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    monkeypatch.setattr(torch.backends.mps, "is_available", boom)
    assert device.cuda_available() is False
    assert device.mps_available() is False
    assert resolve_device_name(GPU, override="") == "cpu"


def test_autocast_device_type() -> None:
    assert device.autocast_device_type(torch.device("cuda")) == "cuda"
    assert device.autocast_device_type(torch.device("cuda", 1)) == "cuda"
    assert device.autocast_device_type(torch.device("mps")) == "mps"
    assert device.autocast_device_type(torch.device("cpu")) == "cpu"
    assert device.autocast_device_type(None) == "cpu"


def test_empty_cache_follows_the_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    _machine(monkeypatch, cuda=True, mps=False)
    device.empty_cache()
    _machine(monkeypatch, cuda=False, mps=True)
    device.empty_cache()
    _machine(monkeypatch, cuda=False, mps=False)
    device.empty_cache()
    assert calls == ["cuda", "mps"]


# --- the per-pass declaration ---------------------------------------------------


def _registered() -> dict[str, Any]:
    from live_action_aov.core.registry import get_registry

    reg = get_registry()
    return {n: reg.get_pass(n) for n in reg.list_passes()}


def test_every_built_in_pass_declares_its_mac_support() -> None:
    passes = {n: c for n, c in _registered().items() if c.__module__.startswith("live_action_aov.")}
    assert passes
    for name, cls in passes.items():
        ds = cls.device_support
        assert isinstance(ds, DeviceSupport), name
        assert ds.mac is not MacSupport.UNKNOWN, name
        # Nothing has been run on a real Mac yet.
        assert ds.mac_verified is False, name
        gpu_on_mac = ds.mac in (MacSupport.MPS, MacSupport.MPS_FP32)
        assert ("mps" in ds.devices) == gpu_on_mac, name
        assert "cuda" in ds.devices or ds.mac is MacSupport.CPU, name


def test_the_mac_light_scope() -> None:
    passes = _registered()
    label = {n: c.device_support.mac for n, c in passes.items()}
    for name in (
        "splat_gsplat",
        "depthcrafter",
        "normalcrafter",
        "marigold_v2_albedo",
        "moge3",
        "video_depth_steady",  # takes its metres from MoGe-3
    ):
        assert label[name] is MacSupport.UNAVAILABLE
    for name in ("depth_anything_v2", "video_depth_anything", "vitmatte_refiner"):
        assert label[name] is MacSupport.MPS_FP32
    # SAM 3 is kept off the Apple GPU: on a Mac it runs on the CPU.
    assert label["sam3_matte"] is MacSupport.CPU_ONLY
    assert passes["sam3_matte"].device_support.devices == ("cuda", "cpu")
    assert label["flow"] is MacSupport.MPS_FP32
    assert label["cryptomatte"] is MacSupport.CPU
    assert label["camera_pycolmap"] is MacSupport.CPU
    # A refiner does not inherit its parent's declaration by accident.
    assert passes["chroma_key"].device_support.devices == ("cpu",)


def test_mac_labels_say_unverified() -> None:
    assert DeviceSupport(mac=MacSupport.MPS_FP32).mac_label() == "Apple GPU, fp32 (unverified)"
    assert DeviceSupport(mac=MacSupport.MPS, mac_verified=True).mac_label() == "Apple GPU"
    assert DeviceSupport(mac=MacSupport.UNAVAILABLE).mac_label() == "not available on Mac"
    assert DeviceSupport().mac_label() == "unknown"


def test_mac_check_and_table() -> None:
    class Splat:
        device_support = DeviceSupport(devices=("cuda",), mac=MacSupport.UNAVAILABLE)

    class Depth:
        device_support = DeviceSupport(devices=GPU, mac=MacSupport.MPS_FP32, mac_notes="fp32.")

    class ThirdParty:  # declares nothing
        pass

    passes = {"splat": Splat, "depth": Depth, "plugin": ThirdParty}
    blocked, unverified = device.mac_check(passes)
    assert blocked == ["splat: not available on Mac"]
    assert unverified == ["depth: Apple GPU, fp32 (unverified) - fp32.", "plugin: unknown"]
    assert [r[0] for r in device.mac_table(passes)] == ["depth", "plugin", "splat"]


# --- hardware, doctor, GUI --------------------------------------------------------


def test_detect_reports_the_apple_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from live_action_aov.core import hardware

    _machine(monkeypatch, cuda=False, mps=True)
    monkeypatch.setattr(hardware, "_apple_chip_name", lambda: "Apple M2 Pro")
    hw = hardware.detect()
    assert hw.backend == "mps" and hw.gpu_name == "Apple M2 Pro" and hw.vram_gb is None
    assert "MPS" in hw.summary() and "experimental" in hw.summary()
    s = hardware.suggest(hw, (3840, 2160), 24)
    assert s.tier == "mac" and s.detector == "sam3" and s.proxy_long_edge in (960, 1280)
    assert any("experimental" in n for n in s.notes)


def test_detect_without_a_gpu_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from live_action_aov.core import hardware

    _machine(monkeypatch, cuda=False, mps=False)
    hw = hardware.detect()
    assert hw.backend is None and hw.gpu_name is None
    assert hardware.suggest(hw, (3840, 2160), 24).tier == "none"


def test_doctor_lists_the_mac_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from live_action_aov.cli.app import app
    from live_action_aov.passes.matte import ltx_alpha

    monkeypatch.setattr(ltx_alpha, "comfy_check", lambda params=None: (False, "not reachable"))
    plain = CliRunner().invoke(app, ["doctor"])
    assert plain.exit_code == 0, plain.stdout
    assert "On a Mac" not in plain.stdout
    result = CliRunner().invoke(app, ["doctor", "--mac"])
    assert result.exit_code == 0, result.stdout
    assert "On a Mac" in result.stdout
    assert "splat_gsplat: not available on Mac" in result.stdout
    assert "depth_anything_v2: Apple GPU, fp32 (unverified)" in result.stdout


def test_gui_mac_check_reads_the_shot_models() -> None:
    from live_action_aov.gui.main_window import _mac_check
    from live_action_aov.gui.shot_state import ShotState

    shot = ShotState(
        name="SH010",
        folder=Path("."),
        sequence_pattern="SH010.%04d.exr",
        frame_range=(1001, 1040),
        resolution=(4096, 2160),
    )
    shot.enabled_models = ["depth_anything_v2", "splat_gsplat"]
    blocked, unverified = _mac_check([shot])
    assert any(b.startswith("splat_gsplat:") for b in blocked)
    assert any(u.startswith("depth_anything_v2:") for u in unverified)
    shot.enabled_models = ["depth_anything_v2", "cryptomatte"]
    blocked, unverified = _mac_check([shot])
    assert blocked == []
    names = {u.split(":")[0] for u in unverified}
    assert names >= {"depth_anything_v2", "sam3_matte", "cryptomatte"}


def test_video_depth_anything_follows_the_device() -> None:
    """The model is told the device its weights are on, and runs fp32 on a Mac."""
    import numpy as np

    from live_action_aov.passes.depth.video_depth_anything import VideoDepthAnythingPass

    seen: dict[str, Any] = {}

    class FakeModel:
        def infer_video_depth(self, frames: Any, **kw: Any) -> tuple[np.ndarray, float]:
            seen.update(kw)
            return np.zeros((1, 2, 2), np.float32), 24.0

    p = VideoDepthAnythingPass({"precision": "fp16"})
    p._model = FakeModel()
    for dev, fp32 in (("cuda", False), ("cpu", False), ("mps", True)):
        p._device = torch.device(dev)
        p._infer_clip(np.zeros((1, 2, 2, 3), np.uint8))
        assert (seen["device"], seen["fp32"]) == (dev, fp32)


def test_forced_device_is_only_an_override_the_run_honours(monkeypatch: pytest.MonkeyPatch) -> None:
    gpu = ("cuda", "mps", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.delenv(device.ENV_VAR, raising=False)
    assert device.forced_device(gpu) is None  # unset
    monkeypatch.setenv(device.ENV_VAR, "auto")
    assert device.forced_device(gpu) is None
    monkeypatch.setenv(device.ENV_VAR, " CUDA:1 ")
    assert device.forced_device(gpu) == "cuda:1"
    assert device.forced_device(DeviceSupport(devices=gpu)) == "cuda:1"
    monkeypatch.setenv(device.ENV_VAR, "cpu")
    assert device.forced_device(gpu) == "cpu"
    monkeypatch.setenv(device.ENV_VAR, "mps")  # not on this machine: ignored
    assert device.forced_device(gpu) is None
    monkeypatch.setenv(device.ENV_VAR, "cuda")
    assert device.forced_device(("cpu",)) is None  # the pass has no GPU model


def test_cuda_index() -> None:
    assert device.cuda_index("cuda:1") == 1
    assert device.cuda_index("cuda") is None
    assert device.cuda_index("cpu") is None
    assert device.cuda_index(None) is None
