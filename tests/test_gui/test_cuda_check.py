"""Compute-capability compatibility logic for the GUI CUDA preflight.

Regression cover for the v0.1.1 false-positive: PyTorch's cu128 wheel
ships sm_86 SASS (not sm_89), and Ada Lovelace runs sm_86 binaries
natively via NVIDIA's binary-compatibility guarantee. The original
strict `cap_str in arches` check rejected every RTX 40-series GPU and
hard-blocked Submit in `main_window._on_submit_clicked`.
"""

from __future__ import annotations

from live_action_aov.gui.cuda_check import (
    CudaState,
    _arch_compatible,
    meets_vram_requirement,
)


def _state(*, available: bool = True, vram: float | None = None) -> CudaState:
    """Build a CudaState for gate tests without touching torch/CUDA."""
    return CudaState(
        available=available,
        torch_version="2.11.0+cu128",
        torch_built_for_cuda=True,
        device_name="Test GPU",
        device_count=1 if available else 0,
        advisory="",
        total_vram_gb=vram,
    )


def test_exact_sass_match() -> None:
    assert _arch_compatible("sm_86", {"sm_80", "sm_86", "sm_90"}) is True


def test_ada_lovelace_falls_back_to_ampere_sass() -> None:
    """RTX 40-series (sm_89) on a cu128 wheel that ships sm_86."""
    arches = {"sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"}
    assert _arch_compatible("sm_89", arches) is True


def test_ada_lovelace_without_ampere_fails() -> None:
    """If the wheel somehow lacks sm_86 too, the fallback can't apply."""
    assert _arch_compatible("sm_89", {"sm_70", "sm_75"}) is False


def test_ptx_jit_path_for_newer_gpu() -> None:
    """Hopper (sm_90) on a wheel with only `compute_80` PTX — JIT'd up."""
    assert _arch_compatible("sm_90", {"sm_70", "compute_80"}) is True


def test_ptx_jit_path_rejects_older_gpu() -> None:
    """sm_60 cannot run compute_80 PTX (you can't JIT downwards)."""
    assert _arch_compatible("sm_60", {"compute_80"}) is False


def test_blackwell_on_legacy_cu124_wheel() -> None:
    """RTX 5090 (sm_120) on a wheel that tops out at sm_90 — the
    original motivating regression for this preflight check."""
    arches = {"sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"}
    assert _arch_compatible("sm_120", arches) is False


def test_blackwell_with_compute_120_ptx() -> None:
    """Same GPU but wheel includes `compute_120` PTX — JIT works."""
    arches = {"sm_50", "sm_70", "sm_80", "sm_86", "sm_90", "compute_120"}
    assert _arch_compatible("sm_120", arches) is True


def test_malformed_cap_string_is_safe() -> None:
    assert _arch_compatible("not_a_sm", {"sm_80"}) is False


def test_malformed_arch_entry_is_skipped() -> None:
    """A junk `compute_` entry shouldn't crash the JIT scan."""
    assert _arch_compatible("sm_90", {"compute_oops", "compute_80"}) is True


# --- VRAM capability gate (portability) ------------------------------


def test_vram_gate_passes_when_above_floor() -> None:
    """RTX 5090 (32 GB) clears a 16 GB heavy-pass floor."""
    assert meets_vram_requirement(_state(vram=32.0), 16.0) is True


def test_vram_gate_passes_at_exact_floor() -> None:
    assert meets_vram_requirement(_state(vram=16.0), 16.0) is True


def test_vram_gate_fails_below_floor() -> None:
    """RTX 4060 (8 GB) can't run a pass that needs 16 GB."""
    assert meets_vram_requirement(_state(vram=8.0), 16.0) is False


def test_vram_gate_permissive_when_unknown() -> None:
    """Unknown VRAM never blocks a pass — telemetry gaps shouldn't hide
    capabilities; the pass OOMs loudly if it truly doesn't fit."""
    assert meets_vram_requirement(_state(vram=None), 16.0) is True


def test_vram_gate_false_when_gpu_unavailable() -> None:
    assert meets_vram_requirement(_state(available=False, vram=32.0), 1.0) is False
