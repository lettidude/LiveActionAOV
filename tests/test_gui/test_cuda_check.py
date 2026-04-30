"""Compute-capability compatibility logic for the GUI CUDA preflight.

Regression cover for the v0.1.1 false-positive: PyTorch's cu128 wheel
ships sm_86 SASS (not sm_89), and Ada Lovelace runs sm_86 binaries
natively via NVIDIA's binary-compatibility guarantee. The original
strict `cap_str in arches` check rejected every RTX 40-series GPU and
hard-blocked Submit in `main_window._on_submit_clicked`.
"""

from __future__ import annotations

from live_action_aov.gui.cuda_check import _arch_compatible


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
