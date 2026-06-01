# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Marigold-IID intrinsic passes — contract tests.

Covers the declarative contract only (channels, license, temporal mode,
produced channels, catalog wiring). No model is loaded — `_load_model` is
lazy, so constructing a pass touches no weights/GPU.
"""

from __future__ import annotations

from live_action_aov.core.pass_base import License, PassType, TemporalMode
from live_action_aov.io import channels as ch
from live_action_aov.passes.intrinsic.marigold import (
    MIN_VRAM_GB,
    MarigoldIntrinsicsAppearancePass,
    MarigoldIntrinsicsLightingPass,
)

_INTRINSIC_PASSES = (MarigoldIntrinsicsLightingPass, MarigoldIntrinsicsAppearancePass)

# --- channel additions ----------------------------------------------


def test_residual_and_material_channels_defined() -> None:
    assert ch.RESIDUAL_CHANNELS == ("residual.r", "residual.g", "residual.b")
    assert ch.MATERIAL_CHANNELS == ("material.roughness", "material.metalness")


def test_new_channels_in_canonical_order_no_dupes() -> None:
    for name in (*ch.RESIDUAL_CHANNELS, *ch.MATERIAL_CHANNELS):
        assert name in ch.CANONICAL_CHANNEL_ORDER
    order = ch.CANONICAL_CHANNEL_ORDER
    assert len(order) == len(set(order))


def test_new_channel_symbols_exported() -> None:
    for sym in (
        "RESIDUAL_CHANNELS",
        "MATERIAL_CHANNELS",
        "CH_RESIDUAL_R",
        "CH_ROUGHNESS",
        "CH_METALNESS",
    ):
        assert sym in ch.__all__


# --- license / contract ----------------------------------------------


def test_intrinsic_passes_commercial_openrail() -> None:
    for cls in _INTRINSIC_PASSES:
        lic = cls.declared_license()
        assert isinstance(lic, License)
        assert lic.commercial_use is True
        assert lic.spdx == "OpenRAIL++-M"


def test_temporal_mode_per_frame() -> None:
    for cls in _INTRINSIC_PASSES:
        assert cls.temporal_mode is TemporalMode.PER_FRAME


def test_pass_type_radiometric() -> None:
    for cls in _INTRINSIC_PASSES:
        assert cls.pass_type is PassType.RADIOMETRIC


def test_lighting_produces_albedo_shading_residual() -> None:
    produced = {c.name for c in MarigoldIntrinsicsLightingPass.produces_channels}
    assert produced == set(ch.ALBEDO_CHANNELS) | set(ch.IRRADIANCE_CHANNELS) | set(
        ch.RESIDUAL_CHANNELS
    )


def test_appearance_produces_albedo_and_materials() -> None:
    produced = {c.name for c in MarigoldIntrinsicsAppearancePass.produces_channels}
    assert produced == set(ch.ALBEDO_CHANNELS) | set(ch.MATERIAL_CHANNELS)


def test_smoothable_channels_match_produced() -> None:
    for cls in _INTRINSIC_PASSES:
        produced = {c.name for c in cls.produces_channels}
        assert set(cls.smoothable_channels) <= produced


def test_temporal_blend_param_default() -> None:
    # Latent propagation: anchor-weight blend in (0, 1).
    for cls in _INTRINSIC_PASSES:
        p = cls()
        b = float(p.params["temporal_blend"])
        assert 0.0 <= b <= 1.0


def test_vram_floor_declared() -> None:
    assert MIN_VRAM_GB <= 8.0  # lightweight vs the 24 GB UniVidX path
    for cls in _INTRINSIC_PASSES:
        assert cls.vram_estimate_gb_fn(1920, 1080) >= 1.0


def test_construct_does_not_load_model() -> None:
    p = MarigoldIntrinsicsLightingPass()
    assert p._pipe is None


# --- GUI catalog wiring ----------------------------------------------


def test_in_gui_catalog() -> None:
    from live_action_aov.gui.pass_catalog import PASS_CATALOG, find_entry

    for key in ("marigold_iid_lighting", "marigold_iid_appearance"):
        assert find_entry(key) is not None
        assert find_entry(key).commercial is True
    assert "Intrinsics" in PASS_CATALOG
    # Marigold normals was dropped (NormalCrafter/DSINE cover normals).
    assert find_entry("marigold_normals") is None
