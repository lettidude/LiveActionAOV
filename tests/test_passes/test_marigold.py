# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Marigold passes — contract tests.

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
from live_action_aov.passes.normals.marigold_normals import MarigoldNormalsPass

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


def test_all_marigold_passes_commercial_openrail() -> None:
    for cls in (
        MarigoldIntrinsicsLightingPass,
        MarigoldIntrinsicsAppearancePass,
        MarigoldNormalsPass,
    ):
        lic = cls.declared_license()
        assert isinstance(lic, License)
        assert lic.commercial_use is True
        assert lic.spdx == "OpenRAIL++-M"


def test_temporal_mode_per_frame() -> None:
    for cls in (
        MarigoldIntrinsicsLightingPass,
        MarigoldIntrinsicsAppearancePass,
        MarigoldNormalsPass,
    ):
        assert cls.temporal_mode is TemporalMode.PER_FRAME


def test_pass_types() -> None:
    assert MarigoldIntrinsicsLightingPass.pass_type is PassType.RADIOMETRIC
    assert MarigoldIntrinsicsAppearancePass.pass_type is PassType.RADIOMETRIC
    assert MarigoldNormalsPass.pass_type is PassType.GEOMETRIC


def test_lighting_produces_albedo_shading_residual() -> None:
    produced = {c.name for c in MarigoldIntrinsicsLightingPass.produces_channels}
    assert produced == set(ch.ALBEDO_CHANNELS) | set(ch.IRRADIANCE_CHANNELS) | set(
        ch.RESIDUAL_CHANNELS
    )


def test_appearance_produces_albedo_and_materials() -> None:
    produced = {c.name for c in MarigoldIntrinsicsAppearancePass.produces_channels}
    assert produced == set(ch.ALBEDO_CHANNELS) | set(ch.MATERIAL_CHANNELS)


def test_normals_produces_xyz() -> None:
    produced = {c.name for c in MarigoldNormalsPass.produces_channels}
    assert produced == {ch.CH_N_X, ch.CH_N_Y, ch.CH_N_Z}


def test_smoothable_channels_match_produced() -> None:
    # Every smoothable channel must be one the pass actually produces.
    for cls in (
        MarigoldIntrinsicsLightingPass,
        MarigoldIntrinsicsAppearancePass,
        MarigoldNormalsPass,
    ):
        produced = {c.name for c in cls.produces_channels}
        assert set(cls.smoothable_channels) <= produced


def test_vram_floor_declared() -> None:
    assert MIN_VRAM_GB <= 8.0  # lightweight vs the 24 GB UniVidX path
    for cls in (
        MarigoldIntrinsicsLightingPass,
        MarigoldIntrinsicsAppearancePass,
        MarigoldNormalsPass,
    ):
        assert cls.vram_estimate_gb_fn(1920, 1080) >= 1.0


def test_construct_does_not_load_model() -> None:
    # Lazy: constructing must not build the diffusers pipe.
    p = MarigoldIntrinsicsLightingPass()
    assert p._pipe is None


# --- GUI catalog wiring ----------------------------------------------


def test_in_gui_catalog() -> None:
    from live_action_aov.gui.pass_catalog import PASS_CATALOG, find_entry

    for key in ("marigold_iid_lighting", "marigold_iid_appearance", "marigold_normals"):
        assert find_entry(key) is not None
    assert "Intrinsics" in PASS_CATALOG
    # All Marigold catalog entries are commercial (no NC consent dialog).
    for key in ("marigold_iid_lighting", "marigold_iid_appearance", "marigold_normals"):
        assert find_entry(key).commercial is True
