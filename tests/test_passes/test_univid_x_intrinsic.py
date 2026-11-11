# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""UniVidX intrinsic pass — contract tests.

The backend isn't wired yet (the 14B model is validated via
scripts/poc_unividx_prep.py before vendoring), so these tests cover the
*declarative contract* only: channel names, license, temporal mode, and
that the unwired backend fails loudly rather than silently producing
garbage.
"""

from __future__ import annotations

import pytest

from live_action_aov.core.pass_base import License, PassType, TemporalMode
from live_action_aov.io import channels as ch
from live_action_aov.passes.intrinsic.univid_x import (
    MIN_VRAM_GB,
    UniVidXIntrinsicPass,
)

# --- channel additions ----------------------------------------------


def test_albedo_irradiance_channels_defined() -> None:
    assert ch.ALBEDO_CHANNELS == ("albedo.r", "albedo.g", "albedo.b")
    assert ch.IRRADIANCE_CHANNELS == ("irradiance.r", "irradiance.g", "irradiance.b")


def test_new_channels_in_canonical_order() -> None:
    for name in (*ch.ALBEDO_CHANNELS, *ch.IRRADIANCE_CHANNELS):
        assert name in ch.CANONICAL_CHANNEL_ORDER


def test_canonical_order_has_no_duplicates() -> None:
    order = ch.CANONICAL_CHANNEL_ORDER
    assert len(order) == len(set(order))


def test_new_channel_constants_exported() -> None:
    for sym in (
        "ALBEDO_CHANNELS",
        "IRRADIANCE_CHANNELS",
        "CH_ALBEDO_R",
        "CH_IRRADIANCE_B",
    ):
        assert sym in ch.__all__


# --- pass contract ---------------------------------------------------


def test_license_is_commercial_safe() -> None:
    lic = UniVidXIntrinsicPass.declared_license()
    assert isinstance(lic, License)
    assert lic.spdx == "Apache-2.0"
    assert lic.commercial_use is True


def test_temporal_mode_is_video_clip() -> None:
    assert UniVidXIntrinsicPass.temporal_mode is TemporalMode.VIDEO_CLIP


def test_pass_type_is_radiometric() -> None:
    assert UniVidXIntrinsicPass.pass_type is PassType.RADIOMETRIC


def test_produces_albedo_and_irradiance() -> None:
    produced = {c.name for c in UniVidXIntrinsicPass.produces_channels}
    assert produced == set(ch.ALBEDO_CHANNELS) | set(ch.IRRADIANCE_CHANNELS)


def test_vram_floor_declared() -> None:
    # Heavy model → must advertise a VRAM floor for the capability gate.
    # 24 GB is the honest minimum (FP8) per the UniVidX_ComfyUI maintainer
    # on the same upstream model; a lower claim would let the GUI offer
    # this pass on a card that then OOM-crashes mid-run.
    assert MIN_VRAM_GB >= 24.0
    assert UniVidXIntrinsicPass.vram_estimate_gb_fn(1920, 1080) >= 24.0


def test_backend_raises_until_wired() -> None:
    """Honest failure: the unwired backend must raise, not no-op."""
    pass_ = UniVidXIntrinsicPass()
    with pytest.raises(NotImplementedError):
        pass_.run_shot(reader=object(), frame_range=(1, 2))


def test_not_in_gui_catalog_yet() -> None:
    """Mirrors the MatAnyone2 precedent — an unfinished pass must not be
    offered to GUI users. Guards against premature catalog exposure."""
    from live_action_aov.gui.pass_catalog import find_entry

    assert find_entry("univid_x_intrinsic") is None
