"""Channel-naming contract tests."""

from __future__ import annotations

from live_action_aov.io.channels import (
    CANONICAL_CHANNEL_ORDER,
    CH_BACK_X,
    CH_MOTION_X,
    CH_N_X,
    CH_Z,
    is_mask_channel,
)


def test_canonical_order_includes_expected_channels() -> None:
    names = set(CANONICAL_CHANNEL_ORDER)
    for required in (CH_Z, CH_N_X, CH_MOTION_X, CH_BACK_X):
        assert required in names


def test_mask_channel_detection() -> None:
    assert is_mask_channel("mask.person")
    assert is_mask_channel("mask.vehicle")
    assert not is_mask_channel("mask.")
    assert not is_mask_channel("matte.r")
