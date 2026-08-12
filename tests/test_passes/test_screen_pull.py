"""ScreenPullPass — whole-plate key pull (real maths, no fakes)."""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import CH_KEY_A, CH_KEY_G, CH_KEY_R
from live_action_aov.passes.matte.screen_pull import ScreenPullPass


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.f = frames

    def read_frame(self, i: int):
        return self.f[i - 1], {}


def _plate(n: int, h: int, w: int, screen: str) -> np.ndarray:
    rgb = np.zeros((n, h, w, 3), np.float32)
    idx = 2 if screen == "blue" else 1
    rgb[..., idx] = 0.7
    rgb[..., [i for i in range(3) if i != idx]] = 0.1
    rgb[:, 8:24, 10:30, :] = 0.4  # neutral subject
    return rgb


def test_pull_green_auto_detected() -> None:
    n, h, w = 2, 32, 48
    p = ScreenPullPass({})
    out = p.run_shot(_Reader(list(_plate(n, h, w, "green"))), (1, 2))
    a = out[1][CH_KEY_A]
    assert a[16, 20] == 1.0  # subject
    assert a[4, 40] < 0.05  # screen -> transparent
    # Premultiplied RGB: screen area is black.
    assert out[1][CH_KEY_R][4, 40] < 0.05
    assert out[1][CH_KEY_G][4, 40] < 0.05


def test_pull_blue_auto_detected() -> None:
    n, h, w = 1, 32, 48
    p = ScreenPullPass({})
    out = p.run_shot(_Reader(list(_plate(n, h, w, "blue"))), (1, 1))
    a = out[1][CH_KEY_A]
    assert a[16, 20] == 1.0 and a[4, 40] < 0.05


def test_despill_tames_green_spill_on_subject() -> None:
    """A subject pixel with green bounce (g above the other channels)
    gets its green clamped toward the despill reference."""
    n, h, w = 1, 32, 48
    plate = _plate(n, h, w, "green")
    plate[:, 16, 15, :] = [0.4, 0.55, 0.4]  # spill: g pumped on the subject
    p = ScreenPullPass({})
    out = p.run_shot(_Reader(list(plate)), (1, 1))
    g = out[1][CH_KEY_G][16, 15]
    # Despilled toward max/mean of (r, b) = 0.4, then premultiplied by ~1.
    assert g <= 0.45, f"green spill not tamed: {g}"


def test_despill_can_be_disabled() -> None:
    n, h, w = 1, 32, 48
    plate = _plate(n, h, w, "green")
    plate[:, 16, 15, :] = [0.4, 0.55, 0.4]
    p = ScreenPullPass({"despill": 0.0})
    out = p.run_shot(_Reader(list(plate)), (1, 1))
    a = out[1][CH_KEY_A][16, 15]
    assert abs(out[1][CH_KEY_G][16, 15] - 0.55 * a) < 1e-4


def test_standalone_contract() -> None:
    """No SAM dependency — the keyer must be runnable on its own."""
    assert ScreenPullPass.requires_artifacts == []
    lic = ScreenPullPass.declared_license()
    assert lic.commercial_use is True and lic.spdx == "MIT"
