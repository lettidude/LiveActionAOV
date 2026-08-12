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


# --- Regression: the 2026-08-12 field failure --------------------------
# Green screen + black glossy car with green reflections + foliage. The
# old whole-frame p90 calibration collapsed d_ref onto the plate content
# and keyed the car; the two-stage calibration must keep everything that
# is not screen essentially opaque.


def _field_plate(h: int = 60, w: int = 80) -> np.ndarray:
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[...] = [0.18, 0.18, 0.18]  # neutral ground/subject
    rgb[0:20, :, :] = [0.06, 0.40, 0.05]  # green screen: top third
    rgb[30:45, 0:40, :] = [0.03, 0.08, 0.02]  # foliage: mildly green
    rgb[45:60, 40:80, :] = [0.02, 0.05, 0.02]  # dark car w/ green bounce
    return rgb


def test_car_and_foliage_survive_the_pull() -> None:
    p = ScreenPullPass({})
    out = p.run_shot(_Reader([_field_plate()]), (1, 1))
    a = out[1][CH_KEY_A]
    assert a[10, 40] < 0.05, "screen must key to transparent"
    assert a[50, 60] > 0.8, "dark car reflection must stay opaque"
    assert a[35, 20] > 0.75, "foliage must stay opaque"
    assert a[25, 40] == 1.0, "neutral subject untouched"


def test_small_screen_still_calibrates() -> None:
    """Screen covering ~4% of frame: whole-frame p90 lands OFF-screen,
    the p99-population calibration still finds it."""
    rgb = np.zeros((50, 50, 3), np.float32)
    rgb[...] = [0.2, 0.2, 0.2]
    rgb[0:10, 0:10, :] = [0.06, 0.40, 0.05]  # 4% screen patch
    p = ScreenPullPass({"screen": "green"})
    out = p.run_shot(_Reader([rgb]), (1, 1))
    a = out[1][CH_KEY_A]
    assert a[5, 5] < 0.05, "small screen must still pull to 0"
    assert a[30, 30] == 1.0


def test_screen_leaving_frame_does_not_retarget_the_pull() -> None:
    """Frame 2 has no screen at all: the shot-level d_ref floor stops the
    calibration from re-targeting onto the foliage."""
    with_screen = _field_plate()
    no_screen = _field_plate()
    no_screen[0:20, :, :] = [0.18, 0.18, 0.18]  # screen gone
    p = ScreenPullPass({"screen": "green"})
    out = p.run_shot(_Reader([with_screen, no_screen]), (1, 2))
    a2 = out[2][CH_KEY_A]
    assert a2[35, 20] > 0.75, "foliage must not become the new 'screen'"
    assert a2[50, 60] > 0.8


def test_key_rgb_keeps_scene_linear_range() -> None:
    """Superbright highlight (linear > 1) must survive into key.rgb —
    no display-style [0,1] clamp on the premultiplied plate."""
    rgb = _field_plate()
    rgb[25, 10, :] = [4.0, 4.0, 4.0]  # speculare hit
    p = ScreenPullPass({"screen": "green"})
    out = p.run_shot(_Reader([rgb]), (1, 1))
    assert out[1][CH_KEY_R][25, 10] > 3.5
