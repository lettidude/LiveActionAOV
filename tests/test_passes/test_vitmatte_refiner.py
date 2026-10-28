"""ViTMatteRefinerPass — contract tests with the model bypassed.

`_load_model` + `_vitmatte_alpha` are the override hooks. The fake returns a
constant 0.5 alpha so we can verify the trimap-obedient composite: certain
foreground stays 1.0, outside the band stays 0.0, and only the unknown band
takes the model's value.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.passes.matte.vitmatte import ViTMatteRefinerPass, guided_trimap


class _FakeViTMatte(ViTMatteRefinerPass):
    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _guide_alpha(self, rgb, binm):  # type: ignore[override]
        return None  # no BiRefNet in CI

    def _vitmatte_alpha(self, crop_rgb, trimap):  # type: ignore[override]
        return np.full(crop_rgb.shape[:2], 0.5, dtype=np.float32)


def test_license_is_commercial_safe() -> None:
    lic = ViTMatteRefinerPass.declared_license()
    assert lic.spdx == "MIT"
    assert lic.commercial_use is True


def test_trimap_obedient_composite() -> None:
    """fg core -> exactly 1.0; outside band -> exactly 0.0; unknown band ->
    the model's alpha. The known regions must be structural, not clamped."""
    T, H, W = 2, 64, 96
    plate = np.full((T, H, W, 3), 0.5, np.float32)
    hard = np.zeros((T, H, W), np.float32)
    hard[:, 20:44, 30:66] = 1.0

    p = _FakeViTMatte({"trimap": "band", "fg_erode": 4, "band_dilate": 8})
    soft = p._refine_instance(plate, hard)

    assert soft.shape == (T, H, W)
    # Deep interior (inside erode(4) of the rect) is exactly 1.0.
    assert float(soft[0, 32, 48]) == 1.0
    # Far outside the dilated band is exactly 0.0.
    assert float(soft[0, 2, 2]) == 0.0
    # The unknown band carries the model's value (0.5), not 0 or 1:
    # 4px outside the hard edge is inside dilate(8) but outside the rect.
    assert abs(float(soft[0, 20 - 4, 48]) - 0.5) < 1e-6


def test_empty_mask_frame_stays_zero() -> None:
    T, H, W = 2, 32, 32
    plate = np.full((T, H, W, 3), 0.5, np.float32)
    hard = np.zeros((T, H, W), np.float32)
    hard[0, 10:20, 10:20] = 1.0  # frame 1 empty
    p = _FakeViTMatte({})
    soft = p._refine_instance(plate, hard)
    assert soft[1].max() == 0.0


def test_accepts_uint8_hard_stack() -> None:
    """The v0.5.2+ artifact contract ships uint8 masks — must work directly."""
    T, H, W = 1, 32, 48
    plate = np.full((T, H, W, 3), 0.5, np.float32)
    hard = np.zeros((T, H, W), np.uint8)
    hard[:, 8:24, 12:36] = 1
    p = _FakeViTMatte({})
    soft = p._refine_instance(plate, hard)
    assert soft.dtype == np.float32
    assert float(soft[0, 16, 24]) == 1.0


# --- guided trimap -------------------------------------------------------


def _sq(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_guided_thin_object_keeps_a_core() -> None:
    """A pole 6 px wide must keep sure-FG (eroding by the band erased it and
    ViTMatte rendered thin shelves semi-transparent)."""
    body = _sq(200, 200, 20, 180, 20, 120)
    pole = _sq(200, 200, 20, 180, 160, 166)
    tri = guided_trimap(body | pole, None)
    assert (tri[100, 160:166] == 255).any()
    assert tri[100, 70] == 255 and tri[5, 5] == 0


def test_guided_follows_birefnet_where_it_agrees() -> None:
    sam = _sq(200, 200, 40, 160, 40, 160)
    guide = sam.astype(np.float32)
    guide[40:60, 40:160] = 0.0  # SAM blob over background (filled hair gap)
    guide[30:40, 40:160] = 1.0  # strands SAM missed, just outside it
    tri = guided_trimap(sam, guide)
    assert tri[50, 100] == 128  # SAM says yes, BiRefNet no -> ViTMatte decides
    assert tri[35, 100] == 128  # outside SAM but BiRefNet sure -> not background
    assert tri[100, 100] == 255 and tri[5, 5] == 0


def test_guided_ignores_birefnet_that_disagrees() -> None:
    """Portrait weights on a rifle: near-zero everywhere -> ignored, the trimap
    is SAM's (a solid core, not 'all background')."""
    sam = _sq(200, 200, 40, 160, 40, 160)
    tri = guided_trimap(sam, np.zeros((200, 200), np.float32))
    assert tri[100, 100] == 255
    assert tri[100, 36] == 128  # band outside SAM still unknown
