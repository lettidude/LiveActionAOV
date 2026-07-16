"""ViTMatteRefinerPass — contract tests with the model bypassed.

`_load_model` + `_vitmatte_alpha` are the override hooks. The fake returns a
constant 0.5 alpha so we can verify the trimap-obedient composite: certain
foreground stays 1.0, outside the band stays 0.0, and only the unknown band
takes the model's value.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.passes.matte.vitmatte import ViTMatteRefinerPass


class _FakeViTMatte(ViTMatteRefinerPass):
    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

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

    p = _FakeViTMatte({"fg_erode": 4, "band_dilate": 8})
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
