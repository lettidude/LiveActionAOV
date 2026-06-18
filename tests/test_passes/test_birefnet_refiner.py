"""BiRefNetRefinerPass — soft-edge refiner, model bypassed.

BiRefNet reuses RVM's refiner machinery (subclass), so the packing/slot
contract is covered by the RVM tests; here we verify the BiRefNet-specific
parts: it stays MIT/commercial, its matte channels are smoothable (per-
frame model → flicker → flow smoother can help), the SAM3+BiRefNet combo
expands correctly, and the real crop→paste→mask-bound logic in
`_refine_instance` works (with `_birefnet_alpha` faked — numpy/cv2 only,
no weights).
"""

from __future__ import annotations

import numpy as np

from live_action_aov.gui.pass_catalog import expand_models
from live_action_aov.io.channels import CH_MATTE_A, CH_MATTE_B, CH_MATTE_G, CH_MATTE_R
from live_action_aov.passes.matte.birefnet import BiRefNetRefinerPass


class _FakeReader:
    def __init__(self, frames: list[np.ndarray], first: int = 1) -> None:
        self._frames = frames
        self._first = first

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _plate_frames(n: int, h: int = 16, w: int = 24) -> list[np.ndarray]:
    return [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n)]


def _rect_stack(n: int, h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    stack = np.zeros((n, h, w), dtype=np.float32)
    stack[:, y0:y1, x0:x1] = 1.0
    return stack


class _FakeBiRefNet(BiRefNetRefinerPass):
    """BiRefNet whose neural call returns a flat alpha of 1.0 for the crop,
    so we can assert the crop → paste → mask-bound geometry exactly."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()
        self._device = "cpu"
        self._dtype = None

    def _birefnet_alpha(self, crop_rgb: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return np.ones(crop_rgb.shape[:2], dtype=np.float32)


def _artifacts_one_hero(n_frames: int, h: int, w: int, rect):
    y0, y1, x0, x1 = rect
    hard = {
        1: {
            "label": "hero",
            "frames": list(range(1, n_frames + 1)),
            "stack": _rect_stack(n_frames, h, w, y0, y1, x0, x1),
        }
    }
    heroes = [
        {"track_id": 1, "slot": "r", "label": "hero", "score": 0.9,
         "frames": list(range(1, n_frames + 1))}
    ]
    return {"sam3_hard_masks": {0: hard}, "sam3_instances": {0: heroes}}


# --- BiRefNet-specific properties -------------------------------------

def test_license_is_mit_commercial() -> None:
    lic = BiRefNetRefinerPass.declared_license()
    assert lic.spdx == "MIT"
    assert lic.commercial_use is True


def test_matte_channels_are_smoothable() -> None:
    # Per-frame model → flicker → the flow smoother should be allowed to act
    # (RVM, recurrent, declares none — this is the key difference).
    assert set(BiRefNetRefinerPass.smoothable_channels) == {
        CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A
    }


def test_inherits_rvm_artifact_contract() -> None:
    reqs = set(BiRefNetRefinerPass.requires_artifacts)
    assert {"sam3_hard_masks", "sam3_instances"} <= reqs
    assert {ch.name for ch in BiRefNetRefinerPass.produces_channels} == {
        CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A
    }


def test_catalog_combo_expands_to_sam3_plus_birefnet() -> None:
    assert expand_models(["sam3_birefnet"]) == ["sam3_matte", "birefnet_refiner"]


# --- real crop → paste → mask-bound geometry --------------------------

def test_refine_instance_crops_pastes_and_bounds() -> None:
    n, h, w = 2, 40, 60
    pass_ = _FakeBiRefNet()
    # A central rectangle as the hard mask.
    hard = _rect_stack(n, h, w, 12, 28, 20, 40)
    plate = np.full((n, h, w, 3), 0.5, dtype=np.float32)
    soft = pass_._refine_instance(plate, hard)
    assert soft.shape == (n, h, w)
    assert soft.dtype == np.float32
    # Inside the seed rectangle → alpha ~1 (fake returns 1, bounded by mask).
    assert soft[0, 20, 30] == 1.0
    # Far outside (and outside the dilation band) → 0.
    assert soft[0, 0, 0] == 0.0
    assert soft[0, -1, -1] == 0.0
    # Bound to the dilated mask: a pixel well beyond the rect stays 0 even
    # though the crop (padded bbox) covered a wider region.
    assert soft[0, 5, 5] == 0.0


def test_refine_instance_empty_mask_returns_zeros() -> None:
    n, h, w = 2, 16, 24
    pass_ = _FakeBiRefNet()
    soft = pass_._refine_instance(
        np.full((n, h, w, 3), 0.5, dtype=np.float32),
        np.zeros((n, h, w), dtype=np.float32),
    )
    assert soft.shape == (n, h, w)
    assert float(soft.sum()) == 0.0


def test_run_shot_packs_matte_channels() -> None:
    n, h, w = 3, 40, 60
    pass_ = _FakeBiRefNet()
    pass_.ingest_artifacts(_artifacts_one_hero(n, h, w, (12, 28, 20, 40)))
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    assert sorted(out) == [1, 2, 3]
    # Hero is slot 'r' → matte.r carries the soft alpha; centre nonzero.
    assert out[1][CH_MATTE_R][20, 30] > 0.0
    # Unused slots stay zero.
    assert float(out[1][CH_MATTE_G].sum()) == 0.0
