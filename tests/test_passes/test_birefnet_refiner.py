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
    def __init__(self, params=None):
        merged = {"fallback_feather": False}
        merged.update(params or {})
        super().__init__(merged)

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
        {
            "track_id": 1,
            "slot": "r",
            "label": "hero",
            "score": 0.9,
            "frames": list(range(1, n_frames + 1)),
        }
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
        CH_MATTE_R,
        CH_MATTE_G,
        CH_MATTE_B,
        CH_MATTE_A,
    }


def test_inherits_rvm_artifact_contract() -> None:
    reqs = set(BiRefNetRefinerPass.requires_artifacts)
    assert {"sam3_hard_masks", "sam3_instances"} <= reqs
    assert {ch.name for ch in BiRefNetRefinerPass.produces_channels} == {
        CH_MATTE_R,
        CH_MATTE_G,
        CH_MATTE_B,
        CH_MATTE_A,
    }


def test_catalog_combo_expands_to_sam3_plus_birefnet() -> None:
    assert expand_models(["sam3_birefnet"]) == ["sam3_matte", "birefnet_refiner"]


# --- real crop → paste → mask-bound geometry --------------------------


def test_refine_instance_crops_pastes_and_bounds() -> None:
    n, h, w = 2, 40, 60
    # Explicit small band: this test checks the crop/paste/bound mechanics
    # on a tiny plate — the production default (20px, sized for flying hair)
    # would reach the corners here and hide what's under test.
    pass_ = _FakeBiRefNet({"hard_mask_dilate": 5})
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


# --- keep_aspect / outline mode (field report) ---------------------------


def test_square_box_keeps_inside_frame() -> None:
    from live_action_aov.passes.matte.birefnet import _square_box

    # Tall box (40x100) in a 200x120 frame -> 100x100 square, centred.
    assert _square_box(80, 120, 10, 110, 200, 120) == (50, 150, 10, 110)
    # Near the right edge: shifted left to stay inside.
    assert _square_box(180, 200, 0, 100, 200, 120) == (100, 200, 0, 100)
    # Taller than the frame is wide: clamped to the frame.
    x0, x1, y0, y1 = _square_box(0, 10, 0, 100, 60, 100)
    assert (x0, x1) == (0, 60) and (y0, y1) == (0, 100)


class _ShapeSpy(BiRefNetRefinerPass):
    """Records what `_birefnet_alpha` hands the network (no weights)."""

    seen: list[tuple[int, int]] = []

    def _load_model(self) -> None:  # type: ignore[override]
        import torch

        class _Net(torch.nn.Module):
            def forward(self, x):  # BiRefNet returns a list of logits
                _ShapeSpy.seen.append(tuple(x.shape[-2:]))
                # A vertical bar in the middle third of the INPUT: if the crop
                # were stretched, it would come back wider than the subject.
                out = torch.full((1, 1, *x.shape[-2:]), -10.0)
                w = x.shape[-1]
                out[..., w // 3 : 2 * w // 3] = 10.0
                return [out]

        self._model = _Net()
        self._device = "cpu"
        self._dtype = torch.float32


def test_keep_aspect_pads_instead_of_stretching() -> None:
    p = _ShapeSpy({"infer_size": 64})
    p._load_model()
    crop = np.zeros((90, 30, 3), np.float32)  # tall person crop, 3:1
    a = p._birefnet_alpha(crop)
    assert a.shape == (90, 30)
    assert _ShapeSpy.seen[-1] == (64, 64)
    # Padded to 90x90: the network's middle third (cols 30..60) is exactly the
    # crop, so the whole crop is foreground. Stretched, it would be 1/3 of it.
    assert float(a.mean()) > 0.9
    p_old = _ShapeSpy({"infer_size": 64, "keep_aspect": False})
    p_old._load_model()
    assert float(p_old._birefnet_alpha(crop).mean()) < 0.5


class _Holey(_FakeBiRefNet):
    """Network says: the subject has a gap down its middle (arm gap SAM
    filled in), or (lost=True) almost nothing at all."""

    lost = False

    def _birefnet_alpha(self, crop_rgb: np.ndarray) -> np.ndarray:  # type: ignore[override]
        a = np.ones(crop_rgb.shape[:2], dtype=np.float32)
        if self.lost:
            return a * 0.0
        a[:, a.shape[1] // 2 - 1 : a.shape[1] // 2 + 1] = 0.0
        return a


def test_outline_refiner_opens_gaps_sam_filled() -> None:
    hard = _rect_stack(1, 40, 40, 5, 35, 5, 35)
    plate = np.full((1, 40, 40, 3), 0.5, np.float32)
    sam_mode = _Holey({"keep_aspect": False})._refine_instance(plate, hard)
    ref_mode = _Holey({"keep_aspect": False, "outline": "refiner"})._refine_instance(plate, hard)
    # Inside SAM's eroded core the "sam" outline is forced solid; "refiner"
    # keeps the gap BiRefNet saw.
    assert sam_mode[0, 20, 19:21].min() == 1.0
    assert ref_mode[0, 20, 19:21].max() == 0.0


def test_outline_refiner_falls_back_when_subject_lost() -> None:
    hard = _rect_stack(1, 40, 40, 5, 35, 5, 35)
    plate = np.full((1, 40, 40, 3), 0.5, np.float32)
    p = _Holey({"keep_aspect": False, "outline": "refiner"})
    p.lost = True
    out = p._refine_instance(plate, hard)
    # Below 75% of SAM's area -> SAM-bounded result (solid core), not empty.
    assert out[0, 20, 20] == 1.0
