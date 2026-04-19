"""RVMRefinerPass — contract tests with the model bypassed.

`_refine_instance` and `_load_model` are the override hooks. The fake
"refines" a hard mask by mild spatial blur — deterministic, numpy-only,
no torch.hub download. We verify:
- MIT license flows through and the gate does NOT block the pass
- requires_artifacts declares sam3_hard_masks + sam3_instances
- matte.{r,g,b,a} channels appear on every frame
- Slots map to channels deterministically
- Frames where the track is absent get zeros
- matte_heroes artifact round-trips for executor metadata wiring
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.passes.matte.rvm import RVMRefinerPass


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


class _FakeRVM(RVMRefinerPass):
    """Refiner that returns a mild floating-point copy of the hard mask."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
        # Pretend soft-alpha by halving the hard mask values — still valid
        # [0,1] but distinguishable from the hard mask for assertions.
        return (hard_stack * 0.5).astype(np.float32, copy=False)


def _artifacts_two_heroes(n_frames: int, h: int, w: int):
    """Build the SAM3-shaped artifact dict the refiner consumes."""
    person_stack = _rect_stack(n_frames, h, w, h // 4, 3 * h // 4, w // 4, 3 * w // 4)
    vehicle_stack = _rect_stack(n_frames, h, w, 0, h // 2, 0, w // 2)
    hard = {
        1: {"label": "person", "frames": list(range(1, n_frames + 1)), "stack": person_stack},
        2: {"label": "vehicle", "frames": list(range(1, n_frames + 1)), "stack": vehicle_stack},
    }
    heroes = [
        {
            "track_id": 1,
            "slot": "r",
            "label": "person",
            "score": 0.9,
            "frames": list(range(1, n_frames + 1)),
        },
        {
            "track_id": 2,
            "slot": "g",
            "label": "vehicle",
            "score": 0.6,
            "frames": list(range(1, n_frames + 1)),
        },
    ]
    return {
        "sam3_hard_masks": {0: hard},
        "sam3_instances": {0: heroes},
    }


def test_license_is_commercial_safe() -> None:
    lic = RVMRefinerPass.declared_license()
    assert lic.spdx == "MIT"
    assert lic.commercial_use is True


def test_requires_artifacts_declares_sam3_dependencies() -> None:
    reqs = set(RVMRefinerPass.requires_artifacts)
    assert "sam3_hard_masks" in reqs
    assert "sam3_instances" in reqs


def test_produces_all_four_matte_channels() -> None:
    names = {ch.name for ch in RVMRefinerPass.produces_channels}
    assert names == {CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A}


def test_run_shot_emits_matte_rgba_every_frame() -> None:
    n = 5
    h, w = 16, 24
    pass_ = _FakeRVM()
    pass_.ingest_artifacts(_artifacts_two_heroes(n, h, w))
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    assert sorted(out) == list(range(1, n + 1))
    for f in out:
        for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
            assert ch in out[f]
            assert out[f][ch].shape == (h, w)
            assert out[f][ch].dtype == np.float32


def test_slot_to_channel_mapping_is_deterministic() -> None:
    """Hero in slot 'r' → matte.r. Slot 'g' → matte.g. Unused slots stay zero."""
    n = 3
    h, w = 16, 24
    pass_ = _FakeRVM()
    pass_.ingest_artifacts(_artifacts_two_heroes(n, h, w))
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))

    # matte.r should reflect the person mask (half-intensity due to the fake).
    r = out[1][CH_MATTE_R]
    # Centre of the person rect must be nonzero.
    assert r[h // 2, w // 2] > 0.0
    # Corner (outside person rect) is zero.
    assert r[0, -1] == 0.0

    # matte.g should reflect the vehicle mask (top-left quadrant).
    g = out[1][CH_MATTE_G]
    assert g[0, 0] > 0.0
    assert g[-1, -1] == 0.0

    # Unused slots (b, a) must be all zeros.
    assert np.all(out[1][CH_MATTE_B] == 0.0)
    assert np.all(out[1][CH_MATTE_A] == 0.0)


def test_instance_absent_frames_zeroed() -> None:
    """A track that's only present in the first 2 frames must leave matte.r
    zero on frames 3+, even if the refiner 'hallucinates' alpha in them."""
    n = 5
    h, w = 16, 24
    hard_stack = _rect_stack(2, h, w, 0, h // 2, 0, w // 2)  # only 2 frames
    artifacts = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": [1, 2], "stack": hard_stack},
            }
        },
        "sam3_instances": {
            0: [
                {"track_id": 1, "slot": "r", "label": "person", "score": 0.9, "frames": [1, 2]},
            ]
        },
    }

    class _HallucinatingRVM(_FakeRVM):
        def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
            # Always return a nonzero mask — the pass MUST zero it out on
            # frames where the track was absent.
            return np.ones_like(hard_stack) * 0.7

    pass_ = _HallucinatingRVM()
    pass_.ingest_artifacts(artifacts)
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))

    # Frames 1 and 2: matte.r has the refined value.
    assert out[1][CH_MATTE_R].max() > 0.0
    assert out[2][CH_MATTE_R].max() > 0.0
    # Frames 3, 4, 5: matte.r must be zero (per-clip slot lock: track absent).
    for f in (3, 4, 5):
        assert np.all(out[f][CH_MATTE_R] == 0.0), f"Frame {f} leaked non-zero matte.r"


def test_emit_artifacts_publishes_matte_heroes_for_metadata() -> None:
    n = 4
    h, w = 16, 24
    pass_ = _FakeRVM()
    pass_.ingest_artifacts(_artifacts_two_heroes(n, h, w))
    pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    artifacts = pass_.emit_artifacts()

    assert "matte_heroes" in artifacts
    heroes = next(iter(artifacts["matte_heroes"].values()))
    assert len(heroes) == 2
    by_slot = {h["slot"]: h for h in heroes}
    assert by_slot["r"]["label"] == "person"
    assert by_slot["r"]["track_id"] == 1
    assert by_slot["g"]["label"] == "vehicle"
    assert by_slot["g"]["track_id"] == 2
    # refined_frames list survives for downstream QC.
    assert set(by_slot["r"]["refined_frames"]) == set(range(1, n + 1))


def test_missing_hard_mask_recorded_not_crashed() -> None:
    """Ranker references a track_id the hard_mask artifact doesn't have
    (pathological; shouldn't happen in practice). Must leave matte.* as
    zeros and record `missing_hard_mask=True` in the emitted artifact."""
    n = 3
    h, w = 16, 24
    artifacts = {
        "sam3_hard_masks": {0: {}},  # empty!
        "sam3_instances": {
            0: [
                {"track_id": 42, "slot": "r", "label": "ghost", "score": 0.5, "frames": [1, 2, 3]},
            ]
        },
    }
    pass_ = _FakeRVM()
    pass_.ingest_artifacts(artifacts)
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    # matte.r stays zero.
    for f in out:
        assert np.all(out[f][CH_MATTE_R] == 0.0)
    # Emitted hero flags missing_hard_mask.
    heroes = next(iter(pass_.emit_artifacts()["matte_heroes"].values()))
    assert heroes[0]["missing_hard_mask"] is True


def test_smoothable_channels_empty() -> None:
    """RVM is recurrent; auto-smoother must not attach."""
    assert RVMRefinerPass.smoothable_channels == []
