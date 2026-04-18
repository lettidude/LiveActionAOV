"""MatAnyone2RefinerPass — contract-parity tests with RVM.

MatAnyone 2 is a drop-in swap for RVM (same `requires_artifacts`, same
produced channels, same `matte_heroes` emission) with two deliberate
divergences:

1. License is `NTU-S-Lab-1.0` with `commercial_use = False` — gate blocks.
2. When the executor writes sidecars, `matte/commercial` must read
   `"false"` (test covered by `test_matanyone2_metadata.py`).

These tests pin the contract: same inputs → identical shapes / frame
keys / slot mapping / absent-frame zeroing as RVM. If the two diverge in
any of those dimensions, the executor's refiner-agnostic discovery
(via `provides_artifacts = ["matte_heroes"]`) quietly breaks — these
tests are the tripwire.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.passes.matte.matanyone2 import MatAnyone2RefinerPass
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


class _FakeMatAnyone(MatAnyone2RefinerPass):
    """Drop in a deterministic soft-alpha (hard_mask * 0.5) — same behavior
    as the RVM fake so we can cross-check parity without two fakes drifting."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
        return (hard_stack * 0.5).astype(np.float32, copy=False)


class _FakeRVM(RVMRefinerPass):
    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
        return (hard_stack * 0.5).astype(np.float32, copy=False)


def _artifacts_two_heroes(n_frames: int, h: int, w: int):
    person_stack = _rect_stack(n_frames, h, w, h // 4, 3 * h // 4, w // 4, 3 * w // 4)
    vehicle_stack = _rect_stack(n_frames, h, w, 0, h // 2, 0, w // 2)
    hard = {
        1: {"label": "person", "frames": list(range(1, n_frames + 1)), "stack": person_stack},
        2: {"label": "vehicle", "frames": list(range(1, n_frames + 1)), "stack": vehicle_stack},
    }
    heroes = [
        {"track_id": 1, "slot": "r", "label": "person", "score": 0.9,
         "frames": list(range(1, n_frames + 1))},
        {"track_id": 2, "slot": "g", "label": "vehicle", "score": 0.6,
         "frames": list(range(1, n_frames + 1))},
    ]
    return {
        "sam3_hard_masks": {0: hard},
        "sam3_instances": {0: heroes},
    }


# ---------------------------------------------------------------------------
# License / metadata-facing
# ---------------------------------------------------------------------------


def test_license_is_noncommercial_gated() -> None:
    lic = MatAnyone2RefinerPass.declared_license()
    assert lic.spdx == "NTU-S-Lab-1.0"
    assert lic.commercial_use is False
    assert lic.commercial_tool_resale is False
    # Notes must mention NC-only intent so the CLI error points somewhere real.
    assert "non-commercial" in lic.notes.lower() or "nc" in lic.notes.lower()


def test_declares_same_dag_contract_as_rvm() -> None:
    """Executor's refiner discovery walks provides_artifacts for
    `matte_heroes`; both backends must match exactly."""
    assert set(MatAnyone2RefinerPass.provides_artifacts) == {"matte_heroes"}
    assert set(MatAnyone2RefinerPass.requires_artifacts) == {
        "sam3_hard_masks",
        "sam3_instances",
    }


def test_produces_all_four_matte_channels() -> None:
    names = {ch.name for ch in MatAnyone2RefinerPass.produces_channels}
    assert names == {CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A}


def test_smoothable_channels_empty() -> None:
    """MatAnyone is memory-based / recurrent; auto-smoother must not attach."""
    assert MatAnyone2RefinerPass.smoothable_channels == []


# ---------------------------------------------------------------------------
# Parity with RVM — same inputs, same outputs
# ---------------------------------------------------------------------------


def test_emits_matte_rgba_every_frame() -> None:
    n, h, w = 4, 16, 24
    pass_ = _FakeMatAnyone()
    pass_.ingest_artifacts(_artifacts_two_heroes(n, h, w))
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    assert sorted(out) == list(range(1, n + 1))
    for f in out:
        for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
            assert ch in out[f]
            assert out[f][ch].shape == (h, w)
            assert out[f][ch].dtype == np.float32


def test_output_bitwise_matches_rvm_for_same_fake_refine() -> None:
    """Given identical `_refine_instance` behavior, MatAnyone and RVM must
    emit identical channel stacks. Any divergence here is a contract leak."""
    n, h, w = 4, 16, 24
    artifacts = _artifacts_two_heroes(n, h, w)
    plates = _plate_frames(n, h, w)

    a = _FakeMatAnyone()
    a.ingest_artifacts(artifacts)
    a_out = a.run_shot(_FakeReader(plates), frame_range=(1, n))

    b = _FakeRVM()
    b.ingest_artifacts(artifacts)
    b_out = b.run_shot(_FakeReader(plates), frame_range=(1, n))

    for f in a_out:
        for ch in (CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A):
            assert np.array_equal(a_out[f][ch], b_out[f][ch]), (
                f"Frame {f} channel {ch} diverges between MatAnyone2 and RVM "
                f"despite identical refine behavior"
            )


def test_instance_absent_frames_zeroed() -> None:
    """Per-clip slot lock: track present frames 1..2 only; frames 3..5
    must be zero in matte.r, even with a hallucinating refiner."""
    n, h, w = 5, 16, 24
    hard_stack = _rect_stack(2, h, w, 0, h // 2, 0, w // 2)
    artifacts = {
        "sam3_hard_masks": {0: {
            1: {"label": "person", "frames": [1, 2], "stack": hard_stack},
        }},
        "sam3_instances": {0: [
            {"track_id": 1, "slot": "r", "label": "person", "score": 0.9,
             "frames": [1, 2]},
        ]},
    }

    class _Hallucinator(_FakeMatAnyone):
        def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
            return np.ones_like(hard_stack) * 0.7

    pass_ = _Hallucinator()
    pass_.ingest_artifacts(artifacts)
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    assert out[1][CH_MATTE_R].max() > 0.0
    assert out[2][CH_MATTE_R].max() > 0.0
    for f in (3, 4, 5):
        assert np.all(out[f][CH_MATTE_R] == 0.0), f"Frame {f} leaked non-zero matte.r"


def test_emit_matte_heroes_for_metadata() -> None:
    n, h, w = 4, 16, 24
    pass_ = _FakeMatAnyone()
    pass_.ingest_artifacts(_artifacts_two_heroes(n, h, w))
    pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    artifacts = pass_.emit_artifacts()
    assert "matte_heroes" in artifacts
    heroes = next(iter(artifacts["matte_heroes"].values()))
    by_slot = {h["slot"]: h for h in heroes}
    assert by_slot["r"]["label"] == "person"
    assert by_slot["g"]["label"] == "vehicle"
    for h in heroes:
        assert h["missing_hard_mask"] is False


def test_missing_hard_mask_recorded_not_crashed() -> None:
    n, h, w = 3, 16, 24
    artifacts = {
        "sam3_hard_masks": {0: {}},
        "sam3_instances": {0: [
            {"track_id": 42, "slot": "r", "label": "ghost", "score": 0.5,
             "frames": [1, 2, 3]},
        ]},
    }
    pass_ = _FakeMatAnyone()
    pass_.ingest_artifacts(artifacts)
    out = pass_.run_shot(_FakeReader(_plate_frames(n, h, w)), frame_range=(1, n))
    for f in out:
        assert np.all(out[f][CH_MATTE_R] == 0.0)
    heroes = next(iter(pass_.emit_artifacts()["matte_heroes"].values()))
    assert heroes[0]["missing_hard_mask"] is True
