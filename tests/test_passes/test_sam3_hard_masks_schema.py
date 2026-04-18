"""`sam3_hard_masks` artifact schema lock — CorridorKey v2c consumer contract.

CorridorKey (spec §21.8, Phase 6) will consume `sam3_hard_masks` directly
to seed its corridor estimation. That consumer doesn't exist yet, so the
contract between SAM 3 and CorridorKey is a **paper** contract; these
tests turn the paper into code.

**What this file protects against.** Someone (future us) refactors SAM 3
and renames `"stack"` → `"masks"`, or flips `frames` from a list to a
range object, or drops to float16 to save memory. All three changes look
harmless in `test_sam3_matte.py` (which asserts *shape-ish* properties —
"has `stack`", "stack is 3D"), but any one of them silently breaks
CorridorKey v2c at runtime, deep in Phase 6, far from where the change
was made.

So this file pins the schema **exactly**: key names, types, dtype, value
ranges, ordering invariants, and the executor-level wrapper shape. If
CorridorKey's consumer expectations change, update these tests first,
then update sam3.py — never the other way round.

Sibling: `sam3_instances` (refiner consumer contract) is also pinned
here since it's another cross-pass artifact whose shape the executor
fans out to both RVM and MatAnyone 2.

Not covered here (by design):
- Pixel content of `stack` — that's the fake's job, not the schema's.
- Track-id semantics — CorridorKey only needs uniqueness + int-ness.
- Label vocabulary — the concept list is job-dependent, not schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.passes.matte.sam3 import SAM3MattePass


# ---------------------------------------------------------------------------
# Deterministic fake — one person + one vehicle, both present every frame.
# ---------------------------------------------------------------------------


class _FakeReader:
    def __init__(self, frames: list[np.ndarray], first: int = 10) -> None:
        self._frames = frames
        self._first = first

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _rect(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class _SchemaFake(SAM3MattePass):
    """Synthesise two tracks (big person + small vehicle) so the schema
    lock test exercises the multi-track branch of the artifact builder."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        h, w, _ = seed_frame.shape
        return [
            (1, "person", _rect(h, w, h // 4, 3 * h // 4, w // 4, 3 * w // 4)),
            (2, "vehicle", _rect(h, w, 0, h // 3, 0, w // 3)),
        ]

    def _track_instance(self, frames, seed_frame_idx, seed_mask):  # type: ignore[override]
        n = frames.shape[0]
        return np.stack([seed_mask] * n, axis=0).astype(np.float32, copy=False)


@pytest.fixture
def _artifacts() -> dict[str, dict[int, object]]:
    """Run a small shot through the fake and return the raw artifact dict.

    Frame range is `(10, 14)` — deliberately non-zero-based so we catch
    any bug where someone assumes frames are indexed from 0 internally.
    """
    n, h, w = 5, 32, 48
    plates = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n)]
    pass_ = _SchemaFake({
        "concepts": ["person", "vehicle"],
        "min_area_fraction": 0.001,
    })
    pass_.run_shot(_FakeReader(plates, first=10), frame_range=(10, 10 + n - 1))
    return pass_.emit_artifacts()


# ---------------------------------------------------------------------------
# sam3_hard_masks — CorridorKey v2c consumption contract
# ---------------------------------------------------------------------------


def test_outer_wrapper_is_frame_keyed_dict_with_one_entry(_artifacts) -> None:
    """Shot-level artifacts land under a single frame key (any frame is
    fine; executors shouldn't depend on which). This mirrors how other
    shot-level artifacts (e.g. `matte_heroes`) are wrapped."""
    hard = _artifacts["sam3_hard_masks"]
    assert isinstance(hard, dict)
    # Exactly one key — shot-level artifact, not per-frame.
    assert len(hard) == 1
    # Key is an int (frame index), value is a dict.
    (only_key,) = hard.keys()
    assert isinstance(only_key, int)
    assert isinstance(hard[only_key], dict)


def test_inner_dict_is_keyed_by_plain_int_track_ids(_artifacts) -> None:
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    assert set(inner.keys()) == {1, 2}
    # Plain Python ints — not numpy scalars (JSON/pickle portability).
    for k in inner.keys():
        assert type(k) is int, f"track_id key {k!r} is {type(k)}, want plain int"


def test_per_track_entry_has_exactly_three_keys(_artifacts) -> None:
    """If someone adds a new field (e.g. `bbox`, `confidence`), update
    this test — CorridorKey needs to know about it before the artifact
    goes live. If someone drops or renames a field, this test is the
    tripwire."""
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    for tid, data in inner.items():
        assert set(data.keys()) == {"label", "frames", "stack"}, (
            f"Track {tid} schema drift: got {sorted(data.keys())!r}, "
            f"want exactly {{'label', 'frames', 'stack'}}"
        )


def test_label_is_plain_str(_artifacts) -> None:
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    for tid, data in inner.items():
        assert type(data["label"]) is str, (
            f"Track {tid} label is {type(data['label'])}, want str"
        )
        assert data["label"], "label is empty"


def test_frames_is_sorted_list_of_plain_ints(_artifacts) -> None:
    """CorridorKey walks `frames` to align `stack[i]` with the plate's
    frame `frames[i]`. That alignment needs: list (not range), sorted
    ascending, plain ints."""
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    for tid, data in inner.items():
        frames = data["frames"]
        assert type(frames) is list, (
            f"Track {tid} frames is {type(frames)}, want list"
        )
        assert all(type(f) is int for f in frames), (
            f"Track {tid} frames contains non-int: {[type(f) for f in frames]}"
        )
        assert frames == sorted(frames), f"Track {tid} frames not sorted: {frames}"
        # Frame indices must be absolute (use the reader's first=10), not
        # local (0-based). This is the biggest CorridorKey footgun.
        assert min(frames) >= 10, (
            f"Track {tid} frames leaked local indexing: {frames[:3]}..."
        )


def test_stack_is_float32_3d_aligned_with_frames(_artifacts) -> None:
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    for tid, data in inner.items():
        stack = data["stack"]
        assert isinstance(stack, np.ndarray), (
            f"Track {tid} stack is {type(stack)}, want np.ndarray"
        )
        assert stack.ndim == 3, f"Track {tid} stack.ndim = {stack.ndim}, want 3"
        # T dimension must match len(frames) exactly — CorridorKey does
        # `stack[i]` → mask at `frames[i]`, so mismatched lengths are fatal.
        assert stack.shape[0] == len(data["frames"]), (
            f"Track {tid} stack has {stack.shape[0]} frames but frames list "
            f"has {len(data['frames'])}"
        )
        assert stack.dtype == np.float32, (
            f"Track {tid} stack dtype is {stack.dtype}, want float32 "
            f"(CorridorKey assumes f32 for in-place memory reuse)"
        )


def test_stack_values_are_in_zero_one_range(_artifacts) -> None:
    """Hard masks are near-binary in [0, 1]. CorridorKey thresholds at
    0.5, so negative values or values > 1 would silently mis-threshold."""
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    for tid, data in inner.items():
        stack = data["stack"]
        assert float(stack.min()) >= -1e-6, (
            f"Track {tid} stack.min={stack.min()} < 0"
        )
        assert float(stack.max()) <= 1.0 + 1e-6, (
            f"Track {tid} stack.max={stack.max()} > 1"
        )


def test_stack_hw_matches_plate_shape(_artifacts) -> None:
    """All track stacks must share the plate (H, W). CorridorKey assumes
    every stack is co-registered with the plate without per-track
    resampling."""
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    shapes = {data["stack"].shape[1:] for data in inner.values()}
    assert len(shapes) == 1, f"Track stacks have mismatched H,W: {shapes}"
    (hw,) = shapes
    assert hw == (32, 48), f"Stack HW {hw} doesn't match plate (32, 48)"


def test_track_id_uniqueness_and_stability(_artifacts) -> None:
    """CorridorKey keys its corridor state by `track_id`. Duplicates would
    cause state collisions. (Also covered by Python dict semantics, but
    pinning it as an explicit invariant keeps the contract loud.)"""
    inner = next(iter(_artifacts["sam3_hard_masks"].values()))
    ids = list(inner.keys())
    assert len(ids) == len(set(ids)), f"Duplicate track_ids in artifact: {ids}"


# ---------------------------------------------------------------------------
# sam3_instances — refiner consumer contract (RVM + MatAnyone 2 both eat this)
# ---------------------------------------------------------------------------


def test_sam3_instances_hero_dict_has_required_keys(_artifacts) -> None:
    """Any refiner (RVM, MatAnyone 2, or a future swap) pulls these five
    keys. Adding fields is safe; removing or renaming is a break."""
    heroes = next(iter(_artifacts["sam3_instances"].values()))
    assert isinstance(heroes, list) and heroes, "heroes list is empty"
    required = {"track_id", "slot", "label", "score", "frames"}
    for h in heroes:
        missing = required - set(h.keys())
        assert not missing, f"Hero {h.get('track_id')!r} missing keys: {missing}"


def test_sam3_instances_slot_values_are_rgba_singletons(_artifacts) -> None:
    heroes = next(iter(_artifacts["sam3_instances"].values()))
    for h in heroes:
        assert h["slot"] in {"r", "g", "b", "a"}, (
            f"Hero {h['track_id']} has invalid slot {h['slot']!r}"
        )


def test_sam3_instances_one_hero_per_slot(_artifacts) -> None:
    """Per-clip slot lock (brainstorm decision #3). Two heroes sharing a
    slot would race in the refiner's channel packer."""
    heroes = next(iter(_artifacts["sam3_instances"].values()))
    slots = [h["slot"] for h in heroes]
    assert len(slots) == len(set(slots)), f"Duplicate slots: {slots}"


def test_sam3_instances_frames_is_subset_of_hard_mask_frames(_artifacts) -> None:
    """Refiner uses `heroes[i]["frames"]` to decide which frames to run
    through the matting net. Those frames must exist in the corresponding
    hard-mask stack or the refiner would index out of bounds."""
    hard = next(iter(_artifacts["sam3_hard_masks"].values()))
    heroes = next(iter(_artifacts["sam3_instances"].values()))
    for h in heroes:
        tid = h["track_id"]
        assert tid in hard, f"Hero {tid} has no hard-mask entry"
        hero_frames = set(h["frames"])
        hard_frames = set(hard[tid]["frames"])
        missing = hero_frames - hard_frames
        assert not missing, (
            f"Hero {tid} references frames {missing} not present in hard-masks"
        )


# ---------------------------------------------------------------------------
# Absent artifacts — empty clip must still pass schema validation
# ---------------------------------------------------------------------------


class _EmptyFake(SAM3MattePass):
    """Nothing detected — CorridorKey must cope with the zero-track case."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        return []

    def _track_instance(self, frames, seed_frame_idx, seed_mask):  # type: ignore[override]
        raise AssertionError("should not be called when no seeds detected")


def test_empty_shot_emits_no_artifacts_rather_than_malformed_skeleton() -> None:
    """If SAM 3 detects nothing, `emit_artifacts` returns `{}` — CorridorKey
    keys off artifact presence to decide whether to run. An empty skeleton
    (e.g. `{"sam3_hard_masks": {0: {}}}`) would trip CorridorKey into a
    no-op with a confusing "empty corridor" warning instead of the clean
    "no matte pass ran" branch."""
    n, h, w = 3, 16, 24
    plates = [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n)]
    pass_ = _EmptyFake({"concepts": ["person"], "min_area_fraction": 0.001})
    pass_.run_shot(_FakeReader(plates, first=1), frame_range=(1, n))
    artifacts = pass_.emit_artifacts()
    # Either totally empty, or all three keys absent. Nothing in between.
    assert artifacts == {}, (
        f"Empty shot leaked artifacts: {list(artifacts.keys())}. "
        f"CorridorKey will see this as 'matte ran with zero heroes' "
        f"rather than 'no matte pass'."
    )
