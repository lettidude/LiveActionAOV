"""Whole + parts ("person" with "hair" / "jacket" inside): the whole keeps
its full mask, parts split it among themselves and stay inside it. A
non-part label inside a bigger object ("person" in a "car") stays exclusive.
Model bypassed with the same fake-subclass convention as test_sam3_clicks."""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.passes.matte.cryptomatte import CryptomattePass
from live_action_aov.passes.matte.sam3 import SAM3MattePass
from live_action_aov.post.matte_layering import MatteLayeringPost

H, W = 40, 60


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class _Reader:
    def __init__(self, n: int) -> None:
        self._f = [np.full((H, W, 3), 0.5, np.float32) for _ in range(n)]

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


class _Fake(SAM3MattePass):
    """Detections are fixed rectangles; tracking repeats the seed mask."""

    DETS: dict[str, np.ndarray] = {
        "person": _rect(4, 40, 20, 40),
        "hair": _rect(2, 12, 20, 40),  # pokes 2 rows above the person mask
        "jacket": _rect(12, 28, 20, 40),
        "car": _rect(0, 40, 10, 50),
    }

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        found = [c for c in concepts if c in self.DETS]
        return [(k + 1, c, self.DETS[c].copy()) for k, c in enumerate(found)]

    def _track_instance(self, frames, seed_frame_idx, seed_mask=None, **_: Any):  # type: ignore[override]
        return np.stack([seed_mask] * frames.shape[0], axis=0)


def _run(concepts: list[str], **extra: Any) -> tuple[_Fake, dict]:
    p = _Fake({"concepts": concepts, "min_area_fraction": 0.001, **extra})
    out = p.run_shot(_Reader(2), frame_range=(1, 2))
    return p, out


def _by_label(p: _Fake) -> dict[str, Any]:
    return {i.label: i for i in p._instances}


def test_parts_hang_on_the_whole() -> None:
    p, out = _run(["person", "hair", "jacket"])
    inst = _by_label(p)
    pid = inst["person"].track_id
    assert p._parent == {inst["hair"].track_id: pid, inst["jacket"].track_id: pid}
    person, hair, jacket = (inst[k].masks[1] > 0 for k in ("person", "hair", "jacket"))
    # Full body: the whole covers every part (even hair SAM put above it).
    assert (person >= hair).all() and (person >= jacket).all()
    assert person[2:4, 20:40].all()
    # Siblings stay exclusive.
    assert not (hair & jacket).any()
    assert out[1]["mask.person"].sum() >= out[1]["mask.hair"].sum() + out[1]["mask.jacket"].sum()
    arts = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    assert arts[inst["hair"].track_id]["parent"] == pid
    assert arts[pid]["parent"] is None


def test_person_in_a_car_is_not_a_part() -> None:
    p, _ = _run(["car", "person"])
    inst = _by_label(p)
    assert p._parent == {}
    # Old behaviour: the smaller object owns the overlap, the car has a hole.
    assert not ((inst["car"].masks[1] > 0) & (inst["person"].masks[1] > 0)).any()


def test_parts_off() -> None:
    p, _ = _run(["person", "hair"], part_containment=0)
    inst = _by_label(p)
    assert p._parent == {}
    assert not ((inst["person"].masks[1] > 0) & (inst["hair"].masks[1] > 0)).any()


def test_explicit_part_labels() -> None:
    p, _ = _run(["car", "person"], part_labels=["person"])
    inst = _by_label(p)
    assert p._parent == {inst["person"].track_id: inst["car"].track_id}


def _family_artifacts() -> dict:
    person = _rect(4, 40, 20, 40).astype(np.uint8)
    hair = _rect(4, 12, 20, 40).astype(np.uint8)
    jacket = _rect(12, 28, 20, 40).astype(np.uint8)
    return {
        "sam3_hard_masks": {
            0: {
                1: {"label": "hair", "frames": [1], "stack": hair[None], "layer": 0, "parent": 3},
                2: {
                    "label": "jacket",
                    "frames": [1],
                    "stack": jacket[None],
                    "layer": 1,
                    "parent": 3,
                },
                3: {
                    "label": "person",
                    "frames": [1],
                    "stack": person[None],
                    "layer": 2,
                    "parent": None,
                },
            }
        }
    }


def test_layering_keeps_parts_inside_and_whole_untouched() -> None:
    whole = _rect(4, 40, 20, 40)
    whole[4:6] = 0.5  # soft top edge (hair strands)
    hair = _rect(4, 12, 20, 40)  # hard-ish part edge, sticks out over the soft top
    jacket = _rect(12, 28, 20, 40)
    hair[10:14] = 0.5
    jacket[10:14] = 0.5  # both fall to 0.5 at the shared seam
    chans = {1: {"mask.person": whole.copy(), "mask.hair": hair, "mask.jacket": jacket}}
    post = MatteLayeringPost()
    post.ingest_artifacts(_family_artifacts())
    out = post.apply(chans, None, "s")[1]
    np.testing.assert_array_equal(out["mask.person"], whole)  # not held out by its parts
    assert (out["mask.hair"] <= whole + 1e-6).all()  # part clamped to the whole
    seam = out["mask.hair"][11:13, 25] + out["mask.jacket"][11:13, 25]
    assert (seam > 0.9).all()  # siblings layered: no dark line between them


def test_cryptomatte_whole_is_what_parts_leave() -> None:
    c = CryptomattePass()
    c.ingest_artifacts(_family_artifacts())
    out = c.run_shot(_Reader(1), frame_range=(1, 1))[1]
    # Coverage ranks sum to <= 1 per pixel (one owner), never 2 inside the hair.
    cov = sum(out[f"CryptoObject{lvl:02d}.{ch}"] for lvl in range(3) for ch in "GA")
    assert float(cov.max()) <= 1.0 + 1e-6


def test_parts_are_cut_from_the_whole_alpha() -> None:
    """Refiner pass: parts are not refined alone, they are cut from the whole's
    matte. Strands outside SAM's hair mask still land in mask.hair, and the
    parts never exceed the whole."""
    from live_action_aov.passes.matte.rvm import RVMRefinerPass

    class _Soft(RVMRefinerPass):
        calls: list[int] = []

        def _load_model(self) -> None:  # type: ignore[override]
            self._model = object()

        def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
            _Soft.calls.append(int(hard_stack.sum()))
            out = hard_stack.astype(np.float32)
            out[:, 2:4, 20:40] = 0.5  # strands above SAM's mask
            return out

    p = _Soft({"refine_all_masks": True})
    p.ingest_artifacts(_family_artifacts())
    out = p.run_shot(_Reader(1), frame_range=(1, 1))[1]
    assert len(_Soft.calls) == 1  # only the whole was refined
    person, hair, jacket = (out[f"mask.{k}"] for k in ("person", "hair", "jacket"))
    assert (hair + jacket <= person + 1e-5).all()
    assert hair[2, 30] > 0.4 and jacket[2, 30] < 0.05  # strand -> nearest part
    assert hair[8, 30] > 0.95 and jacket[20, 30] > 0.95


def test_layer_order_nearer_first_then_smaller() -> None:
    """Thin background shelves are SMALLER than the person but farther: depth
    decides. At about the same depth (rifle in the hands) the smaller wins."""
    from live_action_aov.passes.matte.sam3 import _DetectedInstance, _layer_order

    def inst(tid: int, label: str, area: int) -> _DetectedInstance:
        m = np.zeros((40, 60), np.uint8)
        m.flat[:area] = 1
        return _DetectedInstance(track_id=tid, label=label, masks={1: m})

    person, shelves, rifle = inst(1, "person", 900), inst(2, "shelves", 200), inst(3, "rifle", 100)
    near = {1: 0.8, 2: 0.3, 3: 0.82}
    order = [i.label for i in _layer_order([person, shelves, rifle], {}, near)]
    assert order == ["rifle", "person", "shelves"]
    assert [i.label for i in _layer_order([person, shelves], {})] == ["shelves", "person"]


def test_redetected_part_on_an_owned_part_still_hangs_on_the_whole() -> None:
    """In a joint session the first hair track owns the hair pixels, so a
    hair re-detected later lies on that track, not on the person's own mask.
    It must still become a part of the person (not an independent object
    that splits the alpha 50/50 with it)."""
    from live_action_aov.passes.matte.sam3 import _clip_parents, _DetectedInstance, _part_label_test

    person = _rect(14, 40, 20, 40)  # person mask WITHOUT the hair (owned by track 2)
    hair = _rect(4, 14, 20, 40)
    inst = [
        _DetectedInstance(track_id=1, label="person", masks={1: person}),
        _DetectedInstance(track_id=2, label="hair", masks={1: hair}),
        _DetectedInstance(track_id=2004, label="hair", masks={1: hair.copy()}),
    ]
    out = _clip_parents(inst, {2: 1}, 0.7, _part_label_test("auto"))
    assert out[2004] == 1


def test_stray_same_label_track_does_not_halve_the_whole() -> None:
    """An unrelated 'hair' detection elsewhere (not a part) must not turn the
    label independent: seam normalisation then split the person's hair 50/50
    with the person (PF0100-02)."""
    from live_action_aov.passes.matte.rvm import RVMRefinerPass

    class _Hard(RVMRefinerPass):
        def _load_model(self) -> None:  # type: ignore[override]
            self._model = object()

        def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
            return hard_stack.astype(np.float32)

    arts = _family_artifacts()
    stray = _rect(30, 38, 50, 58).astype(np.uint8)
    arts["sam3_hard_masks"][0][4] = {
        "label": "hair",
        "frames": [1],
        "stack": stray[None],
        "layer": 3,
        "parent": None,
    }
    p = _Hard({"refine_all_masks": True})
    p.ingest_artifacts(arts)
    out = p.run_shot(_Reader(1), frame_range=(1, 1))[1]
    assert out["mask.person"][8, 30] > 0.99  # the person's hair region
    assert out["mask.hair"][8, 30] > 0.99
    post = MatteLayeringPost()
    post.ingest_artifacts(arts)
    lay = post.apply({1: dict(out)}, None, "s")[1]
    assert lay["mask.person"][8, 30] > 0.99  # not held out by its own hair
