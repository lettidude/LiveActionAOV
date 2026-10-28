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
