"""Matte layering post: shared seams between per-object soft mattes."""

from __future__ import annotations

import numpy as np

from live_action_aov.passes.matte.sam3 import _close_seams, _DetectedInstance
from live_action_aov.post.matte_layering import MatteLayeringPost

H, W = 40, 60


def _ramp(lo: int, hi: int, rising: bool) -> np.ndarray:
    """Column profile going 0->1 (rising) or 1->0 across [lo, hi)."""
    x = np.arange(W, dtype=np.float32)
    r = np.clip((x - lo) / max(hi - lo, 1), 0.0, 1.0)
    return r if rising else 1.0 - r


def _scene():
    # Car (back) occupies cols 0..29 on rows 10..39; person (front) cols 30..59.
    car_hard = np.zeros((H, W), np.uint8)
    car_hard[10:, :30] = 1
    person_hard = np.zeros((H, W), np.uint8)
    person_hard[:, 30:] = 1
    # Refined alphas as two independent refiners produce them: both fall
    # to ~0.5 at the contact line (col 30) -> a dark seam when added.
    person = np.tile(_ramp(29, 35, rising=True), (H, 1))
    car = np.tile(_ramp(25, 31, rising=False), (H, 1))
    car[:10] = 0.0
    car[8:12] *= np.linspace(0, 1, 4, dtype=np.float32)[:, None]  # soft top edge vs background
    arts = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": [1], "stack": person_hard[None], "layer": 0},
                2: {"label": "car", "frames": [1], "stack": car_hard[None], "layer": 1},
            }
        }
    }
    chans = {1: {"mask.person": person.copy(), "mask.car": car.copy()}}
    return arts, chans, car


def test_contact_seam_sums_to_one_and_front_edge_is_kept() -> None:
    arts, chans, _ = _scene()
    before = chans[1]["mask.person"] + chans[1]["mask.car"]
    assert before[20, 30] < 0.5  # the dark line we are fixing

    post = MatteLayeringPost({"seam_px": 6})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "shot")[1]
    s = out["mask.person"] + out["mask.car"]
    # Along the contact (rows where both exist) the alphas add up to 1.
    assert np.allclose(s[16:, 20:40], 1.0, atol=1e-5)
    # Front object untouched: its refined edge defines the seam.
    assert np.array_equal(out["mask.person"], chans[1]["mask.person"])
    # Never above 1 anywhere.
    assert float(s.max()) <= 1.0 + 1e-6


def test_edges_against_background_are_untouched() -> None:
    arts, chans, car_in = _scene()
    post = MatteLayeringPost({"seam_px": 6})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "shot")[1]
    # The car's soft top edge, far from the person, is exactly preserved.
    assert np.allclose(out["mask.car"][8:12, :15], car_in[8:12, :15])
    # No car alpha leaks into the empty area above it.
    assert float(out["mask.car"][:8].max()) == 0.0


def test_disabled_or_single_object_is_a_no_op() -> None:
    arts, chans, _ = _scene()
    ref = {k: v.copy() for k, v in chans[1].items()}
    post = MatteLayeringPost({"enabled": False})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "shot")[1]
    assert all(np.array_equal(out[k], ref[k]) for k in ref)


def test_hero_slots_are_layered_by_track() -> None:
    arts, chans, _ = _scene()
    arts["sam3_instances"] = {0: [{"track_id": 1, "slot": "r"}, {"track_id": 2, "slot": "g"}]}
    chans = {1: {"matte.r": chans[1]["mask.person"], "matte.g": chans[1]["mask.car"]}}
    post = MatteLayeringPost({"seam_px": 6})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "shot")[1]
    assert np.allclose((out["matte.r"] + out["matte.g"])[16:, 20:40], 1.0, atol=1e-5)


def test_close_seams_fills_contact_gap_only() -> None:
    a = np.zeros((H, W), np.uint8)
    a[:, :28] = 1  # gap cols 28..31 between the two objects
    b = np.zeros((H, W), np.uint8)
    b[:, 32:] = 1
    c = np.zeros((H, W), np.uint8)  # one object with a narrow notch of its own
    insts = [
        _DetectedInstance(track_id=1, label="car", masks={1: a.copy()}),
        _DetectedInstance(track_id=2, label="person", masks={1: b.copy()}),
    ]
    _close_seams(insts, gap_px=4)
    m1, m2 = insts[0].masks[1] > 0, insts[1].masks[1] > 0
    assert not (m1 & m2).any()
    assert (m1 | m2).all()  # the contact gap is gone
    # A single object's own notch is NOT closed.
    c[5:35, 10:20] = 1
    c[15:25, 14:16] = 0
    solo = [_DetectedInstance(track_id=3, label="x", masks={1: c.copy()})]
    _close_seams(solo, gap_px=3)
    assert np.array_equal(solo[0].masks[1], c)


def test_rejected_sam_spur_is_not_filled_back() -> None:
    """The back object's SAM mask has a spur next to the front object that its
    refiner rejected (alpha ~0) and the front does not claim either: it must
    stay empty, not come back as a hard chip of SAM's shape."""
    arts, chans, _ = _scene()
    car_hard = arts["sam3_hard_masks"][0][2]["stack"][0]
    car_hard[2:6, 26:30] = 1  # spur over the background, touching the person
    chans[1]["mask.person"][2:6, 26:32] = 0.0
    post = MatteLayeringPost({"seam_px": 6})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "shot")[1]
    assert float(out["mask.car"][2:6, 26:30].max()) == 0.0
