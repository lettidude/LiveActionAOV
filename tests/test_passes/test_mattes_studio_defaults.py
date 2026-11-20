"""Hero slot order + slot table, crowd cap, re-seed of an empty seed frame,
the overlap-first owner rule, distance-transform gates, the holdout combine
and the LTX chunk-join check. No model: fakes and synthetic masks."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from live_action_aov.io.cryptomatte import name_to_id
from live_action_aov.passes.matte.boxmask import LazyStack
from live_action_aov.passes.matte.ltx_alpha import (
    chunk_agreement,
    connected_owner,
    grow_disc,
    sam_gate,
)
from live_action_aov.passes.matte.rank import (
    Instance,
    RankWeights,
    mean_coverage,
    rank_and_assign,
)
from live_action_aov.passes.matte.sam3 import (
    SAM3MattePass,
    box_owner,
    crypto_hex,
    crypto_name,
    reseed_frames,
)
from live_action_aov.post.holdout import HoldoutPost, close, grow, holdout_combine, shrink

# ----------------------------------------------------------------------
# Hero order (rank.py)
# ----------------------------------------------------------------------


def _inst(tid: int, area: float, cent: float, frames: int, n: int = 10) -> Instance:
    return Instance(tid, "person", list(range(frames)), area, cent, 0.0)


def test_coverage_order_puts_the_biggest_subject_in_r() -> None:
    # A small, central person beats a big one off-centre on the score; with
    # "coverage" the big one takes matte.r.
    small = _inst(1, 0.005, 0.95, 10)
    big = _inst(2, 0.10, 0.40, 10)
    mid = _inst(3, 0.03, 0.5, 10)
    w = RankWeights()
    by_score = rank_and_assign([small, big, mid], w, 10)
    assert by_score[0].track_id == 1
    by_cover = rank_and_assign([small, big, mid], w, 10, order="coverage")
    assert [h.track_id for h in by_cover] == [2, 3, 1]
    assert by_cover[0].score == pytest.approx(0.10)


def test_coverage_counts_absent_frames_as_zero() -> None:
    brief = _inst(1, 0.2, 0.5, 2)  # big, on 2 of 10 frames -> 0.04
    steady = _inst(2, 0.06, 0.5, 10)  # 0.06 all shot
    assert mean_coverage(brief, 10) == pytest.approx(0.04)
    out = rank_and_assign([brief, steady], RankWeights(), 10, order="coverage")
    assert [h.track_id for h in out] == [2, 1]


def test_unknown_order_is_refused() -> None:
    with pytest.raises(ValueError, match="hero order"):
        rank_and_assign([_inst(1, 0.1, 0.5, 3)], RankWeights(), 3, order="size")


# ----------------------------------------------------------------------
# SAM pass: slot table, hero by name / box, crowd cap, re-seed
# ----------------------------------------------------------------------


class _Reader:
    def __init__(self, frames: list[np.ndarray], first: int = 1) -> None:
        self._frames, self._first = frames, first

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _rect(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class _Fake(SAM3MattePass):
    """`people`: (y0, y1, x0, x1) boxes found on any frame whose mean is
    above `visible_above` (a darker frame shows nobody)."""

    def __init__(self, params: dict[str, Any], people: list[tuple[int, int, int, int]]) -> None:
        super().__init__(params)
        self.people, self.seen_on = people, []
        self.visible_above = -1.0

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        h, w, _ = seed_frame.shape
        self.seen_on.append(float(seed_frame.mean()))
        if "person" not in concepts or float(seed_frame.mean()) <= self.visible_above:
            return []
        return [(i + 1, "person", _rect(h, w, *b)) for i, b in enumerate(self.people)]

    def _track_instance(self, frames, seed_frame_idx, seed_mask=None, **_kw):  # type: ignore[override]
        return np.stack([seed_mask] * frames.shape[0], axis=0)


_PEOPLE = [(10, 14, 22, 26), (2, 30, 30, 46), (4, 20, 2, 14)]  # small central, big, mid


def _run(params: dict[str, Any], n: int = 5, people=_PEOPLE, frames=None) -> _Fake:
    p: dict[str, Any] = {
        "concepts": ["person"],
        "detector": "sam3",
        "min_area_fraction": 0.001,
        "small_objects": "off",
        "exclusive_masks": False,
        "redetect_stride": None,
        **params,
    }
    fake = _Fake(p, people)
    frames = frames or [np.full((32, 48, 3), 0.5, np.float32) for _ in range(n)]
    fake.run_shot(_Reader(frames), (1, len(frames)))
    return fake


def test_slot_table_in_artifacts_and_report() -> None:
    fake = _run({"hero_order": "coverage"})
    heroes = fake.emit_artifacts()["sam3_instances"]
    hero = next(iter(heroes.values()))
    assert hero[0]["slot"] == "r" and hero[0]["track_id"] == 2  # the big one
    assert hero[0]["crypto_name"] == "person_2"
    assert hero[0]["crypto_id"] == name_to_id("person_2")[1]
    assert hero[0]["mean_area"] == pytest.approx(28 * 16 / (32 * 48), abs=1e-5)
    rep = fake.report_fields()
    slots = rep["extra"]["slots"]
    assert [s["slot"] for s in slots] == ["r", "g", "b"]
    assert [s["crypto_name"] for s in slots] == ["person_2", "person_3", "person_1"]
    by_id = {o["id"]: o for o in rep["objects"]}
    assert by_id[2]["slot"] == "r" and by_id[2]["crypto_id"] == crypto_hex("person", 2)


def test_score_order_is_the_default() -> None:
    fake = _run({})
    hero = next(iter(fake.emit_artifacts()["sam3_instances"].values()))
    assert fake.params["hero_order"] == "score"
    # Same as before this change: the score ranking decides.
    ref = rank_and_assign(
        [fake._to_rank_instance(i, 5, 32, 48, 1) for i in fake._instances], RankWeights(), 5
    )
    assert [h["track_id"] for h in hero] == [h.track_id for h in ref]


def test_hero_by_crypto_name_and_by_box() -> None:
    fake = _run(
        {
            "heroes": [
                {"name": "person_1", "slot": "r"},
                {"box": [60, 4, 92, 56], "frame": 2, "slot": "g", "ref_size": [96, 64]},
            ]
        }
    )
    slots = {h.slot: h.track_id for h in fake._heroes}
    assert slots["r"] == 1 and slots["g"] == 2  # the box sits on the big person


def test_box_owner_picks_most_covered() -> None:
    fake = _run({})
    assert box_owner(fake._instances, 1, (22, 10, 26, 14)) == 1
    assert box_owner(fake._instances, 1, (0, 0, 1, 1)) is None


def test_crowd_cap_keeps_the_biggest_and_warns() -> None:
    fake = _run({"crowd_cap": 2})
    ids = sorted(i.track_id for i in fake._instances)
    assert ids == [2, 3]
    rep = fake.report_fields()
    assert any("crowd" in w for w in rep["warnings"])
    assert rep["extra"]["crowd_dropped"] == ["person_1"]


def test_crowd_warning_only_by_default() -> None:
    fake = _run({"crowd_warn_above": 2})
    assert len(fake._instances) == 3
    assert any("more than 2" in w for w in fake.report_fields()["warnings"])


def test_empty_seed_frame_is_reseeded() -> None:
    # The middle (seed) frame shows nobody; the first frame does.
    frames = [np.full((32, 48, 3), 0.9, np.float32) for _ in range(5)]
    frames[2] = np.full((32, 48, 3), 0.1, np.float32)
    fake = _Fake(
        {
            "concepts": ["person"],
            "detector": "sam3",
            "min_area_fraction": 0.001,
            "small_objects": "off",
            "exclusive_masks": False,
            "redetect_stride": None,  # SAM 3's re-detection would find them too
        },
        _PEOPLE,
    )
    fake.visible_above = 0.5
    fake.run_shot(_Reader(frames), (1, 5))
    assert len(fake._instances) == 3
    assert "re-seeded" in fake.report_fields()["warnings"][0]


def test_reseed_off_keeps_the_empty_result() -> None:
    frames = [np.full((32, 48, 3), 0.9, np.float32) for _ in range(5)]
    frames[2] = np.full((32, 48, 3), 0.1, np.float32)
    fake = _Fake(
        {
            "concepts": ["person"],
            "detector": "sam3",
            "min_area_fraction": 0.001,
            "small_objects": "off",
            "reseed_empty": False,
            "redetect_stride": None,
        },
        _PEOPLE,
    )
    fake.visible_above = 0.5
    fake.run_shot(_Reader(frames), (1, 5))
    assert fake._instances == []


def test_reseed_frames_order() -> None:
    assert reseed_frames(101, 50) == [25, 75, 0, 100]
    assert reseed_frames(1, 0) == []
    assert crypto_name("", 4) == "object_4"


# ----------------------------------------------------------------------
# Overlap-first owner rule
# ----------------------------------------------------------------------


def _owner_setup() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray]:
    import cv2

    h, w = 60, 100
    a = np.zeros((h, w), np.uint8)
    a[10:50, 5:30] = 1  # person A
    b = np.zeros((h, w), np.uint8)
    b[10:50, 60:85] = 1  # person B
    fg = np.zeros((h, w), np.float32)
    fg[a > 0] = 1
    fg[b > 0] = 1
    fg[25:28, 30:55] = 1  # A's outstretched arm, reaching to 5 px from B
    d = np.stack([cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) for m in (a, b)])
    near = np.argmin(d, 0).astype(np.int16)
    return near, fg, [a, b], d


def test_outstretched_arm_goes_to_its_person() -> None:
    owner, fg, bodies, d = _owner_setup()
    assert (owner[25:28, 50:55] == 1).all()  # nearest gives the hand to B
    n = connected_owner(owner, fg, bodies, d)
    assert n == 1
    assert (owner[25:28, 30:55] == 0).all()


def test_piece_joining_two_people_is_split_between_them_only() -> None:
    owner, fg, bodies, d = _owner_setup()
    fg[25:28, 55:60] = 1  # the arm now touches B
    c = np.zeros_like(bodies[0])
    c[52:58, 90:98] = 1  # a third person nearby, not in that piece
    d3 = np.concatenate([d, np.full((1, *c.shape), np.inf, np.float32)])
    owner[...] = np.argmin(d, 0)
    connected_owner(owner, fg, [*bodies, None], d3)
    assert (owner[25:28, 31:42] == 0).all() and (owner[25:28, 50:60] == 1).all()


def test_piece_with_no_mask_and_big_piece_keep_nearest() -> None:
    owner, fg, bodies, d = _owner_setup()
    fg[55:60, 40:50] = 1  # a loose blob, no mask in it
    before = owner.copy()
    connected_owner(owner, fg, bodies, d, max_ratio=0.01)  # arm too big for the cap
    assert (owner == before).all()


# ----------------------------------------------------------------------
# Distance-transform growth
# ----------------------------------------------------------------------


def test_grow_disc_matches_the_disc_kernel() -> None:
    import cv2

    m = np.zeros((400, 600), np.uint8)
    m[150:250, 200:260] = 1
    m[300:320, 400:500] = 1
    for r in (3, 17, 60):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        ref = cv2.dilate(m, k) > 0
        got = grow_disc(m, r)
        assert np.count_nonzero(ref ^ got) <= 0.01 * np.count_nonzero(ref)


def test_grow_disc_is_fast_on_a_wide_gate() -> None:
    m = np.zeros((1080, 1920), np.uint8)
    m[400:700, 800:1000] = 1
    t0 = time.perf_counter()
    g = grow_disc(m, 450)
    assert time.perf_counter() - t0 < 2.0
    assert g[550, 900 + 540] and not g[550, 1000 + 460]


def test_sam_gate_distance_close_to_kernel() -> None:
    st = np.zeros((2, 120, 160), np.uint8)
    st[0, 40:80, 60:100] = 1
    a = sam_gate(st, 0.05, "kernel")
    b = sam_gate(st, 0.05, "distance")
    assert float(np.abs(a - b).max()) < 0.05
    assert not b[1].any()


# ----------------------------------------------------------------------
# Holdout combine
# ----------------------------------------------------------------------


def test_morphology_helpers() -> None:
    m = np.zeros((40, 40), bool)
    m[10:30, 10:18] = True
    m[10:30, 20:30] = True  # 2 px gap
    assert close(m, 2)[20, 19] and not m[20, 19]
    assert shrink(grow(m, 3), 3)[20, 19]
    assert not shrink(m, 5)[12, 12]


def test_holdout_cuts_the_neighbour_and_keeps_the_rest() -> None:
    a = np.ones((50, 80), np.float32)
    b = np.zeros((50, 80), np.uint8)
    b[10:40, 50:70] = 1
    out = holdout_combine(a, b, grow_px=2, feather_px=0)
    assert out[25, 60] == 0 and out[25, 47] == 1 and out[25, 49] == 0
    assert (holdout_combine(a, np.zeros_like(b)) == a).all()


def test_grey_opening_only_in_the_band_and_outside_the_protect_zone() -> None:
    a = np.zeros((60, 120), np.float32)
    a[:, :30] = 1.0  # the hero, solid
    a[:, 44:47] = 0.6  # a thin soft strip along the neighbour
    a[5:8, 100:103] = 0.8  # a speck far from the neighbour
    b = np.zeros((60, 120), np.uint8)
    b[:, 50:70] = 1  # the neighbour
    own = np.zeros_like(b)
    own[:, :30] = 1
    out = holdout_combine(a, b, protect=own, protect_px=1, open_px=4, open_band_px=12)
    assert out[30, 45] < 0.05  # strip next to the neighbour removed
    assert out[6, 101] == pytest.approx(0.8)  # speck outside the band untouched
    assert (out[:, :30] == 1.0).all()  # hero untouched
    # A protect zone wide enough to reach the strip keeps it.
    kept = holdout_combine(a, b, protect=own, protect_px=20, open_px=4, open_band_px=12)
    assert kept[30, 45] == pytest.approx(0.6)


def test_holdout_post_writes_one_channel_per_hero() -> None:
    T, h, w = 2, 40, 60
    s1 = np.zeros((T, h, w), np.uint8)
    s1[:, 5:35, 5:25] = 1
    s2 = np.zeros((T, h, w), np.uint8)
    s2[:, 5:35, 30:55] = 1
    arts = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": [1, 2], "stack": LazyStack(list(s1), (h, w))},
                2: {"label": "person", "frames": [1, 2], "stack": LazyStack(list(s2), (h, w))},
            }
        },
        "sam3_instances": {0: [{"track_id": 1, "slot": "r"}, {"track_id": 2, "slot": "g"}]},
    }
    soft = np.zeros((h, w), np.float32)
    soft[5:35, 5:32] = 1.0  # r spills 2 px onto person 2
    chans = {f: {"matte.r": soft.copy(), "matte.g": np.zeros((h, w), np.float32)} for f in (1, 2)}
    post = HoldoutPost({"close_px": 0, "grow_px": 0, "feather_px": 0})
    post.ingest_artifacts(arts)
    out = post.apply(chans, None, "sh")
    r = out[1]["holdout.person_1"]
    assert r[20, 31] == 0 and r[20, 20] == 1
    assert "holdout.person_2" in out[1]
    assert (out[1]["matte.r"] == soft).all()  # inputs untouched


# ----------------------------------------------------------------------
# LTX chunk joins
# ----------------------------------------------------------------------


def test_chunk_agreement() -> None:
    x = np.zeros((10, 10), np.float32)
    x[2:8, 2:8] = 1
    y = x.copy()
    assert chunk_agreement([x], [y]) == (1.0, 0.0)
    y[2:8, 2:5] = 0
    iou, da = chunk_agreement([x], [y])
    assert iou == pytest.approx(0.5) and da == pytest.approx(0.18)
    z = np.zeros((10, 10), np.float32)
    assert chunk_agreement([z], [z])[0] == 1.0


def test_connected_never_extends_past_the_nearest_reach() -> None:
    # A reflection / spray trail joined to A but past `far_frac` (unowned by
    # the nearest rule) stays unowned: the rule changes owners, not coverage.
    owner, fg, bodies, d = _owner_setup()
    owner[25:28, 45:55] = -1  # the arm's tip is past the reach
    connected_owner(owner, fg, bodies, d)
    assert (owner[25:28, 45:55] == -1).all()
    assert (owner[25:28, 30:45] == 0).all()


def test_refiners_carry_the_slot_table() -> None:
    from live_action_aov.passes.matte.rvm import slot_table_fields

    h = {"track_id": 2, "crypto_name": "person_2", "crypto_id": "28b3f53c", "mean_area": 0.1}
    assert slot_table_fields(h) == {
        "crypto_name": "person_2",
        "crypto_id": "28b3f53c",
        "mean_area": 0.1,
    }
    assert slot_table_fields({"track_id": 1}) == {}
