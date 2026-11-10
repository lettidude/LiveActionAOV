"""Ranker unit tests — pure Python, no torch/OIIO.

The ranker is the most testable piece of the matte pipeline: given
`Instance` objects with known area/centrality/motion/duration, we can
assert the top-N and the RGBA slot assignment analytically. Spec §13.1
Phase 3 brainstorm decision #7 makes this the canonical test surface.
"""

from __future__ import annotations

import pytest

from live_action_aov.passes.matte.rank import (
    SLOT_ORDER,
    HeroOverride,
    Instance,
    RankWeights,
    is_scenery_label,
    rank_and_assign,
    score_instance,
)


def _inst(
    track_id: int,
    label: str = "person",
    area: float = 0.1,
    centrality: float = 0.5,
    motion: float = 0.2,
    frames: list[int] | None = None,
    user_priority: float = 0.0,
    scenery: bool = False,
) -> Instance:
    return Instance(
        track_id=track_id,
        label=label,
        frames=frames if frames is not None else list(range(10)),
        area_fraction=area,
        centrality=centrality,
        motion_energy=motion,
        user_priority=user_priority,
        scenery=scenery,
    )


def test_slot_order_is_rgba() -> None:
    assert SLOT_ORDER == ("r", "g", "b", "a")


def test_score_instance_weighted_sum_matches_default_weights() -> None:
    weights = RankWeights()  # area=0.4, centrality=0.2, motion=0.2, duration=0.2
    inst = _inst(track_id=1, area=0.5, centrality=0.5, motion=0.5, frames=list(range(10)))
    s = score_instance(inst, weights, n_clip_frames=10)
    # area*0.5 + centrality*0.5 + motion*0.5 + duration(=1.0)*0.2
    expected = 0.4 * 0.5 + 0.2 * 0.5 + 0.2 * 0.5 + 0.2 * 1.0
    assert s == pytest.approx(expected)


def test_score_with_zero_clip_length_short_circuits() -> None:
    assert score_instance(_inst(1), RankWeights(), n_clip_frames=0) == 0.0


def test_top_four_selected_by_score_descending() -> None:
    """Six instances with varying area; top 4 claim RGBA."""
    insts = [_inst(track_id=i, area=area) for i, area in enumerate([0.05, 0.5, 0.3, 0.1, 0.4, 0.2])]
    heroes = rank_and_assign(insts, RankWeights(), n_clip_frames=10, max_heroes=4)
    assert len(heroes) == 4
    # Slots returned in canonical r,g,b,a order:
    assert [h.slot for h in heroes] == ["r", "g", "b", "a"]
    # Highest area → r, then g, then b, then a.
    assert [h.track_id for h in heroes] == [1, 4, 2, 5]


def test_override_claims_slot_before_score_ranking() -> None:
    insts = [
        _inst(track_id=1, area=0.5),
        _inst(track_id=2, area=0.3),
        _inst(track_id=3, area=0.05),  # tiny — would never naturally rank
    ]
    heroes = rank_and_assign(
        insts,
        RankWeights(),
        n_clip_frames=10,
        max_heroes=3,
        overrides=[HeroOverride(track_id=3, slot="r")],
    )
    by_slot = {h.slot: h.track_id for h in heroes}
    assert by_slot["r"] == 3  # override wins
    # The remaining two slots fill by score order.
    assert by_slot["g"] == 1
    assert by_slot["b"] == 2


def test_unknown_override_track_is_silently_skipped() -> None:
    """Spec: overrides referencing a missing track_id are skipped, not raised."""
    insts = [_inst(track_id=1, area=0.5), _inst(track_id=2, area=0.3)]
    heroes = rank_and_assign(
        insts,
        RankWeights(),
        n_clip_frames=10,
        max_heroes=2,
        overrides=[HeroOverride(track_id=99, slot="r")],
    )
    # No crash; overrides for missing tracks do nothing.
    assert sorted(h.track_id for h in heroes) == [1, 2]


def test_ties_broken_by_area_then_track_id() -> None:
    """Identical scores: larger area wins; if area ties too, smaller track_id."""
    # Zero weights on non-area features so score depends only on area.
    w = RankWeights(area=0.0, centrality=0.0, motion=0.0, duration=0.0, user_priority=0.0)
    a = _inst(track_id=5, area=0.3)
    b = _inst(track_id=2, area=0.3)  # same area, smaller id → wins the tie
    c = _inst(track_id=7, area=0.1)
    heroes = rank_and_assign([a, b, c], w, n_clip_frames=10, max_heroes=3)
    assert [h.track_id for h in heroes] == [2, 5, 7]


def test_fewer_instances_than_slots_returns_short_list() -> None:
    heroes = rank_and_assign(
        [_inst(track_id=1)],
        RankWeights(),
        n_clip_frames=10,
        max_heroes=4,
    )
    assert len(heroes) == 1
    assert heroes[0].slot == "r"


def test_max_heroes_gt_slots_raises() -> None:
    with pytest.raises(ValueError, match="max_heroes"):
        rank_and_assign([_inst(1)], RankWeights(), n_clip_frames=10, max_heroes=5)


def test_determinism_same_input_same_assignment() -> None:
    """Phase 3 brainstorm: same inputs → same slots. Bit-stable."""
    insts = [_inst(track_id=i, area=0.1 * (i + 1)) for i in range(6)]
    a = rank_and_assign(insts, RankWeights(), n_clip_frames=10, max_heroes=4)
    b = rank_and_assign(insts, RankWeights(), n_clip_frames=10, max_heroes=4)
    assert [(h.track_id, h.slot) for h in a] == [(h.track_id, h.slot) for h in b]


def test_duration_contributes_to_score() -> None:
    """An instance present in all frames beats one present in only half,
    all else equal."""
    long_inst = _inst(track_id=1, frames=list(range(10)))
    short_inst = _inst(track_id=2, frames=list(range(5)))
    heroes = rank_and_assign(
        [short_inst, long_inst],
        RankWeights(area=0.0, centrality=0.0, motion=0.0, duration=1.0),
        n_clip_frames=10,
        max_heroes=2,
    )
    # Long-duration hero claims r.
    assert heroes[0].slot == "r"
    assert heroes[0].track_id == 1


def test_user_priority_contributes_when_weighted() -> None:
    low = _inst(track_id=1, area=0.5, user_priority=0.0)
    high = _inst(track_id=2, area=0.05, user_priority=1.0)
    # Huge weight on user_priority ensures the low-area one wins.
    weights = RankWeights(area=0.1, centrality=0.0, motion=0.0, duration=0.0, user_priority=5.0)
    heroes = rank_and_assign([low, high], weights, n_clip_frames=10, max_heroes=2)
    assert heroes[0].track_id == 2 and heroes[0].slot == "r"


# ---------------------------------------------------------------------------
# Scenery never takes a hero slot while a subject is unslotted (A65 plate,
# 2026-11-09: car, person, building, tree got RGBA; the second person none).
# ---------------------------------------------------------------------------


def test_is_scenery_label_matches_words_not_substrings() -> None:
    assert is_scenery_label("sky")
    assert is_scenery_label("old brick building")
    assert is_scenery_label("Tree")
    assert is_scenery_label("water")
    assert not is_scenery_label("person")
    assert not is_scenery_label("vehicle")
    assert not is_scenery_label("skyler")  # a name, not the sky
    assert not is_scenery_label("treehouse keeper")


def test_scenery_ranks_after_every_subject() -> None:
    """The A65 car plate: big sky / buildings / trees outscore the second
    person on area, but the four slots go to the three subjects first."""
    insts = [
        _inst(1, "person", area=0.04),
        _inst(2, "person", area=0.11),
        _inst(3, "vehicle", area=0.30),
        _inst(4, "tree", area=0.12, scenery=True),
        _inst(8, "building", area=0.25, scenery=True),
        _inst(11, "sky", area=0.35, scenery=True),
    ]
    heroes = rank_and_assign(insts, RankWeights(), n_clip_frames=10, max_heroes=4)
    by_slot = {h.slot: h.track_id for h in heroes}
    # Subjects in score order: vehicle, person_2, person_1 ...
    assert [by_slot[s] for s in ("r", "g", "b")] == [3, 2, 1]
    # ... then the best scenery fills the last slot (there is nothing else).
    assert by_slot["a"] == 11


def test_scenery_fills_slots_only_when_subjects_run_out() -> None:
    insts = [_inst(1, "person", area=0.1), _inst(2, "sky", area=0.5, scenery=True)]
    heroes = rank_and_assign(insts, RankWeights(), n_clip_frames=10, max_heroes=4)
    assert [(h.slot, h.track_id) for h in heroes] == [("r", 1), ("g", 2)]
    # Four scenery tracks and no subject: all four get slots, by score.
    only_scenery = [_inst(i, "tree", area=0.1 * i, scenery=True) for i in range(1, 6)]
    heroes = rank_and_assign(only_scenery, RankWeights(), n_clip_frames=10, max_heroes=4)
    assert [h.track_id for h in heroes] == [5, 4, 3, 2]


def test_override_can_still_put_scenery_in_a_slot() -> None:
    """User overrides come first, scenery or not."""
    insts = [
        _inst(1, "person", area=0.3),
        _inst(2, "person", area=0.2),
        _inst(3, "sky", area=0.6, scenery=True),
    ]
    heroes = rank_and_assign(
        insts,
        RankWeights(),
        n_clip_frames=10,
        max_heroes=2,
        overrides=[HeroOverride(track_id=3, slot="r")],
    )
    assert [(h.slot, h.track_id) for h in heroes] == [("r", 3), ("g", 1)]


def test_scenery_tie_break_is_the_same_deterministic_one() -> None:
    w = RankWeights(area=0.0, centrality=0.0, motion=0.0, duration=0.0, user_priority=0.0)
    insts = [
        _inst(9, "tree", area=0.3, scenery=True),
        _inst(4, "building", area=0.3, scenery=True),  # same area, smaller id wins
        _inst(7, "sky", area=0.1, scenery=True),
    ]
    heroes = rank_and_assign(insts, w, n_clip_frames=10, max_heroes=3)
    assert [h.track_id for h in heroes] == [4, 9, 7]
