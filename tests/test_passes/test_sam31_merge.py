"""SAM 3.1 small-objects mode: seed frames, the track merge and its guards,
all on synthetic masks (no model, no GPU)."""

from __future__ import annotations

import numpy as np

from live_action_aov.passes.matte import sam31_merge
from live_action_aov.passes.matte.sam31_merge import MergeConfig, make_track, merge_tracks

N, H, W = 12, 60, 80
CFG = MergeConfig()


def _box(y0: int, y1: int, x0: int, x1: int, frames: range | None = None) -> np.ndarray:
    st = np.zeros((N, H, W), np.uint8)
    for f in frames if frames is not None else range(N):
        st[f, y0:y1, x0:x1] = 1
    return st


def _tracks(*specs: tuple[np.ndarray, int, int], label: str = "person") -> list:
    """(stack, session, obj_id) -> RawTracks (seed = session for the record)."""
    out = []
    for st, session, oid in specs:
        t = make_track(st, label, session, session, oid, lambda st=st: st, CFG)
        assert t is not None
        out.append(t)
    return out


# --- seed frames ---------------------------------------------------------------


def test_seed_frames_primary_first_then_ends_then_every_n() -> None:
    assert sam31_merge.seed_frames(200, 100, 40) == [100, 0, 199, 40, 80, 120, 160]
    assert sam31_merge.seed_frames(100, 50, 40) == [50, 0, 99]  # 40 / 80: near another seed
    assert sam31_merge.seed_frames(200, 100, 0) == [100, 0, 199]
    assert sam31_merge.seed_frames(1, 0, 40) == [0]
    assert sam31_merge.seed_frames(0, 0, 40) == []
    assert sam31_merge.seed_frames(10, 99, 4)[0] == 9  # primary clamped into the clip


def test_glitch_frames_drop_a_sudden_whole_frame_reading() -> None:
    cov = np.full(20, 0.004)
    cov[3:6] = 0.4  # a small object read as most of the frame
    cov[10] = 0.0  # not seen: not a glitch
    assert sam31_merge.glitch_frames(cov, 6.0, 0.05) == [3, 4, 5]
    assert sam31_merge.glitch_frames(cov, 0.0, 0.05) == []  # off
    steady = np.full(20, 0.3)
    steady[7] = 0.35  # a big object growing a little: kept
    assert sam31_merge.glitch_frames(steady, 6.0, 0.05) == []


def test_make_track_skips_glitch_frames_and_empty_tracks() -> None:
    st = _box(20, 26, 10, 14)
    st[2] = 1  # whole frame on one frame
    t = make_track(st, "person", 0, 6, 1, lambda: st, CFG)
    assert t is not None and t.glitches == [2] and 2 not in t.small
    empty = np.zeros((N, H, W), np.uint8)
    assert make_track(empty, "person", 0, 6, 1, lambda: empty, CFG) is None


# --- merge ---------------------------------------------------------------------


def test_one_object_seen_by_two_sessions_becomes_one_track() -> None:
    a = _box(10, 50, 10, 30, range(0, 9))  # primary: loses it after frame 8
    b = _box(11, 50, 10, 31, range(3, 12))  # late seed: finds it from frame 3
    other = _box(10, 50, 50, 70)  # someone else, primary only
    tracks = _tracks((a, 0, 0), (b, 1, 5), (other, 0, 1))
    clusters, rep = merge_tracks(tracks, CFG)
    assert clusters == [[0, 1], [2]] and rep.linked == 1
    stack = sam31_merge.assemble(tracks, clusters[0], CFG)
    assert all(stack[f].any() for f in range(N))  # frames 9-11 filled by the late seed
    assert stack[4, 10, 10] == 1 and stack[4, 10, 30] == 0  # primary's mask where both exist


def test_two_people_who_coexist_are_never_linked_through_a_third_track() -> None:
    left = _box(10, 50, 10, 30)
    right = _box(10, 50, 30, 50)
    both = _box(10, 50, 10, 50)  # a later session read the pair as one object
    tracks = _tracks((left, 0, 0), (right, 0, 1), (both, 1, 0))
    clusters, _ = merge_tracks(tracks, MergeConfig(link_iou=0.45, dup_frames=0.0))
    assert [0, 1] not in clusters and not any({0, 1} <= set(c) for c in clusters)


def test_fragments_of_one_object_are_stitched() -> None:
    a = _box(10, 50, 10, 30, range(0, 5))
    b = _box(10, 50, 12, 32, range(7, 12))  # re-found 2 frames later, in the same place
    far = _box(10, 50, 55, 75, range(7, 12))  # starts then too, elsewhere
    clusters, rep = merge_tracks(_tracks((a, 0, 0), (b, 0, 1), (far, 0, 2)), CFG)
    assert clusters == [[0, 1], [2]] and rep.stitched == 1
    late = _box(10, 50, 12, 32, range(10, 12))  # gap of 5 frames
    clusters, _ = merge_tracks(_tracks((a, 0, 0), (late, 0, 1)), MergeConfig(stitch_gap=3))
    assert clusters == [[0], [1]]


def test_a_duplicate_of_comparable_size_is_merged() -> None:
    whole = _box(10, 50, 10, 30)  # 40 x 20
    dup = _box(12, 48, 11, 29)  # the same person tracked twice, slightly tighter
    clusters, rep = merge_tracks(_tracks((whole, 0, 0), (dup, 0, 1)), CFG)
    assert clusters == [[0, 1]] and rep.duplicates == 1


def test_a_body_part_is_not_merged_into_its_whole() -> None:
    """A hand (or a bag, a glove) found as its own object lies entirely
    inside the body: containment alone would merge it. The size-ratio
    guard keeps it a separate object (the part / whole logic handles it)."""
    body = _box(10, 50, 10, 30)  # 800 px
    hand = _box(30, 36, 24, 30)  # 36 px, inside the body on every frame
    for hand_session in (0, 1):  # same session or another seed's
        tracks = _tracks((body, 0, 0), (hand, hand_session, 1))
        clusters, rep = merge_tracks(tracks, CFG)
        assert clusters == [[0], [1]] and rep.duplicates == 0
    # ...and it is the guard doing it: without it the part is swallowed.
    clusters, _ = merge_tracks(_tracks((body, 0, 0), (hand, 0, 1)), MergeConfig(dup_size_ratio=0))
    assert clusters == [[0, 1]]


def test_labels_never_merge() -> None:
    a = _tracks((_box(10, 50, 10, 30), 0, 0), label="person")
    b = _tracks((_box(10, 50, 10, 30), 1, 0), label="dog")
    clusters, _ = merge_tracks(a + b, CFG)
    assert clusters == [[0], [1]]


def test_output_order_primary_ids_first_then_first_frame() -> None:
    late = _box(5, 15, 60, 70, range(6, 12))  # secondary session only, appears at 6
    early = _box(5, 15, 40, 50, range(1, 12))  # secondary session only, appears at 1
    p1 = _box(30, 50, 10, 20)
    p0 = _box(30, 50, 25, 35)
    tracks = _tracks((late, 1, 0), (early, 2, 0), (p1, 0, 1), (p0, 0, 0))
    clusters, _ = merge_tracks(tracks, CFG)
    assert clusters == [[3], [2], [1], [0]]


def test_assembly_prefers_a_bigger_member_that_contains_the_primary_one() -> None:
    torso = _box(20, 40, 10, 30)  # primary saw only the torso
    full = _box(10, 50, 10, 30)  # a later seed saw the whole person
    tracks = _tracks((torso, 0, 0), (full, 1, 0))
    clusters, _ = merge_tracks(tracks, CFG)
    assert clusters == [[0, 1]]
    stack = sam31_merge.assemble(tracks, clusters[0], CFG)
    assert stack[5, 12, 15] == 1  # the head, from the whole
