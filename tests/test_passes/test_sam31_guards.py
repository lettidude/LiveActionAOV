"""SAM 3.1 guards: the over-merge guard of the track merge, the free-VRAM
preflight before several sessions, and the card-size gate of `detector`
"auto". Synthetic masks and fakes only (no model, no GPU)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from live_action_aov.passes.matte import sam3 as sam3mod
from live_action_aov.passes.matte import sam31_bridge, sam31_merge
from live_action_aov.passes.matte.sam3 import SAM3MattePass
from live_action_aov.passes.matte.sam31_merge import MergeConfig, make_track, merge_tracks

N, H, W = 12, 60, 80
GUARD = MergeConfig(split_guard=0.25)


def _box(y0: int, y1: int, x0: int, x1: int, frames: range | None = None) -> np.ndarray:
    st = np.zeros((N, H, W), np.uint8)
    for f in frames if frames is not None else range(N):
        st[f, y0:y1, x0:x1] = 1
    return st


def _tracks(*specs: tuple[np.ndarray, int, int], cfg: MergeConfig = GUARD) -> list:
    out = []
    for st, session, oid in specs:
        t = make_track(st, "person", session, session, oid, lambda st=st: st, cfg)
        assert t is not None
        out.append(t)
    return out


def _two_people_late() -> tuple[np.ndarray, np.ndarray]:
    """A: the left person. B (another seed): the left person, but on the
    last 3 frames it reads the left AND the right person as one object."""
    a = _box(10, 50, 10, 30)
    b = _box(10, 50, 10, 30)
    b[9:, 10:50, 50:70] = 1
    return a, b


def test_large_blobs_counts_separate_big_parts_only() -> None:
    m = np.zeros((40, 60), bool)
    m[5:25, 5:15] = True  # 200 px
    m[5:25, 40:50] = True  # 200 px, apart
    m[35, 30] = True  # a speck
    c = sam31_merge.crop_of(m)
    assert c is not None
    assert sam31_merge.large_blobs(c, 0.15 * c.area) == 2
    assert sam31_merge.large_blobs(c, 0.9 * c.area) == 0


def test_without_the_guard_one_id_takes_two_people() -> None:
    a, b = _two_people_late()
    tracks = _tracks((a, 0, 0), (b, 1, 0), cfg=MergeConfig())
    clusters, _ = merge_tracks(tracks, MergeConfig())
    assert clusters == [[0, 1]]
    stack = sam31_merge.assemble(tracks, clusters[0], MergeConfig())
    assert stack[10, 30, 60] == 1  # the right person ends up in the left person's ID


def test_the_guard_refuses_a_link_that_splits_into_two_people() -> None:
    a, b = _two_people_late()
    clusters, rep = merge_tracks(_tracks((a, 0, 0), (b, 1, 0)), GUARD)
    assert clusters == [[0], [1]] and rep.refused_split >= 1 and rep.linked == 0
    # ...and on fewer split frames than asked for, it lets it through.
    clusters, _ = merge_tracks(
        _tracks((a, 0, 0), (b, 1, 0)), MergeConfig(split_guard=0.15, split_frames=4)
    )
    assert clusters == [[0, 1]]


def test_the_guard_keeps_a_real_link_and_a_person_cut_by_a_pole() -> None:
    a = _box(10, 50, 10, 30, range(0, 9))
    b = _box(11, 50, 10, 31, range(3, 12))
    clusters, rep = merge_tracks(_tracks((a, 0, 0), (b, 1, 5)), GUARD)
    assert clusters == [[0, 1]] and rep.refused_split == 0
    pole_a = _box(10, 50, 10, 40)
    pole_a[:, :, 24:27] = 0  # a pole cuts the person in two, in both tracks
    pole_b = pole_a.copy()
    pole_b[:, 10:12, :] = 0
    clusters, rep = merge_tracks(_tracks((pole_a, 0, 0), (pole_b, 1, 0)), GUARD)
    assert clusters == [[0, 1]] and rep.refused_split == 0


def test_the_guard_keeps_a_link_where_one_track_has_a_loose_limb() -> None:
    """Another session's track of the same person also holds a hand that
    comes loose from the body: one small extra blob, not a second person."""
    a = _box(10, 50, 10, 30)
    b = _box(10, 50, 10, 30)
    b[:, 30:36, 33:37] = 1  # a hand, apart from the body, only in b
    clusters, rep = merge_tracks(_tracks((a, 0, 0), (b, 1, 0)), GUARD)
    assert clusters == [[0, 1]] and rep.refused_split == 0


def test_the_guard_also_stops_a_duplicate_merge() -> None:
    whole = _box(10, 50, 10, 30)
    dup = _box(12, 48, 11, 29)
    dup[9:, 12:48, 50:60] = 1  # on 3 frames it also takes someone else
    clusters, rep = merge_tracks(_tracks((whole, 0, 0), (dup, 0, 1)), GUARD)
    assert clusters == [[0], [1]] and rep.duplicates == 0 and rep.refused_split >= 1


def test_the_pass_turns_the_guard_on_for_max_only_through_its_config() -> None:
    assert SAM3MattePass({})._sam31_merge_config().split_guard == 0.25
    assert SAM3MattePass({"sam31_split_guard": 0})._sam31_merge_config().split_guard == 0.0
    assert MergeConfig().split_guard == 0.0  # the module alone: off


# --- free-VRAM preflight -------------------------------------------------------


def test_preflight_warns_when_the_gpu_is_short(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sam3mod, "_free_vram_gb", lambda: (12.0, 31.8))
    with caplog.at_level(logging.WARNING):
        SAM3MattePass({"small_objects": "max"})._sam31_vram_preflight(3)
    assert "12.0 GB free" in caplog.text and "small_objects auto" in caplog.text


def test_preflight_is_quiet_with_room_one_session_or_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(sam3mod, "_free_vram_gb", lambda: (30.5, 31.8))
    with caplog.at_level(logging.WARNING):
        SAM3MattePass({})._sam31_vram_preflight(3)
    monkeypatch.setattr(sam3mod, "_free_vram_gb", lambda: (5.0, 31.8))
    with caplog.at_level(logging.WARNING):
        SAM3MattePass({})._sam31_vram_preflight(1)  # the default single session
        SAM3MattePass({"sam31_max_vram_check": "off"})._sam31_vram_preflight(3)
    monkeypatch.setattr(sam3mod, "_free_vram_gb", lambda: None)
    with caplog.at_level(logging.WARNING):
        SAM3MattePass({})._sam31_vram_preflight(3)  # no card info: nothing to say
    assert "GB free" not in caplog.text


def test_preflight_refuse_stops_sam31(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sam3mod, "_free_vram_gb", lambda: (12.0, 31.8))
    with pytest.raises(RuntimeError, match="GB free"):
        SAM3MattePass({"sam31_max_vram_check": "refuse"})._sam31_vram_preflight(3)


# --- card-size gate --------------------------------------------------------------


@pytest.mark.parametrize(("vram", "want"), [(31.8, "sam3.1"), (23.7, "sam3"), (16.0, "sam3")])
def test_auto_picks_sam31_only_on_a_card_of_at_least_24_gb(
    monkeypatch: pytest.MonkeyPatch, vram: float, want: str
) -> None:
    monkeypatch.setattr(sam31_bridge, "available", lambda p: (True, "py312"))
    monkeypatch.setattr(sam3mod, "_total_vram_gb", lambda: vram)
    got, why = SAM3MattePass({"detector": "auto"})._resolve_detector()
    assert got == want
    if want == "sam3":
        assert "24 GB card" in why
