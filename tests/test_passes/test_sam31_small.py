"""SAM 3.1 with `small_objects` on: object cap, several seed frames, tiles,
the relative floor and the duplicate merge, on synthetic masks (the SAM 3.1
environment is replaced by fakes)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.passes.matte import sam31_bridge
from live_action_aov.passes.matte.boxmask import BoxMask
from live_action_aov.passes.matte.sam3 import SAM3MattePass
from live_action_aov.passes.matte.sam31_bridge import Sam31Track, keep_small, merge_duplicates

N, H, W = 20, 120, 200


def _track(y0: int, y1: int, x0: int, x1: int, frames=range(N), label="person", src=0, seed=0):  # type: ignore[no-untyped-def]
    masks = []
    for k in range(N):
        if k in frames:
            m = np.zeros((H, W), np.uint8)
            m[y0:y1, x0:x1] = 1
            masks.append(BoxMask.from_dense(m))
        else:
            masks.append(BoxMask((H, W)))
    return Sam31Track(label, 0, src, seed, masks)


# --- duplicate merge ----------------------------------------------------------


def test_merge_joins_one_person_found_by_two_seeds() -> None:
    """Lost at frame 12 by the first session, found again by the last-frame
    seed: one track over the whole clip."""
    a = _track(20, 80, 50, 70, frames=range(0, 12))
    b = _track(21, 80, 50, 71, frames=range(8, N), seed=N - 1)
    kept, merged = merge_duplicates([a, b])
    assert len(kept) == 1 and len(merged) == 1
    assert all(m.crop.size for m in kept[0].masks)  # frames 0..19 all covered


def test_merge_keeps_a_tight_crowd_apart() -> None:
    """The negative case: a row of people shoulder to shoulder (neighbours
    overlap a little), a "group" mask over four of them and one real
    duplicate. Only the duplicate goes, the group is not collapsed."""
    people = [_track(30, 90, 10 + 14 * i, 26 + 14 * i) for i in range(13)]  # 2 px overlap
    group = _track(28, 92, 10, 72)  # covers people 0..3
    dup = _track(31, 90, 10 + 14 * 6, 26 + 14 * 6, frames=range(3, N), src=2)
    kept, merged = merge_duplicates([*people, group, dup])
    assert len(kept) == 15 - 1
    assert len(merged) == 1 and merged[0][0] == 14  # the duplicate, nothing else


def test_merge_does_not_chain() -> None:
    """A~B and B~C but not A~C: B goes into A, C stays (a merge that grew
    with each union would swallow C too)."""
    a = _track(20, 80, 40, 80)
    b = _track(20, 80, 52, 92)  # 70 % inside a
    c = _track(20, 80, 64, 104)  # 70 % inside b, 40 % inside a
    kept, merged = merge_duplicates([a, b, c])
    assert len(kept) == 2 and merged == [(1, 0)]


def test_merge_never_crosses_labels_or_brief_overlaps() -> None:
    hand = _track(40, 60, 50, 66, label="hand")
    person = _track(40, 62, 50, 68)
    # Two people crossing: overlap on 3 of 20 shared frames only.
    p1 = _track(20, 80, 120, 140)
    p2 = _track(20, 80, 120, 140, frames=range(0, 3))
    p2.masks[3:] = _track(20, 80, 170, 190, frames=range(3, N)).masks[3:]
    kept, _ = merge_duplicates([hand, person, p1, p2])
    assert len(kept) == 4


# --- floor --------------------------------------------------------------------


def test_small_floor_keeps_small_people_and_drops_specks() -> None:
    person = _track(40, 52, 100, 105)  # 12 px tall, 0.25 % of frame
    assert keep_small(person, 10, 0.0003, 3)
    speck = _track(40, 43, 100, 103)  # 3 px speck
    assert not keep_small(speck, 10, 0.0003, 3)
    brief = _track(40, 60, 100, 110, frames=range(0, 2))
    assert not keep_small(brief, 10, 0.0003, 3)
    # Today's floor would drop the 12 px person (0.25 % < 0.5 %).
    stack = np.stack([m.dense() for m in person.masks])
    assert not sam31_bridge.keep_object(stack, 0.005, 3)


def test_seed_frames() -> None:
    p = SAM3MattePass({})
    assert p._sam31_seeds(100, 50) == [0, 50, 99]
    assert SAM3MattePass({"sam31_seed_every": 30})._sam31_seeds(100, 50) == [0, 30, 50, 60, 90, 99]
    assert SAM3MattePass({"sam31_seed_frames": "single"})._sam31_seeds(100, 50) == [50]


def test_tile_windows_cover_the_plate_with_overlap() -> None:
    win = sam31_bridge.tile_windows(1000, 2000, 2, 0.2)
    assert len(win) == 4 and win[0][:2] == (0, 0)
    assert max(w[2] for w in win) == 1000 and max(w[3] for w in win) == 2000
    assert win[1][1] < win[0][3]  # neighbours overlap
    assert sam31_bridge.tile_windows(1000, 2000, 1, 0.2) == []


# --- bridge: worker command, crops read back, tiles placed -------------------


def _load_worker():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("sam31_worker_t", sam31_bridge.WORKER)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_run_small_tiles_come_back_on_the_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proxy frames (40 x 60) of a 2x plate; one object in the whole frame,
    one in tile 2 at its local (5, 6): placed on the plate, then halved."""
    worker = _load_worker()
    seen: dict = {}

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        a = dict(zip(cmd[3::2], cmd[4::2], strict=False))
        tiles = json.loads(cmd[cmd.index("--tiles") + 1])
        seen.update(cmd=cmd, seeds=json.loads(a["--seeds"]), tiles=tiles)
        seen["tile_px"] = [
            np.asarray(__import__("PIL.Image").Image.open(Path(t) / "00000.jpg")).shape
            for t in tiles
        ]
        out = Path(a["--out"])
        out.mkdir()
        c = np.ones((10, 4), np.uint8)
        worker._save_crops(str(out / "a.npz"), dict.fromkeys(range(3), (8, 12, c, 40, 60)), 3)
        worker._save_crops(str(out / "b.npz"), {1: (5, 6, np.ones((20, 8), np.uint8), 48, 72)}, 3)
        objs = [
            {"file": "a.npz", "concept": "person", "obj_id": 0, "src": 0, "seed": 0},
            {"file": "b.npz", "concept": "person", "obj_id": 3, "src": 2, "seed": 2},
        ]
        (out / "result.json").write_text(json.dumps({"objects": objs, "seconds": 1}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sam31_bridge.subprocess, "run", fake_run)
    frames = np.full((3, 40, 60, 3), 0.5, np.float32)
    plate = np.full((80, 120, 3), 0.25, np.float32)
    tiles = sam31_bridge.tile_windows(80, 120, 2, 0.2)
    got = sam31_bridge.run_sam31_small(
        frames,
        ["person"],
        [0, 1, 2],
        "py312",
        max_objects=64,
        tiles=tiles,
        plate_frame=lambda k: plate,
        plate_hw=(80, 120),
    )
    assert seen["seeds"] == [0, 1, 2] and "--crops" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--max-objects") + 1] == "64"
    assert seen["tile_px"][0][:2] == (tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1])
    a, b = got
    assert a.masks[0].box == (8, 12, 18, 16) and a.present() == [0, 1, 2]
    ty, tx = tiles[1][:2]  # src 2 = the second tile
    assert b.present() == [1] and b.masks[1].shape == (40, 60)
    assert b.masks[1].box[:2] == ((ty + 5) // 2, (tx + 6) // 2)


def test_run_small_without_cap_or_tiles_omits_those_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir()
        (out / "result.json").write_text(json.dumps({"objects": []}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sam31_bridge.subprocess, "run", fake_run)
    sam31_bridge.run_sam31_small(np.zeros((2, 8, 8, 3), np.float32), ["person"], [1], "py312")
    assert "--max-objects" not in seen["cmd"] and "--tiles" not in seen["cmd"]


# --- the matte pass ----------------------------------------------------------


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


class _NoSAM3(SAM3MattePass):
    def _load_model(self) -> None:  # type: ignore[override]
        raise AssertionError("SAM 3 loaded")


def test_pass_small_objects_keeps_small_people_and_merges(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict = {}
    small = _track(40, 52, 20, 25)  # 12 px, 0.25 %: dropped by the old floor
    hero = _track(10, 110, 60, 100)
    hero_again = _track(11, 110, 60, 101, frames=range(5, N), seed=N - 1)
    speck = _track(5, 7, 150, 152)

    def fake_small(frames, concepts, seeds, python, **kw):  # type: ignore[no-untyped-def]
        calls.update(seeds=seeds, **kw)
        return [small, hero, hero_again, speck]

    def no_default(*a, **kw):  # type: ignore[no-untyped-def]
        raise AssertionError("the small_objects run must not call run_sam31")

    monkeypatch.setattr(sam31_bridge, "run_sam31_small", fake_small)
    monkeypatch.setattr(sam31_bridge, "run_sam31", no_default)
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    p = _NoSAM3({"detector": "sam3.1", "concepts": ["person"], "layer_order": "area"})
    assert p.params["small_objects"] == "auto"
    frames = [np.full((H, W, 3), 0.5, np.float32) for _ in range(N)]
    p.run_shot(_Reader(frames), frame_range=(1, N))
    assert calls["seeds"] == [0, N // 2, N - 1] and calls["max_objects"] == 64
    assert calls["tiles"] == []  # 200 px frames: no tiles under "auto"
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    assert len(hard) == 2  # small person + hero; duplicate merged, speck dropped
    heights = sorted(int(np.flatnonzero(d["stack"][10].any(1)).size) for d in hard.values())
    assert heights == [12, 100]


def test_pass_small_objects_off_is_the_old_run(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_small(*a, **kw):  # type: ignore[no-untyped-def]
        raise AssertionError("small_objects off must not call run_sam31_small")

    stack = np.zeros((N, H, W), np.uint8)
    stack[:, 10:110, 60:100] = 1
    monkeypatch.setattr(sam31_bridge, "run_sam31_small", no_small)
    monkeypatch.setattr(
        sam31_bridge, "run_sam31", lambda *a, **kw: [sam31_bridge.Sam31Object("person", 0, stack)]
    )
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    p = _NoSAM3(
        {
            "detector": "sam3.1",
            "concepts": ["person"],
            "layer_order": "area",
            "small_objects": "off",
        }
    )
    p.run_shot(_Reader([np.full((H, W, 3), 0.5, np.float32)] * N), frame_range=(1, N))
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    assert len(hard) == 1


def test_pass_proxy_tiles_read_the_plate(monkeypatch: pytest.MonkeyPatch) -> None:
    """1K-style proxy of a big plate: tiles are cut from the full-res plate."""
    calls: dict = {}

    class _ProxyReader(_Reader):
        _base = type("B", (), {"full_res_reader": None})()

        def read_frame_full_res(self, idx: int) -> tuple[np.ndarray, dict]:
            calls.setdefault("full", []).append(idx)
            return np.full((1200, 2000, 3), 0.5, np.float32), {}

    def fake_small(frames, concepts, seeds, python, **kw):  # type: ignore[no-untyped-def]
        calls.update(kw)
        calls["plate0"] = kw["plate_frame"](0).shape if kw["plate_frame"] else None
        return []

    monkeypatch.setattr(sam31_bridge, "run_sam31_small", fake_small)
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    p = _NoSAM3({"detector": "sam3.1", "concepts": ["person"], "layer_order": "area"})
    monkeypatch.setattr(SAM3MattePass, "_full_hw", lambda self: (1200, 2000))
    p.run_shot(_ProxyReader([np.full((H, W, 3), 0.5, np.float32)] * N), frame_range=(1, N))
    assert calls["plate_hw"] == (1200, 2000) and len(calls["tiles"]) == 4
    assert calls["plate0"] == (1200, 2000, 3)
