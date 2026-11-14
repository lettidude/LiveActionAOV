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
    assert SAM3MattePass({})._sam31_seeds(100, 50) == [50]  # default: the sample frame
    # "max" seeds several frames, the sample frame (the primary session) first.
    assert SAM3MattePass({"small_objects": "max"})._sam31_seeds(200, 100) == [
        100,
        0,
        199,
        40,
        80,
        120,
        160,
    ]
    multi = {"sam31_seed_frames": "multi"}
    assert SAM3MattePass(multi)._sam31_seeds(100, 50) == [50, 0, 99]
    every = {**multi, "sam31_seed_every": 30}
    assert SAM3MattePass(every)._sam31_seeds(100, 50) == [50, 0, 99, 30]
    single = {"small_objects": "max", "sam31_seed_frames": "single"}
    assert SAM3MattePass(single)._sam31_seeds(100, 50) == [50]


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
    p = _NoSAM3(
        {
            "detector": "sam3.1",
            "concepts": ["person"],
            "layer_order": "area",
            "sam31_seed_frames": "multi",
        }
    )
    assert p.params["small_objects"] == "auto"
    frames = [np.full((H, W, 3), 0.5, np.float32) for _ in range(N)]
    p.run_shot(_Reader(frames), frame_range=(1, N))
    assert calls["seeds"] == [N // 2, 0, N - 1] and calls["max_objects"] == 64
    assert calls["tiles"] == [] and calls["work_size"] == 1024  # tiles: opt-in only
    assert calls["work_above"] == 1920
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
    p = _NoSAM3(
        {"detector": "sam3.1", "concepts": ["person"], "layer_order": "area", "sam31_tiles": "on"}
    )
    monkeypatch.setattr(SAM3MattePass, "_full_hw", lambda self: (1200, 2000))
    p.run_shot(_ProxyReader([np.full((H, W, 3), 0.5, np.float32)] * N), frame_range=(1, N))
    assert calls["plate_hw"] == (1200, 2000) and len(calls["tiles"]) == 4
    assert calls["plate0"] == (1200, 2000, 3)


def test_work_size_shrinks_what_sam31_sees_and_masks_come_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frames 40 x 60 with `work_size` 30: the worker sees 20 x 30 JPEGs and
    its 20 x 30 masks come back on 40 x 60; a tile's mask goes back to its
    window before it is placed."""
    worker = _load_worker()
    seen: dict = {}

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        a = dict(zip(cmd[3::2], cmd[4::2], strict=False))
        dirs = [a["--frames"], *json.loads(cmd[cmd.index("--tiles") + 1])]
        img = __import__("PIL.Image").Image
        seen["px"] = [np.asarray(img.open(Path(d) / "00000.jpg")).shape[:2] for d in dirs]
        out = Path(a["--out"])
        out.mkdir()
        th, tw = seen["px"][1]
        worker._save_crops(str(out / "a.npz"), {0: (4, 6, np.ones((5, 2), np.uint8), 20, 30)}, 2)
        worker._save_crops(str(out / "b.npz"), {0: (0, 0, np.ones((2, 2), np.uint8), th, tw)}, 2)
        objs = [
            {"file": "a.npz", "concept": "person", "obj_id": 0, "src": 0, "seed": 0},
            {"file": "b.npz", "concept": "person", "obj_id": 1, "src": 1, "seed": 0},
        ]
        (out / "result.json").write_text(json.dumps({"objects": objs, "seconds": 1}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sam31_bridge.subprocess, "run", fake_run)
    frames = np.full((2, 40, 60, 3), 0.5, np.float32)
    tiles = sam31_bridge.tile_windows(40, 60, 2, 0.2)
    a, b = sam31_bridge.run_sam31_small(frames, ["person"], [0], "py312", tiles=tiles, work_size=30)
    assert seen["px"][0] == (20, 30) and max(seen["px"][1]) == 30
    assert a.masks[0].shape == (40, 60) and a.masks[0].box == (8, 12, 18, 16)
    ty, tx, ty1, tx1 = tiles[0]
    sy = (ty1 - ty) / seen["px"][1][0]
    assert b.masks[0].shape == (40, 60) and b.masks[0].box[:2] == (ty, tx)
    assert b.masks[0].box[2] - ty == pytest.approx(2 * sy, abs=1)


def test_work_size_zero_or_bigger_keeps_frames() -> None:
    img = np.zeros((40, 60, 3), np.float32)
    assert sam31_bridge._fit(img, 0) is img and sam31_bridge._fit(img, 60) is img
    assert sam31_bridge._fit(img, 30).shape == (20, 30, 3)


def test_work_size_leaves_input_up_to_the_threshold_untouched() -> None:
    """A 1920 proxy runs as is; only bigger clips (4K) are shrunk."""
    hd = np.zeros((108, 192, 3), np.float32)  # stands for 1920 x 1080
    big = np.zeros((216, 410, 3), np.float32)  # stands for a 4K plate
    assert sam31_bridge._fit(hd, 102, above=192) is hd
    assert sam31_bridge._fit(big, 102, above=192).shape[1] == 102
    assert sam31_bridge._fit(hd, 102, above=0).shape[1] == 102  # no threshold: as before


def test_small_floor_of_a_tile_track_is_relative_to_its_tile() -> None:
    """60 px is 0.25 % of the 120 x 200 frame (under a 0.4 % floor) but 0.6 %
    of a 100 x 100 tile; without the tile's area the frame's floor applies."""
    t = _track(10, 20, 10, 16, src=1)
    assert not keep_small(t, 5, 0.004, 3)
    assert keep_small(t, 5, 0.004, 3, seen_area=100 * 100)


# --- several sessions: skip guard, track merge, "max" ------------------------


class _FakePredictor:
    """Meta's multiplex predictor, enough for the worker: one object per
    session (a box moving one px per frame). `fail` = (folder, seed frame)
    pairs whose session raises like SAM 3.1's "No points are provided",
    during propagation."""

    def __init__(self, n: int, fail: set[tuple[str, int]]) -> None:
        self.n, self.fail = n, fail
        self.sessions: dict[str, tuple[str, int]] = {}
        self.closed: list[str] = []
        self.model = type("M", (), {"init_state": staticmethod(lambda **kw: None)})()

    def handle_request(self, req: dict) -> dict:
        if req["type"] == "start_session":
            sid = f"s{len(self.sessions)}"
            self.sessions[sid] = (Path(req["resource_path"]).name, -1)
            return {"session_id": sid}
        if req["type"] == "add_prompt":
            sid = req["session_id"]
            self.sessions[sid] = (self.sessions[sid][0], req["frame_index"])
            return {"outputs": {"out_obj_ids": [7]}}
        self.closed.append(req["session_id"])
        return {}

    def handle_stream_request(self, req: dict):  # type: ignore[no-untyped-def]
        if self.sessions[req["session_id"]] in self.fail:
            raise RuntimeError("No points are provided; please add points first")
        for f in range(self.n):
            m = np.zeros((16, 24), np.uint8)
            m[4:12, f : f + 6] = 1
            yield {"frame_index": f, "outputs": {"out_obj_ids": [7], "out_binary_masks": [m]}}


def _run_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: set[tuple[str, int]]
) -> tuple[int, Path, _FakePredictor]:
    import sys
    import types

    n = 6
    dirs = [tmp_path / "frames", tmp_path / "tile00"]
    for d in dirs:
        d.mkdir()
        for i in range(n):
            (d / f"{i:05d}.jpg").write_bytes(b"")
    pred = _FakePredictor(n, fail)
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_multiplex_video_predictor = lambda **kw: pred  # type: ignore[attr-defined]
    for name, mod in {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model.decoder": types.ModuleType("sam3.model.decoder"),
        "sam3.model_builder": builder,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    worker = _load_worker()
    monkeypatch.setattr(worker, "_shims", lambda: None)
    out = tmp_path / "out"
    argv = ["--frames", str(dirs[0]), "--out", str(out), "--concepts", json.dumps(["person"])]
    argv += ["--seed", "3", "--seeds", "[3, 0, 5]", "--tiles", json.dumps([str(dirs[1])])]
    argv += ["--crops", "--checkpoint", "fake.pt"]
    return worker.main(argv), out, pred


def test_worker_skips_a_failing_tile_or_secondary_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc, out, pred = _run_worker(monkeypatch, tmp_path, {("tile00", 3), ("frames", 0)})
    res = json.loads((out / "result.json").read_text())
    assert rc == 0 and len(pred.closed) == 6  # failed sessions are closed too
    skipped = [(s["src"], s["seed"]) for s in res["sessions"] if "skipped" in s]
    assert skipped == [(0, 0), (1, 3)]
    got = [(o["src"], o["seed"], o["session"]) for o in res["objects"]]
    assert got == [(0, 3, 0), (0, 5, 2), (1, 0, 4), (1, 5, 5)]  # the primary is session 0


def test_worker_primary_session_failure_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="No points"):
        _run_worker(monkeypatch, tmp_path, {("frames", 3)})


def test_bridge_logs_a_skipped_session(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir()
        skip = {"src": 1, "concept": "person", "seed": 0, "skipped": "No points"}
        (out / "result.json").write_text(json.dumps({"objects": [], "sessions": [skip]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(sam31_bridge.subprocess, "run", fake_run)
    frames = np.zeros((2, 8, 8, 3), np.float32)
    with caplog.at_level("WARNING"):
        assert sam31_bridge.run_sam31_small(frames, ["person"], [1], "py") == []
    assert "skipped" in caplog.text and "No points" in caplog.text


def _session(t: Sam31Track, session: int) -> Sam31Track:
    return Sam31Track(t.label, t.obj_id, t.src, t.seed, t.masks, session)


def test_merge_sessions_links_one_object_across_sessions_and_keeps_its_frames() -> None:
    from live_action_aov.passes.matte.sam31_merge import MergeConfig

    a = _session(_track(10, 110, 60, 100, frames=range(0, 12)), 0)  # primary loses it at 12
    b = _session(_track(11, 110, 60, 101, frames=range(6, N), seed=N - 1), 1)
    other = _session(_track(10, 60, 150, 170), 0)
    merged, rep, frames = sam31_bridge.merge_sessions([a, b, other], MergeConfig())
    assert len(merged) == 2 and rep.linked == 1
    assert merged[0].present() == list(range(N)) and frames[0] == (N, 0, N)
    assert merged[0].masks[3].box == a.masks[3].box  # the primary's mask where both exist


def test_merge_sessions_keeps_a_thin_mask_frame_on_a_wide_plate() -> None:
    """A person seen through a narrow gap (2 px wide on a 1920 px plate) is
    under one px in the 480 px comparison copy: the frame must still count."""
    from live_action_aov.passes.matte.sam31_merge import MergeConfig

    n, h, w = 6, 270, 1920
    masks = []
    for k in range(n):
        m = np.zeros((h, w), np.uint8)
        m[50:250, 901 : 903 if k < 3 else 960] = 1
        masks.append(BoxMask.from_dense(m))
    track = Sam31Track("person", 0, 0, 0, masks)
    merged, _, frames = sam31_bridge.merge_sessions([track], MergeConfig())
    assert merged[0].present() == list(range(n)) and frames == [(n, 0, n)]


def test_a_person_walking_up_to_the_camera_is_not_a_glitch() -> None:
    from live_action_aov.passes.matte import sam31_merge

    cov = np.r_[np.full(12, 0.002), np.full(8, 0.2)]  # far away, then close
    assert sam31_merge.glitch_frames(cov, 6.0, 0.05, 8) == []
    assert sam31_merge.glitch_frames(cov, 6.0, 0.05, 0) == list(range(12, 20))  # track median
    spike = np.full(20, 0.002)
    spike[9] = 0.2  # one frame read as a big scattered mask
    assert sam31_merge.glitch_frames(spike, 6.0, 0.05, 8) == [9]


def test_dense_track_keeps_a_thin_mask_frame() -> None:
    from live_action_aov.passes.matte.sam31_merge import MergeConfig, make_track

    st = np.zeros((4, 270, 1920), np.uint8)
    st[:, 50:250, 901:903] = 1
    t = make_track(st, "person", 0, 0, 1, lambda: st, MergeConfig())
    assert t is not None and sorted(t.small) == [0, 1, 2, 3]


def test_pass_max_seeds_several_frames_without_tiles_and_merges_tracks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}
    hero = _session(_track(10, 110, 60, 100, frames=range(0, 12)), 0)
    hero_late = _session(_track(11, 110, 60, 101, frames=range(6, N), seed=N - 1), 1)
    late = _session(_track(40, 52, 20, 25, frames=range(14, N), seed=N - 1), 1)  # enters late

    def fake_small(frames, concepts, seeds, python, **kw):  # type: ignore[no-untyped-def]
        calls.update(seeds=seeds, **kw)
        return [hero, hero_late, late]

    monkeypatch.setattr(sam31_bridge, "run_sam31_small", fake_small)
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    params = {"detector": "sam3.1", "concepts": ["person"], "layer_order": "area"}
    p = _NoSAM3({**params, "small_objects": "max"})
    frames = [np.full((H, W, 3), 0.5, np.float32) for _ in range(N)]
    p.run_shot(_Reader(frames), frame_range=(1, N))
    assert calls["seeds"] == [N // 2, 0, N - 1] and calls["tiles"] == []
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    held = sorted(int(d["stack"].reshape(N, -1).any(1).sum()) for d in hard.values())
    assert held == [N - 14, N]  # the hero holds the whole clip, the late entrant is kept


def test_glitch_rule_defaults_to_the_whole_track_median() -> None:
    assert SAM3MattePass({})._sam31_merge_config().glitch_window == 0
