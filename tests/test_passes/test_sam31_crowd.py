"""SAM 3.1 crowd mode (`sam31_tiles` "on"): tile absorb, swap guard, the
tile sessions' own cap, dense-tile split, tile work size and the run-report
line, on synthetic masks (the SAM 3.1 environment is replaced by fakes)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.passes.matte import sam31_bridge
from live_action_aov.passes.matte.boxmask import BoxMask
from live_action_aov.passes.matte.sam3 import CROWD_MODE_WARNING, SAM3MattePass, _crowd_summary
from live_action_aov.passes.matte.sam31_bridge import Sam31Track
from live_action_aov.passes.matte.sam31_crowd import absorb_tile_tracks, swap_guard

from .test_sam31_small import _load_worker, _run_worker

N, H, W = 20, 120, 200


def _box(y0: int, y1: int, x0: int, x1: int) -> BoxMask:
    m = np.zeros((H, W), np.uint8)
    m[y0:y1, x0:x1] = 1
    return BoxMask.from_dense(m)


def _track(boxes: dict[int, tuple[int, int, int, int]], src: int = 0, label: str = "person"):  # type: ignore[no-untyped-def]
    masks = [_box(*boxes[k]) if k in boxes else BoxMask((H, W)) for k in range(N)]
    return Sam31Track(label, len(boxes), src, 0, masks)


def _const(y0: int, y1: int, x0: int, x1: int, src: int = 0, label: str = "person"):  # type: ignore[no-untyped-def]
    return _track(dict.fromkeys(range(N), (y0, y1, x0, x1)), src, label)


# Two tiles side by side on the plate (= the frames here), overlapping 80-120.
TILES = [(0, 0, 120, 120), (0, 80, 120, 200)]


# --- tile absorb ---------------------------------------------------------------


def test_absorb_takes_the_pieces_of_a_big_person_cut_by_a_tile() -> None:
    big = _const(10, 110, 60, 140)  # whole-frame track over the tile seam
    left = _const(10, 110, 60, 120, src=1)  # tile 1 cut at x 120 (inner edge)
    right = _const(10, 110, 80, 140, src=2)  # tile 2 cut at x 80
    kept, rep = absorb_tile_tracks([big, left, right], TILES, (H, W))
    assert [t.src for t in kept] == [0]
    assert rep.absorbed == 2 and rep.tile_tracks == 2 and rep.pairs == [(1, 0), (2, 0)]


def test_absorb_keeps_small_people_next_to_or_inside_a_group() -> None:
    group = _const(10, 60, 10, 70)  # a whole-frame "group" mask over a few heads
    head_in = _const(20, 30, 20, 30, src=1)  # inside it, away from the tile edges, small
    head_out = _const(80, 90, 20, 30, src=1)  # outside it
    other = _const(20, 30, 40, 50, src=1, label="dog")  # another label
    kept, rep = absorb_tile_tracks([group, head_in, head_out, other], TILES, (H, W))
    assert len(kept) == 4 and rep.absorbed == 0


def test_absorb_takes_a_whole_copy_found_inside_one_tile() -> None:
    big = _const(10, 100, 20, 60)
    copy = _const(11, 100, 20, 60, src=1)  # same person seen whole in tile 1
    kept, rep = absorb_tile_tracks([big, copy], TILES, (H, W))
    assert len(kept) == 1 and rep.absorbed == 1


def test_absorb_off_or_without_tiles_changes_nothing() -> None:
    big = _const(10, 110, 60, 140)
    piece = _const(10, 110, 60, 120, src=1)
    assert absorb_tile_tracks([big, piece], TILES, (H, W), contain=0)[0] == [big, piece]
    assert absorb_tile_tracks([big, piece], [], (H, W))[0] == [big, piece]


def test_absorb_maps_plate_tiles_onto_proxy_frames() -> None:
    """Tiles on a 2x plate, tracks on the frames: the cut edge still found."""
    big = _const(10, 110, 60, 140)
    left = _const(10, 110, 60, 120, src=1)
    plate_tiles = [(0, 0, 240, 240), (0, 160, 240, 400)]
    kept, rep = absorb_tile_tracks([big, left], plate_tiles, (2 * H, 2 * W))
    assert rep.absorbed == 1


# --- swap guard -----------------------------------------------------------------


def test_swap_guard_gives_two_swapped_neighbours_their_ids_back() -> None:
    small, big = (20, 30, 20, 30), (20, 50, 150, 170)
    a = _track({k: small if k < 10 else big for k in range(N)})
    b = _track({k: big if k < 10 else small for k in range(N)})
    out, rep = swap_guard([a, b])
    assert rep.events == 2 and rep.splits == 2 and rep.rejoined == 2
    assert len(out) == 2
    for t in out:  # each output ID stays on one place the whole clip
        boxes = {t.masks[k].box for k in range(N)}
        assert len(boxes) == 1 and len(t.present()) == N


def test_swap_guard_drops_a_short_excursion() -> None:
    home, away = (60, 70, 40, 50), (60, 110, 100, 140)
    t = _track({k: away if k in (5, 6, 7) else home for k in range(N)})
    out, rep = swap_guard([t])
    assert rep.dropped_stretches == 1 and rep.dropped_frames == 3 and rep.splits == 0
    assert out[0].present() == [k for k in range(N) if k not in (5, 6, 7)]


def test_swap_guard_leaves_occlusion_and_walking_alone() -> None:
    # A person whose legs go behind something (area / 4, centroid inside the
    # old box) and one walking 3 px a frame: nothing flagged, same objects.
    hidden = _track({k: (20, 80, 100, 120) if k < 10 else (20, 35, 100, 120) for k in range(N)})
    walker = _track({k: (40, 90, 10 + 3 * k, 30 + 3 * k) for k in range(N)})
    tracks = [hidden, walker]
    out, rep = swap_guard(tracks)
    assert rep.events == 0 and out[0] is hidden and out[1] is walker


def test_swap_guard_off() -> None:
    small, big = (20, 30, 20, 30), (20, 50, 150, 170)
    a = _track({k: small if k < 10 else big for k in range(N)})
    out, rep = swap_guard([a], jump=0)
    assert out == [a] and rep.events == 0


# --- worker: tile cap, per-session stats, no primary ------------------------------


def test_worker_tile_sessions_get_their_own_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys
    import types

    caps: list[tuple[str, int]] = []
    built: dict = {}
    n = 4
    dirs = [tmp_path / "frames", tmp_path / "tile00"]
    for d in dirs:
        d.mkdir()
        for i in range(n):
            (d / f"{i:05d}.jpg").write_bytes(b"")

    class Pred:
        def __init__(self) -> None:
            self.model = type("M", (), {"max_num_objects": 0})()
            self.cur = ""

        def handle_request(self, req: dict) -> dict:
            if req["type"] == "start_session":
                self.cur = Path(req["resource_path"]).name
                caps.append((self.cur, self.model.max_num_objects))
                return {"session_id": "s"}
            if req["type"] == "add_prompt":
                k = 3 if self.cur == "tile00" else 2  # the tile is "dense"
                return {"outputs": {"out_obj_ids": list(range(k))}}
            return {}

        def handle_stream_request(self, req: dict):  # type: ignore[no-untyped-def]
            m = np.zeros((8, 8), np.uint8)
            m[1:3, 1:3] = 1
            for f in range(n):
                yield {"frame_index": f, "outputs": {"out_obj_ids": [0], "out_binary_masks": [m]}}

    pred = Pred()
    builder = types.ModuleType("sam3.model_builder")

    def build(**kw):  # type: ignore[no-untyped-def]
        built.update(kw)
        return pred

    builder.build_sam3_multiplex_video_predictor = build  # type: ignore[attr-defined]
    for name, mod in {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model.decoder": types.ModuleType("sam3.model.decoder"),
        "sam3.model_builder": builder,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    worker = _load_worker()
    monkeypatch.setattr(worker, "_shims", lambda: None)
    monkeypatch.setattr(worker, "_drop_unknown_kwargs", lambda p: None)
    out = tmp_path / "out"
    argv = ["--frames", str(dirs[0]), "--out", str(out), "--concepts", '["person"]']
    argv += ["--seed", "1", "--tiles", json.dumps([str(dirs[1])]), "--crops"]
    argv += ["--max-objects", "2", "--tile-max-objects", "3", "--free-between"]
    argv += ["--checkpoint", "fake.pt"]
    assert worker.main(argv) == 0
    assert built["max_num_objects"] == 3  # built with the larger cap
    assert caps == [("frames", 2), ("tile00", 3)]
    res = json.loads((out / "result.json").read_text())
    ss = res["sessions"]
    assert [(s["src"], s["cap"], s["cap_hit"]) for s in ss] == [(0, 2, True), (1, 3, True)]
    assert all("seconds" in s and "peak_gb" in s for s in ss)


def test_worker_without_tile_cap_keeps_the_model_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc, out, _ = _run_worker(monkeypatch, tmp_path, set())
    res = json.loads((out / "result.json").read_text())
    assert rc == 0 and all(s["cap"] == 0 and not s["cap_hit"] for s in res["sessions"])


# --- bridge: dense split, tile work size, info -------------------------------------


def _fake_worker(calls: list, dense_src: int | None):  # type: ignore[no-untyped-def]
    worker = _load_worker()

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir()
        tiles = json.loads(cmd[cmd.index("--tiles") + 1]) if "--tiles" in cmd else []
        frames = cmd[cmd.index("--frames") + 1]
        px = np.asarray(__import__("PIL.Image").Image.open(Path(tiles[0]) / "00000.jpg")).shape
        calls.append({"cmd": cmd, "tiles": tiles, "frames": frames, "tile_px": px})
        c = np.ones((4, 4), np.uint8)
        worker._save_crops(str(out / "a.npz"), dict.fromkeys(range(2), (2, 2, c, *px[:2])), 2)
        objs = [{"file": "a.npz", "concept": "person", "obj_id": 1, "src": 1, "seed": 0}]
        first = len(calls) == 1
        sessions = [
            {"src": s, "concept": "person", "seed": 0, "seeded": 1, "tracked": 1, "cap": 2}
            | {"cap_hit": first and s == dense_src, "seconds": 1.0, "peak_gb": 3.0}
            for s in range(1 + len(tiles))
        ]
        res = {"objects": objs, "sessions": sessions, "seconds": 5, "peak_vram_gb": 3.0}
        (out / "result.json").write_text(json.dumps(res))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def test_split_dense_tracks_a_capped_tile_again_as_smaller_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list = []
    monkeypatch.setattr(sam31_bridge.subprocess, "run", _fake_worker(calls, dense_src=2))
    frames = np.full((2, 60, 100, 3), 0.5, np.float32)
    tiles = sam31_bridge.tile_windows(60, 100, 2, 0.2)
    info: dict = {}
    got = sam31_bridge.run_sam31_small(
        frames,
        ["person"],
        [0],
        "py312",
        max_objects=2,
        tiles=tiles,
        split_dense=True,
        tile_max_objects=2,
        info=info,
    )
    assert len(calls) == 2
    assert "--no-primary" in calls[1]["cmd"] and "--no-primary" not in calls[0]["cmd"]
    assert len(calls[1]["tiles"]) == 3  # 2x2 sub-tiles of tile 2: one as --frames, 3 more
    assert len(info["tiles"]) == 4 + 4 and info["tiles"][:4] == tiles
    sub0 = info["tiles"][4]
    y0, x0, y1, x1 = tiles[1]
    assert sub0[:2] == (y0, x0) and sub0[2] < y1 and sub0[3] < x1
    # first run: object in tile 1; second run: its src 1 = sub-tile 2 (= tile 6)
    assert [t.src for t in got] == [1, 4 + 2]
    assert sum(1 for s in info["sessions"] if s.get("split")) == 4
    assert info["seconds"] == 10.0


def test_no_split_when_no_tile_hit_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(sam31_bridge.subprocess, "run", _fake_worker(calls, dense_src=None))
    frames = np.full((2, 60, 100, 3), 0.5, np.float32)
    tiles = sam31_bridge.tile_windows(60, 100, 2, 0.2)
    sam31_bridge.run_sam31_small(
        frames, ["person"], [0], "py312", max_objects=2, tiles=tiles, split_dense=True
    )
    assert len(calls) == 1


def test_tile_work_size_shrinks_tile_clips_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list = []
    monkeypatch.setattr(sam31_bridge.subprocess, "run", _fake_worker(calls, dense_src=None))
    frames = np.full((2, 60, 100, 3), 0.5, np.float32)
    tiles = sam31_bridge.tile_windows(60, 100, 2, 0.2)
    sam31_bridge.run_sam31_small(
        frames, ["person"], [0], "py312", tiles=tiles, tile_work_size=20, free_between=True
    )
    assert max(calls[0]["tile_px"][:2]) == 20
    assert "--free-between" in calls[0]["cmd"] and "--tile-max-objects" not in calls[0]["cmd"]
    calls.clear()
    sam31_bridge.run_sam31_small(frames, ["person"], [0], "py312", tiles=tiles)
    assert calls[0]["tile_px"][:2] == (tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1])
    assert "--free-between" not in calls[0]["cmd"]


# --- the pass: crowd mode wiring + report ----------------------------------------


def test_crowd_summary_line() -> None:
    from live_action_aov.passes.matte.sam31_crowd import SwapReport

    info = {
        "sessions": [
            {"src": 0, "cap": 64, "cap_hit": False, "seconds": 20.0},
            {"src": 1, "cap": 64, "cap_hit": True, "seconds": 40.0},
            {"src": 2, "empty": True, "seconds": 1.0},
            {"src": 3, "skipped": "oom", "seconds": 2.0},
        ],
        "seconds": 63.0,
        "peak_vram_gb": 21.5,
    }
    s = _crowd_summary(info, 3, 5, SwapReport(events=4))
    assert s["sessions"] == 4 and s["sessions_cap_hit"] == 1 and s["sessions_skipped"] == 1
    assert s["sessions_empty"] == 1 and s["tile_tracks_absorbed"] == 5
    assert "3 tiles, 4 sessions, 1 hit the per-session object cap (64)" in s["line"]
    assert "2 skipped or empty" in s["line"] and "swap guard 4 flagged" in s["line"]


def test_crowd_mode_is_off_by_default_and_not_part_of_max() -> None:
    p = SAM3MattePass({"small_objects": "max"})
    assert p.params["sam31_tiles"] == "off"
    assert "much longer" in CROWD_MODE_WARNING


class _Pass(SAM3MattePass):
    def _load_model(self) -> None:  # type: ignore[override]
        raise AssertionError("SAM 3 loaded")


def _run_pass(monkeypatch: pytest.MonkeyPatch, params: dict, found: list) -> tuple:  # type: ignore[type-arg]
    seen: dict = {}

    def fake_small(frames, concepts, seeds, python, **kw):  # type: ignore[no-untyped-def]
        seen.update(kw)
        if "info" in kw:
            kw["info"].update(
                {"sessions": [{"src": 0, "cap_hit": False}], "seconds": 3, "peak_vram_gb": 2.0}
            )
        return found

    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    monkeypatch.setattr(sam31_bridge, "run_sam31_small", fake_small)
    p = _Pass(params)
    frames = np.zeros((N, H, W, 3), np.float32)
    p._add_sam31_small(frames, ["person"], 0, 1)
    return p, seen


def test_pass_without_crowd_mode_sends_no_crowd_options(monkeypatch: pytest.MonkeyPatch) -> None:
    p, seen = _run_pass(monkeypatch, {"small_objects": "max"}, [_const(10, 110, 60, 140)])
    for k in ("tile_max_objects", "tile_work_size", "free_between", "split_dense", "info"):
        assert k not in seen
    assert p._crowd_mode == {} and "crowd_mode" not in p.report_fields()["extra"]


def test_pass_crowd_mode_absorbs_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    big = _const(10, 110, 60, 140)
    pieces = [_const(10, 110, 60, 110, src=1), _const(10, 110, 90, 140, src=2)]
    head = _const(80, 92, 10, 22, src=1)
    params = {"small_objects": "auto", "sam31_tiles": "on", "sam31_tile_max_objects": 128}
    p, seen = _run_pass(monkeypatch, params, [big, *pieces, head])
    assert seen["tile_max_objects"] == 128 and seen["tile_work_size"] == 1008
    assert seen["free_between"] is True and seen["split_dense"] is False
    assert len(p._instances) == 2  # the big person + the small head, no tile pieces
    rf = p.report_fields()
    cm = rf["extra"]["crowd_mode"]
    assert cm["tiles"] == 4 and cm["sessions"] == 1 and cm["tile_tracks_absorbed"] == 2
    assert any(w.startswith("Crowd mode (SAM 3.1 tiles):") for w in rf["warnings"])
