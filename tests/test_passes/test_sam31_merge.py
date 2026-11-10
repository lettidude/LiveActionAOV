"""SAM 3.1 small-objects mode: seed frames, the track merge and its guards,
the size floor, and the worker's multi-seed sessions, all on synthetic
masks with SAM 3.1 replaced by fakes (no model, no GPU)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.passes.matte import sam31_bridge, sam31_merge
from live_action_aov.passes.matte.sam3 import SAM3MattePass
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


# --- bridge --------------------------------------------------------------------


def _write_worker_out(
    out: Path, objs: list[tuple[str, int, int, int, np.ndarray]], **extra
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for concept, si, seed, oid, st in objs:
        name = f"obj_00_s{si:02d}_{oid:04d}.npz"
        np.savez_compressed(out / name, mask=st)
        entries.append(
            {"file": name, "concept": concept, "obj_id": oid, "session": si, "seed": seed}
        )
    (out / "result.json").write_text(json.dumps({"seed": 6, "objects": entries, **extra}))


def test_merge_results_logs_a_skipped_session_and_merges(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    a = _box(10, 50, 10, 30, range(0, 9))
    b = _box(11, 50, 10, 31, range(3, 12))
    sessions = [
        {"concept": "person", "seed": 6, "seeded": 1, "tracked": 1},
        {"concept": "person", "seed": 0, "skipped": "No points are provided"},
        {"concept": "person", "seed": 11, "seeded": 1, "tracked": 1},
    ]
    _write_worker_out(
        tmp_path,
        [("person", 0, 6, 0, a), ("person", 2, 11, 3, b)],
        seeds=[6, 0, 11],
        sessions=sessions,
    )
    with caplog.at_level("INFO"):
        objs = sam31_bridge.merge_results(tmp_path, CFG)
    assert "skipped" in caplog.text and "No points are provided" in caplog.text
    assert (
        len(objs) == 1 and objs[0].stack.shape == (N, H, W) and objs[0].stack.any(axis=(1, 2)).all()
    )


def test_merge_results_empty_is_an_explicit_outcome(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_worker_out(tmp_path, [], seeds=[6, 0, 11], sessions=[])
    with caplog.at_level("WARNING"):
        assert sam31_bridge.merge_results(tmp_path, CFG) == []
    assert "no objects found in 3 session(s)" in caplog.text


def _fake_worker_run(seen: dict):  # type: ignore[no-untyped-def]
    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        seen["cmd"] = cmd
        a = dict(zip(cmd[3::2], cmd[4::2], strict=False))
        st = np.zeros((4, 24, 32), np.uint8)
        st[:, 2:20, 4:12] = 1
        if "--extra-seeds" in cmd:
            _write_worker_out(Path(a["--out"]), [("person", 0, 2, 0, st), ("person", 1, 0, 0, st)])
        else:
            out = Path(a["--out"])
            out.mkdir()
            np.savez_compressed(out / "obj_00_0000.npz", mask=st)
            (out / "result.json").write_text(
                json.dumps(
                    {"objects": [{"file": "obj_00_0000.npz", "concept": "person", "obj_id": 0}]}
                )
            )
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return fake_run


def test_run_sam31_single_seed_command_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}
    monkeypatch.setattr(sam31_bridge.subprocess, "run", _fake_worker_run(seen))
    objs = sam31_bridge.run_sam31(np.zeros((4, 24, 32, 3), np.float32), ["person"], 2, "py312")
    assert "--extra-seeds" not in seen["cmd"] and len(objs) == 1
    assert seen["cmd"][3:] == [
        "--frames",
        seen["cmd"][4],
        "--out",
        seen["cmd"][6],
        "--concepts",
        '["person"]',
        "--seed",
        "2",
    ]


def test_run_sam31_small_objects_seeds_more_frames_and_merges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}
    monkeypatch.setattr(sam31_bridge.subprocess, "run", _fake_worker_run(seen))
    objs = sam31_bridge.run_sam31(
        np.zeros((4, 24, 32, 3), np.float32), ["person"], 2, "py312", extra_seeds=[0, 3], merge=CFG
    )
    i = seen["cmd"].index("--extra-seeds")
    assert json.loads(seen["cmd"][i + 1]) == [0, 3]
    assert len(objs) == 1  # the same person from two sessions: one object


def test_peak_height() -> None:
    st = _box(10, 50, 10, 30, range(0, 3))
    st[5, 0:2, 0:2] = 1
    assert sam31_bridge.peak_height(st) == 40
    assert sam31_bridge.peak_height(np.zeros((2, 4, 4), np.uint8)) == 0


# --- matte pass wiring -----------------------------------------------------------


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


class _NoSAM3(SAM3MattePass):
    def _load_model(self) -> None:  # type: ignore[override]
        raise AssertionError("SAM 3 loaded")


def _run_pass(monkeypatch: pytest.MonkeyPatch, params: dict) -> tuple[dict, dict]:
    n, h, w = 5, 120, 160
    big = np.zeros((n, h, w), np.uint8)
    big[:, 20:100, 20:60] = 1  # 16.7 % of the frame
    small = np.zeros((n, h, w), np.uint8)
    small[:, 50:62, 100:104] = 1  # 48 px = 0.25 % of the frame, 12 px tall
    speck = np.zeros((n, h, w), np.uint8)
    speck[:, 5:7, 150:152] = 1  # 4 px: under both floors
    found = [
        sam31_bridge.Sam31Object("person", 0, big),
        sam31_bridge.Sam31Object("person", 1, small),
        sam31_bridge.Sam31Object("person", 2, speck),
    ]
    calls: dict = {}

    def fake(frames, concepts, seed, python, **kw):  # type: ignore[no-untyped-def]
        calls.update(seed=seed, kw=kw)
        return found

    monkeypatch.setattr(sam31_bridge, "run_sam31", fake)
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    p = _NoSAM3({"detector": "sam3.1", "concepts": ["person"], "layer_order": "area", **params})
    frames = [np.full((h, w, 3), 0.5, np.float32) for _ in range(n)]
    p.run_shot(_Reader(frames), frame_range=(1, n))
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    return calls, hard


@pytest.mark.parametrize("small_objects", ["off", "auto"])
def test_default_follow_keeps_the_single_seed_path(
    monkeypatch: pytest.MonkeyPatch, small_objects: str
) -> None:
    calls, hard = _run_pass(monkeypatch, {"small_objects": small_objects})
    assert set(calls["kw"]) == {"timeout_s", "checkpoint"}  # today's call, nothing added
    assert len(hard) == 1  # only the big one passes the 0.5 % floor


def test_small_objects_max_turns_on_multi_seed_and_the_low_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, hard = _run_pass(monkeypatch, {"small_objects": "max"})
    assert calls["seed"] == 2 and calls["kw"]["extra_seeds"] == [0, 4]
    assert isinstance(calls["kw"]["merge"], MergeConfig)
    assert len(hard) == 2  # the small person kept, the speck still dropped


def test_sam31_small_objects_forces_either_way(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, hard = _run_pass(monkeypatch, {"small_objects": "off", "sam31_small_objects": "on"})
    assert "merge" in calls["kw"] and len(hard) == 2
    calls, hard = _run_pass(monkeypatch, {"small_objects": "max", "sam31_small_objects": "off"})
    assert "merge" not in calls["kw"] and len(hard) == 1


def test_small_objects_height_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    _, hard = _run_pass(monkeypatch, {"sam31_small_objects": "on", "min_object_px": 20})
    assert len(hard) == 1  # the 12 px person is under a 20 px floor


# --- worker ----------------------------------------------------------------------


def _load_worker() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_sam31_worker_test", sam31_bridge.WORKER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakePredictor:
    """Meta's multiplex predictor API, enough for the worker: one object per
    session, a box that moves one px per frame; `fail_seeds` raise on
    add_prompt like a SAM 3.1 crash."""

    def __init__(self, n: int, fail_seeds: set[int]) -> None:
        self.n, self.fail = n, fail_seeds
        self.seed: dict[str, int] = {}
        self.closed: list[str] = []
        self.model = types.SimpleNamespace(init_state=lambda **kw: None)

    def handle_request(self, req: dict) -> dict:
        if req["type"] == "start_session":
            sid = f"s{len(self.seed)}"
            self.seed[sid] = -1
            return {"session_id": sid}
        if req["type"] == "add_prompt":
            if req["frame_index"] in self.fail:
                raise RuntimeError("No points are provided")
            self.seed[req["session_id"]] = req["frame_index"]
            return {"outputs": {"out_obj_ids": [7]}}
        self.closed.append(req["session_id"])
        return {}

    def handle_stream_request(self, req: dict):  # type: ignore[no-untyped-def]
        for f in range(self.n):
            m = np.zeros((16, 24), np.uint8)
            m[4:12, f : f + 6] = 1
            yield {"frame_index": f, "outputs": {"out_obj_ids": [7], "out_binary_masks": [m]}}


def _run_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail: set[int], extra: str | None
) -> tuple[int, Path, _FakePredictor]:
    n = 6
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(n):
        (frames / f"{i:05d}.jpg").write_bytes(b"")
    pred = _FakePredictor(n, fail)
    builder = types.ModuleType("sam3.model_builder")
    builder.build_sam3_multiplex_video_predictor = lambda **kw: pred  # type: ignore[attr-defined]
    decoder = types.ModuleType("sam3.model.decoder")
    for name, mod in {
        "sam3": types.ModuleType("sam3"),
        "sam3.model": types.ModuleType("sam3.model"),
        "sam3.model.decoder": decoder,
        "sam3.model_builder": builder,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    worker = _load_worker()
    out = tmp_path / "out"
    argv = ["--frames", str(frames), "--out", str(out), "--concepts", '["person"]']
    argv += ["--seed", "3", "--checkpoint", "fake.pt"]
    if extra is not None:
        argv += ["--extra-seeds", extra]
    return worker.main(argv), out, pred


def test_worker_single_seed_output_is_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc, out, pred = _run_worker(monkeypatch, tmp_path, set(), None)
    res = json.loads((out / "result.json").read_text())
    assert rc == 0 and len(pred.seed) == 1
    assert res["objects"] == [{"file": "obj_00_0007.npz", "concept": "person", "obj_id": 7}]
    assert "sessions" not in res and "seeds" not in res


def test_worker_skips_a_failing_secondary_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rc, out, pred = _run_worker(monkeypatch, tmp_path, {0}, "[0, 5]")
    res = json.loads((out / "result.json").read_text())
    assert rc == 0 and res["seeds"] == [3, 0, 5]
    assert [s.get("skipped") for s in res["sessions"]] == [None, "No points are provided", None]
    assert [(o["session"], o["seed"]) for o in res["objects"]] == [(0, 3), (2, 5)]
    assert len(pred.closed) == 3  # the failed session is closed too
    for o in res["objects"]:
        with np.load(out / o["file"]) as z:
            assert z["mask"].shape == (6, 16, 24)


def test_worker_primary_session_failure_still_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="No points"):
        _run_worker(monkeypatch, tmp_path, {3}, "[0, 5]")
