"""Per-frame referee in the camera json (`quality.frames`): every frame of the
solve gets the held-out median (plate px), the number of points it is over
and `ok`. Synthetic cameras with known poses: a few wrong frames inside a
good shot must show up frame by frame while the shot verdict stays as it was."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from live_action_aov.io.camera_schema import CameraSolve, Quality, RefereeFrame
from live_action_aov.passes.camera import pnp
from live_action_aov.passes.camera.nodal import NodalSolve
from live_action_aov.passes.camera.pycolmap_track import (
    build_solve_from_nodal,
    build_solve_from_pnp,
    referee_frames,
)

W, H, N_FR = 640, 360, 40
FX = pnp.focal_from_fov(W, 55.0)
STEP = 0.04  # metres per frame, camera moving right
BAD = (11, 12, 27)  # local frames whose pose is wrong
FIRST = 1001


def _K() -> np.ndarray:
    return np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])


def _true_poses() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {t: (np.eye(3), np.array([-(t - N_FR // 2) * STEP, 0.0, 0.0])) for t in range(N_FR)}


def _tracks(n: int = 1500, seed: int = 0) -> np.ndarray:
    """Static points seen through the true camera, 0.2 px noise; points that
    leave the frame are lost (NaN), like KLT tracks."""
    rng = np.random.default_rng(seed)
    P = np.c_[rng.uniform(-4, 4, n), rng.uniform(-2.5, 2.5, n), rng.uniform(4, 15, n)]
    tracks = np.full((N_FR, n, 2), np.nan, np.float32)
    for t, (R, tv) in _true_poses().items():
        uv = cv2.projectPoints(P, cv2.Rodrigues(R)[0], tv, _K(), None)[0][:, 0]
        uv += rng.normal(0, 0.2, uv.shape)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        tracks[t, inside] = uv[inside]
    return tracks


def _bad_poses(
    yaw_deg: float = 0.3, shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = _true_poses()
    for t in BAD:
        yaw = cv2.Rodrigues(np.array([0.0, np.radians(yaw_deg), 0.0]))[0]
        poses[t] = (yaw, poses[t][1] + np.array(shift))
    return poses


def _frame_list() -> list[tuple[int, str]]:
    return [(FIRST + t, f"frame_{FIRST + t:06d}.png") for t in range(N_FR)]


def test_bad_frames_hidden_in_a_good_shot_show_frame_by_frame() -> None:
    tracks = _tracks()
    v = pnp.assess_poses(_K(), _bad_poses(), tracks, N_FR // 2, W, N_FR)
    # The shot median hides them: the shot verdict is still GOOD ...
    assert v.verdict == "GOOD"
    good_px = 0.0011 * W
    # ... but every wrong frame stands out, and no good frame is flagged.
    good_max = max(v.frame_heldout_px[t] for t in range(N_FR) if t not in BAD)
    assert good_max < good_px
    for t in BAD:
        assert v.frame_heldout_px[t] > 4 * good_max, (t, v.frame_heldout_px[t], good_max)
    assert min(v.frame_points.values()) >= 20
    frames = referee_frames(v, _frame_list(), {f for f, _ in _frame_list()}, 2.0, {})
    assert sorted(frames) == [f for f, _ in _frame_list()]
    worst = sorted(frames, key=lambda f: -(frames[f].heldout_median_px or 0.0))[: len(BAD)]
    assert sorted(worst) == [FIRST + t for t in BAD]
    # Plate px = solve px x scale; a tested frame with a big error stays ok.
    f = FIRST + BAD[0]
    assert frames[f].heldout_median_px == pytest.approx(v.frame_heldout_px[BAD[0]] * 2.0, abs=1e-3)
    assert frames[f].ok and frames[f].n_points == v.frame_points[BAD[0]]


def test_far_off_frames_still_rank_worst_when_they_spoil_the_shot() -> None:
    """Frames off by a lot also pull their good neighbours up (the referee
    triangulates every point from all the other frames): the shot is no
    longer GOOD, and the wrong frames are still the worst ones, by far."""
    v = pnp.assess_poses(_K(), _bad_poses(1.5, (0.3, 0.1, 0.0)), _tracks(), N_FR // 2, W, N_FR)
    assert v.verdict != "GOOD"
    e = v.frame_heldout_px
    worst = sorted(e, key=lambda t: -e[t])[: len(BAD)]
    assert sorted(worst) == sorted(BAD)
    assert min(e[t] for t in BAD) > 3 * max(e[t] for t in e if t not in BAD)


def test_the_shot_verdict_is_the_same_with_or_without_the_point_counts() -> None:
    """The counts are a side output: `heldout_from_poses` gives the same
    errors with them, so the verdict cannot move."""
    tracks = _tracks()
    counts: dict[int, int] = {}
    a, pa = pnp.heldout_from_poses(_K(), _bad_poses(), tracks, N_FR // 2)
    b, pb = pnp.heldout_from_poses(_K(), _bad_poses(), tracks, N_FR // 2, counts=counts)
    assert a == b and np.array_equal(pa, pb, equal_nan=True)
    assert set(counts) == set(a) and all(c > 0 for c in counts.values())


def test_unposed_untestable_and_thin_frames_are_not_ok() -> None:
    v = pnp.PnPVerdict(
        verdict="GOOD",
        heldout_px=0.3,
        reliable=[0, 1],
        travel_m=1.0,
        depth_m=5.0,
        nodal=False,
        frame_heldout_px={0: 0.3, 1: 0.4, 2: float("inf"), 3: 0.2, 4: 9.0},
        frame_points={0: 300, 1: 300, 2: 0, 3: 5, 4: 300},
    )
    fl = [(10 + k, str(k)) for k in range(6)]
    posed = {10, 11, 12, 13}  # frame 14 measured but left out of the camera, 15 never posed
    fr = referee_frames(v, fl, posed, 1.5, {})
    assert fr[10] == RefereeFrame(heldout_median_px=0.45, n_points=300, ok=True)
    assert fr[12] == RefereeFrame(heldout_median_px=None, n_points=0, ok=False)  # untestable
    assert not fr[13].ok and fr[13].heldout_median_px == pytest.approx(0.3)  # 5 points < 20
    assert not fr[14].ok and fr[14].heldout_median_px == pytest.approx(13.5)  # no pose
    assert fr[15] == RefereeFrame()  # nothing at all
    assert referee_frames(v, fl, posed, 1.5, {"referee_min_points": 5})[13].ok


def test_pnp_camera_json_carries_the_frames_block() -> None:
    sol = pnp.PnPSolve(K=_K(), anchor=0, points=np.ones((5, 3)), colors=np.zeros((5, 3), np.uint8))
    sol.poses = {0: (np.eye(3), np.zeros(3)), 2: (np.eye(3), np.array([-1.0, 0.0, 0.0]))}
    sol.frame_error_px = {0: 0.0, 2: 0.5}
    v = pnp.PnPVerdict(
        verdict="WEAK",
        heldout_px=0.5,
        reliable=[0, 2],
        travel_m=1.0,
        depth_m=5.0,
        nodal=False,
        frame_heldout_px={0: 0.0, 1: 3.0, 2: 0.5},
        frame_points={0: 100, 1: 90, 2: 80},
    )
    fl = [(10, "a"), (11, "b"), (12, "c")]
    s = build_solve_from_pnp(
        sol, frame_list=fl, plate_wh=(2 * W, 2 * H), solve_wh=(W, H), params={}, verdict=v
    )
    assert s.quality.verdict == "WEAK"  # unchanged
    d = json.loads(s.model_dump_json())
    frames = d["quality"]["frames"]
    assert list(frames) == ["10", "11", "12"]  # plate frames, as strings in the json
    assert frames["11"] == {"heldout_median_px": 6.0, "n_points": 90, "ok": False}  # no pose
    assert frames["12"] == {"heldout_median_px": 1.0, "n_points": 80, "ok": True}
    # The block reads back, and a reader that only knows `frames` + `verdict`
    # (pnp_report.json shape) finds what it needs inside `quality`.
    assert CameraSolve.model_validate(d).quality.frames[12].ok
    q = d["quality"]
    assert q["verdict"] == "WEAK" and q["frames"]["12"]["heldout_median_px"] == 1.0


def test_nodal_camera_json_carries_the_frames_block() -> None:
    sol = NodalSolve(
        focal_px=FX,
        rotations={0: np.eye(3), 1: np.eye(3)},
        frame_error_px={0: 0.1, 1: 0.1},
        frame_match_count={0: 200, 1: 200},
    )
    v = pnp.PnPVerdict(
        verdict="GOOD",
        heldout_px=0.2,
        reliable=[0, 1],
        travel_m=0.0,
        depth_m=float("inf"),
        nodal=True,
        frame_heldout_px={0: 0.2, 1: 0.2},
        frame_points={0: 50, 1: 50},
    )
    s = build_solve_from_nodal(
        sol,
        frame_list=[(1, "a"), (2, "b"), (3, "c")],
        plate_wh=(W, H),
        solve_wh=(W, H),
        params={},
        verdict=v,
    )
    assert s.quality.frames[1].ok and s.quality.frames[2].ok
    assert s.quality.frames[3] == RefereeFrame()  # never posed


def test_old_json_without_frames_still_reads() -> None:
    q = Quality.model_validate({"verdict": "GOOD", "heldout_error_px": 0.5})
    assert q.frames == {}
