"""Depth-anchored PnP camera fallback, on synthetic scenes (no MoGe-3,
no COLMAP): known depth + known camera motion in, the same motion out."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from live_action_aov.passes.camera import pnp
from live_action_aov.passes.camera.pycolmap_track import PyColmapCameraPass, build_solve_from_pnp

W, H, N_FR, ANCHOR = 320, 192, 9, 4
FX = pnp.focal_from_fov(W, 60.0)
TX = 0.05  # metres per frame, camera moving right
Z_TOP, Z_BOTTOM = 4.0, 8.0


def _texture(seed: int, w: int, h: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    return (255 * (t - t.min()) / (t.max() - t.min())).astype(np.uint8)


def _scene() -> tuple[list[np.ndarray], np.ndarray]:
    """Two fronto-parallel planes (top half 4 m, bottom half 8 m): the camera
    translating right by TX shifts each half left by FX * TX / Z px."""
    pad = 60
    top, bot = _texture(1, W + 2 * pad, H // 2), _texture(2, W + 2 * pad, H // 2)
    frames = []
    for t in range(N_FR):
        c = (t - ANCHOR) * TX
        f = np.zeros((H, W), np.uint8)
        for half, tex, z in ((slice(0, H // 2), top, Z_TOP), (slice(H // 2, H), bot, Z_BOTTOM)):
            m = np.float32([[1, 0, -pad - FX * c / z], [0, 1, 0]])
            f[half] = cv2.warpAffine(tex, m, (W, H // 2), flags=cv2.INTER_CUBIC)
        frames.append(f)
    depth = np.zeros((H, W), np.float32)
    depth[: H // 2], depth[H // 2 :] = Z_TOP, Z_BOTTOM
    return frames, depth


def _K() -> np.ndarray:
    return np.array([[FX, 0, W / 2], [0, FX, H / 2], [0, 0, 1]])


def test_depth_edges_are_flagged_with_the_window_around_them() -> None:
    _, depth = _scene()
    bad = pnp.near_depth_edges(depth, radius=5)
    assert bad[H // 2 - 5 : H // 2 + 5].all()
    assert not bad[: H // 2 - 6].any() and not bad[H // 2 + 6 :].any()


def test_back_project_round_trips_through_the_camera() -> None:
    uv = np.array([[10.0, 20.0], [200.0, 150.0]], np.float32)
    depth = np.full((H, W), 5.0, np.float32)
    P = pnp.back_project(uv, depth, _K())
    proj, _ = cv2.projectPoints(P, np.zeros(3), np.zeros(3), _K(), None)
    np.testing.assert_allclose(proj[:, 0], uv, atol=1e-3)
    assert np.allclose(P[:, 2], 5.0)


def test_solve_poses_recovers_motion_despite_outliers() -> None:
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-3, 3, (400, 2)), rng.uniform(3, 12, 400)]
    sol = pnp.PnPSolve(K=_K(), anchor=0, points=P, colors=np.zeros((400, 3), np.uint8))
    tracks = np.zeros((3, 400, 2), np.float32)
    truth = {1: np.array([0.1, 0.0, 0.0]), 2: np.array([0.2, -0.05, 0.1])}
    tracks[0] = cv2.projectPoints(P, np.zeros(3), np.zeros(3), _K(), None)[0][:, 0]
    for t, c in truth.items():
        tracks[t] = cv2.projectPoints(P, np.zeros(3), -c, _K(), None)[0][:, 0]
        tracks[t] += rng.normal(0, 0.3, tracks[t].shape)
        tracks[t, :60] += rng.uniform(-40, 40, (60, 2))  # 15 % bad tracks
    pnp.solve_poses(sol, tracks)
    for t, c in truth.items():
        R, tv = sol.poses[t]
        centre = -R.T @ tv
        np.testing.assert_allclose(centre, c, atol=0.02)
        assert sol.frame_error_px[t] < 1.0


def test_klt_follows_known_shifts_and_drops_points_on_tracked_objects() -> None:
    frames, _ = _scene()
    uv = np.array([[100.0, 40.0], [200.0, 150.0], [160.0, 50.0]], np.float32)
    static = [None] * N_FR
    blocked = np.ones((H, W), bool)
    blocked[:, 140:155] = False  # a "person" that the third point slides onto
    static[ANCHOR + 2] = blocked
    tr = pnp.track_klt(frames, ANCHOR, uv, static)
    for t in (0, 8):
        d = (t - ANCHOR) * TX
        np.testing.assert_allclose(tr[t, 0], uv[0] - [FX * d / Z_TOP, 0], atol=0.3)
        np.testing.assert_allclose(tr[t, 1], uv[1] - [FX * d / Z_BOTTOM, 0], atol=0.3)
    assert np.isnan(tr[ANCHOR + 2 :, 2]).all() and np.isfinite(tr[ANCHOR + 1, 2]).all()


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, f: int) -> tuple[np.ndarray, dict]:
        g = self._f[f - 1001].astype(np.float32) / 255.0
        return np.repeat(g[..., None], 3, axis=2), {}


def _run_fallback(tmp_path: Path, **params: object) -> object:
    frames, depth = _scene()
    frame_list = [(1001 + t, f"frame_{1001 + t:06d}.png") for t in range(N_FR)]
    for (_, name), g in zip(frame_list, frames, strict=True):
        cv2.imwrite(str(tmp_path / name), g)
    p = PyColmapCameraPass({"pnp_max_corners": 800, **params})
    p._anchor_depth = lambda img: (depth, 60.0)  # type: ignore[method-assign]
    p._pnp_fov_source = "FOV 60.0 deg (test)"
    return p._solve_pnp(_Reader(frames), frame_list, (2 * W, 2 * H), (W, H), tmp_path, None)


def test_fallback_recovers_the_camera_in_metres(tmp_path: Path) -> None:
    solve = _run_fallback(tmp_path)
    assert solve is not None
    assert solve.backend.mapper == "depth_pnp"
    assert solve.quality.registered_frame_count == N_FR
    assert solve.intrinsics.fx_px == pytest.approx(2 * FX, rel=1e-6)  # plate res = 2x solve
    pos = {e.frame: e.translation for e in solve.extrinsics}
    assert pos[1001 + ANCHOR] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    for t in range(N_FR):
        x, y, z = pos[1001 + t]
        assert x == pytest.approx((t - ANCHOR) * TX, abs=0.01)  # right = +X in the Y-up world
        assert abs(y) < 0.01 and abs(z) < 0.03
    q = solve.quality
    assert q.verdict == "GOOD" and q.warnings[0].startswith("GOOD CANDIDATE - translated")
    assert q.heldout_error_px is not None and q.heldout_error_px < 0.0011 * 2 * W
    assert solve.points and "Depth-anchored PnP" in q.warnings[1]
    assert solve.conventions.units == "metres"
    # Points in front of the camera = -Z in the Y-up world.
    assert all(p.xyz[2] < 0 for p in solve.points)


def test_a_failed_camera_is_labelled_or_refused(tmp_path: Path) -> None:
    strict = {"pnp_good_frac": 1e-9, "pnp_weak_frac": 1e-9}
    solve = _run_fallback(tmp_path, **strict)
    assert solve is not None and solve.quality.verdict == "FAILED"
    assert solve.quality.warnings[0].endswith("DO NOT USE")
    assert _run_fallback(tmp_path, pnp_keep_failed=False, **strict) is None


def _plane_tracks(P: np.ndarray, centres: dict[int, np.ndarray], n_frames: int) -> np.ndarray:
    tracks = np.zeros((n_frames, len(P), 2), np.float32)
    for t in range(n_frames):
        c = centres.get(t, np.zeros(3))
        tracks[t] = cv2.projectPoints(P, np.zeros(3), -c, _K(), None)[0][:, 0]
    return tracks


def test_held_out_points_catch_a_camera_the_fit_error_hides() -> None:
    """The fit set agrees with itself; the held-out points say the camera is
    wrong (they follow another motion, like tracks on things that move)."""
    rng = np.random.default_rng(1)
    P = np.c_[rng.uniform(-3, 3, (500, 2)), rng.uniform(3, 12, 500)]
    held = pnp.holdout_mask(len(P))
    centres = {1: np.array([0.1, 0.0, 0.0]), 2: np.array([0.2, 0.0, 0.0])}
    tracks = _plane_tracks(P, centres, 3)
    other = _plane_tracks(P, {1: np.array([-0.4, 0.2, 0.0]), 2: np.array([-0.8, 0.4, 0.0])}, 3)
    tracks[:, held] = other[:, held]
    sol = pnp.PnPSolve(K=_K(), anchor=0, points=P, colors=np.zeros((len(P), 3), np.uint8))
    pnp.solve_poses(sol, tracks, held=held)
    assert max(sol.frame_error_px.values()) < 0.1  # the fit looks perfect
    assert min(sol.frame_heldout_px[t] for t in (1, 2)) > 5.0
    assert pnp.assess(sol, W, 3).verdict == "FAILED"


def test_assess_travel_uses_reliable_frames_and_flags_nodal() -> None:
    P = np.c_[np.zeros((10, 2)), np.full(10, 5.0)]
    sol = pnp.PnPSolve(K=_K(), anchor=0, points=P, colors=np.zeros((10, 3), np.uint8))
    for t in range(10):
        sol.poses[t] = (np.eye(3), np.array([-0.01 * t, 0.0, 0.0]))  # 1 cm per frame
        sol.frame_heldout_px[t] = 0.1
    sol.poses[9] = (np.eye(3), np.array([-2625.0, 0.0, 0.0]))  # a frame that went wild
    sol.frame_heldout_px[9] = 50.0
    v = pnp.assess(sol, W, 10)
    assert v.verdict == "GOOD" and 9 not in v.reliable
    assert v.travel_m == pytest.approx(0.08)
    assert v.nodal  # 8 cm over a 5 m set: under 2 % parallax
    sol.frame_heldout_px.update(dict.fromkeys(range(1, 9), 0.6))  # between the GOOD and WEAK limits
    assert pnp.assess(sol, W, 10).verdict == "WEAK"


def test_build_solve_keeps_unsolved_frames_out() -> None:
    sol = pnp.PnPSolve(K=_K(), anchor=0, points=np.ones((5, 3)), colors=np.zeros((5, 3), np.uint8))
    sol.poses = {0: (np.eye(3), np.zeros(3)), 2: (np.eye(3), np.array([-1.0, 0.0, 0.0]))}
    sol.frame_error_px = {0: 0.0, 2: 0.5}
    fl = [(10, "a"), (11, "b"), (12, "c")]
    s = build_solve_from_pnp(sol, frame_list=fl, plate_wh=(W, H), solve_wh=(W, H), params={})
    assert s.unregistered_frames == [11]
    assert [e.frame for e in s.extrinsics] == [10, 12]
    assert s.extrinsics[1].translation[0] == pytest.approx(1.0)
