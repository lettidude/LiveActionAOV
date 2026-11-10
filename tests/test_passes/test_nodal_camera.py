"""Rotation-only (nodal) camera: the solver on synthetic pans with known
focal and rotations, the fallback's acceptance rules, the CameraSolve it
builds (camera at the origin, plate-resolution intrinsics) and the KLT
referee's verdict it must carry."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.passes.camera import nodal, pnp
from live_action_aov.passes.camera.pycolmap_track import build_solve_from_nodal, lens_focal_mm

W, H = 2048, 1152


def _rotations(n: int, yaw_per_frame: float) -> list[np.ndarray]:
    """cam_from_world of a tripod pan: yaw plus a slow tilt, frame 0 = world."""
    return [
        nodal.rotvec_to_matrix(np.array([0.0, np.radians(yaw_per_frame * k), 0.0]))
        @ nodal.rotvec_to_matrix(np.array([np.radians(0.1 * k), 0.0, 0.0]))
        for k in range(n)
    ]


def _scene(n: int, yaw_per_frame: float, f: float = 1800.0, noise: float = 0.3, seed: int = 1):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    c = (W / 2, H / 2)
    Rs = _rotations(n, yaw_per_frame)

    def obs(k: int) -> tuple[np.ndarray, np.ndarray]:
        cam = dirs @ Rs[k].T
        p = nodal.project(cam, f, c)
        ok = (cam[:, 2] > 0.2) & (p[:, 0] > 0) & (p[:, 0] < W) & (p[:, 1] > 0) & (p[:, 1] < H)
        return ok, p

    pairs = []
    for i in range(n):
        oi, pi = obs(i)
        for j in range(i + 1, min(n, i + 4)):
            oj, pj = obs(j)
            m = oi & oj
            pairs.append(
                (
                    i,
                    j,
                    pi[m] + rng.normal(0, noise, (m.sum(), 2)),
                    pj[m] + rng.normal(0, noise, (m.sum(), 2)),
                )
            )
    return pairs, Rs


def _tracks(
    Rs: list[np.ndarray], f: float = 1800.0, n_pts: int = 600, noise: float = 0.3, seed: int = 7
) -> np.ndarray:
    """What a KLT referee sees on that pan: points at infinity (the scene is
    far away, the camera only turns) in view on every frame, (T, N, 2)."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(20 * n_pts, 3))
    dirs[:, 2] = np.abs(dirs[:, 2]) + 1.0  # mostly in front of the camera
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    c = (W / 2, H / 2)
    uv = np.stack([nodal.project(dirs @ R.T, f, c) for R in Rs])  # (T, M, 2)
    inside = (uv[..., 0] > 2) & (uv[..., 0] < W - 3) & (uv[..., 1] > 2) & (uv[..., 1] < H - 3)
    keep = np.nonzero(inside.all(axis=0))[0][:n_pts]
    assert len(keep) >= n_pts // 2
    return (uv[:, keep] + rng.normal(0, noise, uv[:, keep].shape)).astype(np.float32)


def _K(f: float) -> np.ndarray:
    return np.array([[f, 0.0, W / 2], [0.0, f, H / 2], [0.0, 0.0, 1.0]])


def _poses(rotations: dict[int, np.ndarray]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {k: (R, np.zeros(3)) for k, R in rotations.items()}


def test_rotvec_roundtrip() -> None:
    for v in ([0.1, -0.2, 0.3], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]):
        R = nodal.rotvec_to_matrix(np.array(v))
        assert np.allclose(nodal.rotvec_to_matrix(nodal.matrix_to_rotvec(R)), R, atol=1e-8)


def test_pan_recovers_focal_and_rotations() -> None:
    pairs, Rs = _scene(30, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    assert s.focal_px == pytest.approx(1800.0, rel=0.005)
    worst = max(
        np.degrees(np.linalg.norm(nodal.matrix_to_rotvec(s.rotations[k] @ Rs[k].T)))
        for k in range(30)
    )
    assert worst < 0.05
    assert np.median(list(s.frame_error_px.values())) < 1.0


def test_fixed_focal_is_kept_when_asked() -> None:
    """A locked-off camera cannot measure the focal: the prior is kept."""
    pairs, _ = _scene(10, 0.0)
    R0 = nodal.chain_rotations(pairs, 1500.0, (W / 2, H / 2), list(range(10)))
    s = nodal.refine(pairs, 1500.0, R0, (W, H), fix_focal=True)
    assert s.focal_px == pytest.approx(1500.0)


def test_lens_metadata_focal() -> None:
    assert lens_focal_mm({"nuke/arri/camera/FocalLength": "150.0"}) == 150.0
    assert lens_focal_mm({"FocalLength": "-"}) is None
    assert lens_focal_mm({}) is None


def test_nodal_camera_solve_sits_at_the_origin_at_plate_res() -> None:
    pairs, Rs = _scene(8, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    frames = [(1001 + k, f"frame_{1001 + k:06d}.png") for k in range(8)]
    verdict = pnp.assess_poses(_K(s.focal_px), _poses(s.rotations), _tracks(Rs), 4, W, 8)
    solve = build_solve_from_nodal(
        s,
        frame_list=frames,
        plate_wh=(4096, 2304),
        solve_wh=(W, H),
        params={},
        pairs=pairs,
        verdict=verdict,
    )
    assert solve.backend.mapper == "nodal"
    assert solve.intrinsics.fx_px == pytest.approx(2 * s.focal_px)
    assert all(e.translation == (0.0, 0.0, 0.0) for e in solve.extrinsics)
    assert [e.frame for e in solve.extrinsics] == [f for f, _ in frames]
    assert solve.points and solve.quality.registered_frame_count == 8
    assert any("nodal pan" in w for w in solve.quality.warnings)
    # The referee's verdict is in the JSON, plate px, and leads the warnings.
    assert solve.quality.verdict == "GOOD"
    assert solve.quality.heldout_error_px == pytest.approx(2 * verdict.heldout_px)
    assert solve.quality.warnings[0].startswith("GOOD CANDIDATE - NODAL: held-out error")


def test_a_nodal_camera_without_a_verdict_is_impossible() -> None:
    """Like the SfM camera: no verdict, no CameraSolve (the GUI light, the
    Nuke label and `liveaov qc` must never see an empty one again)."""
    pairs, _ = _scene(8, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    frames = [(1001 + k, f"frame_{1001 + k:06d}.png") for k in range(8)]
    kw = dict(frame_list=frames, plate_wh=(W, H), solve_wh=(W, H), params={})
    with pytest.raises(TypeError):
        build_solve_from_nodal(s, **kw)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="verdict"):
        build_solve_from_nodal(s, verdict=None, **kw)


def test_referee_calls_a_pure_yaw_good_and_nodal() -> None:
    """Points at infinity under the solver's own rotations (zero translation):
    every track reprojects from the rotation alone, so GOOD, and the camera
    does not travel, so NODAL. Resolution-scaled into the camera JSON."""
    pairs, Rs = _scene(8, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    tracks = _tracks(Rs)
    v = pnp.assess_poses(_K(s.focal_px), _poses(s.rotations), tracks, 4, W, 8)
    assert v.verdict == "GOOD" and v.nodal
    assert v.reliable == list(range(8)) and v.travel_m == 0.0
    assert np.isfinite(v.heldout_px) and v.heldout_px < 0.0011 * W


def test_referee_fails_a_rotation_the_plate_disagrees_with() -> None:
    """Rotations at twice the pan over tracks of the real pan: no scene at
    any distance explains them, so the held-out error is large and finite
    and the camera is FAILED - labelled DO NOT USE in the JSON."""
    pairs, Rs = _scene(8, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    doubled = {k: R @ R for k, R in s.rotations.items()}
    v = pnp.assess_poses(_K(s.focal_px), _poses(doubled), _tracks(Rs), 4, W, 8)
    assert v.verdict == "FAILED"
    assert np.isfinite(v.heldout_px) and v.heldout_px > 0.0027 * W
    frames = [(1001 + k, f"frame_{1001 + k:06d}.png") for k in range(8)]
    solve = build_solve_from_nodal(
        s, frame_list=frames, plate_wh=(W, H), solve_wh=(W, H), params={}, verdict=v
    )
    assert solve.quality.verdict == "FAILED"
    assert solve.quality.heldout_error_px == pytest.approx(v.heldout_px)
    assert solve.quality.warnings[0].startswith("FAILED CANDIDATE - NODAL")
    assert solve.quality.warnings[0].endswith("DO NOT USE")
