"""Rotation-only (nodal) camera: the solver on synthetic pans with known
focal and rotations, the fallback's acceptance rules, and the CameraSolve
it builds (camera at the origin, plate-resolution intrinsics)."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.passes.camera import nodal
from live_action_aov.passes.camera.pycolmap_track import build_solve_from_nodal, lens_focal_mm

W, H = 2048, 1152


def _scene(n: int, yaw_per_frame: float, f: float = 1800.0, noise: float = 0.3, seed: int = 1):
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    c = (W / 2, H / 2)
    Rs = [
        nodal.rotvec_to_matrix(np.array([0.0, np.radians(yaw_per_frame * k), 0.0]))
        @ nodal.rotvec_to_matrix(np.array([np.radians(0.1 * k), 0.0, 0.0]))
        for k in range(n)
    ]

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
    pairs, _ = _scene(8, 0.6)
    s = nodal.solve_nodal(pairs, (W, H))
    frames = [(1001 + k, f"frame_{1001 + k:06d}.png") for k in range(8)]
    solve = build_solve_from_nodal(
        s, frame_list=frames, plate_wh=(4096, 2304), solve_wh=(W, H), params={}, pairs=pairs
    )
    assert solve.backend.mapper == "nodal"
    assert solve.intrinsics.fx_px == pytest.approx(2 * s.focal_px)
    assert all(e.translation == (0.0, 0.0, 0.0) for e in solve.extrinsics)
    assert [e.frame for e in solve.extrinsics] == [f for f, _ in frames]
    assert solve.points and solve.quality.registered_frame_count == 8
    assert any("nodal pan" in w for w in solve.quality.warnings)
