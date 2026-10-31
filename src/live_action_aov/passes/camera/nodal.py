# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Rotation-only (nodal) camera solve, for shots with no parallax.

A pan / tilt / roll on a tripod gives COLMAP nothing to triangulate, so its
mapper gives up. The matches it already computed still pin the camera down:
with no translation every pair of frames is related by `x_j ~ K R_j R_i^T
K^-1 x_i`, one focal length and one rotation per frame. This module solves
exactly that from the inlier matches in the COLMAP database:

1. focal: grid search (+ local refinement) of the median pair residual when
   each pair is fit by its best rotation (Kabsch on the bearing vectors);
2. rotations: chained along the strongest pairs from the first frame;
3. joint least squares (soft-L1) over log-focal and every rotation.

Pure numpy (no scipy in the dependency set): golden-section search for the
focal, a block Levenberg-Marquardt with soft-L1 reweighting for the joint
refinement. The pass reads the database and builds the solve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

#: One verified pair: frame indices (i < j) and their inlier pixel matches.
Pair = tuple[int, int, np.ndarray, np.ndarray]


@dataclass
class NodalSolve:
    focal_px: float
    #: cam_from_world rotations (OpenCV), one per frame index; world = frame 0
    #: of the solved set. Missing = frame not connected to the rest.
    rotations: dict[int, np.ndarray]
    #: Mean pixel residual of the pairs ending on each frame (solve res).
    frame_error_px: dict[int, float] = field(default_factory=dict)
    frame_match_count: dict[int, int] = field(default_factory=dict)


def bearings(xy: np.ndarray, f: float, c: tuple[float, float]) -> np.ndarray:
    """Pixels -> unit camera rays (OpenCV: x right, y down, z forward)."""
    v = np.column_stack([(xy[:, 0] - c[0]) / f, (xy[:, 1] - c[1]) / f, np.ones(len(xy))])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def project(rays: np.ndarray, f: float, c: tuple[float, float]) -> np.ndarray:
    z = np.where(np.abs(rays[:, 2]) < 1e-9, 1e-9, rays[:, 2])
    return np.column_stack([f * rays[:, 0] / z + c[0], f * rays[:, 1] / z + c[1]])


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation R minimising |R a - b| over rows (unit vectors)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0, 1.0, d]) @ vt


def _pair_error(p: Pair, f: float, c: tuple[float, float]) -> float:
    _, _, xi, xj = p
    ri, rj = bearings(xi, f, c), bearings(xj, f, c)
    R = kabsch(ri, rj)
    return float(np.median(np.linalg.norm(project(ri @ R.T, f, c) - xj, axis=1)))


def estimate_focal(
    pairs: Sequence[Pair], size: tuple[int, int], lo: float = 0.3, hi: float = 6.0
) -> float:
    """Focal (px) whose per-pair best rotations fit the matches best. Needs
    some rotation in the shot: a locked-off camera does not constrain it."""
    w, h = size
    c = (w / 2.0, h / 2.0)
    long_edge = float(max(w, h))
    # The widest pairs carry the focal; a handful is enough and fast.
    use = sorted(pairs, key=lambda p: -float(np.abs(p[3] - p[2]).mean()))[:24]

    def cost(logf: float) -> float:
        f = float(np.exp(logf))
        return float(np.mean([_pair_error(p, f, c) for p in use]))

    grid = np.log(np.geomspace(lo * long_edge, hi * long_edge, 48))
    k = int(np.argmin([cost(g) for g in grid]))
    a, b = float(grid[max(0, k - 1)]), float(grid[min(len(grid) - 1, k + 1)])
    g = (np.sqrt(5.0) - 1.0) / 2.0  # golden section on [a, b]
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = cost(x1), cost(x2)
    while b - a > 1e-4:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = cost(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = cost(x2)
    return float(np.exp((a + b) / 2.0))


def chain_rotations(
    pairs: Sequence[Pair], f: float, c: tuple[float, float], frames: Sequence[int]
) -> dict[int, np.ndarray]:
    """cam_from_world for every frame reachable from frames[0], along the
    pairs with the most matches (maximum spanning tree)."""
    rel = {}
    for i, j, xi, xj in pairs:
        rel[(i, j)] = (len(xi), kabsch(bearings(xi, f, c), bearings(xj, f, c)))
    root = frames[0]
    R = {root: np.eye(3)}
    edges = sorted(rel.items(), key=lambda kv: -kv[1][0])
    grown = True
    while grown:
        grown = False
        for (i, j), (_, Rij) in edges:  # Rij maps rays of i to rays of j
            if i in R and j not in R:
                R[j] = Rij @ R[i]
                grown = True
            elif j in R and i not in R:
                R[i] = Rij.T @ R[j]
                grown = True
    return R


def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    """Rodrigues."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    cos = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    th = float(np.arccos(cos))
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-6:  # 180 deg: axis from the symmetric part
        k = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        k *= np.sign(np.array([1.0, R[0, 1] + 1e-12, R[0, 2] + 1e-12]))
        return th * k / np.linalg.norm(k)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * w / (2.0 * np.sin(th))


def refine(
    pairs: Sequence[Pair],
    f0: float,
    R0: dict[int, np.ndarray],
    size: tuple[int, int],
    max_matches: int = 150,
    iterations: int = 30,
    soft_px: float = 1.5,
    rng_seed: int = 0,
    fix_focal: bool = False,
) -> NodalSolve:
    """Joint least squares over log-focal and every rotation (the first
    frame stays fixed): block Levenberg-Marquardt, each pair touching only
    the focal and its two rotations, soft-L1 reweighting at `soft_px`."""
    w, h = size
    c = (w / 2.0, h / 2.0)
    frames = sorted(R0)
    root = frames[0]
    col = {k: 1 + 3 * n for n, k in enumerate(k for k in frames if k != root)}
    n_par = 1 + 3 * len(col)
    rng = np.random.default_rng(rng_seed)
    data = []
    for i, j, xi, xj in pairs:
        if i not in R0 or j not in R0:
            continue
        if len(xi) > max_matches:
            sel = rng.choice(len(xi), max_matches, replace=False)
            xi, xj = xi[sel], xj[sel]
        data.append((i, j, xi, xj))
    x = np.zeros(n_par)
    x[0] = np.log(f0)
    for k, cidx in col.items():
        x[cidx : cidx + 3] = matrix_to_rotvec(R0[k])

    def rot(xv: np.ndarray, k: int) -> np.ndarray:
        return np.eye(3) if k == root else rotvec_to_matrix(xv[col[k] : col[k] + 3])

    def pair_res(
        logf: float, Ri: np.ndarray, Rj: np.ndarray, xi: np.ndarray, xj: np.ndarray
    ) -> np.ndarray:
        f = float(np.exp(logf))
        return (project(bearings(xi, f, c) @ (Rj @ Ri.T).T, f, c) - xj).ravel()

    def local(xv: np.ndarray, i: int, j: int) -> tuple[np.ndarray, list[int]]:
        idx = [0]
        p = [xv[0]]
        for k in (i, j):
            if k in col:
                idx += list(range(col[k], col[k] + 3))
                p += list(xv[col[k] : col[k] + 3])
        return np.array(p), idx

    def eval_pair(pv: np.ndarray, i: int, j: int, xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        n = 1
        Ri = Rj = np.eye(3)
        if i in col:
            Ri = rotvec_to_matrix(pv[n : n + 3])
            n += 3
        if j in col:
            Rj = rotvec_to_matrix(pv[n : n + 3])
        return pair_res(pv[0], Ri, Rj, xi, xj)

    def weights(r: np.ndarray) -> np.ndarray:
        e = np.repeat(np.linalg.norm(r.reshape(-1, 2), axis=1), 2)
        return 1.0 / np.sqrt(1.0 + (e / soft_px) ** 2)  # soft-L1 IRLS

    def total_cost(xv: np.ndarray) -> float:
        tot = 0.0
        for i, j, xi, xj in data:
            e = np.linalg.norm(
                pair_res(xv[0], rot(xv, i), rot(xv, j), xi, xj).reshape(-1, 2), axis=1
            )
            tot += float(np.sum(2.0 * soft_px**2 * (np.sqrt(1.0 + (e / soft_px) ** 2) - 1.0)))
        return tot

    lam = 1e-3
    cost = total_cost(x)
    for _ in range(iterations):
        JtJ = np.zeros((n_par, n_par))
        Jtr = np.zeros(n_par)
        for i, j, xi, xj in data:
            pv, idx = local(x, i, j)
            r = eval_pair(pv, i, j, xi, xj)
            J = np.empty((r.size, pv.size))
            for q in range(pv.size):
                d = np.zeros(pv.size)
                d[q] = 1e-6
                J[:, q] = (eval_pair(pv + d, i, j, xi, xj) - eval_pair(pv - d, i, j, xi, xj)) / 2e-6
            wr = weights(r)
            Jw = J * wr[:, None]
            JtJ[np.ix_(idx, idx)] += J.T @ Jw
            Jtr[idx] += Jw.T @ r
        improved = False
        for _try in range(8):
            A = JtJ + lam * np.diag(np.maximum(np.diag(JtJ), 1e-9))
            g = Jtr.copy()
            if fix_focal:  # locked-off: the focal is not observable, keep it
                A[0, :] = 0.0
                A[:, 0] = 0.0
                A[0, 0] = 1.0
                g[0] = 0.0
            step = np.linalg.solve(A, -g)
            x_new = x + step
            c_new = total_cost(x_new)
            if c_new < cost:
                x, lam = x_new, max(lam / 3.0, 1e-9)
                improved = cost - c_new > 1e-7 * cost
                cost = c_new
                break
            lam *= 10.0
        if not improved:
            break

    f = float(np.exp(x[0]))
    rotations = {k: rot(x, k) for k in frames}
    per: dict[int, list[float]] = {}
    count: dict[int, int] = {}
    for i, j, xi, xj in data:
        e = np.linalg.norm(pair_res(x[0], rot(x, i), rot(x, j), xi, xj).reshape(-1, 2), axis=1)
        per.setdefault(j, []).extend(e.tolist())
        count[j] = count.get(j, 0) + len(xi)
    frame_err = {k: float(np.median(v)) for k, v in per.items()}
    if root not in frame_err and frame_err:
        frame_err[root] = float(np.median(list(frame_err.values())))
        count[root] = max(count.values())
    return NodalSolve(f, rotations, frame_err, count)


def solve_nodal(pairs: Sequence[Pair], size: tuple[int, int]) -> NodalSolve:
    """Focal + rotations for a no-parallax shot from verified pair matches."""
    if not pairs:
        raise ValueError("nodal solve: no verified pairs")
    w, h = size
    frames = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    f0 = estimate_focal(pairs, size)
    R0 = chain_rotations(pairs, f0, (w / 2.0, h / 2.0), frames)
    return refine(pairs, f0, R0, size)


__all__ = [
    "NodalSolve",
    "bearings",
    "chain_rotations",
    "estimate_focal",
    "kabsch",
    "matrix_to_rotvec",
    "rotvec_to_matrix",
    "solve_nodal",
]
