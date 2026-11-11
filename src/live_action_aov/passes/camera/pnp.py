# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Depth-anchored PnP camera: the fallback when SfM cannot triangulate.

COLMAP needs parallax between frames it can match. Fast handheld action
(motion blur, people filling the frame, a short shot) often gives it none it
can use - it registers a handful of frames - while the camera clearly
moves, so the rotation-only (nodal) fallback does not fit either. Here the
3D structure comes from ONE frame instead of from triangulation:

1. MoGe-3 metric depth of the anchor frame (the middle one), back-projected
   with its focal: 3D points of the static set (tracked objects masked out),
   in metres, in the anchor camera's frame (= the world).
2. Corners away from depth edges (occlusion edges track badly and have the
   least reliable depth) tracked through the shot with pyramidal
   Lucas-Kanade, frame to frame both ways from the anchor, dropped when the
   forward-backward check fails, they leave the frame or land on a tracked
   object.
3. Per frame, `solvePnPRansac` (EPnP) + Levenberg-Marquardt refine.
4. Held-out check: every `holdout_every`-th point never enters the solve and
   is reprojected afterwards. The fit error alone can lie - RANSAC can fit a
   small consistent subset (e.g. tracks on things that move) with a low
   error while the camera is wrong - so the verdict (GOOD / WEAK / FAILED),
   the travel and the nodal flag come from the held-out points.

The scale is MoGe-3's metric estimate (approximate), the poses are as good as
the anchor depth: good enough to line up a point cloud, a card or a matte
painting, not a survey. Frames whose points are all gone (a whip pan away
from the anchor view) stay unregistered.

The same held-out verdict judges the SfM camera (`heldout_from_poses`,
`assess_poses`): COLMAP's own reprojection error is measured on the SIFT
features it fitted, so a solve that drifted or split can still report a
clean number. The referee tracks independent KLT corners through the shot
and, with COLMAP's poses held FIXED, triangulates every track from all the
frames it is seen in except the one under test, then reprojects it into
that frame (leave-one-frame-out). A frame whose pose disagrees with the
rest of the camera shows up there whatever the fit error says.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class PnPSolve:
    K: np.ndarray  # (3, 3) at solve resolution
    anchor: int  # local frame index of the anchor (= world)
    points: np.ndarray  # (N, 3) anchor-camera OpenCV coordinates, metres
    colors: np.ndarray  # (N, 3) uint8 RGB
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # R_cw, t_cw
    frame_error_px: dict[int, float] = field(default_factory=dict)  # median, solve res
    frame_inliers: dict[int, int] = field(default_factory=dict)
    # median reprojection error of the held-out points, solve res
    frame_heldout_px: dict[int, float] = field(default_factory=dict)


@dataclass
class PnPVerdict:
    verdict: str  # GOOD / WEAK / FAILED
    heldout_px: float  # median of the per-frame held-out errors, solve res
    reliable: list[int]  # local frame indices whose held-out error is <= the WEAK limit
    travel_m: float  # extent of the camera centres over the reliable frames
    depth_m: float  # median anchor depth of the static points
    nodal: bool  # travel < `nodal_parallax` x depth

    @property
    def parallax(self) -> float:
        return self.travel_m / max(self.depth_m, 1e-9)


def holdout_mask(n: int, every: int = 5) -> np.ndarray:
    """True for the points kept out of the solve (every `every`-th one)."""
    return (np.arange(n) % max(int(every), 2)) == 0


def focal_from_fov(width: int, fov_x_deg: float) -> float:
    return (width / 2.0) / float(np.tan(np.radians(fov_x_deg) / 2.0))


def anchor_corners(
    gray: np.ndarray, usable: np.ndarray, max_corners: int, min_distance: int
) -> np.ndarray:
    """Trackable corners (N, 2) float32 inside `usable` (bool, same size)."""
    import cv2

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(max_corners),
        qualityLevel=0.005,
        minDistance=int(min_distance),
        mask=usable.astype(np.uint8) * 255,
        blockSize=7,
    )
    return np.zeros((0, 2), np.float32) if pts is None else pts.reshape(-1, 2).astype(np.float32)


def near_depth_edges(depth: np.ndarray, radius: int, rel_jump: float = 0.1) -> np.ndarray:
    """True where a window of `radius` px spans a depth jump of more than
    `rel_jump` (relative), or invalid depth. A corner there sits on an
    occlusion edge: its tracking window mixes two depths, and the depth map
    is least reliable there (on a synthetic two-plane scene these corners
    alone biased the camera by 6 %)."""
    import cv2

    valid = depth > 0
    logd = np.where(valid, np.log(np.maximum(depth, 1e-6)), 0.0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    span = cv2.dilate(logd, k) - cv2.erode(logd, k)
    bad = span > np.log1p(rel_jump)
    bad |= cv2.dilate((~valid).astype(np.uint8), k) > 0
    return bad


def back_project(uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) with their depth map values -> (N, 3) camera points."""
    h, w = depth.shape
    xi = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    z = depth[yi, xi].astype(np.float64)
    x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
    y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
    return np.stack([x, y, z], 1)


def track_klt(
    gray: list[np.ndarray],
    anchor: int,
    uv: np.ndarray,
    static: list[np.ndarray | None],
    fb_max_px: float = 1.0,
    win: int = 21,
    levels: int = 4,
    guide: Callable[[int, int, np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray:
    """Track `uv` (N, 2) from frame `anchor` through `gray` (uint8 frames)
    both ways. Returns (T, N, 2), NaN once a point is lost (it never comes
    back: a re-found point could be another one).

    `guide(a, b, pts)` (see `flow_guide`) predicts each step, forward and
    backward; LK then refines from the prediction on a shallow pyramid."""
    cv2: Any = importlib.import_module("cv2")  # its stubs reject the None arguments

    T, N = len(gray), len(uv)
    out = np.full((T, N, 2), np.nan, np.float32)
    out[anchor] = uv
    h, w = gray[anchor].shape[:2]
    lk = {
        "winSize": (win, win),
        "maxLevel": levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    guided = {**lk, "maxLevel": 1, "flags": cv2.OPTFLOW_USE_INITIAL_FLOW}
    for step in (1, -1):
        alive = np.ones(N, bool)
        prev = uv.copy()
        t = anchor + step
        while 0 <= t < T and alive.any():
            idx = np.nonzero(alive)[0]
            p0 = prev[idx].reshape(-1, 1, 2)
            g1 = guide(t - step, t, prev[idx]) if guide is not None else None
            if g1 is None:
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray[t - step], gray[t], p0, None, **lk)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - step], p1, None, **lk)
            else:
                p1 = g1.astype(np.float32).reshape(-1, 1, 2)
                p1, st1, _ = cv2.calcOpticalFlowPyrLK(gray[t - step], gray[t], p0, p1, **guided)
                gb = guide(t, t - step, p1.reshape(-1, 2)) if guide is not None else None
                pb = (gb if gb is not None else prev[idx]).astype(np.float32).reshape(-1, 1, 2)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[t - step], p1, pb, **guided)
            p1, pb = p1.reshape(-1, 2), pb.reshape(-1, 2)
            ok = (st1[:, 0] == 1) & (st2[:, 0] == 1)
            ok &= np.linalg.norm(pb - prev[idx], axis=1) <= fb_max_px
            ok &= (p1[:, 0] >= 1) & (p1[:, 0] < w - 2) & (p1[:, 1] >= 1) & (p1[:, 1] < h - 2)
            m = static[t]
            if m is not None:
                xi = np.clip(np.round(p1[:, 0]).astype(int), 0, w - 1)
                yi = np.clip(np.round(p1[:, 1]).astype(int), 0, h - 1)
                ok &= m[yi, xi]
            alive[idx[~ok]] = False
            prev[idx[ok]] = p1[ok]
            out[t, idx[ok]] = p1[ok]
            t += step
    return out


def flow_guide(
    fwd: dict[int, np.ndarray],
    bwd: dict[int, np.ndarray],
    frames: list[int],
    scale: float = 1.0,
) -> Callable[[int, int, np.ndarray], np.ndarray | None]:
    """Predict where points at frame index `a` are at index `b` (= a +- 1)
    from a flow pass's vectors: `fwd[f]` / `bwd[f]` are (2, h, w) flow from
    frame f to f+1 / f-1 in flow px, `scale` = solve px per flow px.

    Only a guess for `track_klt`: chaining the vectors alone drifts (a
    0.2 px/frame bias is 15 px after 60 frames), while LK started from the
    guess stays locked to the texture and still follows motion far beyond
    its own search window."""

    def guide(a: int, b: int, pts: np.ndarray) -> np.ndarray | None:
        field = (fwd if b > a else bwd).get(frames[a])
        if field is None:
            return None
        return pts + _sample_flow(field, pts / scale) * scale

    return guide


def _sample_flow(flow: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Bilinear (N, 2) sample of a (2, h, w) flow field at `pts` (x, y)."""
    cv2: Any = importlib.import_module("cv2")
    mx = pts[:, 0].astype(np.float32).reshape(-1, 1)
    my = pts[:, 1].astype(np.float32).reshape(-1, 1)
    u = cv2.remap(
        flow[0].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    v = cv2.remap(
        flow[1].astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return np.concatenate([u, v], axis=1)


def solve_poses(
    sol: PnPSolve,
    tracks: np.ndarray,
    min_inliers: int = 30,
    reproj_px: float = 3.0,
    held: np.ndarray | None = None,
) -> PnPSolve:
    """Per-frame PnP of `sol.points` against `tracks` (T, N, 2). Points in
    `held` (bool, N) are not used by the solve; their median reprojection
    error goes to `frame_heldout_px` (inf when none is visible)."""
    cv2: Any = importlib.import_module("cv2")  # its stubs reject the None arguments

    n = len(sol.points)
    held = np.zeros(n, bool) if held is None else held
    for t in range(tracks.shape[0]):
        vis = np.isfinite(tracks[t, :, 0])
        if t == sol.anchor:
            sol.poses[t] = (np.eye(3), np.zeros(3))
            sol.frame_error_px[t] = 0.0
            sol.frame_inliers[t] = int(vis.sum())
            sol.frame_heldout_px[t] = 0.0
            continue
        fit = vis & ~held
        if fit.sum() < max(min_inliers, 6):
            continue
        obj = sol.points[fit].astype(np.float64)
        img = tracks[t, fit].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj,
            img,
            sol.K,
            None,
            iterationsCount=2000,
            reprojectionError=float(reproj_px),
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inl is None or len(inl) < min_inliers:
            continue
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], sol.K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, sol.K, None)
        err = np.linalg.norm(proj[:, 0] - img[inl], axis=1)
        R, _ = cv2.Rodrigues(rvec)
        sol.poses[t] = (R, tvec.reshape(3))
        sol.frame_error_px[t] = float(np.median(err))
        sol.frame_inliers[t] = int(len(inl))
        test = vis & held
        if test.any():
            hp, _ = cv2.projectPoints(sol.points[test].astype(np.float64), rvec, tvec, sol.K, None)
            herr = np.linalg.norm(hp[:, 0] - tracks[t, test], axis=1)
            sol.frame_heldout_px[t] = float(np.median(herr))
        else:
            sol.frame_heldout_px[t] = float("inf")
    return sol


def assess(
    sol: PnPSolve,
    width: int,
    n_frames: int,
    good_frac: float = 0.0011,
    weak_frac: float = 0.0027,
    nodal_parallax: float = 0.02,
) -> PnPVerdict:
    """Verdict from the held-out errors. Limits are fractions of the solve
    width (resolution independent). GOOD: median held-out error <= good
    limit and >= 90 % of the shot's frames reliable; WEAK: <= weak limit and
    >= 60 %; else FAILED. Travel uses reliable frames only, since an
    unreliable frame can put the camera anywhere."""
    good_px, weak_px = good_frac * width, weak_frac * width
    errs = {t: sol.frame_heldout_px.get(t, float("inf")) for t in sol.poses}
    solved = [e for t, e in errs.items() if t != sol.anchor]
    med = float(np.median(solved)) if solved else float("inf")
    reliable = sorted(t for t, e in errs.items() if e <= weak_px)
    frac = len(reliable) / max(n_frames, 1)
    if med <= good_px and frac >= 0.9:
        verdict = "GOOD"
    elif med <= weak_px and frac >= 0.6:
        verdict = "WEAK"
    else:
        verdict = "FAILED"
    centres = np.array([-sol.poses[t][0].T @ sol.poses[t][1] for t in reliable]).reshape(-1, 3)
    travel = float(np.linalg.norm(centres.max(0) - centres.min(0))) if len(centres) > 1 else 0.0
    depth = float(np.median(sol.points[:, 2])) if len(sol.points) else 0.0
    return PnPVerdict(
        verdict=verdict,
        heldout_px=med,
        reliable=reliable,
        travel_m=travel,
        depth_m=depth,
        nodal=travel < nodal_parallax * depth,
    )


def heldout_from_poses(
    K: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    tracks: np.ndarray,
    anchor: int,
    min_views: int = 3,
    inf_tol_px: float = 0.5,
) -> tuple[dict[int, float], np.ndarray]:
    """Held-out check of FIXED poses (an SfM camera) against independent
    tracks (T, N, 2), NaN where lost. `poses` maps a local frame index to
    `(R_cw, t_cw)` (OpenCV, solve resolution, like `PnPSolve.poses`).

    Leave-one-frame-out: each track is triangulated (DLT on normalised
    coordinates) from every posed frame it is seen in EXCEPT the frame under
    test (`min_views` of them, or 2 when the shot has no more), then
    reprojected into that frame. A track whose finite point explains the
    frames no better than a direction (within `inf_tol_px` per view: no
    parallax, or a camera that only rotates) is a point at infinity, which
    the rotation alone reprojects. Returns the per-frame median error in
    solve px (inf where nothing could be tested, or the points land behind
    the camera) and the triangulated points (N, 3) in the anchor camera's
    frame: NaN where a point could not be triangulated, z = inf for a point
    at infinity, so that `assess` reads the depth and the nodal flag from
    them."""
    T, N = int(tracks.shape[0]), int(tracks.shape[1])
    frames = sorted(t for t in poses if 0 <= t < T)
    errs = {t: float("inf") for t in frames}
    pts = np.full((N, 3), np.nan)
    if N == 0 or len(frames) < 2 or anchor not in poses:
        return errs, pts
    K = np.asarray(K, dtype=np.float64)
    Kinv = np.linalg.inv(K)
    tol = (float(inf_tol_px) / float(K[0, 0])) ** 2  # algebraic cost of one view's noise
    P = {
        t: np.hstack(
            [np.asarray(R, dtype=np.float64), np.asarray(tv, dtype=np.float64).reshape(3, 1)]
        )
        for t, (R, tv) in poses.items()
        if t in errs
    }
    vis = np.zeros((T, N), bool)
    norm = np.zeros((T, N, 2))
    for t in frames:
        v = np.isfinite(tracks[t, :, 0]) & np.isfinite(tracks[t, :, 1])
        xy = np.c_[np.nan_to_num(tracks[t].astype(np.float64)), np.ones(N)] @ Kinv.T
        vis[t], norm[t] = v, xy[:, :2]

    def rows(t: int) -> np.ndarray:
        """(N, 4, 4) normal-equation block of frame t (zero where unseen)."""
        r1 = norm[t, :, :1] * P[t][2] - P[t][0]
        r2 = norm[t, :, 1:2] * P[t][2] - P[t][1]
        r = np.stack([r1, r2], 1) * vis[t][:, None, None]
        return np.einsum("nki,nkj->nij", r, r)

    S = np.zeros((N, 4, 4))
    for t in frames:
        S += rows(t)
    cnt = vis[frames].sum(0)
    # Two views triangulate; `min_views` asks for more where the shot has them.
    need = max(2, min(int(min_views), len(frames) - 1))
    Pf = np.stack([P[t] for t in frames])  # (F, 3, 4)
    use_all = vis[frames]  # (F, N)

    def triangulate(S_sel: np.ndarray, use_sel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Homogeneous points (M, 4) and an at-infinity flag (M,), sign fixed
        so the fitted frames see them in front."""
        cost_fin, V4 = np.linalg.eigh(S_sel)
        X = V4[..., :, 0]
        cost_inf, V3 = np.linalg.eigh(S_sel[:, :3, :3])
        X_inf = np.c_[V3[..., :, 0], np.zeros(len(X))]
        xc = np.einsum("fij,mj->fmi", Pf, X) * use_sel[..., None]  # (F, M, 3)
        # A finite point that beats the direction by less than the noise, or
        # one sitting on the (common) camera centre, is a point at infinity.
        no_gain = cost_inf[:, 0] - cost_fin[:, 0] <= tol * use_sel.sum(0)
        on_centre = np.linalg.norm(xc, axis=2).sum(0) <= 1e-6 * use_sel.sum(0)
        at_inf = no_gain | on_centre
        X = np.where(at_inf[:, None], X_inf, X)
        z = np.einsum("fj,mj->fm", Pf[:, 2], X) * use_sel  # depth in each fitted frame
        sign = np.where(z.sum(0) < 0, -1.0, 1.0)
        return X * sign[:, None], at_inf

    def project(t: int, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (M, 2) of homogeneous points in frame t and an in-front flag."""
        xc = X @ P[t].T
        z = xc[:, 2]
        front = z > 1e-9 * np.linalg.norm(xc, axis=1)
        uv = np.full((len(X), 2), np.inf)
        uv[front] = ((xc[front] / z[front, None]) @ K.T)[:, :2]
        return uv, front

    for k, t in enumerate(frames):
        other = cnt - vis[t]
        ok = vis[t] & (other >= need)
        if not ok.any():
            continue
        use = use_all[:, ok].copy()
        use[k] = False
        X, _inf = triangulate(S[ok] - rows(t)[ok], use)
        uv, front = project(t, X)
        e = np.linalg.norm(uv - tracks[t, ok].astype(np.float64), axis=1)
        e[~front] = np.inf
        errs[t] = float(np.median(e))

    ok = cnt >= need
    if ok.any():
        X, at_inf = triangulate(S[ok], use_all[:, ok])
        xc = X @ P[anchor].T
        w = X[:, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            cart = xc / w[:, None]
        cart[~at_inf & ~(cart[:, 2] > 0)] = np.nan  # behind the anchor: not a scene point
        cart[at_inf] = (np.nan, np.nan, np.inf)
        pts[ok] = cart
    return errs, pts


def assess_poses(
    K: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    tracks: np.ndarray,
    anchor: int,
    width: int,
    n_frames: int,
    good_frac: float = 0.0011,
    weak_frac: float = 0.0027,
    nodal_parallax: float = 0.02,
) -> PnPVerdict:
    """`assess` for FIXED poses (an SfM camera): the held-out errors come
    from `heldout_from_poses`, the travel from `poses`, the depth from the
    triangulated tracks seen from the anchor. Same limits, same verdict."""
    errs, pts = heldout_from_poses(K, poses, tracks, anchor)
    keep = ~np.isnan(pts[:, 2])
    sol = PnPSolve(
        K=np.asarray(K, dtype=np.float64),
        anchor=anchor,
        points=pts[keep],
        colors=np.zeros((int(keep.sum()), 3), np.uint8),
        poses={t: poses[t] for t in errs},
        frame_heldout_px=errs,
    )
    return assess(
        sol,
        width,
        n_frames,
        good_frac=good_frac,
        weak_frac=weak_frac,
        nodal_parallax=nodal_parallax,
    )


__all__ = [
    "PnPSolve",
    "PnPVerdict",
    "anchor_corners",
    "assess",
    "assess_poses",
    "back_project",
    "focal_from_fov",
    "heldout_from_poses",
    "holdout_mask",
    "near_depth_edges",
    "solve_poses",
    "track_klt",
]
