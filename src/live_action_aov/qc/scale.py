# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""How much a depth pass's global scale steps from frame to frame.

Two numbers per pair of frames (t-1, t), both on the static background:
pixels with valid depth on both frames and outside every matte / mask
(grown by `object_px`) on both frames.

- ``rel_change``: median of |Z_t - Z_t-1| / Z_t, everything that moves;
- ``scale_step``: |median of Z_t / Z_t-1 - 1|, the global scale part;
- ``residual``: median |ln(Z_t / Z_t-1) - ln(1 + step)|, what is left once
  the global scale is taken out (local flicker).

By default the two frames are compared pixel for pixel, which is exact on
a locked-off shot. With ``warp=True`` and the run's motion vectors, frame t
is warped onto t-1 first and only pixels whose vectors survive a
forward/backward round trip count, so a camera move is measured too.
Values are percent: median, 95th percentile and max over the pairs.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import cv2
import numpy as np

from live_action_aov.io.channels import CH_BACK_X, CH_BACK_Y, CH_MOTION_X, CH_MOTION_Y, CH_Z
from live_action_aov.qc.stability import Frame, _objects, _valid, _warp

#: Fewer static pixels than this in a pair: the pair is not measured.
MIN_STATIC_PX = 1000

Grid = tuple[np.ndarray, np.ndarray]


def pixel_grid(shape: tuple[int, ...]) -> Grid:
    h, w = shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return gx, gy


def pair_static(
    a: Frame,
    b: Frame,
    *,
    warp: bool,
    kernel: np.ndarray,
    sky: float,
    grid: Grid,
    round_trip_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth of `a`, depth of `b` brought onto `a`'s pixels, and the static
    mask, for one pair of frames (float64 depths, NaN where not valid)."""
    za = a[CH_Z].astype(np.float64)
    zb32 = np.where(_valid(b, sky), b[CH_Z], np.nan).astype(np.float32)
    obj_b = _objects(b, kernel).astype(np.float32)
    ok_rt = np.ones(za.shape, bool)
    if warp and CH_MOTION_X in a and CH_BACK_X in b:
        mx, my = a[CH_MOTION_X], a[CH_MOTION_Y]
        bx = _warp(b[CH_BACK_X], mx, my, grid)
        by = _warp(b[CH_BACK_Y], mx, my, grid)
        ok_rt = np.nan_to_num(np.hypot(mx + bx, my + by), nan=np.inf) < round_trip_px
        zb32 = _warp(zb32, mx, my, grid)
        obj_b = _warp(obj_b, mx, my, grid)
    zb = zb32.astype(np.float64)
    static = (
        ok_rt
        & _valid(a, sky)
        & np.isfinite(zb)
        & (zb > 0)
        & ~_objects(a, kernel)
        & (np.nan_to_num(obj_b, nan=1.0) < 0.5)
    )
    return za, zb, static


def _summary(name: str, v: list[float]) -> dict[str, float]:
    if not v:
        return {}
    a = np.asarray(v, dtype=np.float64) * 100
    return {
        f"{name}_med": round(float(np.median(a)), 4),
        f"{name}_p95": round(float(np.percentile(a, 95)), 4),
        f"{name}_max": round(float(a.max()), 4),
    }


def scale_steps(
    frames: list[int],
    get: Callable[[int], Frame],
    *,
    warp: bool = False,
    object_px: int = 15,
    sky_m: float = 5000.0,
) -> tuple[dict[str, float], list[str]]:
    """The two numbers (and the residual) for one depth pass.

    `get(frame)` returns the frame's channels: Z (+ depth.confidence), any
    mask / matte channels, and the motion / back vectors when `warp`."""
    frames = sorted(frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(object_px) + 1,) * 2)
    rel: list[float] = []
    step: list[float] = []
    resid: list[float] = []
    share: list[float] = []
    grid: Grid | None = None
    prev = get(frames[0]) if frames else None
    for t0, t1 in pairwise(frames):
        a, b = prev, get(t1)
        prev = b
        if a is None or t1 != t0 + 1:
            continue
        if grid is None:
            grid = pixel_grid(a[CH_Z].shape)
        # (t-1, t) = (a, b): Z_t over Z_t-1 on the static set.
        za, zb, static = pair_static(a, b, warp=warp, kernel=kernel, sky=sky_m, grid=grid)
        share.append(float(static.mean()))
        if static.sum() < MIN_STATIC_PX:
            continue
        lr = np.log(zb[static]) - np.log(za[static])
        s = float(np.exp(np.median(lr))) - 1.0
        rel.append(float(np.median(np.abs(zb[static] - za[static]) / zb[static])))
        step.append(abs(s))
        resid.append(float(np.median(np.abs(lr - np.log1p(s)))))
    out: dict[str, float] = {
        "pairs": float(len(step)),
        "static_share": round(float(np.median(share)) * 100, 2) if share else 0.0,
    }
    out.update(_summary("rel_change", rel))
    out.update(_summary("scale_step", step))
    out.update(_summary("residual", resid))
    warnings = [] if step else ["scale check: too little static background to measure"]
    return out, warnings


def grade(scale_step_med_pct: float, good: float = 1.0, weak: float = 2.0) -> str:
    """GOOD / WEAK / FAILED on the median global scale step (percent)."""
    if scale_step_med_pct <= good:
        return "GOOD"
    return "WEAK" if scale_step_med_pct <= weak else "FAILED"


__all__ = ["MIN_STATIC_PX", "grade", "pair_static", "pixel_grid", "scale_steps"]
