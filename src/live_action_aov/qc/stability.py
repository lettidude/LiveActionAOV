# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""How steady a delivered depth pass is over time (run-report QC block).

Reads the sidecars as written (float32), so it measures what the artist
gets. For every pair of frames (t, t+1) the depth of t+1 is warped back onto
t with the run's own motion vectors, then compared on two pixel sets:

- static: the vectors survive a forward/backward round trip, outside every
  matte / mask (grown), depth valid on both frames;
- edge band: pixels on a depth edge (|grad ln Z| above `edge_grad`, grown by
  `edge_px`) where the round trip holds. Edges are where flicker shows most
  and where a softer depth hides it, so the band's area is reported too.

Two units side by side, because they rank depth passes differently:
- ``dlnz``: |ln Z(t+1, warped) - ln Z(t)|, the relative change (scale-free);
- ``range``: |Z(t+1, warped) - Z(t)| over the shot's depth range (p1-p99 of
  every valid pixel of the whole clip, frozen once per shot, so windows and
  before/after runs share it). It under-weights the far range.
``jitter`` is the whole-frame scale pumping: |median of ln Z(t+1)/Z(t)| over
the static set. Values are percent: median and 95th percentile over every
pair. No verdict: numbers only.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from live_action_aov.io.channels import (
    CH_BACK_X,
    CH_BACK_Y,
    CH_DEPTH_CONFIDENCE,
    CH_MOTION_X,
    CH_MOTION_Y,
    CH_Z,
)

DEFAULTS: dict[str, float] = {
    "round_trip_px": 1.0,  # forward + backward vectors must close within this
    "object_px": 15,  # grow mattes / masks by this before excluding them
    "edge_grad": 0.05,  # |grad ln Z| per pixel that marks a depth edge
    "edge_px": 4,  # edge band half-width
    "min_round_trip": 0.5,  # warn below this share of round-trip-valid pixels
    "sky_m": 5000.0,  # depths at or beyond this are sky fill, not measured
}
_SAMPLES = 50_000  # pixels kept per pair and set (deterministic stride)

Frame = dict[str, np.ndarray]


def _warp(
    img: np.ndarray, fx: np.ndarray, fy: np.ndarray, grid: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """`img` sampled at p + (fx, fy); NaN where that lands outside the frame."""
    gx, gy = grid
    x, y = gx + fx, gy + fy
    out = cv2.remap(img.astype(np.float32), x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    h, w = img.shape
    inside = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return np.where(inside, out, np.nan).astype(np.float32)


def _valid(f: Frame, sky: float) -> np.ndarray:
    z = f[CH_Z]
    ok = np.isfinite(z) & (z > 0) & (z < sky)
    conf = f.get(CH_DEPTH_CONFIDENCE)
    return ok & (conf > 0.5) if conf is not None else ok


def _objects(f: Frame, kernel: np.ndarray) -> np.ndarray:
    m = [v for k, v in f.items() if k.startswith(("mask", "matte."))]
    if not m:
        return np.zeros(f[CH_Z].shape, bool)
    u = (np.max(np.stack(m), axis=0) > 0.02).astype(np.uint8)
    return cv2.dilate(u, kernel).astype(bool)


def _sub(a: np.ndarray) -> np.ndarray:
    return a[:: max(1, a.size // _SAMPLES)]


def frozen_range(frames: list[int], get: Callable[[int], Frame], sky: float) -> tuple[float, float]:
    """p1 / p99 of every valid depth pixel of the whole clip (strided)."""
    vals = []
    for f in frames:
        fr = get(f)
        z = fr[CH_Z][_valid(fr, sky)]
        vals.append(z[:: max(1, z.size // 20_000)])
    pooled = np.concatenate(vals) if vals else np.zeros(0)
    if pooled.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(pooled, [1.0, 99.0])
    return float(lo), float(hi)


def depth_stability(
    frames: list[int], get: Callable[[int], Frame], **params: float
) -> tuple[dict[str, float], list[str]]:
    """Metrics (percent unless named otherwise) and warnings for one depth pass.

    `get(frame)` returns the frame's channels: Z (+ depth.confidence), the
    motion / back vectors in pixels, and any mask / matte channels."""
    p = {**DEFAULTS, **params}
    sky = float(p["sky_m"])
    frames = sorted(frames)
    if len(frames) < 2:
        return {}, ["stability QC needs at least two frames"]
    first = get(frames[0])
    if CH_MOTION_X not in first or CH_BACK_X not in first:
        return {}, ["stability QC needs the flow pass in the run (motion vectors)"]
    # One read per frame: the range (p1-p99 of the whole clip, frozen once)
    # is gathered on the way and applied at the end (|dZ| / range).
    zvals: list[np.ndarray] = []

    def _zsample(fr: Frame) -> None:
        z = fr[CH_Z][_valid(fr, sky)]
        zvals.append(z[:: max(1, z.size // 20_000)])

    h, w = first[CH_Z].shape
    gx0, gy0 = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = (gx0, gy0)
    k_obj = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(p["object_px"]) + 1,) * 2)
    k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(p["edge_px"]) + 1,) * 2)
    acc: dict[str, list[np.ndarray]] = {k: [] for k in ("s_dlnz", "s_rng", "e_dlnz", "e_rng")}
    jitter, rt_share, edge_share, static_share = [], [], [], []
    pairs = 0
    prev = first
    _zsample(first)
    for t, t1 in pairwise(frames):
        a = prev
        b = get(t1)
        prev = b
        _zsample(b)
        if t1 != t + 1:
            continue
        mx, my = a[CH_MOTION_X], a[CH_MOTION_Y]
        bx = _warp(b[CH_BACK_X], mx, my, grid)
        by = _warp(b[CH_BACK_Y], mx, my, grid)
        rt = np.nan_to_num(np.hypot(mx + bx, my + by), nan=np.inf) < float(p["round_trip_px"])
        rt_share.append(float(rt.mean()))
        za = a[CH_Z].astype(np.float64)
        zb = _warp(np.where(_valid(b, sky), b[CH_Z], np.nan), mx, my, grid).astype(np.float64)
        both = rt & _valid(a, sky) & np.isfinite(zb) & (zb > 0)
        obj_b = _warp(_objects(b, k_obj).astype(np.float32), mx, my, grid)
        static = both & ~_objects(a, k_obj) & (np.nan_to_num(obj_b, nan=1.0) < 0.5)
        lnz = np.log(np.where(_valid(a, sky), za, np.nan))
        gy_, gx_ = np.gradient(lnz)
        edge = np.nan_to_num(np.hypot(gx_, gy_), nan=0.0) > float(p["edge_grad"])
        edge = cv2.dilate(edge.astype(np.uint8), k_edge).astype(bool)
        band = both & edge
        static_share.append(float(static.mean()))
        edge_share.append(float(edge.mean()))
        if static.sum() >= 1000:
            d = np.log(zb[static]) - np.log(za[static])
            acc["s_dlnz"].append(_sub(np.abs(d)))
            acc["s_rng"].append(_sub(np.abs(zb[static] - za[static])))
            jitter.append(float(abs(np.median(d))))
            pairs += 1
        if band.sum() >= 1000:
            acc["e_dlnz"].append(_sub(np.abs(np.log(zb[band]) - np.log(za[band]))))
            acc["e_rng"].append(_sub(np.abs(zb[band] - za[band])))

    pooled = np.concatenate(zvals) if zvals else np.zeros(0)
    rng = float(np.subtract(*np.percentile(pooled, [99.0, 1.0]))) if pooled.size else 0.0
    out: dict[str, float] = {
        "pairs": float(pairs),
        "range_m": round(rng, 4),
        "round_trip_valid": round(float(np.median(rt_share)) * 100, 2) if rt_share else 0.0,
        "static_share": round(float(np.median(static_share)) * 100, 2) if static_share else 0.0,
        "edge_share": round(float(np.median(edge_share)) * 100, 2) if edge_share else 0.0,
    }
    names = {
        "s_dlnz": "static_dlnz",
        "s_rng": "static_range",
        "e_dlnz": "edge_dlnz",
        "e_rng": "edge_range",
    }
    for k, name in names.items():
        if acc[k] and (k.endswith("dlnz") or rng > 0):
            v = np.concatenate(acc[k]) / (rng if k.endswith("rng") else 1.0)
            out[f"{name}_med"] = round(float(np.median(v)) * 100, 4)
            out[f"{name}_p95"] = round(float(np.percentile(v, 95)) * 100, 4)
    if jitter:
        out["jitter_med"] = round(float(np.median(jitter)) * 100, 4)
    warnings = []
    if rt_share and np.median(rt_share) < float(p["min_round_trip"]):
        warnings.append(
            "stability QC: the motion vectors fail the round trip on most of the frame, "
            "so these numbers are unreliable"
        )
    if pairs == 0:
        warnings.append("stability QC: too few static pixels to measure")
    return out, warnings


def read_frame(paths: dict[str, Path], wanted: tuple[str, ...]) -> Frame:
    """The channels of one frame that start with `wanted`, from the files
    they were written to (one multilayer EXR, or one per pass), float32."""
    import OpenImageIO as oiio

    out: Frame = {}
    for path in sorted(set(paths.values())):
        buf = oiio.ImageBuf(str(path))
        spec = buf.spec()
        names = list(spec.channelnames)
        # A channel listed in `paths` is read from its own file only.
        keep = [
            i for i, n in enumerate(names) if n.startswith(wanted) and paths.get(n, path) == path
        ]
        if not keep:
            continue
        px = buf.get_pixels(oiio.FLOAT, roi=spec.roi)
        for i in keep:
            out.setdefault(names[i], np.asarray(px[..., i], np.float32))
    return out


def depth_stability_from_files(
    index: dict[int, dict[str, Path]], **params: float
) -> tuple[dict[str, float], list[str]]:
    """`depth_stability` on the written sidecars (`index[frame][channel] = path`)."""
    wanted = (CH_Z, CH_DEPTH_CONFIDENCE, "motion.", "back.", "mask", "matte.")
    cache: dict[int, Frame] = {}

    def get(f: int) -> Frame:
        if f not in cache:
            if len(cache) > 2:
                cache.clear()
            fr = read_frame(index[f], wanted)
            fr.pop("Z_raw", None)
            cache[f] = fr
        return cache[f]

    frames = [f for f in sorted(index) if CH_Z in index[f]]
    return depth_stability(frames, get, **params)


__all__ = ["DEFAULTS", "depth_stability", "depth_stability_from_files", "frozen_range"]
