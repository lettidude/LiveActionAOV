# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Steady metric depth: Video Depth Anything's shape over time, in metres.

Video Depth Anything (Small, Apache-2.0) is a video model: its depth is
steady from frame to frame, but relative (no unit). MoGe-3 (MIT) is a
per-frame model: it gives metres and the shot FOV, but its depth moves from
frame to frame. This pass keeps the first's steadiness and the second's
metres:

1. VDA runs on the whole clip (relative disparity, or the metric Small
   weights with `vda_source: metric`).
2. MoGe-3 runs on a reference frame every `ref_every` frames, with the same
   shot FOV and letterbox the MoGe-3 pass would use.
3. On the pixels both see, VDA is fitted to MoGe-3: scale + shift in
   disparity (relative source) or one scale (metric source). The fit is
   robust (worst residuals dropped) so moving subjects and MoGe-3's own
   outliers do not drive it. `align_window` = 0 pools every reference into
   ONE fit for the clip (nothing can flicker, the default); N > 0 pools the
   references within N frames, then blends the fits linearly. Windows follow
   MoGe-3's own slow scale wander too, so keep 0 unless a long dolly needs it.

   `fit: mono` (relative source, one fit for the clip) maps VDA to MoGe-3
   with a monotone curve instead of a straight line: medians of quantile
   bins made non-decreasing (pool-adjacent-violators), piecewise linear,
   still one function for the whole clip. It follows MoGe-3 better when a
   straight line squashes the far range. Beyond its end nodes it continues
   with the curve's overall slope, held so it never turns over.

`Z` is metres; P comes from this Z with MoGe-3's FOV (`depth_intrinsics`),
so P and Z agree. The camera track keeps its own MoGe-3 depth: this pass
does not touch it.

Guards:
- A shot where the fit cannot be trusted (too few usable pixels, or VDA
  and MoGe-3 that do not agree on what is near) ships RELATIVE depth: one
  monotone function of VDA per clip, median 1, never inverted, flagged in
  the run report and not stamped metric.
- With `keep_moge_z` MoGe-3's own per-frame metres are written as a
  separate `moge.Z` layer, for camera and metric checks.
- When MoGe-3's own scale wanders through the shot, the one fit follows its
  average: the run report says so.
- VDA is softer than MoGe-3: thin and distant detail is lost.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_Z, CH_Z_MOGE

_log = logging.getLogger(__name__)

#: Pixels sampled per reference frame for the fit (deterministic stride).
_FIT_SAMPLES = 40_000

#: Mono fit: quantile bins, and the smallest slope a segment may have (a
#: share of the end-to-end chord) so no flat step merges different depths.
_MONO_BINS = 32
_MONO_CHORD = 0.05


class SteadyMetricDepthPass(UtilityPass):
    name = "video_depth_steady"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0 / MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Video Depth Anything Small (relative or metric weights, Apache-2.0) "
            "aligned to MoGe-3 ViT-L (MIT). Base / Large VDA weights are "
            "CC-BY-NC-4.0 and never used here."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    temporal_window = 32
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Depth in metres, steady over time"),
        ChannelSpec(name=CH_DEPTH_CONFIDENCE, description="Valid-pixel mask (0 = sky / far)"),
        ChannelSpec(name=CH_Z_MOGE, description="MoGe-3 depth per frame, metres (keep_moge_z)"),
    ]
    provides_artifacts = ["depth_metric", "depth_intrinsics"]
    optional_artifacts = ["camera_solve"]
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "vda_source": "relative",  # "relative" (disparity) | "metric" (metric Small)
        "input_size": 518,
        "precision": "fp16",
        "ref_every": 8,  # MoGe-3 reference frame every N frames (+ the last)
        "align_window": 0,  # frames; 0 = one fit for the whole clip
        # "mono" (monotone curve; relative source, one fit) | "affine" (scale + shift).
        # Mono asked with the metric source or windows runs affine, and says so.
        "fit": "mono",
        "fit_keep": 0.8,  # fraction of best-fitting pixels kept by the robust fit
        # Below this share of the picture usable for the fit, ship relative depth.
        "min_fit_fraction": 0.1,
        # Run-report warning when MoGe-3's own scale varies more than this
        # (coefficient of variation over the reference frames).
        "scale_drift_warn": 0.15,
        "keep_moge_z": True,  # write MoGe-3's per-frame Z as its own layer (moge.Z)
        "far_factor": 4.0,  # depth beyond far_factor x MoGe's 99th pct = sky
        "sky_depth_m": 10000.0,
        # Passed to MoGe-3 (same meaning as there).
        "fov_x_deg": None,
        "fov_note": "",
        "letterbox_box": None,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if self.params["vda_source"] not in ("relative", "metric"):
            raise ValueError("vda_source must be 'relative' or 'metric'")
        if self.params["fit"] not in ("affine", "mono"):
            raise ValueError("fit must be 'affine' or 'mono'")
        self._fit = str(self.params["fit"])
        self._fit_note = ""
        if self._fit == "mono" and (
            self.params["vda_source"] != "relative" or int(self.params["align_window"]) > 0
        ):
            self._fit = "affine"
            self._fit_note = "fit 'mono' needs the relative source and one fit per clip: ran affine"
        self._camera_artifacts: dict[str, dict[int, Any]] = {}
        self._first: int | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self._metric = True
        self.fit_report: dict[str, Any] = {}
        self._warnings: list[str] = []

    def prefetch_weights(self) -> str:
        """Download both checkpoints (VDA Small of the chosen source, MoGe-3)
        without building the models (`liveaov prefetch` runs on CPU)."""
        from huggingface_hub import hf_hub_download

        from live_action_aov.core.runtime_env import load_local_first
        from live_action_aov.passes.depth import video_depth_anything as vda
        from live_action_aov.passes.normals.moge3 import MoGe3Pass

        metric = self.params["vda_source"] == "metric"
        repo = (vda._METRIC_REPOS if metric else vda._VARIANT_REPOS)["vits"]
        name = (vda._METRIC_WEIGHTS if metric else vda._VARIANT_WEIGHTS)["vits"]
        load_local_first(hf_hub_download, repo_id=repo, filename=name)
        return MoGe3Pass({}).prefetch_weights()

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        self._camera_artifacts = {"camera_solve": artifacts.get("camera_solve") or {}}

    # VIDEO_CLIP: everything goes through run_shot.
    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        raise NotImplementedError("SteadyMetricDepthPass is VIDEO_CLIP; use run_shot")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError("SteadyMetricDepthPass is VIDEO_CLIP; use run_shot")

    def run_shot(
        self, reader: Any, frame_range: tuple[int, int]
    ) -> dict[int, dict[str, np.ndarray]]:
        from live_action_aov.io.channels import CH_Z_RAW
        from live_action_aov.passes.depth.video_depth_anything import VideoDepthAnythingPass
        from live_action_aov.passes.normals.moge3 import MoGe3Pass

        first, last = frame_range
        frames = list(range(first, last + 1))
        self._first = first
        metric_src = self.params["vda_source"] == "metric"

        vda = VideoDepthAnythingPass(
            {
                "variant": "vits",
                "metric": metric_src,
                "input_size": self.params["input_size"],
                "precision": self.params["precision"],
            }
        )
        try:
            vda_out = vda.run_shot(reader, frame_range)
        finally:
            vda.unload()
            _free_gpu()
        raw = {f: vda_out[f][CH_Z_RAW] for f in frames}
        del vda_out

        refs = reference_frames(frames, int(self.params["ref_every"]))
        keep_moge = bool(self.params["keep_moge_z"])
        moge = MoGe3Pass(
            {
                "emit_normals": False,
                "fov_x_deg": self.params.get("fov_x_deg"),
                "fov_note": self.params.get("fov_note", ""),
                "letterbox_box": self.params.get("letterbox_box"),
            }
        )
        moge.ingest_artifacts(self._camera_artifacts)
        try:
            moge_depth = moge.depth_at(reader, frame_range, frames if keep_moge else refs)
            self._intrinsics = moge._intrinsics
            box = moge.letterbox
        finally:
            moge.unload()
            _free_gpu()
        sky = float(self.params["sky_depth_m"])

        # Letterbox bars are not picture: invalid on every frame, as in MoGe-3.
        h, w = raw[frames[0]].shape
        picture = np.zeros((h, w), bool)
        x0, y0, cw, ch = box or (0, 0, w, h)
        picture[y0 : y0 + ch, x0 : x0 + cw] = True

        samples = {}
        far_ref = []
        for f in refs:
            z_ref, mask = moge_depth[f]
            x, y = fit_samples(raw[f], z_ref, mask, metric_src)
            samples[f] = (x, y)
            if mask.any():
                far_ref.append(float(np.percentile(z_ref[mask], 99.0)))
        n_fit = sum(
            int((moge_depth[f][1] & np.isfinite(raw[f]) & (raw[f] > 0)).sum()) for f in refs
        )
        fit_fraction = n_fit / max(1, len(refs) * int(picture.sum()))
        keep = float(self.params["fit_keep"])
        window = int(self.params["align_window"])
        params = None
        mono = None
        degenerate: list[str] = []
        if not far_ref or not any(len(s[0]) for s in samples.values()):
            degenerate.append("MoGe-3 found no valid depth on the reference frames")
        else:
            if fit_fraction < float(self.params["min_fit_fraction"]):
                degenerate.append(f"only {fit_fraction:.0%} of the picture usable for the fit")
            params = window_fits(samples, frames, window, metric_src, keep)
            if not all(np.isfinite(p[0]) and p[0] > 0 for p in params.values()):
                degenerate.append(
                    "VDA and MoGe-3 disagree on what is near (fit slope not positive)"
                )
            if not degenerate and self._fit == "mono":
                x = np.concatenate([s[0] for s in samples.values()])
                y = np.concatenate([s[1] for s in samples.values()])
                mono = mono_fit(x, y)
                if mono is None:
                    degenerate.append("the monotone fit is flat")

        self._metric = not degenerate
        self._warnings = [self._fit_note] if self._fit_note else []
        if degenerate:
            msg = "depth is RELATIVE, not metres: " + "; ".join(degenerate)
            self._warnings.append(msg)
            _log.warning("Steady depth: %s", msg)
        far = float(self.params["far_factor"]) * float(np.median(far_ref)) if far_ref else 0.0
        drift = None
        if self._metric:
            drift = scale_drift(samples, params, mono, metric_src)
            if drift is not None and drift > float(self.params["scale_drift_warn"]):
                self._warnings.append(
                    f"MoGe-3's own scale wanders through the shot (variation {drift:.2f}): "
                    "the metres follow its average"
                )
        self.fit_report = {
            "fit": ("mono" if mono is not None else "affine") if self._metric else "relative",
            "references": len(refs),
            "fit_fraction": round(fit_fraction, 4),
            "far_m": far,
            "moge_scale_variation": None if drift is None else round(drift, 4),
        }
        if params:
            self.fit_report["a_range"] = (
                min(p[0] for p in params.values()),
                max(p[0] for p in params.values()),
            )
        _log.info(
            "Steady depth: VDA (%s) fitted to MoGe-3 on %d reference frames, %s, fit %s",
            "metric" if metric_src else "relative",
            len(refs),
            "one fit for the clip" if window <= 0 else f"{window}-frame windows",
            self.fit_report["fit"],
        )
        rel = (1.0, 0.0)
        if not self._metric:
            rel = relative_scale(raw, frames, picture, metric_src, float(self.params["far_factor"]))
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in frames:
            if not self._metric:
                z, valid = apply_relative(raw[f], rel[0], metric_src, rel[1])
            elif mono is not None:
                z, valid = apply_mono(raw[f], mono, far)
            else:
                assert params is not None
                a, b = params[f]
                z, valid = apply_fit(raw[f], a, b, metric_src, far)
            valid &= picture
            out[f] = {
                CH_Z: np.where(valid, z, sky).astype(np.float32),
                CH_DEPTH_CONFIDENCE: valid.astype(np.float32),
            }
            if keep_moge:
                zm, mm = moge_depth[f]
                out[f][CH_Z_MOGE] = np.where(mm & picture, zm, sky).astype(np.float32)
        return out

    def report_fields(self) -> dict[str, Any]:
        if not self.fit_report:
            return {}
        metrics = {
            k: float(v)
            for k, v in self.fit_report.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        return {
            "engine": f"steady depth, fit {self.fit_report['fit']}",
            "metrics": metrics,
            "warnings": list(self._warnings),
        }

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if self._first is None or self._intrinsics is None:
            return {}
        out = {"depth_intrinsics": {self._first: np.asarray(self._intrinsics, dtype=np.float32)}}
        if self._metric:
            out["depth_metric"] = {self._first: np.asarray([1.0], dtype=np.float32)}
        return out


# ----------------------------------------------------------------------
# Pure helpers (unit-tested without the models)
# ----------------------------------------------------------------------


def _free_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def reference_frames(frames: list[int], every: int) -> list[int]:
    """Every `every`-th frame from the first, plus the last."""
    if not frames:
        return []
    every = max(1, every)
    refs = frames[::every]
    if refs[-1] != frames[-1]:
        refs.append(frames[-1])
    return refs


def fit_samples(
    vda: np.ndarray, z_ref: np.ndarray, mask: np.ndarray, metric_src: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) pairs for the fit: x = VDA value, y = what it must map to
    (MoGe inverse depth for disparity, MoGe log depth for metric)."""
    if vda.shape != z_ref.shape:
        raise ValueError(f"VDA {vda.shape} and MoGe {z_ref.shape} depth differ in size")
    ok = mask & np.isfinite(vda) & np.isfinite(z_ref) & (z_ref > 0)
    ok &= vda > 0
    idx = np.flatnonzero(ok)
    if idx.size > _FIT_SAMPLES:
        idx = idx[:: idx.size // _FIT_SAMPLES][:_FIT_SAMPLES]
    x = vda.ravel()[idx].astype(np.float64)
    zr = z_ref.ravel()[idx].astype(np.float64)
    if metric_src:
        return np.log(x), np.log(zr)
    return x, 1.0 / zr


def robust_fit(x: np.ndarray, y: np.ndarray, metric_src: bool, keep: float) -> tuple[float, float]:
    """Disparity: y = a x + b with b >= 0, least squares on the RELATIVE
    residual (near and far pixels count alike) over the `keep` fraction of
    pixels that fit best (3 rounds). b >= 0 keeps every surface the video
    model sees at a finite depth. Metric: one log offset (median), returned
    as (scale, 0)."""
    if x.size == 0:
        return (1.0, 0.0) if metric_src else (float("nan"), float("nan"))
    if metric_src:
        return float(np.exp(np.median(y - x))), 0.0
    sel = np.ones(x.size, dtype=bool)
    a = b = 0.0
    for _ in range(3):
        w = 1.0 / np.maximum(y[sel], 1e-9)
        xs = x[sel] * w
        A = np.stack([xs, w], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, np.ones(int(sel.sum())), rcond=None)
        if b < 0.0:
            a, b = float(np.dot(xs, np.ones_like(xs)) / max(np.dot(xs, xs), 1e-12)), 0.0
        r = np.abs(a * x + b - y) / np.maximum(y, 1e-9)
        sel = r <= np.quantile(r, keep)
    return float(a), float(b)


def window_fits(
    samples: dict[int, tuple[np.ndarray, np.ndarray]],
    frames: list[int],
    window: int,
    metric_src: bool,
    keep: float,
) -> dict[int, tuple[float, float]]:
    """Fit parameters for every frame: one pooled fit (window <= 0), or a
    pooled fit per reference frame over the references within `window`
    frames, blended linearly between references."""
    refs = sorted(samples)

    def pooled(sel: list[int]) -> tuple[float, float]:
        x = np.concatenate([samples[r][0] for r in sel])
        y = np.concatenate([samples[r][1] for r in sel])
        return robust_fit(x, y, metric_src, keep)

    if window <= 0 or len(refs) < 2:
        p = pooled(refs)
        return dict.fromkeys(frames, p)
    half = window / 2.0
    fits = [pooled([q for q in refs if abs(q - r) <= half] or [r]) for r in refs]
    a = np.interp(frames, refs, [p[0] for p in fits])
    b = np.interp(frames, refs, [p[1] for p in fits])
    return {f: (float(a[i]), float(b[i])) for i, f in enumerate(frames)}


def apply_fit(
    vda: np.ndarray, a: float, b: float, metric_src: bool, far: float
) -> tuple[np.ndarray, np.ndarray]:
    """Depth in metres and a validity mask (False = sky / beyond `far`)."""
    if metric_src:
        z = vda.astype(np.float64) * a
    else:
        disp = a * vda.astype(np.float64) + b
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(disp > 0, 1.0 / disp, np.inf)
    valid = np.isfinite(z) & (z > 0) & (z <= far)
    return np.where(valid, z, 0.0).astype(np.float32), valid


MonoFit = tuple[np.ndarray, np.ndarray, float, float]


def _pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the weighted least-squares non-decreasing fit."""
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for yi, wi in zip(y.tolist(), w.tolist(), strict=True):
        vals.append(yi)
        wts.append(wi)
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wsum = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wsum
            wts[-2] = wsum
            cnt[-2] += cnt[-1]
            vals.pop()
            wts.pop()
            cnt.pop()
    return np.repeat(np.asarray(vals), cnt)


def mono_fit(x: np.ndarray, y: np.ndarray, bins: int = _MONO_BINS) -> MonoFit | None:
    """Monotone map disparity -> inverse depth: medians of quantile bins of
    x, made non-decreasing (PAV), flat steps given a small slope so the
    curve is strictly increasing. Returns (nodes x, nodes y, slope below,
    slope above), or None when the curve is flat or there is too little.
    Below the first node the slope is held so inverse depth stays >= 0 at
    zero disparity: far surfaces stay finite, nothing turns over."""
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if x.size < 4 * bins:
        return None
    edges = np.quantile(x, np.linspace(0.0, 1.0, bins + 1))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, bins - 1)
    xs, ys, ws = [], [], []
    for k in range(bins):
        sel = idx == k
        if sel.sum() >= 4:
            xs.append(float(np.median(x[sel])))
            ys.append(float(np.median(y[sel])))
            ws.append(float(sel.sum()))
    if len(xs) < 2:
        return None
    xa, ya = np.asarray(xs), _pav(np.asarray(ys), np.asarray(ws))
    xu, inv = np.unique(xa, return_inverse=True)
    yu = np.full(xu.size, -np.inf)
    np.maximum.at(yu, inv, ya)
    if xu.size < 2:
        return None
    chord = (yu[-1] - yu[0]) / (xu[-1] - xu[0])
    if not np.isfinite(chord) or chord <= 0.0:
        return None
    for i in range(1, yu.size):  # lift flat steps only; rising parts are kept as fitted
        yu[i] = max(yu[i], yu[i - 1] + _MONO_CHORD * chord * (xu[i] - xu[i - 1]))
    s_lo = min(chord, yu[0] / xu[0])
    return xu, yu, float(s_lo), float(chord)


def mono_eval(d: np.ndarray, fit: MonoFit) -> np.ndarray:
    """Inverse depth from disparity through the monotone map (linear tails)."""
    xs, ys, s_lo, s_hi = fit
    d = d.astype(np.float64)
    y = np.interp(d, xs, ys)
    y = np.where(d < xs[0], ys[0] + s_lo * (d - xs[0]), y)
    return np.where(d > xs[-1], ys[-1] + s_hi * (d - xs[-1]), y)


def apply_mono(vda: np.ndarray, fit: MonoFit, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Depth in metres through the monotone map, and the validity mask."""
    disp = mono_eval(vda, fit)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(disp > 0, 1.0 / disp, np.inf)
    valid = np.isfinite(z) & (z > 0) & (z <= far)
    return np.where(valid, z, 0.0).astype(np.float32), valid


def scale_drift(
    samples: dict[int, tuple[np.ndarray, np.ndarray]],
    params: dict[int, tuple[float, float]] | None,
    mono: MonoFit | None,
    metric_src: bool,
) -> float | None:
    """How much MoGe-3's own scale varies over the reference frames against
    the fit: coefficient of variation of the per-frame median ratio."""
    ratios = []
    for r, (x, y) in samples.items():
        if x.size < 100:
            continue
        if metric_src:
            assert params is not None
            ratios.append(float(np.exp(np.median(y - x - np.log(params[r][0])))))
            continue
        if mono is not None:
            pred = mono_eval(x, mono)
        else:
            assert params is not None
            pred = params[r][0] * x + params[r][1]
        ok = pred > 0
        if ok.sum() >= 100:
            ratios.append(float(np.median(y[ok] / pred[ok])))
    if len(ratios) < 2:
        return None
    arr = np.asarray(ratios)
    return float(arr.std() / max(arr.mean(), 1e-9))


def relative_scale(
    raw: dict[int, np.ndarray],
    frames: list[int],
    picture: np.ndarray,
    metric_src: bool,
    far_factor: float,
) -> tuple[float, float]:
    """For relative output: one clip-wide scale so the median depth is 1,
    and the far limit (far_factor x the 99th percentile)."""
    vals = []
    step = max(1, len(frames) // 16)
    for f in frames[::step]:
        v = raw[f][picture]
        v = v[np.isfinite(v) & (v > 0)]
        vals.append(v[:: max(1, v.size // 20_000)])
    pooled = np.concatenate(vals) if vals else np.ones(1)
    if pooled.size == 0:
        pooled = np.ones(1)
    depth = pooled if metric_src else 1.0 / pooled
    med = float(np.median(depth))
    return med, far_factor * float(np.percentile(depth, 99.0)) / med


def apply_relative(
    vda: np.ndarray, scale: float, metric_src: bool, far: float
) -> tuple[np.ndarray, np.ndarray]:
    """Relative depth (clip median = 1), the same function on every frame,
    monotone in VDA: never inverted."""
    v = vda.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = v / scale if metric_src else np.where(v > 0, 1.0 / (v * scale), np.inf)
    valid = np.isfinite(z) & (z > 0) & (z <= far)
    return np.where(valid, z, 0.0).astype(np.float32), valid


__all__ = ["SteadyMetricDepthPass"]
