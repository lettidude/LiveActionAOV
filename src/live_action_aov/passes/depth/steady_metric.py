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

`Z` is metres; P comes from this Z with MoGe-3's FOV (`depth_intrinsics`),
so P and Z agree. The camera track keeps its own MoGe-3 depth: this pass
does not touch it.
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
from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_Z

_log = logging.getLogger(__name__)

#: Pixels sampled per reference frame for the fit (deterministic stride).
_FIT_SAMPLES = 40_000


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
        "fit_keep": 0.8,  # fraction of best-fitting pixels kept by the robust fit
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
        self._camera_artifacts: dict[str, dict[int, Any]] = {}
        self._first: int | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self.fit_report: dict[str, Any] = {}

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
            ref_depth = moge.depth_at(reader, frame_range, refs)
            self._intrinsics = moge._intrinsics
            box = moge.letterbox
        finally:
            moge.unload()
            _free_gpu()

        samples = {}
        far_ref = []
        for f, (z_ref, mask) in ref_depth.items():
            x, y = fit_samples(raw[f], z_ref, mask, metric_src)
            samples[f] = (x, y)
            if mask.any():
                far_ref.append(float(np.percentile(z_ref[mask], 99.0)))
        if not far_ref or not any(len(s[0]) for s in samples.values()):
            raise RuntimeError("MoGe-3 found no valid depth on the reference frames")
        params = window_fits(
            samples,
            frames,
            int(self.params["align_window"]),
            metric_src,
            float(self.params["fit_keep"]),
        )
        far = float(self.params["far_factor"]) * float(np.median(far_ref))
        sky = float(self.params["sky_depth_m"])
        self.fit_report = {
            "references": len(refs),
            "far_m": far,
            "a_range": (min(p[0] for p in params.values()), max(p[0] for p in params.values())),
        }
        _log.info(
            "Steady depth: VDA (%s) fitted to MoGe-3 on %d reference frames, %s",
            "metric" if metric_src else "relative",
            len(refs),
            "one fit for the clip"
            if int(self.params["align_window"]) <= 0
            else f"{self.params['align_window']}-frame windows",
        )
        # Letterbox bars are not picture: invalid on every frame, as in MoGe-3.
        h, w = raw[frames[0]].shape
        picture = np.zeros((h, w), bool)
        x0, y0, cw, ch = box or (0, 0, w, h)
        picture[y0 : y0 + ch, x0 : x0 + cw] = True
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in frames:
            a, b = params[f]
            z, valid = apply_fit(raw[f], a, b, metric_src, far)
            valid &= picture
            out[f] = {
                CH_Z: np.where(valid, z, sky).astype(np.float32),
                CH_DEPTH_CONFIDENCE: valid.astype(np.float32),
            }
        return out

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if self._first is None or self._intrinsics is None:
            return {}
        return {
            "depth_metric": {self._first: np.asarray([1.0], dtype=np.float32)},
            "depth_intrinsics": {self._first: np.asarray(self._intrinsics, dtype=np.float32)},
        }


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


__all__ = ["SteadyMetricDepthPass"]
