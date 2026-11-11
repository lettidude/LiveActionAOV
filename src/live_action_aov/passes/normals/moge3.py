# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""MoGe-3 geometry pass: metric depth + normals (+ P via PositionFromDepth).

Backend: Microsoft MoGe-3 ViT-L (`Ruicheng/moge-3-vitl`, MIT; code vendored in
`vendored/moge`). One forward pass gives a metric point map, depth, normals
and the camera FOV; MoGe-3's sparse 3D refiner makes the GEOMETRY sharp (no
halo / double edge around people, which MoGe-2's points have). Its normal
head equals MoGe-2's, already 3-8x steadier frame to frame than DSINE
(POC 2026-11-01, `_dev/moge_poc`). ~120 ms/frame and 2.6 GB on a 5090.

Per-frame models wobble, so the pass fixes the shot-level parts:

- **One FOV per shot.** Per frame MoGe's FOV drifts up to ~10 deg on a
  close-up. Priority: `fov_x_deg` param > the camera track (`camera_solve`,
  unless its focal was a guess) > the median of MoGe's own estimate over
  `fov_samples` frames. Every frame then runs with that FOV.
- **Scale.** The metric scale jitters 5-10% frame to frame. `Z` is divided
  by the frame's median-depth ratio to a rolling median over
  `scale_window` frames: jitter goes, a real dolly stays. `Z_raw` keeps the
  per-frame values.
- **Letterbox.** Black bars confuse the model: they are cropped before
  inference and filled back as invalid.

Outputs (`emit_depth` / `emit_normals` pick which, so MoGe-3 can serve the
Depth slot, the Normals slot or both without overwriting another backend):
- `Z`, `Z_raw` - metric depth in metres; sky / invalid = `sky_depth_m`.
- `depth.confidence` - MoGe's validity mask (0 = sky / letterbox).
- `N.x/y/z` - camera-space normals, OpenGL (+Y up, +Z toward camera), like
  DSINE and NormalCrafter; invalid = (0, 0, 1).
- artifacts `depth_metric` + `depth_intrinsics` (normalised fx, fy, cx, cy)
  so PositionFromDepth builds P with the same FOV the model used.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.runtime_env import load_local_first
from live_action_aov.io.channels import (
    CH_DEPTH_CONFIDENCE,
    CH_N_X,
    CH_N_Y,
    CH_N_Z,
    CH_Z,
    CH_Z_RAW,
)
from live_action_aov.passes.normals.dsine import _convert_axes

_log = logging.getLogger(__name__)

_MISSING_DEPS = (
    "MoGe-3 needs its extra dependencies (FlexGEMM + Triton). Run update.bat, "
    "or: uv sync --extra all"
)


class MoGe3Pass(UtilityPass):
    name = "moge3"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes="MoGe-3 (Microsoft, 2026). MIT code + weights; DINOv2 backbone Apache-2.0.",
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    window_safe = True
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Metric depth in metres, scale-stabilised"),
        ChannelSpec(name=CH_Z_RAW, description="Metric depth in metres, per frame"),
        ChannelSpec(name=CH_DEPTH_CONFIDENCE, description="Valid-pixel mask (0 = sky)"),
        ChannelSpec(name=CH_N_X, description="Camera-space normal x (OpenGL), unit-length"),
        ChannelSpec(name=CH_N_Y, description="Camera-space normal y (OpenGL)"),
        ChannelSpec(name=CH_N_Z, description="Camera-space normal z (OpenGL)"),
    ]
    provides_artifacts = ["depth_metric", "depth_intrinsics"]
    optional_artifacts = ["camera_solve"]
    smoothable_channels = [CH_Z, CH_N_X, CH_N_Y, CH_N_Z]

    @classmethod
    def smoothable_for(cls, params: dict[str, Any]) -> list[str]:
        out = [CH_Z] if params.get("emit_depth", True) else []
        return out + ([CH_N_X, CH_N_Y, CH_N_Z] if params.get("emit_normals", True) else [])

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "Ruicheng/moge-3-vitl",
        "emit_depth": True,
        "emit_normals": True,
        "fov_x_deg": None,  # explicit horizontal FOV of the plate; None = auto
        "fov_note": "",  # where a pinned `fov_x_deg` came from (the log, the header)
        # Explicit letterbox box (x, y, w, h); None = detect (`crop_letterbox`).
        "letterbox_box": None,
        "fov_samples": 8,
        "refine_steps": 3,
        "resolution_level": 9,
        "precision": "fp16",
        "scale_window": 9,  # frames; 0 = no scale stabilisation
        "sky_depth_m": 10000.0,
        "crop_letterbox": True,
        "smooth": "auto",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._camera_solve: Any = None
        self._first: int | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        self.fov_source = ""

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def prefetch_weights(self) -> str:
        """Download the checkpoint only (`liveaov prefetch` runs on CPU, where
        the MoGe-3 refiner cannot even be imported)."""
        from huggingface_hub import hf_hub_download

        return str(load_local_first(hf_hub_download, str(self.params["model_id"]), "model.pt"))

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch

        try:
            from live_action_aov.vendored.moge.model.v3 import MoGeModel
        except ImportError as e:
            raise RuntimeError(f"{_MISSING_DEPS} ({e})") from e
        ckpt = self.prefetch_weights()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = MoGeModel.from_pretrained(ckpt).to(self._device).eval()

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        solves = artifacts.get("camera_solve") or {}
        self._camera_solve = next(iter(solves.values()), None)

    # ------------------------------------------------------------------
    # Single-frame lifecycle
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """(1, H, W, 3) float32 sRGB-display -> (3, H, W) tensor on device."""
        import torch

        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(f"MoGe3Pass preprocess expects (1, H, W, 3), got {frames.shape}")
        self._load_model()
        img = np.ascontiguousarray(np.clip(frames[0], 0.0, 1.0), dtype=np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).to(self._device)

    def infer(self, tensor: Any, fov_x: float | None = None) -> dict[str, np.ndarray]:
        """Returns numpy depth (H, W), normal (H, W, 3, OpenCV), mask (H, W)
        and the horizontal FOV in degrees the prediction used."""
        import torch

        assert self._model is not None
        fp16 = self.params["precision"] == "fp16" and self._device.type == "cuda"
        with torch.no_grad():
            r = self._model.infer(
                tensor,
                fov_x=fov_x,
                refine_steps=int(self.params["refine_steps"]),
                resolution_level=int(self.params["resolution_level"]),
                use_fp16=fp16,
            )
        k = r["intrinsics"].float().cpu().numpy()
        mask = r["mask"].cpu().numpy().astype(bool)
        depth = r["depth"].float().cpu().numpy()
        mask &= np.isfinite(depth) & (depth > 0)
        normal = r["normal"].float().cpu().numpy() if "normal" in r else None
        return {
            "depth": np.where(mask, depth, 0.0).astype(np.float32),
            "normal": normal,
            "mask": mask,
            "fov_x": float(np.degrees(2.0 * np.arctan(0.5 / k[0, 0]))),
        }

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """`infer` output (already padded to the frame) -> channels."""
        out: dict[str, np.ndarray] = {}
        mask = tensor["mask"]
        if self.params["emit_depth"]:
            z = np.where(mask, tensor["depth"], float(self.params["sky_depth_m"]))
            out[CH_Z] = z.astype(np.float32)
            out[CH_Z_RAW] = z.astype(np.float32)
            out[CH_DEPTH_CONFIDENCE] = mask.astype(np.float32)
        if self.params["emit_normals"]:
            n = tensor["normal"]
            if n is None:
                raise RuntimeError("this MoGe checkpoint has no normal head")
            n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
            n = _convert_axes(np.moveaxis(n, -1, 0), src="opencv", dst="opengl")
            n[:, ~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)[:, None]
            n = np.clip(n, -1.0, 1.0).astype(np.float32)
            out[CH_N_X], out[CH_N_Y], out[CH_N_Z] = n[0], n[1], n[2]
        return out

    # ------------------------------------------------------------------
    # Shot level
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        frames = list(range(first, last + 1))
        self._first = first
        sample_imgs, run = self._shot_runner(reader, frames)

        out: dict[int, dict[str, np.ndarray]] = {}
        medians: list[float] = []
        for f in frames:
            img = sample_imgs.pop(f, None)
            if img is None:
                img, _ = reader.read_frame(f)
            r = run(img)
            valid = r["depth"][r["mask"]]
            medians.append(float(np.median(valid)) if valid.size else float("nan"))
            out[f] = self.postprocess(r)
        if self.params["emit_depth"]:
            gains = scale_gains(medians, int(self.params["scale_window"]))
            sky = float(self.params["sky_depth_m"])
            for f, g in zip(frames, gains, strict=True):
                if g != 1.0:
                    z = out[f][CH_Z]
                    out[f][CH_Z] = np.where(out[f][CH_DEPTH_CONFIDENCE] > 0, z * g, sky).astype(
                        np.float32
                    )
        return out

    def _shot_runner(
        self, reader: Any, frames: list[int]
    ) -> tuple[dict[int, np.ndarray], Callable[[np.ndarray], dict[str, Any]]]:
        """The shot-level setup (letterbox box, one FOV, intrinsics) and a
        function that runs one full frame with it."""
        sample_imgs, (h, w), box = self._clip_setup(reader, frames)
        samples = list(sample_imgs)
        x0, y0, cw, ch = box
        if box != (0, 0, w, h):
            _log.info("MoGe-3: letterbox cropped to %dx%d at (%d, %d)", cw, ch, x0, y0)

        def run(img: np.ndarray, fov: float | None) -> dict[str, Any]:
            r = self.infer(self.preprocess(img[None, y0 : y0 + ch, x0 : x0 + cw]), fov)
            return pad_result(r, box, (h, w))

        fov = self._shot_fov(
            cw, w, {f: run(sample_imgs[f], None) for f in samples} if self._needs_own_fov() else {}
        )
        fx = cw / (2.0 * math.tan(math.radians(fov) / 2.0))
        self._intrinsics = (fx / w, fx / h, (x0 + cw / 2.0) / w, (y0 + ch / 2.0) / h)
        _log.info("MoGe-3: horizontal FOV %.2f deg (%s)", fov, self.fov_source)
        return sample_imgs, lambda img: run(img, fov)

    def depth_at(
        self, reader: Any, frame_range: tuple[int, int], frames: list[int]
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Metric depth and validity mask of `frames` only, with the same
        shot-level FOV and letterbox `run_shot` uses (a metric reference for
        another depth backend). Sets the intrinsics artifact too."""
        first, last = frame_range
        self._first = first
        sample_imgs, run = self._shot_runner(reader, list(range(first, last + 1)))
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for f in frames:
            img = sample_imgs.get(f)
            if img is None:
                img, _ = reader.read_frame(f)
            r = run(img)
            out[f] = (r["depth"], r["mask"])
        return out

    def _clip_setup(
        self, reader: Any, frames: list[int]
    ) -> tuple[dict[int, np.ndarray], tuple[int, int], tuple[int, int, int, int]]:
        """The frames the shot-level decisions are taken on, the frame size
        and the letterbox box."""
        samples = sample_frames(frames, int(self.params["fov_samples"]))
        sample_imgs = {f: reader.read_frame(f)[0] for f in samples}
        h, w = next(iter(sample_imgs.values())).shape[:2]
        pinned = self.params.get("letterbox_box")
        if pinned:
            box = (int(pinned[0]), int(pinned[1]), int(pinned[2]), int(pinned[3]))
        elif self.params["crop_letterbox"]:
            box = letterbox_box(list(sample_imgs.values()))
        else:
            box = (0, 0, w, h)
        return sample_imgs, (h, w), box

    def probe_clip(self, reader: Any, frame_range: tuple[int, int]) -> dict[str, Any]:
        """One FOV and one letterbox box for the whole clip, so every window
        of a long shot runs with the same ones."""
        frames = list(range(frame_range[0], frame_range[1] + 1))
        sample_imgs, (h, w), box = self._clip_setup(reader, frames)
        out: dict[str, Any] = {"letterbox_box": list(box)}
        if self._needs_own_fov():
            x0, y0, cw, ch = box
            est = {
                f: pad_result(
                    self.infer(self.preprocess(img[None, y0 : y0 + ch, x0 : x0 + cw]), None),
                    box,
                    (h, w),
                )
                for f, img in sample_imgs.items()
            }
            crop_fov = self._shot_fov(cw, w, est)
            fx = cw / (2.0 * math.tan(math.radians(crop_fov) / 2.0))
            out["fov_x_deg"] = math.degrees(2.0 * math.atan(w / (2.0 * fx)))
            out["fov_note"] = self.fov_source
        return out

    def _needs_own_fov(self) -> bool:
        return self.params.get("fov_x_deg") is None and camera_fov(self._camera_solve) is None

    def _shot_fov(self, crop_w: int, frame_w: int, estimates: dict[int, dict[str, Any]]) -> float:
        """Horizontal FOV (degrees) of the CROPPED image, one per shot."""

        def crop_fov(full_fov: float) -> float:
            fx = frame_w / (2.0 * math.tan(math.radians(full_fov) / 2.0))
            return math.degrees(2.0 * math.atan(crop_w / (2.0 * fx)))

        if self.params.get("fov_x_deg") is not None:
            self.fov_source = str(self.params.get("fov_note") or "") or "fov_x_deg param"
            return crop_fov(float(self.params["fov_x_deg"]))
        cam = camera_fov(self._camera_solve)
        if cam is not None:
            self.fov_source = "camera track"
            return crop_fov(cam)
        fovs = [e["fov_x"] for e in estimates.values()]
        self.fov_source = f"MoGe median of {len(fovs)} frames ({min(fovs):.1f}-{max(fovs):.1f})"
        return float(np.median(fovs))

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if self._first is None or not self.params["emit_depth"] or self._intrinsics is None:
            return {}
        return {
            "depth_metric": {self._first: np.asarray([1.0], dtype=np.float32)},
            "depth_intrinsics": {self._first: np.asarray(self._intrinsics, dtype=np.float32)},
        }


# ----------------------------------------------------------------------
# Pure helpers (unit-tested without the model)
# ----------------------------------------------------------------------


def sample_frames(frames: list[int], n: int) -> list[int]:
    """`n` frames spread evenly over the shot (all of them if fewer)."""
    if n <= 0 or len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def letterbox_box(
    images: list[np.ndarray], thresh: float = 0.04, min_frac: float = 0.02
) -> tuple[int, int, int, int]:
    """(x0, y0, w, h) of the active picture: rows / columns black in EVERY
    sample, contiguous from an edge and at least `min_frac` thick, are bars."""
    h, w = images[0].shape[:2]
    peak = np.max(np.stack([im.max(axis=-1) for im in images]), axis=0)
    rows, cols = peak.max(axis=1) > thresh, peak.max(axis=0) > thresh
    if not rows.any() or not cols.any():
        return (0, 0, w, h)

    def span(active: np.ndarray, n: int) -> tuple[int, int]:
        lo = int(np.argmax(active))
        hi = n - int(np.argmax(active[::-1]))
        if lo < min_frac * n:
            lo = 0
        if n - hi < min_frac * n:
            hi = n
        return lo, hi

    y0, y1 = span(rows, h)
    x0, x1 = span(cols, w)
    return (x0, y0, x1 - x0, y1 - y0)


def pad_result(
    r: dict[str, Any], box: tuple[int, int, int, int], size: tuple[int, int]
) -> dict[str, Any]:
    """Put a cropped prediction back into the full frame (outside = invalid)."""
    x0, y0, cw, ch = box
    h, w = size
    if (x0, y0, cw, ch) == (0, 0, w, h):
        return r
    depth = np.zeros((h, w), np.float32)
    mask = np.zeros((h, w), bool)
    depth[y0 : y0 + ch, x0 : x0 + cw] = r["depth"]
    mask[y0 : y0 + ch, x0 : x0 + cw] = r["mask"]
    normal = None
    if r.get("normal") is not None:
        normal = np.zeros((h, w, 3), np.float32)
        normal[..., 2] = -1.0  # OpenCV "toward camera"
        normal[y0 : y0 + ch, x0 : x0 + cw] = r["normal"]
    return {**r, "depth": depth, "mask": mask, "normal": normal}


def camera_fov(solve: Any) -> float | None:
    """Horizontal FOV (degrees, full plate) from a camera solve, or None when
    there is none or its focal is a guess."""
    if solve is None:
        return None
    intr = solve.intrinsics
    if getattr(intr, "source", "") == "guess" or intr.fx_px <= 0:
        return None
    return math.degrees(2.0 * math.atan(intr.width_px / (2.0 * intr.fx_px)))


def scale_gains(medians: list[float], window: int) -> list[float]:
    """Per-frame depth gain: rolling median of log(median depth) over
    `window` frames minus the frame's own -> removes jitter, keeps slow
    real changes (a dolly). Frames with no valid depth get 1.0."""
    m = np.log(np.asarray(medians, dtype=np.float64))
    if window <= 1 or len(m) < 3:
        return [1.0] * len(m)
    half = window // 2
    gains = []
    for i in range(len(m)):
        if not np.isfinite(m[i]):
            gains.append(1.0)
            continue
        seg = m[max(0, i - half) : i + half + 1]
        seg = seg[np.isfinite(seg)]
        gains.append(float(np.exp(np.median(seg) - m[i])))
    return gains


__all__ = ["MoGe3Pass"]
