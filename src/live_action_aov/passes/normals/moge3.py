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
  unless its focal was a guess or no real lens, or the camera check said FAILED
  or nodal: `camera_fov_from`) > the median of MoGe's own estimate over
  `fov_samples` frames. Every frame then runs with that FOV.
- **Scale.** The metric scale jitters from frame to frame. `scale_ref`:
  - `static` (default when the run has mattes or motion vectors): each
    frame's scale step is measured on what does not move only, then
    smoothed. When the motion vectors say the background is still (a
    locked-off shot): every pixel whose vectors stay near zero, masked or
    not, smoothed over the long `still_window` (a nodal camera gets it
    too; short enough that a long clip run in windows gives the same
    depth as in one go, 0 = one scale for the whole run). Otherwise: outside every SAM mask, through the motion vectors
    when there are some, smoothed over `scale_window` frames so a real
    push-in stays. No camera needed. When most of the shot has too
    little static background, it falls back to `frame` and the run report
    says the scale is not stabilised.
  - `frame` (the previous behaviour): the frame's median-depth ratio to a
    rolling median over `scale_window` frames. Moving people near camera
    move that median, so it does little on a locked-off shot.
  `Z_raw` keeps the per-frame values.
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
from itertools import pairwise
from typing import Any

import numpy as np

from live_action_aov.core.hardware import WINDOW_OVERLAP
from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.core.runtime_env import load_local_first
from live_action_aov.io.camera_schema import PLAUSIBLE_FOV_X_DEG, plausible_fov_x
from live_action_aov.io.channels import (
    CH_BACK_X,
    CH_BACK_Y,
    CH_DEPTH_CONFIDENCE,
    CH_MOTION_X,
    CH_MOTION_Y,
    CH_N_X,
    CH_N_Y,
    CH_N_Z,
    CH_Z,
    CH_Z_RAW,
)
from live_action_aov.passes.normals.dsine import _convert_axes
from live_action_aov.qc.scale import MIN_STATIC_PX, pair_static, pixel_grid
from live_action_aov.qc.stability import _valid

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
    optional_artifacts = ["camera_solve", "sam3_hard_masks", "forward_flow", "backward_flow"]
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
        # "auto" = "static" when the run has SAM masks or motion vectors, else
        # "frame" (the previous behaviour, kept as is).
        "scale_ref": "auto",
        "static_min_share": 0.5,  # share of frame pairs the static set must measure
        "still_px": 0.5,  # median background motion (px) below which the shot is still
        # Frames of the scale smoothing on a still background or a nodal camera.
        # Up to 2 x the window overlap + 1, a clip run in windows matches one go.
        "still_window": 2 * WINDOW_OVERLAP + 1,
        "object_px": 15,  # grow the SAM masks by this before leaving them out
        # "usable" = never a camera the check marked FAILED or nodal;
        # "any" = the previous behaviour (only a guessed focal is refused).
        "camera_fov_from": "usable",
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
        self._tracks: list[tuple[dict[int, int], Any]] = []
        self._fwd: dict[int, Any] = {}
        self._bwd: dict[int, Any] = {}
        self._report: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._first: int | None = None
        self._intrinsics: tuple[float, float, float, float] | None = None
        # Letterbox box (x, y, w, h) of the last shot setup, in full-frame pixels.
        self.letterbox: tuple[int, int, int, int] | None = None
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
        hard = artifacts.get("sam3_hard_masks") or {}
        tracks = next(iter(hard.values()), None) or {}
        self._tracks = [
            ({fr: i for i, fr in enumerate(t.get("frames") or [])}, t.get("stack"))
            for t in tracks.values()
            if t.get("stack") is not None
        ]
        self._fwd = dict(artifacts.get("forward_flow") or {})
        self._bwd = dict(artifacts.get("backward_flow") or {})

    def report_fields(self) -> dict[str, Any]:
        if not self._report and not self._warnings:
            return {}
        metrics = {
            k: float(v)
            for k, v in self._report.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        }
        notes = {k: v for k, v in self._report.items() if isinstance(v, str)}
        return {"metrics": metrics, "extra": notes, "warnings": list(self._warnings)}

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
            gains = self._scale_gains(frames, out, medians)
            sky = float(self.params["sky_depth_m"])
            for f, g in zip(frames, gains, strict=True):
                if g != 1.0:
                    z = out[f][CH_Z]
                    out[f][CH_Z] = np.where(out[f][CH_DEPTH_CONFIDENCE] > 0, z * g, sky).astype(
                        np.float32
                    )
        return out

    def _scale_gains(
        self, frames: list[int], out: dict[int, dict[str, np.ndarray]], medians: list[float]
    ) -> list[float]:
        """Per-frame depth gain (see the module docstring, `scale_ref`)."""
        window = int(self.params["scale_window"])
        old = scale_gains(medians, window)
        mode = str(self.params.get("scale_ref", "auto"))
        if mode == "auto":
            mode = "static" if (self._tracks or self._fwd) else "frame"
        self._report["fov_source"] = self.fov_source
        if mode != "static" or window <= 0 or len(frames) < 3:
            self._report["scale_ref"] = "frame"
            return old
        data = [self._static_frame(f, out[f]) for f in frames]
        still_px = float(self.params["still_px"])
        still = background_still(data, still_px)
        steps = static_log_steps(
            data,
            warp=bool(self._fwd),
            object_px=int(self.params["object_px"]),
            sky=float(self.params["sky_depth_m"]),
            still_px=still_px if still else None,
        )
        measured = sum(s is not None for s in steps) / max(1, len(steps))
        self._report["static_pairs_measured"] = round(measured, 3)
        if measured < float(self.params["static_min_share"]):
            self._report["scale_ref"] = "frame"
            self._warnings.append(
                "MoGe-3: scale not stabilised on the static background (too little of it: "
                "most of the frame moves or is masked); the frame-median scale is used"
            )
            return old
        nodal = getattr(getattr(self._camera_solve, "quality", None), "nodal", None) is True
        if bool(still) or nodal:
            window = max(0, int(self.params["still_window"]))
            why = "still background" if still else "nodal camera"
            self._report["scale_ref"] = (
                f"static, {window} frames ({why})" if window else f"static, whole run ({why})"
            )
        else:
            self._report["scale_ref"] = f"static, {window} frames"
        m = np.log(np.asarray(medians, dtype=np.float64))
        fallback = [float(v) for v in np.diff(m)]
        return static_scale_gains(steps, fallback, window)

    def _static_frame(self, f: int, ch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """What the static-set measure needs of one frame: Z, validity, the
        union of the SAM masks and the motion vectors (when in the run)."""
        z = ch[CH_Z_RAW] if CH_Z_RAW in ch else ch[CH_Z]
        fr: dict[str, np.ndarray] = {CH_Z: z, CH_DEPTH_CONFIDENCE: ch[CH_DEPTH_CONFIDENCE]}
        union = _mask_union(self._tracks, f, z.shape)
        if union is not None:
            fr["mask.objects"] = union
        fw, bw = self._fwd.get(f), self._bwd.get(f)
        if fw is not None and bw is not None and np.shape(fw)[-2:] == z.shape:
            fr[CH_MOTION_X], fr[CH_MOTION_Y] = np.asarray(fw[0]), np.asarray(fw[1])
            fr[CH_BACK_X], fr[CH_BACK_Y] = np.asarray(bw[0]), np.asarray(bw[1])
        return fr

    def _shot_runner(
        self, reader: Any, frames: list[int]
    ) -> tuple[dict[int, np.ndarray], Callable[[np.ndarray], dict[str, Any]]]:
        """The shot-level setup (letterbox box, one FOV, intrinsics) and a
        function that runs one full frame with it."""
        sample_imgs, (h, w), box = self._clip_setup(reader, frames)
        samples = list(sample_imgs)
        self.letterbox = box
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
        trust = str(self.params.get("camera_fov_from", "usable"))
        return (
            self.params.get("fov_x_deg") is None and camera_fov(self._camera_solve, trust) is None
        )

    def _shot_fov(self, crop_w: int, frame_w: int, estimates: dict[int, dict[str, Any]]) -> float:
        """Horizontal FOV (degrees) of the CROPPED image, one per shot."""

        def crop_fov(full_fov: float) -> float:
            fx = frame_w / (2.0 * math.tan(math.radians(full_fov) / 2.0))
            return math.degrees(2.0 * math.atan(crop_w / (2.0 * fx)))

        if self.params.get("fov_x_deg") is not None:
            self.fov_source = str(self.params.get("fov_note") or "") or "fov_x_deg param"
            return crop_fov(float(self.params["fov_x_deg"]))
        trust = str(self.params.get("camera_fov_from", "usable"))
        cam = camera_fov(self._camera_solve, trust)
        if cam is not None:
            self.fov_source = "camera track"
            return crop_fov(cam)
        refused = camera_refused(self._camera_solve) if trust != "any" else ""
        raw = _raw_camera_fov(self._camera_solve)
        fovs = [e["fov_x"] for e in estimates.values()]
        self.fov_source = f"MoGe median of {len(fovs)} frames ({min(fovs):.1f}-{max(fovs):.1f})"
        if not refused and raw is not None:
            # Neither FAILED nor nodal, yet refused: the lens itself is impossible.
            self.fov_source += "; camera track ignored (no real lens)"
            msg = (
                f"MoGe-3: the camera track's lens (horizontal FOV {raw:.1f} deg) is not a real "
                "lens - ignored, MoGe-3 estimates the FOV itself"
            )
            _log.warning(msg)
            if msg not in self._warnings:
                self._warnings.append(msg)
        if refused:
            self.fov_source += f"; camera track ignored ({refused})"
            msg = (
                f"MoGe-3: the camera track is {refused}, so its lens is not used; "
                "MoGe-3 estimates the FOV itself"
            )
            _log.warning(msg)
            if msg not in self._warnings:
                self._warnings.append(msg)
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


def camera_refused(solve: Any) -> str:
    """Why a camera's lens is not trusted for depth: "FAILED" (the camera
    check says do not use it), "nodal" (no parallax), or "" (usable)."""
    q = getattr(solve, "quality", None)
    if q is None:
        return ""
    if str(getattr(q, "verdict", "") or "").strip().upper().startswith("FAILED"):
        return "FAILED"
    if getattr(q, "nodal", None) is True:
        return "nodal"
    return ""


def camera_fov(solve: Any, trust: str = "usable") -> float | None:
    """Horizontal FOV (degrees, full plate) from a camera solve, or None when
    there is none, its focal is a guess or no real lens (a failed track), or
    (`trust="usable"`) the camera check marked it FAILED or nodal. `trust="any"`
    skips the FAILED / nodal check only: an impossible lens is never used."""
    if solve is None:
        return None
    if trust != "any" and camera_refused(solve):
        return None
    fov = _raw_camera_fov(solve)
    # The range the camera pass judged it with (its `lens_fov_x_deg_range`).
    bounds = getattr(solve, "_lens_fov_x_deg_range", PLAUSIBLE_FOV_X_DEG)
    return fov if plausible_fov_x(fov, bounds) else None


def _raw_camera_fov(solve: Any) -> float | None:
    """The solve's horizontal FOV before any trust check, None without one."""
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


def _mask_union(
    tracks: list[tuple[dict[int, int], Any]], frame: int, shape: tuple[int, ...]
) -> np.ndarray | None:
    """Union of the SAM hard masks on `frame` (float 0/1), None when none."""
    out: np.ndarray | None = None
    for index, stack in tracks:
        i = index.get(frame)
        if i is None:
            continue
        m = np.asarray(stack[i]) > 0
        if m.shape != tuple(shape[:2]):
            import cv2

            small = m.astype(np.uint8)
            m = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        out = m if out is None else (out | m)
    return None if out is None else out.astype(np.float32)


def static_log_steps(
    frames: list[dict[str, np.ndarray]],
    *,
    warp: bool,
    object_px: int,
    sky: float,
    still_px: float | None = None,
) -> list[float | None]:
    """Median ln(Z_t+1 / Z_t) on the static part of each consecutive pair;
    None where too few static pixels.

    Static is, by default: outside every mask (grown by `object_px`), valid
    on both frames, through the motion vectors when `warp`. With `still_px`
    (a shot whose background does not move): every valid pixel whose motion
    vectors stay below `still_px` on both frames, masked or not, so a parked
    car or a standing person counts and only what moves is left out."""
    import cv2

    if not frames:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(object_px) + 1,) * 2)
    grid = pixel_grid(frames[0][CH_Z].shape)
    out: list[float | None] = []
    for a, b in pairwise(frames):
        if still_px is not None and CH_MOTION_X in a and CH_BACK_X in b:
            za = a[CH_Z].astype(np.float64)
            zb = b[CH_Z].astype(np.float64)
            st = (
                _valid(a, sky)
                & _valid(b, sky)
                & (np.hypot(a[CH_MOTION_X], a[CH_MOTION_Y]) < still_px)
                & (np.hypot(b[CH_BACK_X], b[CH_BACK_Y]) < still_px)
            )
        else:
            za, zb, st = pair_static(a, b, warp=warp, kernel=kernel, sky=sky, grid=grid)
        if st.sum() < MIN_STATIC_PX:
            out.append(None)
        else:
            out.append(float(np.median(np.log(zb[st]) - np.log(za[st]))))
    return out


def static_scale_gains(
    steps: list[float | None], fallback: list[float], window: int
) -> list[float]:
    """Per-frame gains from the static-background scale steps.

    The steps chain into a log scale per frame (`fallback[i]`, the
    whole-frame step, where a pair was not measured). `window` 0 = one scale
    for all the frames given (the median frame's level, so the metres stay);
    otherwise a centred rolling mean over `window` frames is kept, so a slow
    real change (a push-in) survives and the jitter goes. A rolling window
    only looks `window // 2` frames each way, so a clip run in windows with
    at least that much context gives the same gains as in one go."""
    c = np.zeros(len(steps) + 1, dtype=np.float64)
    for i, s in enumerate(steps):
        v = s if s is not None else fallback[i]
        c[i + 1] = c[i] + (v if np.isfinite(v) else 0.0)
    if window <= 0:
        target = np.full_like(c, float(np.median(c)))
    else:
        half = window // 2
        target = np.array([c[max(0, i - half) : i + half + 1].mean() for i in range(len(c))])
    return [float(g) for g in np.exp(target - c)]


def background_still(frames: list[dict[str, np.ndarray]], still_px: float) -> bool | None:
    """True when the static background does not move (median motion over the
    shot below `still_px`), False when it does, None without motion vectors."""
    mags = []
    for fr in frames:
        if CH_MOTION_X not in fr:
            continue
        mv = np.hypot(fr[CH_MOTION_X], fr[CH_MOTION_Y])
        keep = fr[CH_DEPTH_CONFIDENCE] > 0.5
        obj = fr.get("mask.objects")
        if obj is not None:
            keep &= obj < 0.5
        if keep.any():
            mags.append(float(np.median(mv[keep])))
    if not mags:
        return None
    return float(np.median(mags)) < still_px


__all__ = [
    "MoGe3Pass",
    "background_still",
    "camera_fov",
    "camera_refused",
    "static_log_steps",
    "static_scale_gains",
]
