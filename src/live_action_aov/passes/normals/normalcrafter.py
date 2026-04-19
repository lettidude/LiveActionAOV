"""NormalCrafter pass (VIDEO_CLIP, temporal-native normals, spec §13.1 Phase 2).

Backend: `diffusers` + the NormalCrafter weights (Stable-X/NormalCrafter),
built on Stable Video Diffusion. Same license story as DepthCrafter — the
weights are Apache-2.0 but the SVD backbone keeps the whole pipeline behind
`--allow-noncommercial` until upstream clears it. Weights + diffusers are
an optional extra: `pip install live-action-aov[normalcrafter]`.

Lifecycle mirrors DepthCrafter:
1. `run_shot` reads every frame in the requested range.
2. Frames are tiled into windows (`plan_window_starts`).
3. Each window is preprocessed / inferred / postprocessed — the model
   returns camera-space normals of shape `(window, 3, H, W)`.
4. `stitch_windowed_predictions` blends the overlap regions with the
   trapezoid weighting used by DepthCrafter.
5. After stitching, two spec traps bite:
   - **Trap 2**: stitching is a weighted *blend* of already-unit normals,
     which breaks unit length in the overlap. We renormalize each frame
     in the stitched clip by `N / max(||N||, eps)`.
   - **Axis convention**: NormalCrafter publishes its normals in OpenCV
     camera-space (+X right, +Y down, +Z forward-into-scene). Spec §10.3
     requires OpenGL/Maya (+X right, +Y up, +Z toward camera). The output
     is converted with the same helper DSINE uses (flip Y and Z).

Tests bypass `diffusers` via `_load_model` + `_infer_window` subclass
overrides so CI stays fast and offline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_N_X, CH_N_Y, CH_N_Z
from live_action_aov.passes.normals.dsine import _convert_axes
from live_action_aov.shared.video_clip import (
    plan_window_starts,
    stitch_windowed_predictions,
)


class NormalCrafterPass(UtilityPass):
    name = "normalcrafter"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0+SVD-NC",
        commercial_use=False,
        commercial_tool_resale=False,
        notes=(
            "Weights are Apache-2.0 but NormalCrafter is built on Stable "
            "Video Diffusion, whose commercial terms are non-permissive. "
            "Gated behind --allow-noncommercial. Verify with upstream "
            "authors before any commercial deployment."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    temporal_window = 14
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_N_X, description="Camera-space normal x, [-1,1] unit-length"),
        ChannelSpec(name=CH_N_Y, description="Camera-space normal y"),
        ChannelSpec(name=CH_N_Z, description="Camera-space normal z"),
    ]
    smoothable_channels: list[str] = []   # VIDEO_CLIP: already temporally coherent

    DEFAULT_PARAMS: dict[str, Any] = {
        "window": 14,
        "overlap": 2,
        "num_inference_steps": 10,
        "guidance_scale": 1.0,
        "inference_short_edge": 576,
        "precision": "fp16",             # SVD-family pipeline: fp16 default
        "model_id": "Yanrui95/NormalCrafter",
        # Axis convention — NormalCrafter = OpenCV, spec = OpenGL.
        "input_axes": "opencv",
        "output_axes": "opengl",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if int(self.params["overlap"]) >= int(self.params["window"]):
            raise ValueError(
                f"overlap ({self.params['overlap']}) must be < window "
                f"({self.params['window']})"
            )
        self._pipeline: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._frame_keys: list[int] = []
        self._plate_shape: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Model lifecycle (diffusers — heavy; tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import diffusers  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "NormalCrafter requires the `diffusers` package. "
                "Install via: pip install live-action-aov[normalcrafter]"
            ) from e
        import torch
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(
            str(self.params["model_id"]),
            custom_pipeline=str(self.params["model_id"]),
            torch_dtype=torch.float16 if self.params["precision"] == "fp16" else torch.float32,
            trust_remote_code=True,
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params["precision"] == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        pipe = pipe.to(self._device)
        self._pipeline = pipe

    # ------------------------------------------------------------------
    # Per-window lifecycle (one sliding window at a time)
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Input: (W_frames, H, W_px, 3) float32 sRGB-display in [0,1]."""
        import torch

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"NormalCrafter preprocess expects (W, H, W_px, 3), got {frames.shape}"
            )
        self._load_model()
        n, plate_h, plate_w, _ = frames.shape
        short = int(self.params["inference_short_edge"])
        scale = short / max(min(plate_h, plate_w), 1)
        inf_h = max(64, int(round(plate_h * scale / 8)) * 8)
        inf_w = max(64, int(round(plate_w * scale / 8)) * 8)

        img = np.clip(frames, 0.0, 1.0).astype(np.float32, copy=False)
        t = torch.from_numpy(img).permute(0, 3, 1, 2)   # (N, 3, H, W)
        t = torch.nn.functional.interpolate(
            t, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        return {
            "video": t.to(self._device, dtype=self._dtype),
            "plate_shape": (plate_h, plate_w),
            "inf_shape": (inf_h, inf_w),
            "n_frames": n,
        }

    def infer(self, tensor: Any) -> Any:
        """Delegates to `_infer_window` so tests can inject a fake without
        reimplementing the surrounding plumbing."""
        pred = self._infer_window(tensor)
        return {
            "normals": pred,
            "plate_shape": tensor["plate_shape"],
        }

    def _infer_window(self, tensor: Any) -> Any:
        """Run the diffusers pipeline on a window. Overridden in tests."""
        import torch

        assert self._pipeline is not None
        with torch.no_grad():
            result = self._pipeline(
                video=tensor["video"],
                num_inference_steps=int(self.params["num_inference_steps"]),
                guidance_scale=float(self.params["guidance_scale"]),
            )
        # NormalCrafter pipelines typically return `.normals` or `.frames`;
        # normalize to a torch.Tensor of shape (N, 3, H, W).
        normals = getattr(result, "normals", None)
        if normals is None and hasattr(result, "frames"):
            normals = result.frames
        if normals is None:
            raise RuntimeError(
                "NormalCrafter pipeline returned no `.normals`/`.frames` attribute"
            )
        return normals

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale window normals to plate res, renormalize to unit length
        (trap 2), return the raw axis-convention tensors as an (N, 3, H, W)
        array keyed by CH_N_{X,Y,Z}.

        Axis conversion is deferred to `run_shot` so it happens once after
        stitching (the trapezoid blend is done in the native axis space to
        avoid double-flipping).
        """
        import torch

        plate_h, plate_w = tensor["plate_shape"]
        n = tensor["normals"]
        if n.ndim != 4 or n.shape[1] != 3:
            raise ValueError(
                f"Expected normals (B, 3, h, w), got {tuple(n.shape)}"
            )
        n_up = torch.nn.functional.interpolate(
            n, size=(plate_h, plate_w), mode="bilinear", align_corners=False
        )
        mag = torch.sqrt((n_up ** 2).sum(dim=1, keepdim=True)).clamp_min(1e-6)
        n_unit = n_up / mag
        arr = n_unit.cpu().numpy().astype(np.float32)          # (N, 3, H, W)
        return {
            CH_N_X: arr[:, 0],
            CH_N_Y: arr[:, 1],
            CH_N_Z: arr[:, 2],
        }

    # ------------------------------------------------------------------
    # Shot-level: window plan → stitch → renormalize → axis convert
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        n_frames = last - first + 1
        window = min(int(self.params["window"]), n_frames)
        overlap = min(int(self.params["overlap"]), max(window - 1, 0))

        frames = np.stack(
            [reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0
        )
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])

        starts = plan_window_starts(n_frames, window, overlap)
        # Accumulate each channel separately so the stitcher can broadcast
        # 2-D per-frame predictions.
        nx_preds: list[np.ndarray] = []
        ny_preds: list[np.ndarray] = []
        nz_preds: list[np.ndarray] = []
        for s in starts:
            clip = frames[s : s + window]
            model_in = self.preprocess(clip)
            model_out = self.infer(model_in)
            partial = self.postprocess(model_out)
            nx_preds.append(partial[CH_N_X])
            ny_preds.append(partial[CH_N_Y])
            nz_preds.append(partial[CH_N_Z])

        nx = stitch_windowed_predictions(nx_preds, starts, n_frames, overlap)
        ny = stitch_windowed_predictions(ny_preds, starts, n_frames, overlap)
        nz = stitch_windowed_predictions(nz_preds, starts, n_frames, overlap)

        # Trap 2 again: weighted blend over the overlap broke unit length.
        stacked = np.stack([nx, ny, nz], axis=1)              # (N, 3, H, W)
        mag = np.sqrt((stacked ** 2).sum(axis=1, keepdims=True))
        mag = np.maximum(mag, 1e-6)
        stacked = stacked / mag

        # Axis convention (opencv → opengl) applied once post-stitch.
        src = str(self.params.get("input_axes", "opencv"))
        dst = str(self.params.get("output_axes", "opengl"))
        if src.lower() != dst.lower():
            converted = np.empty_like(stacked)
            for i in range(stacked.shape[0]):
                converted[i] = _convert_axes(stacked[i], src=src, dst=dst)
            stacked = converted

        stacked = np.clip(stacked, -1.0, 1.0).astype(np.float32, copy=False)

        self._frame_keys = list(range(first, last + 1))
        self._plate_shape = (plate_h, plate_w)
        out: dict[int, dict[str, np.ndarray]] = {}
        for i, f in enumerate(self._frame_keys):
            out[f] = {
                CH_N_X: stacked[i, 0],
                CH_N_Y: stacked[i, 1],
                CH_N_Z: stacked[i, 2],
            }
        return out


__all__ = ["NormalCrafterPass"]
