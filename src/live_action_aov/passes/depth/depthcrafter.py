"""DepthCrafter pass (VIDEO_CLIP, temporal-native depth, spec §13.1 Phase 2).

Backend: `diffusers` + the Tencent DepthCrafter weights (built on Stable
Video Diffusion). The license stays non-commercial (SVD terms) until the
upstream authors clear it, so the CLI gates this pass behind
`--allow-noncommercial`. Weights and the `diffusers` dependency are
optional extras — install via `pip install live-action-aov[depthcrafter]`.

Lifecycle:
1. `run_shot` reads every frame in the requested range into memory.
2. Frames are tiled into windows of `window` frames with `overlap` frames
   of overlap (`plan_window_starts`).
3. Each window is preprocessed, inferred, postprocessed — the backend
   returns a `(window, H, W)` relative-depth tensor.
4. `stitch_windowed_predictions` blends overlap regions with a trapezoid
   weighting so the crossfade is smooth and clip endpoints are not
   attenuated.
5. Per-clip normalization (spec §2.2, trap 5): single `(d_min, d_max)`
   across the entire stitched clip → `Z = 1 - (d - d_min)/(d_max - d_min)`.

Tests bypass the real diffusers pipeline via `_load_model` + `_infer_window`
subclass overrides, which keeps CI fast and offline.
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
from live_action_aov.io.channels import CH_Z, CH_Z_RAW
from live_action_aov.shared.video_clip import (
    plan_window_starts,
    stitch_windowed_predictions,
)


class DepthCrafterPass(UtilityPass):
    name = "depthcrafter"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0+SVD-NC",
        commercial_use=False,
        commercial_tool_resale=False,
        notes=(
            "Weights are Apache-2.0 but DepthCrafter is built on Stable Video "
            "Diffusion, whose commercial terms are non-permissive. Gated "
            "behind --allow-noncommercial. Verify with upstream authors "
            "before any commercial deployment."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    temporal_window = 110
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Relative depth, per-clip normalized [0,1] (1=near)"),
        ChannelSpec(name=CH_Z_RAW, description="Raw stitched model output (un-normalized)"),
    ]
    smoothable_channels: list[str] = []  # VIDEO_CLIP: already temporally coherent

    DEFAULT_PARAMS: dict[str, Any] = {
        "window": 110,
        "overlap": 25,
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        # DepthCrafter's own preprocessing (see vendored utils.py
        # `read_video_frames`) scales based on the LONG edge:
        #   scale = max_res / max(H, W)
        #   (H, W) = round(H*scale/64)*64, round(W*scale/64)*64
        # This is NOT the same as short-edge scaling — on wide plates
        # (2048×1080, 3840×2160, etc.) the two produce very different
        # inference shapes, only one of which matches the SVD UNet's
        # trained skip-connection sizes. `max_res=1024` is DepthCrafter's
        # documented default. Kept the legacy `inference_short_edge`
        # alias so existing YAML configs keep working.
        "max_res": 1024,
        "inference_short_edge": 1024,  # legacy alias, now means max_res
        "precision": "fp16",  # fp16 default — SVD is VRAM-heavy on fp32
        # DepthCrafter is distributed as a UNet *swap* for Stable Video
        # Diffusion. Two HF repos are involved: the UNet weights (at
        # `tencent/DepthCrafter` — `config.json` + `.safetensors` only)
        # and the rest of the SVD pipeline (text/image encoders, VAE,
        # scheduler) from `stabilityai/stable-video-diffusion-img2vid-xt`.
        "unet_model_id": "tencent/DepthCrafter",
        "svd_base_model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        # Legacy alias preserved so existing YAMLs using `model_id` keep
        # working — `_load_model` reads `unet_model_id` first.
        "model_id": "tencent/DepthCrafter",
        "cpu_offload": None,  # None | "model" | "sequential"
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        if int(self.params["overlap"]) >= int(self.params["window"]):
            raise ValueError(
                f"overlap ({self.params['overlap']}) must be < window ({self.params['window']})"
            )
        self._pipeline: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Per-clip normalization constants recorded by run_shot.
        self._norm_min: float = 0.0
        self._norm_max: float = 1.0
        self._frame_keys: list[int] = []

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
                "DepthCrafter requires the `diffusers` package. "
                "Install via: pip install live-action-aov[depthcrafter]"
            ) from e
        import torch

        from live_action_aov.vendored.depthcrafter.depth_crafter_ppl import (
            DepthCrafterPipeline,
        )
        from live_action_aov.vendored.depthcrafter.unet import (
            DiffusersUNetSpatioTemporalConditionModelDepthCrafter,
        )

        unet_id = str(self.params.get("unet_model_id") or self.params["model_id"])
        svd_base = str(self.params["svd_base_model_id"])
        use_fp16 = str(self.params["precision"]).lower() == "fp16"
        weight_dtype = torch.float16 if use_fp16 else torch.float32

        # Step 1: DepthCrafter's custom UNet swaps into an SVD base. The
        # repo at `tencent/DepthCrafter` holds ONLY the UNet weights
        # (config.json + .safetensors at root), not a full diffusers
        # pipeline — that's why DiffusionPipeline.from_pretrained was
        # hitting a 404 looking for model_index.json. Load the UNet
        # directly with no subfolder.
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_id,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        # Step 2: the rest of the pipeline (VAE, scheduler, image encoder)
        # comes from the SVD-xt base, with DepthCrafter's UNet plugged in.
        pipe = DepthCrafterPipeline.from_pretrained(
            svd_base,
            unet=unet,
            torch_dtype=weight_dtype,
            variant="fp16" if use_fp16 else None,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = weight_dtype

        # SVD pipelines eat VRAM; expose the standard offload hooks so a
        # 5090/4090 user can run fp16 inline but a smaller GPU can still
        # drive the pass via sequential offload.
        offload = self.params.get("cpu_offload")
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            # xFormers optional — attention runs in PyTorch native if absent.
            pass
        self._pipeline = pipe

    # ------------------------------------------------------------------
    # Per-window lifecycle. `preprocess`/`infer`/`postprocess` operate on a
    # single window (shape (W_frames, H, W, 3)) so external callers can
    # still drive the pass one window at a time.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Input: (W, H, W_px, 3) float32 sRGB-display in [0,1]. Returns a
        dict ready for `infer`. Stores plate shape so `postprocess` can
        upscale the prediction back to it."""
        import torch

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"DepthCrafter preprocess expects (W, H, W_px, 3), got {frames.shape}")
        import math

        self._load_model()
        n, plate_h, plate_w, _ = frames.shape
        # Long-edge scaling, CEIL to /64, min 576 on both dims.
        # SVD's UNet has 3 spatial downsamples × 8-pixel VAE patch =
        # 64 px per stride, AND skip connections assume spatial dims
        # ≥ the 576 × 1024 training size. Rounding NEAREST (what
        # DepthCrafter's own util does) produces 512 × 1024 on 2048×
        # 1080 plates, which the UNet rejects:
        #   "Expected size 72 but got size 64 for tensor number 1"
        # Ceiling to /64 pads up to 576 instead; slight aspect
        # distortion at the input is undone when we upscale the
        # depth output back to plate shape.
        max_res = int(self.params.get("max_res") or self.params["inference_short_edge"])
        long_edge = max(plate_h, plate_w, 1)
        scale = max_res / long_edge if long_edge > max_res else 1.0
        inf_h = max(576, math.ceil(plate_h * scale / 64) * 64)
        inf_w = max(576, math.ceil(plate_w * scale / 64) * 64)

        img = np.clip(frames, 0.0, 1.0).astype(np.float32, copy=False)
        t = torch.from_numpy(img).permute(0, 3, 1, 2)  # (N, 3, H, W)
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
            "depth": pred,
            "plate_shape": tensor["plate_shape"],
        }

    def _infer_window(self, tensor: Any) -> Any:
        """Run the diffusers pipeline on a window. Overridden in tests."""
        import numpy as np
        import torch

        assert self._pipeline is not None
        with torch.no_grad():
            # `output_type="np"` skips the default PIL conversion in
            # `video_processor.postprocess_video` — that conversion
            # gives us `List[List[PIL.Image]]` which crashed the
            # downstream `.ndim` checks. We want raw floats.
            result = self._pipeline(
                video=tensor["video"],
                num_inference_steps=int(self.params["num_inference_steps"]),
                guidance_scale=float(self.params["guidance_scale"]),
                output_type="np",
            )
        depth = getattr(result, "depth", None)
        if depth is None and hasattr(result, "frames"):
            depth = result.frames
        if depth is None:
            raise RuntimeError("DepthCrafter pipeline returned no `.depth`/`.frames` attribute")
        # `np` output from postprocess_video is still batched as a list
        # at the outer level (one entry per batch item in the SVD
        # pipeline). Peel the first (only) batch; the inner array is
        # (N, H, W, 3) with depth replicated across 3 channels.
        if isinstance(depth, list):
            if not depth:
                raise RuntimeError("DepthCrafter pipeline returned an empty frames list")
            depth = depth[0]
        depth = np.asarray(depth)
        if depth.ndim == 4 and depth.shape[-1] == 3:
            # 3 channels are identical (depth replicated); collapse to 1.
            depth = depth.mean(axis=-1)
        return torch.from_numpy(depth.astype(np.float32))

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale window depth to plate res. Returns Z_raw + Z_raw (same
        values — per-clip normalization happens in `run_shot`).
        """
        import torch

        plate_h, plate_w = tensor["plate_shape"]
        d = tensor["depth"]
        if d.ndim == 3:
            d = d.unsqueeze(1)  # (N, 1, h, w)
        elif d.ndim == 4 and d.shape[1] != 1:
            d = d.mean(dim=1, keepdim=True)
        d_up = torch.nn.functional.interpolate(
            d, size=(plate_h, plate_w), mode="bilinear", align_corners=False
        )
        raw = d_up.squeeze(1).cpu().numpy().astype(np.float32)  # (N, H, W)
        # Emit a per-frame dict keyed by local window index (caller reassembles).
        return {CH_Z_RAW: raw, CH_Z: raw}

    # ------------------------------------------------------------------
    # Shot-level: window plan → stitch → per-clip normalize
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

        frames = np.stack([reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0)
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])

        starts = plan_window_starts(n_frames, window, overlap)
        predictions: list[np.ndarray] = []
        for s in starts:
            clip = frames[s : s + window]
            model_in = self.preprocess(clip)
            model_out = self.infer(model_in)
            partial = self.postprocess(model_out)
            predictions.append(partial[CH_Z_RAW])  # (window, H, W)

        raw_clip = stitch_windowed_predictions(predictions, starts, n_frames, overlap)  # (N, H, W)

        # Per-clip normalization (trap 5); flip so larger Z == nearer.
        d_min = float(raw_clip.min())
        d_max = float(raw_clip.max())
        span = max(d_max - d_min, 1e-6)
        z_clip = (1.0 - (raw_clip - d_min) / span).astype(np.float32)

        self._norm_min = d_min
        self._norm_max = d_max
        self._frame_keys = list(range(first, last + 1))
        out: dict[int, dict[str, np.ndarray]] = {}
        for i, f in enumerate(self._frame_keys):
            out[f] = {
                CH_Z: z_clip[i],
                CH_Z_RAW: raw_clip[i].astype(np.float32, copy=False),
            }
        # Reference the plate shape so downstream error messages can verify.
        self._plate_shape = (plate_h, plate_w)
        return out

    def emit_artifacts(self) -> dict[str, dict[int, np.ndarray]]:
        if not self._frame_keys:
            return {}
        any_frame = self._frame_keys[0]
        return {
            "depth_norm_min": {any_frame: np.asarray([self._norm_min], dtype=np.float32)},
            "depth_norm_max": {any_frame: np.asarray([self._norm_max], dtype=np.float32)},
        }


__all__ = ["DepthCrafterPass"]
