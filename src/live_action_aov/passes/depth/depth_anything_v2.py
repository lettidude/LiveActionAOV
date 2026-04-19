"""Depth Anything V2 pass (commercial-safe fallback, spec §13.1 Phase 2).

Backend: `transformers.pipeline("depth-estimation", "depth-anything/Depth-Anything-V2-Small-hf")`.
The Small + Base variants ship Apache-2.0; Large and Giant are CC-BY-NC-4.0
and must be declared as such by a separate pass. This class defaults to Small
and refuses to load Large/Giant identifiers — a dedicated `DepthAnythingV2LargePass`
(not shipped in Phase 2) would carry the non-commercial license flag instead.

Outputs (spec §5.1, channels.py):
- `Z`      — relative depth, normalized **per-clip** (spec §2.2 trap 5: one
             min/max across all frames, not per-frame, to keep temporal
             consistency through the smoother).
- `Z_raw`  — raw model output, un-normalized, for users who want to
             renormalize differently or run their own analysis.

Temporal: PER_FRAME. The executor auto-wires `TemporalSmoother` when
`smooth: auto` and a flow pass exists in the same job.

Implementation notes:
- The DA V2 model expects multiples of 14 on each spatial dim (ViT patch
  size = 14). We resize the short edge to `inference_short_edge` (default
  518, the stock DA V2 training size), round both edges to the next 14×,
  then bilinearly upscale the output back to plate resolution. Depth is a
  scalar field, so no vector rescaling is needed (unlike RAFT).
- sRGB display-referred input is what the HF pipeline expects; the executor
  is supposed to hand that over already (Phase 1 uses the same gamma 2.2
  shim as RAFT for consistency).
- Per-clip normalization runs in `run_shot` after all frames are inferred:
  `Z = 1 - (d - d_min) / (d_max - d_min + eps)`. The "1 -" flip makes Z
  larger for nearer objects, which is what Nuke's PositionFromDepth and
  most compers expect. `Z_raw` is stored without the flip so downstream
  tools that assume "larger depth = farther" still work.
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

# HF model IDs. Small/Base are Apache-2.0 (commercial OK); Large/Giant are
# CC-BY-NC-4.0 — we refuse to load them from this class.
_COMMERCIAL_VARIANTS: dict[str, str] = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}
_NONCOMMERCIAL_VARIANTS: set[str] = {"large", "giant"}


class DepthAnythingV2Pass(UtilityPass):
    name = "depth_anything_v2"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Depth Anything V2 Small/Base are Apache-2.0. Large/Giant are "
            "CC-BY-NC-4.0 and must be loaded through a separate non-commercial "
            "pass — this pass refuses those identifiers."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Relative depth, per-clip normalized [0,1] (1=near)"),
        ChannelSpec(name=CH_Z_RAW, description="Raw model output (un-normalized)"),
    ]
    # Smooth Z (the comp-visible normalized depth). Z_raw is deliberately left
    # untouched so downstream tools have access to the un-smoothed signal.
    smoothable_channels = [CH_Z]

    DEFAULT_PARAMS: dict[str, Any] = {
        "variant": "small",  # "small" | "base"
        "inference_short_edge": 518,  # ViT-14 trained size
        "precision": "fp32",  # "fp32" | "fp16" (fp16 only if cuda)
        "smooth": "auto",  # executor uses this to auto-wire TemporalSmoother
        "depth_space": "relative",  # vs. "metric" (Depth Pro) — metadata hint
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        variant = str(self.params["variant"]).lower()
        if variant in _NONCOMMERCIAL_VARIANTS:
            raise ValueError(
                f"DepthAnythingV2Pass refuses variant {variant!r} — Large/Giant "
                f"are CC-BY-NC-4.0. Use a non-commercial pass or pick 'small'/'base'."
            )
        if variant not in _COMMERCIAL_VARIANTS:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from {sorted(_COMMERCIAL_VARIANTS)}."
            )
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Raw per-frame depth maps captured during run_shot; normalized in-place
        # once the whole clip is inferred.
        self._raw: dict[int, np.ndarray] = {}
        # Diagnostic dump state — gated by env var LAAOV_DEBUG_DA_V2=<dir>;
        # only dumps on the first frame so repeated inference stays quiet.
        self._debug_dumped = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = _COMMERCIAL_VARIANTS[str(self.params["variant"]).lower()]
        self._processor = AutoImageProcessor.from_pretrained(repo)
        model = AutoModelForDepthEstimation.from_pretrained(repo)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params["precision"] == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        model.to(self._device).eval()
        if self._dtype == torch.float16:
            model.half()
        self._model = model

    # ------------------------------------------------------------------
    # Single-frame lifecycle (PER_FRAME)
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Input: (1, H, W, 3) float32 sRGB-display in [0,1]. Returns a dict
        with the processor-preprocessed pixel tensor + plate shape for upscale.
        """

        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(
                f"DepthAnythingV2Pass preprocess expects (1, H, W, 3), got {frames.shape}"
            )
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        img = np.clip(frames[0], 0.0, 1.0)
        # HF processor takes uint8 HxWxC or PIL; we pass a list of numpy arrays
        # as float32 and let the processor handle the resize + normalization.
        # DA V2's processor respects `size` at inference; we pass the target
        # short edge so it picks the matching 14-multiple interior size.
        # DPT's processor in transformers >=5 rejects `shortest_edge` and only
        # accepts `{"height": ..., "width": ...}`; with its default
        # `keep_aspect_ratio=True` + `ensure_multiple_of=14`, setting both to
        # the target short edge yields "short edge == edge, long edge scaled
        # and rounded to a 14-multiple" — which is what we want. Passing
        # `shortest_edge` triggers an AttributeError masking this contract.
        edge = int(self.params["inference_short_edge"])
        # do_rescale=False is critical: HF `AutoImageProcessor` defaults to
        # `rescale_factor = 1/255` because it assumes uint8 [0, 255]. We
        # feed float32 in [0, 1] post display transform — without this flag
        # the processor divides by 255 again, so sRGB 0.5 becomes 0.002,
        # then ImageNet-normalised to ~-2.1, and the model sees pure noise
        # and falls back to a radial depth prior (the diagnostic PNG dump
        # showed this exactly — "02_processor_output.png" was solid black
        # despite a healthy 0.30–0.58 input range).
        proc = self._processor(
            images=img,
            return_tensors="pt",
            do_resize=True,
            size={"height": edge, "width": edge},
            do_rescale=False,
        )
        pixel_values = proc["pixel_values"].to(self._device, dtype=self._dtype)
        self._debug_dump_preprocess(img, proc, pixel_values)
        return {
            "pixel_values": pixel_values,
            "plate_shape": (plate_h, plate_w),
        }

    def _debug_dump_preprocess(self, img: np.ndarray, proc: Any, pixel_values: Any) -> None:
        """Dump the display-transformed input and the processor's normalized
        tensor (de-normalized back to image) to the directory in
        LAAOV_DEBUG_DA_V2 on the first frame only. No-op otherwise.
        """
        import os

        dump_dir = os.environ.get("LAAOV_DEBUG_DA_V2", "")
        if not dump_dir or self._debug_dumped:
            return
        self._debug_dumped = True
        try:
            from PIL import Image
        except Exception:
            return
        os.makedirs(dump_dir, exist_ok=True)
        Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)).save(
            os.path.join(dump_dir, "01_input_to_processor.png")
        )
        pv_np = pixel_values[0].detach().float().cpu().numpy()
        mean = np.asarray(
            getattr(self._processor, "image_mean", [0.5, 0.5, 0.5]), dtype=np.float32
        )[:, None, None]
        std = np.asarray(getattr(self._processor, "image_std", [0.5, 0.5, 0.5]), dtype=np.float32)[
            :, None, None
        ]
        pv_img = np.clip(pv_np * std + mean, 0.0, 1.0)
        pv_img = np.transpose(pv_img, (1, 2, 0))
        Image.fromarray((pv_img * 255).astype(np.uint8)).save(
            os.path.join(dump_dir, "02_processor_output.png")
        )
        with open(os.path.join(dump_dir, "shapes.txt"), "w", encoding="utf-8") as f:
            f.write(f"input HxW (post display transform): {img.shape[0]}x{img.shape[1]}\n")
            f.write(f"pixel_values shape: {tuple(pixel_values.shape)}\n")
            f.write(f"pixel_values dtype: {pixel_values.dtype}\n")
            f.write(f"input_min/max: {float(img.min()):.4f} / {float(img.max()):.4f}\n")
            f.write(f"processor image_mean: {list(mean.flatten())}\n")
            f.write(f"processor image_std:  {list(std.flatten())}\n")
            f.write(
                f"size param: {{'height': {self.params['inference_short_edge']}, 'width': {self.params['inference_short_edge']}}}\n"
            )

    def _debug_dump_depth(self, raw: np.ndarray) -> None:
        """Dump the normalized raw depth on the first frame as a PNG for
        quick eyeballing. Called from run_shot on frame 0 only.
        """
        import os

        dump_dir = os.environ.get("LAAOV_DEBUG_DA_V2", "")
        if not dump_dir:
            return
        try:
            from PIL import Image
        except Exception:
            return
        os.makedirs(dump_dir, exist_ok=True)
        d = raw.astype(np.float32)
        d_min, d_max = float(d.min()), float(d.max())
        span = max(d_max - d_min, 1e-6)
        norm = (d - d_min) / span
        Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(
            os.path.join(dump_dir, "03_raw_depth_frame0.png")
        )
        with open(os.path.join(dump_dir, "shapes.txt"), "a", encoding="utf-8") as f:
            f.write(f"frame0 raw depth HxW: {raw.shape[0]}x{raw.shape[1]}\n")
            f.write(f"frame0 raw depth min/max: {d_min:.4f} / {d_max:.4f}\n")

    def infer(self, tensor: Any) -> Any:
        import torch

        assert self._model is not None
        with torch.no_grad():
            outputs = self._model(pixel_values=tensor["pixel_values"])
        # transformers DepthEstimationOutput: `.predicted_depth` is (B, h, w)
        pred = outputs.predicted_depth.float()
        return {
            "depth": pred,
            "plate_shape": tensor["plate_shape"],
        }

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale to plate resolution. No per-clip normalization here — that
        happens in `run_shot` after all frames are inferred (trap 5).
        Single-pair callers just get `Z_raw` and an un-normalized `Z` (= raw).
        """
        import torch

        plate_h, plate_w = tensor["plate_shape"]
        d = tensor["depth"]
        if d.ndim == 3:
            d = d.unsqueeze(1)  # (B, 1, h, w)
        d_up = torch.nn.functional.interpolate(
            d, size=(plate_h, plate_w), mode="bilinear", align_corners=False
        )
        raw = d_up[0, 0].cpu().numpy().astype(np.float32)
        return {CH_Z_RAW: raw, CH_Z: raw}

    # ------------------------------------------------------------------
    # Shot-level iteration: per-frame inference + per-clip Z normalization
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        per_frame_raw: dict[int, np.ndarray] = {}
        for f in range(first, last + 1):
            frame, _ = reader.read_frame(f)
            model_in = self.preprocess(frame[None, ...])
            model_out = self.infer(model_in)
            partial = self.postprocess(model_out)
            per_frame_raw[f] = partial[CH_Z_RAW]
            if f == first:
                self._debug_dump_depth(partial[CH_Z_RAW])

        # Per-clip normalization (trap 5). Single min/max across all frames;
        # flip so larger Z == nearer (Nuke PositionFromDepth convention).
        stacked = np.stack(list(per_frame_raw.values()), axis=0)
        d_min = float(stacked.min())
        d_max = float(stacked.max())
        span = max(d_max - d_min, 1e-6)

        out: dict[int, dict[str, np.ndarray]] = {}
        self._raw = per_frame_raw
        for f, raw in per_frame_raw.items():
            z = 1.0 - (raw - d_min) / span
            out[f] = {
                CH_Z: z.astype(np.float32, copy=False),
                CH_Z_RAW: raw.astype(np.float32, copy=False),
            }
        # Stash normalization constants so emit_artifacts can expose them.
        self._norm_min = d_min
        self._norm_max = d_max
        return out

    def emit_artifacts(self) -> dict[str, dict[int, np.ndarray]]:
        """Expose per-clip normalization constants as scalars on frame 0.

        Downstream tools (and the executor's metadata emitter) read these to
        record `depth/normalization/min` and `depth/normalization/max`.
        """
        if not self._raw:
            return {}
        any_frame = next(iter(self._raw))
        return {
            "depth_norm_min": {any_frame: np.asarray([self._norm_min], dtype=np.float32)},
            "depth_norm_max": {any_frame: np.asarray([self._norm_max], dtype=np.float32)},
        }


__all__ = ["DepthAnythingV2Pass"]
