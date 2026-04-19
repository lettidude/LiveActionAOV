"""Depth Pro pass (metric depth, spec §13.1 Phase 2 round 2).

Backend: Apple ML Research's Depth Pro (`apple/DepthPro` on HuggingFace, via
`transformers`). Produces **metric** depth in meters plus a per-pixel
confidence. Unlike relative-depth backends, no per-clip normalization is
applied — `Z` and `Z_raw` are both emitted in meters so downstream tools
can choose whether to compress the range themselves.

License: Apple ML Research License. Non-commercial only; gated behind
`--allow-noncommercial`. See the upstream repo for the full text; the
short version is "research + personal use, not for shipping products".

Outputs (spec §5.1):
- `Z`                — metric depth in meters (raw, no flip/normalization).
- `Z_raw`            — identical to `Z` (kept for schema parity with the
                       relative-depth backends where Z_raw is the
                       un-normalized source).
- `depth.confidence` — per-pixel confidence in [0, 1].

Temporal: PER_FRAME. The executor auto-wires `TemporalSmoother` when
`smooth: auto` and a flow pass exists in the same job. Only `Z` is
smoothed — smoothing the confidence channel would be noise-propagation.

Emits a `depth_metric` scalar artifact so the executor can stamp
`depth/space=metric` + `depth/unit=meters` in sidecar metadata (instead
of the `relative` / `normalized_per_clip` wiring used for DA V2 / DepthCrafter).

Implementation notes:
- DepthPro's HF pipeline uses a DPT-style backbone; we drive it through
  `AutoModelForDepthEstimation` + `AutoImageProcessor` the same way DA V2
  does, so the fake-model test plumbing lines up.
- Tests override `_load_model` + `infer` to avoid the ~2 GB weight
  download; only `preprocess`/`postprocess`/`run_shot` run real code.
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
from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_Z, CH_Z_RAW


class DepthProPass(UtilityPass):
    name = "depthpro"
    version = "0.1.0"
    license = License(
        spdx="Apple-ML-Research",
        commercial_use=False,
        commercial_tool_resale=False,
        notes=(
            "Apple Depth Pro is released under the Apple ML Research "
            "License (non-commercial research use only). Gated behind "
            "--allow-noncommercial. Consult the upstream LICENSE before "
            "any commercial deployment."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Metric depth in meters"),
        ChannelSpec(name=CH_Z_RAW, description="Metric depth in meters (raw; identical to Z)"),
        ChannelSpec(name=CH_DEPTH_CONFIDENCE, description="Per-pixel confidence, [0,1]"),
    ]
    # Smooth metric Z; keep Z_raw and confidence untouched so analysis tools
    # can see the un-smoothed signal and raw uncertainty.
    smoothable_channels = [CH_Z]

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "apple/DepthPro",
        "inference_short_edge": 1536,  # DepthPro's stock inference size
        "precision": "fp16",  # fp16 on CUDA; fp32 on CPU
        "smooth": "auto",
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._processor: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._frame_keys: list[int] = []

    # ------------------------------------------------------------------
    # Model lifecycle (transformers — tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        repo = str(self.params["model_id"])
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
        """Input: (1, H, W, 3) float32 sRGB-display in [0,1]."""
        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(f"DepthProPass preprocess expects (1, H, W, 3), got {frames.shape}")
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        img = np.clip(frames[0], 0.0, 1.0)
        # transformers >=5 processors expect `{"height": ..., "width": ...}`
        # and reject `shortest_edge` with a misleading SizeDict.keys() crash.
        # DepthPro defaults to a square 1536x1536; we mirror that shape while
        # letting the user override the edge size.
        edge = int(self.params["inference_short_edge"])
        # do_rescale=False: our input is float32 [0, 1] after the display
        # transform; the HF processor's default rescale_factor=1/255 would
        # otherwise divide it by 255 again (same bug fixed in
        # DepthAnythingV2Pass — see that file's comment for the full trace).
        proc = self._processor(
            images=img,
            return_tensors="pt",
            do_resize=True,
            size={"height": edge, "width": edge},
            do_rescale=False,
        )
        pixel_values = proc["pixel_values"].to(self._device, dtype=self._dtype)
        return {
            "pixel_values": pixel_values,
            "plate_shape": (plate_h, plate_w),
        }

    def infer(self, tensor: Any) -> Any:
        """Run the model and return `{depth, confidence, plate_shape}`.

        DepthPro's HF head exposes `.predicted_depth` (meters) and an
        optional confidence tensor. Some versions expose confidence as
        `.confidence`, others fold it into the output dict — try both.
        """
        import torch

        assert self._model is not None
        with torch.no_grad():
            outputs = self._model(pixel_values=tensor["pixel_values"])
        depth = outputs.predicted_depth.float()
        confidence = getattr(outputs, "confidence", None)
        if confidence is None and hasattr(outputs, "depth_confidence"):
            confidence = outputs.depth_confidence
        if confidence is not None:
            confidence = confidence.float()
        return {
            "depth": depth,
            "confidence": confidence,
            "plate_shape": tensor["plate_shape"],
        }

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale to plate resolution. Metric values pass through unchanged.

        When the backend doesn't expose a confidence tensor we synthesize a
        field of 1.0 so downstream code always sees the channel.
        """
        import torch

        plate_h, plate_w = tensor["plate_shape"]
        d = tensor["depth"]
        if d.ndim == 3:
            d = d.unsqueeze(1)  # (B, 1, h, w)
        d_up = torch.nn.functional.interpolate(
            d, size=(plate_h, plate_w), mode="bilinear", align_corners=False
        )
        z = d_up[0, 0].cpu().numpy().astype(np.float32)

        conf_t = tensor.get("confidence")
        if conf_t is not None:
            if conf_t.ndim == 3:
                conf_t = conf_t.unsqueeze(1)
            conf_up = torch.nn.functional.interpolate(
                conf_t, size=(plate_h, plate_w), mode="bilinear", align_corners=False
            )
            conf = conf_up[0, 0].cpu().numpy().astype(np.float32)
            conf = np.clip(conf, 0.0, 1.0)
        else:
            conf = np.ones((plate_h, plate_w), dtype=np.float32)

        return {
            CH_Z: z,
            CH_Z_RAW: z.copy(),
            CH_DEPTH_CONFIDENCE: conf,
        }

    # ------------------------------------------------------------------
    # Shot-level iteration — straight PER_FRAME loop, no per-clip norm.
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            frame, _ = reader.read_frame(f)
            model_in = self.preprocess(frame[None, ...])
            model_out = self.infer(model_in)
            out[f] = self.postprocess(model_out)
        self._frame_keys = list(range(first, last + 1))
        return out

    def emit_artifacts(self) -> dict[str, dict[int, np.ndarray]]:
        """Flag the output as metric so the executor writes the right
        `depth/space` + `depth/unit` metadata.
        """
        if not self._frame_keys:
            return {}
        any_frame = self._frame_keys[0]
        return {
            "depth_metric": {any_frame: np.asarray([1.0], dtype=np.float32)},
        }


__all__ = ["DepthProPass"]
