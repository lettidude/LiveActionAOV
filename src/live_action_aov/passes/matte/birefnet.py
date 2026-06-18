# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""BiRefNet soft-edge matte refiner (Phase 3 round 1b).

Backend: `ZhengPeng7/BiRefNet` — a high-resolution dichotomous image
segmentation / matting network. **MIT-licensed and commercial-safe**, and
production-proven (CorridorKey uses it internally), which is what makes it
the commercial-clean path to roto-grade *soft* edges — hair, motion blur,
semi-transparency — that SAM 3's hard masks can't give.

This pass reuses RVM's identical refiner machinery (`ingest_artifacts`,
`run_shot`, slot packing into `matte.r/g/b/a`, `emit_artifacts`) by
subclassing `RVMRefinerPass`; only the model and `_refine_instance`
differ. Keeping RVM untouched means zero regression risk to the shipped
default refiner.

### How it differs from RVM
- BiRefNet is **per-frame** (no recurrent state). To get its best edges we
  crop tightly around the SAM 3 mask's bounding box (BiRefNet shines at
  high resolution on a single object), run inference on the crop, paste the
  alpha back, and bound it by the dilated hard mask so it can't pick a
  different salient object inside the crop.
- Because it's per-frame, the `matte.*` channels ARE `smoothable` — the
  flow-guided `temporal_smooth` post pass can stabilise residual flicker
  (RVM, being recurrent, declared none).

`_load_model` / `_refine_instance` / `_birefnet_alpha` are override hooks
so CI can inject a numpy-only fake without pulling BiRefNet weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import License
from live_action_aov.io.channels import (
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)
from live_action_aov.passes.matte.rvm import RVMRefinerPass


class BiRefNetRefinerPass(RVMRefinerPass):
    name = "birefnet_refiner"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "BiRefNet (ZhengPeng7/BiRefNet) is MIT-licensed — inference and "
            "outputs are commercial-safe, no usage gate. Produces soft alpha "
            "(roto-grade edges); per-frame, so pair with temporal_smooth on "
            "noisy plates."
        ),
    )

    # Per-frame model → flicker is possible → let the flow-guided smoother
    # stabilise the matte channels (RVM is recurrent and declared none).
    smoothable_channels = [CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A]

    DEFAULT_PARAMS: dict[str, Any] = {
        # Default to the general BiRefNet weights (definitely present, soft
        # sigmoid output). Matting-specialised weights (e.g. a *-matting
        # repo) can be set here for hair-grade alpha once verified.
        "model_id": "ZhengPeng7/BiRefNet",
        "infer_size": 1024,  # BiRefNet's native inference resolution
        "precision": "fp16",
        "hard_mask_dilate": 5,  # grow the seed before bounding the alpha
        "crop_pad_fraction": 0.12,  # pad the mask bbox before cropping
    }

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageSegmentation

            model = AutoModelForImageSegmentation.from_pretrained(
                str(self.params["model_id"]), trust_remote_code=True
            )
        except ImportError as e:
            # BiRefNet's remote code pulls timm / einops / kornia.
            raise RuntimeError(
                "BiRefNet requires extra dependencies (timm, einops, kornia). "
                "Install via: pip install live-action-aov[birefnet]"
            ) from e
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params.get("precision") == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        model.to(self._device, dtype=self._dtype).eval()
        self._model = model

    # ------------------------------------------------------------------
    # Soft-alpha inference — crop → BiRefNet → paste → bound by mask
    # ------------------------------------------------------------------

    def _birefnet_alpha(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Run BiRefNet on a single RGB crop (H, W, 3) float32 [0,1] →
        alpha (H, W) float32 [0,1] at the crop's resolution."""
        import cv2
        import torch
        from PIL import Image

        size = int(self.params.get("infer_size", 1024))
        h, w = crop_rgb.shape[:2]
        pil = Image.fromarray((np.clip(crop_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
        pil = pil.resize((size, size), Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        ten = (
            torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device, self._dtype)
        )
        with torch.no_grad():
            pred = self._model(ten)[-1].sigmoid().float().cpu().numpy()[0, 0]
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        """Per-frame: crop around the seed, run BiRefNet, paste + bound.

        Input:  plate_stack (T, H, W, 3) float32 sRGB [0,1];
                hard_stack  (T, H, W)    float32 [0,1].
        Output: (T, H, W) float32 [0,1] soft alpha.
        """
        import cv2

        self._load_model()
        assert self._model is not None
        T, H, W, _ = plate_stack.shape
        dilate = max(int(self.params.get("hard_mask_dilate", 5)), 0)
        pad_frac = float(self.params.get("crop_pad_fraction", 0.12))
        kernel = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8) if dilate > 0 else None

        out = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            binm = (hard_stack[t] > 0.5).astype(np.uint8)
            if int(binm.sum()) == 0:
                continue
            dil = cv2.dilate(binm, kernel) if kernel is not None else binm
            ys, xs = np.nonzero(dil)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pw = int((x1 - x0) * pad_frac)
            ph = int((y1 - y0) * pad_frac)
            x0, x1 = max(0, x0 - pw), min(W, x1 + pw)
            y0, y1 = max(0, y0 - ph), min(H, y1 + ph)

            crop = plate_stack[t, y0:y1, x0:x1]
            alpha_crop = self._birefnet_alpha(crop)
            full = np.zeros((H, W), dtype=np.float32)
            full[y0:y1, x0:x1] = alpha_crop
            # Bound to our object so BiRefNet can't leak onto a neighbour.
            out[t] = np.clip(full * dil.astype(np.float32), 0.0, 1.0)
        return out


__all__ = ["BiRefNetRefinerPass"]
