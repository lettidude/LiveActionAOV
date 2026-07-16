# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""ViTMatte trimap-guided matte refiner.

Backend: `hustvl/vitmatte-*-composition-1k` — ViT-based deep image matting,
MIT licensed (code AND weights), and **native in HF transformers**
(`VitMatteForImageMatting`, no trust_remote_code).

### Why trimap-guided matters here
BiRefNet is a salient-object model: it decides *by itself* what the object
is, and we have to bound it (dilated clamp + eroded-core guarantee) so it
can't eat the interior or leak onto neighbours. ViTMatte is **trimap-
obedient by construction**: we hand it fg / bg / unknown regions derived
from SAM 3's hard mask, and it only estimates alpha inside the unknown
band — known regions are respected structurally, not by post-hoc clamps.
That is exactly the contract our SAM3-guided refiner design wants.

Trimap from the hard mask (per frame):
    fg      = erode(hard, fg_erode_px)        -> 255
    unknown = dilate(hard, band_dilate_px) - fg -> 128
    bg      = the rest                         -> 0

Provenance note: trained on Composition-1K (Adobe DIM, research dataset) —
same accepted bar as BiRefNet-portrait/P3M; stamped in sidecar metadata.

Per-frame (no temporal state) → `matte.*` channels are smoothable, pair
with the flow-guided temporal smoother on noisy plates, same as BiRefNet.

`_load_model` / `_vitmatte_alpha` / `_refine_instance` are override hooks
so CI can inject a numpy-only fake without pulling weights.
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


class ViTMatteRefinerPass(RVMRefinerPass):
    name = "vitmatte_refiner"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "ViTMatte (hustvl) — MIT code and weights, native HF transformers "
            "loader. Trimap-guided: only estimates alpha in the unknown band "
            "derived from SAM 3's hard mask; known regions are respected by "
            "construction. Trained on Composition-1K (Adobe DIM research "
            "dataset) — same provenance bar as the BiRefNet fine-tunes."
        ),
    )

    # Per-frame model → flicker possible → smoothable (same as BiRefNet).
    smoothable_channels = [CH_MATTE_R, CH_MATTE_G, CH_MATTE_B, CH_MATTE_A]

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "hustvl/vitmatte-base-composition-1k",
        "precision": "fp32",  # ViTMatte is light; fp32 avoids edge banding
        # Trimap construction from the SAM hard mask, in pixels at plate res.
        "fg_erode": 5,  # certain-foreground core
        "band_dilate": 15,  # unknown band reach beyond the hard edge
        "crop_pad_fraction": 0.12,  # pad the band bbox before cropping
        "max_infer_long_edge": 1600,  # downscale huge crops before the ViT
    }

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        model_id = str(self.params["model_id"])
        self._processor = VitMatteImageProcessor.from_pretrained(model_id)
        model = VitMatteForImageMatting.from_pretrained(model_id)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params.get("precision") == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        model.to(self._device, dtype=self._dtype).eval()
        self._model = model

    def _vitmatte_alpha(self, crop_rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """ViTMatte on one crop. crop_rgb (H, W, 3) float32 [0,1];
        trimap (H, W) uint8 {0, 128, 255} -> alpha (H, W) float32 [0,1]."""
        import cv2
        import torch
        from PIL import Image

        h, w = crop_rgb.shape[:2]
        # Bound the ViT's quadratic attention on huge crops.
        long_edge = max(h, w)
        max_edge = int(self.params.get("max_infer_long_edge", 1600))
        if long_edge > max_edge:
            s = max_edge / long_edge
            iw, ih = max(1, int(w * s)), max(1, int(h * s))
        else:
            iw, ih = w, h
        img = Image.fromarray(
            (np.clip(crop_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
        ).resize((iw, ih), Image.Resampling.BILINEAR)
        tri = Image.fromarray(trimap, "L").resize((iw, ih), Image.Resampling.NEAREST)

        inputs = self._processor(images=img, trimaps=tri, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.no_grad():
            alphas = self._model(**inputs).alphas  # (1, 1, H_pad, W_pad)
        a = alphas[0, 0, :ih, :iw].float().cpu().numpy()  # crop the pad
        if (iw, ih) != (w, h):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(a.astype(np.float32), 0.0, 1.0)

    # ------------------------------------------------------------------
    # Per-frame refine: hard mask -> trimap -> ViTMatte -> composite
    # ------------------------------------------------------------------

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        """Input: plate (T, H, W, 3) f32 sRGB [0,1]; hard (T, H, W) [0,1]/uint8.
        Output: (T, H, W) float32 soft alpha."""
        import cv2

        self._load_model()
        assert self._model is not None
        T, H, W, _ = plate_stack.shape
        fg_erode = max(int(self.params.get("fg_erode", 5)), 0)
        band = max(int(self.params.get("band_dilate", 15)), 1)
        pad_frac = float(self.params.get("crop_pad_fraction", 0.12))
        k_fg = np.ones((2 * fg_erode + 1, 2 * fg_erode + 1), np.uint8) if fg_erode else None
        k_band = np.ones((2 * band + 1, 2 * band + 1), np.uint8)

        out = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            binm = (hard_stack[t] > 0.5).astype(np.uint8)
            if int(binm.sum()) == 0:
                continue
            fg = cv2.erode(binm, k_fg) if k_fg is not None else binm
            dil = cv2.dilate(binm, k_band)
            trimap = np.zeros((H, W), np.uint8)
            trimap[dil > 0] = 128
            trimap[fg > 0] = 255

            ys, xs = np.nonzero(dil)
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            pw, ph = int((x1 - x0) * pad_frac), int((y1 - y0) * pad_frac)
            x0, x1 = max(0, x0 - pw), min(W, x1 + pw)
            y0, y1 = max(0, y0 - ph), min(H, y1 + ph)

            alpha_crop = self._vitmatte_alpha(
                plate_stack[t, y0:y1, x0:x1], trimap[y0:y1, x0:x1]
            )
            full = np.zeros((H, W), dtype=np.float32)
            full[y0:y1, x0:x1] = alpha_crop
            # Trimap-obedient composite: known regions are exact, the model
            # only fills the unknown band. No post-hoc clamps needed.
            out[t] = np.where(fg > 0, 1.0, np.where(dil > 0, full, 0.0)).astype(np.float32)
        return out


__all__ = ["ViTMatteRefinerPass"]
