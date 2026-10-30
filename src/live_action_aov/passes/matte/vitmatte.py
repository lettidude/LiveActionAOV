# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

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

Trimap (per frame), `trimap="guided"` (default):
    SAM 3 says WHERE the object is, BiRefNet says how sure each pixel is
    (only when it agrees with SAM - IoU >= `guide_agree_iou`; on objects it
    does not understand it is ignored), ViTMatte decides every unsure pixel.
    fg      = deep inside SAM (never deeper than half the local thickness,
              so poles / a detached arm keep a core) + where BiRefNet is
              certain AND SAM agrees                        -> 255
    bg      = far from SAM, or outside SAM where BiRefNet is certain it is
              background                                    -> 0
    unknown = the rest: a WIDE band that follows the uncertainty (hair,
              blur, a blobby SAM edge), narrow on crisp edges -> 128
    SAM's own edge is never trusted as the final outline: a blobby SAM mask
    (hair volume filled in) used to stay blobby because only a thin band
    around it was refined (field report, 2026-10-28).

`trimap="band"` is the old construction (fg = erode(hard, fg_erode),
unknown = dilate(hard, band) - fg).

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
from live_action_aov.core.runtime_env import load_local_first
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
        "band_dilate": "auto",  # adaptive: fraction of object bbox diag (resolution-independent)
        "crop_pad_fraction": 0.12,  # pad the band bbox before cropping
        "max_infer_long_edge": 1600,  # downscale huge crops before the ViT
        # "guided" (SAM + BiRefNet evidence, see module doc) | "band" (legacy).
        "trimap": "guided",
        "guide_model_id": "ZhengPeng7/BiRefNet-portrait",
        "guide_agree_iou": 0.75,  # below this BiRefNet is ignored for the frame
        "band_frac": 0.05,  # unknown-band radius = frac * sqrt(object area)
        "band_min": 6,
        "band_max": 40,
    }

    def _trust_outline(self) -> bool:
        # The guided trimap departs from SAM's edge on purpose; the edge
        # guardrail (compares against a feathered SAM edge) would undo it.
        return str(self.params.get("trimap", "guided")) == "guided"

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import VitMatteForImageMatting, VitMatteImageProcessor

        model_id = str(self.params["model_id"])
        self._processor = load_local_first(VitMatteImageProcessor.from_pretrained, model_id)
        model = load_local_first(VitMatteForImageMatting.from_pretrained, model_id)
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
        img = Image.fromarray((np.clip(crop_rgb, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB").resize(
            (iw, ih), Image.Resampling.BILINEAR
        )
        tri = Image.fromarray(trimap, "L").resize((iw, ih), Image.Resampling.NEAREST)

        inputs = self._processor(images=img, trimaps=tri, return_tensors="pt").to(
            self._device, self._dtype
        )
        with torch.no_grad():
            alphas = self._model(**inputs).alphas  # (1, 1, H_pad, W_pad)
        a = alphas[0, 0, :ih, :iw].float().cpu().numpy()  # crop the pad
        del inputs, alphas
        # Every crop has its own size, so the CUDA caching allocator keeps
        # reserving new blocks (25 GiB reserved for 5 GiB used on a 16-frame
        # test). Next to the other models of a run that overflowed the 32 GB
        # card into shared system memory and one object took ~10 min. Hand the
        # cache back once it is far above what is really in use.
        if self._device.type == "cuda" and torch.cuda.memory_reserved() > 3 * max(
            torch.cuda.memory_allocated(), 1 << 30
        ):
            torch.cuda.empty_cache()
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
        from live_action_aov.passes.matte.rvm import adaptive_band_px

        fg_erode = max(int(self.params.get("fg_erode", 5)), 0)
        band_param = self.params.get("band_dilate", "auto")
        pad_frac = float(self.params.get("crop_pad_fraction", 0.12))
        k_fg = np.ones((2 * fg_erode + 1, 2 * fg_erode + 1), np.uint8) if fg_erode else None

        guided = str(self.params.get("trimap", "guided")) == "guided"
        out = np.zeros((T, H, W), dtype=np.float32)
        for t in range(T):
            binm = (hard_stack[t] > 0.5).astype(np.uint8)
            if int(binm.sum()) == 0:
                continue
            if guided:
                out[t] = self._refine_guided(plate_stack[t], binm)
                continue
            band = (
                adaptive_band_px(binm, 0.18, 8, 72)
                if band_param == "auto"
                else max(int(band_param), 1)
            )
            k_band = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
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

            alpha_crop = self._vitmatte_alpha(plate_stack[t, y0:y1, x0:x1], trimap[y0:y1, x0:x1])
            full = np.zeros((H, W), dtype=np.float32)
            full[y0:y1, x0:x1] = alpha_crop
            # Trimap-obedient composite: known regions are exact, the model
            # only fills the unknown band. No post-hoc clamps needed.
            out[t] = np.where(fg > 0, 1.0, np.where(dil > 0, full, 0.0)).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Guided trimap (default)
    # ------------------------------------------------------------------

    def _guide_alpha(self, rgb: np.ndarray, binm: np.ndarray) -> np.ndarray | None:
        """BiRefNet alpha for the object's square crop, full-frame (H, W), or
        None when no guide is configured. Tests override."""
        if not self.params.get("guide_model_id"):
            return None
        import cv2

        from live_action_aov.passes.matte.birefnet import BiRefNetRefinerPass, _square_box

        if getattr(self, "_guide", None) is None:
            self._guide = BiRefNetRefinerPass({"model_id": self.params["guide_model_id"]})
        H, W = binm.shape
        ys, xs = np.nonzero(cv2.dilate(binm, np.ones((11, 11), np.uint8)))
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        pw, ph = int((x1 - x0) * 0.12), int((y1 - y0) * 0.12)
        x0, x1, y0, y1 = _square_box(
            max(0, x0 - pw), min(W, x1 + pw), max(0, y0 - ph), min(H, y1 + ph), W, H
        )
        self._guide._load_model()
        full = np.zeros((H, W), np.float32)
        full[y0:y1, x0:x1] = self._guide._birefnet_alpha(rgb[y0:y1, x0:x1])
        return full

    def _refine_guided(self, rgb: np.ndarray, binm: np.ndarray) -> np.ndarray:
        trimap = guided_trimap(
            binm,
            self._guide_alpha(rgb, binm),
            frac=float(self.params.get("band_frac", 0.05)),
            r_min=int(self.params.get("band_min", 6)),
            r_max=int(self.params.get("band_max", 40)),
            agree_iou=float(self.params.get("guide_agree_iou", 0.75)),
        )
        H, W = binm.shape
        ys, xs = np.nonzero(trimap)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad = float(self.params.get("crop_pad_fraction", 0.12))
        pw, ph = int((x1 - x0) * pad), int((y1 - y0) * pad)
        x0, x1 = max(0, x0 - pw), min(W, x1 + pw)
        y0, y1 = max(0, y0 - ph), min(H, y1 + ph)
        full = np.zeros((H, W), dtype=np.float32)
        full[y0:y1, x0:x1] = self._vitmatte_alpha(rgb[y0:y1, x0:x1], trimap[y0:y1, x0:x1])
        return np.where(trimap == 255, 1.0, np.where(trimap == 128, full, 0.0)).astype(np.float32)


def guided_trimap(
    binm: np.ndarray,
    guide: np.ndarray | None,
    frac: float = 0.05,
    r_min: int = 6,
    r_max: int = 40,
    agree_iou: float = 0.75,
) -> np.ndarray:
    """SAM hard mask (+ optional BiRefNet alpha) -> trimap uint8 {0,128,255}.
    See the module doc for the rules."""
    import cv2

    def disk(r: int) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    m = (binm > 0).astype(np.uint8)
    r = int(np.clip(frac * np.sqrt(float(m.sum())), r_min, r_max))
    # Sure-FG spine: 3r deep, capped at half each component's own thickness.
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(m)
    thick = np.zeros(n, np.float32)
    np.maximum.at(thick, lab.ravel().astype(np.intp), dist.ravel())
    fg = (dist > np.minimum(3 * r, 0.5 * thick[lab])) & (m > 0)
    agree = False
    if guide is not None:
        g = guide > 0.5
        union = int((g | (m > 0)).sum())
        agree = union > 0 and (g & (m > 0)).sum() / union >= agree_iou
    if agree:
        assert guide is not None
        gate = cv2.dilate(m, disk(2 * r)) > 0
        near = cv2.dilate(m, disk(max(r // 2, 2))) > 0
        fg |= (cv2.erode((guide > 0.97).astype(np.uint8), disk(3)) > 0) & (m > 0)
        bg = ~gate | ((guide < 0.02) & ~near)
    else:
        bg = ~(cv2.dilate(m, disk(r)) > 0)
    tri = np.full(m.shape, 128, np.uint8)
    tri[bg] = 0
    tri[fg] = 255
    return tri


__all__ = ["ViTMatteRefinerPass", "guided_trimap"]
