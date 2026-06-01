# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Marigold surface-normals pass (spec §13.1).

Backend: PRS-ETH Marigold Normals (diffusers `MarigoldNormalsPipeline`),
SD2-based (~1B). A third normals backend alongside DSINE and NormalCrafter,
filling a real gap:

- **DSINE** (MIT): fast, but single-image -> flickers on sequences.
- **NormalCrafter** (CC-BY-NC): temporally consistent, but non-commercial.
- **Marigold** (OpenRAIL++-M, this pass): cleaner license than NormalCrafter,
  higher quality than DSINE. Single-image, so a fixed per-frame seed + the
  flow-guided temporal smoother (`smooth: auto`) handle consistency.

Axis convention: Marigold predicts in screen-space camera frame with +X right,
+Y up, +Z toward the viewer — which is exactly the spec's locked convention
(§10.3, OpenGL/Maya). So, unlike DSINE (OpenCV), NO axis flip is needed. The
optional `flip_y`/`flip_z` params exist only as an escape hatch.

Outputs (channels.py): `N.x`, `N.y`, `N.z` — camera-space, [-1, 1],
unit-length per pixel (renormalized after any resize, spec §11.3 trap 2).
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

MIN_VRAM_GB = 4.0


class MarigoldNormalsPass(UtilityPass):
    name = "marigold_normals"
    version = "0.1.0"
    license = License(
        spdx="OpenRAIL++-M",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Marigold Normals (PRS-ETH / ETH Zurich), SD2 backbone. CreativeML "
            "OpenRAIL++-M permits commercial use (use-restricted). Cleaner-licensed "
            "than NormalCrafter; temporally stabilized via fixed seed + flow smoother."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    input_colorspace = "srgb_display"

    @staticmethod
    def vram_estimate_gb_fn(w: int, h: int) -> float:
        del w, h
        return MIN_VRAM_GB

    produces_channels = [
        ChannelSpec(name=CH_N_X, description="Camera-space normal x, [-1,1] unit-length"),
        ChannelSpec(name=CH_N_Y, description="Camera-space normal y"),
        ChannelSpec(name=CH_N_Z, description="Camera-space normal z"),
    ]
    smoothable_channels = [CH_N_X, CH_N_Y, CH_N_Z]

    DEFAULT_PARAMS: dict[str, Any] = {
        "num_inference_steps": 4,
        "ensemble_size": 1,
        "seed": 42,
        "precision": "fp16",
        "smooth": "auto",
        # Marigold already matches the spec's OpenGL axes; flips are an escape hatch.
        "flip_y": False,
        "flip_z": False,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._pipe: Any = None
        self._device: Any = None

    def _load_model(self) -> None:
        if self._pipe is not None:
            return
        import diffusers
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fp16 = self.params["precision"] == "fp16" and self._device.type == "cuda"
        dtype = torch.float16 if fp16 else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if fp16:
            kwargs["variant"] = "fp16"
        try:
            pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
                "prs-eth/marigold-normals-v1-1", **kwargs
            )
        except Exception:
            kwargs.pop("variant", None)
            pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
                "prs-eth/marigold-normals-v1-1", **kwargs
            )
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe

    def preprocess(self, frames: np.ndarray) -> Any:
        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(
                f"MarigoldNormalsPass preprocess expects (1, H, W, 3), got {frames.shape}"
            )
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        u8 = (np.clip(frames[0], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return {"image": u8, "plate_shape": (plate_h, plate_w)}

    def infer(self, tensor: Any) -> Any:
        import torch
        from PIL import Image

        assert self._pipe is not None
        gen = torch.Generator(device=self._device).manual_seed(int(self.params["seed"]))
        with torch.no_grad():
            out = self._pipe(
                Image.fromarray(tensor["image"]),
                num_inference_steps=int(self.params["num_inference_steps"]),
                ensemble_size=int(self.params["ensemble_size"]),
                generator=gen,
                match_input_resolution=True,
            )
        return {"prediction": np.asarray(out.prediction), "plate_shape": tensor["plate_shape"]}

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        n = tensor["prediction"]
        if n.ndim == 4:  # (1, H, W, 3) -> (H, W, 3)
            n = n[0]
        if n.ndim != 3 or n.shape[-1] != 3:
            raise ValueError(f"Expected Marigold normals (H, W, 3), got {n.shape}")
        plate_h, plate_w = tensor["plate_shape"]
        if n.shape[:2] != (plate_h, plate_w):
            import cv2

            n = cv2.resize(n.astype(np.float32), (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
        n = n.astype(np.float32, copy=False)
        # Trap 2: renormalize to unit length after any resize.
        mag = np.sqrt((n**2).sum(axis=-1, keepdims=True)).clip(min=1e-6)
        n = n / mag
        if bool(self.params.get("flip_y")):
            n[..., 1] *= -1.0
        if bool(self.params.get("flip_z")):
            n[..., 2] *= -1.0
        n = np.clip(n, -1.0, 1.0)
        return {
            CH_N_X: np.ascontiguousarray(n[..., 0]),
            CH_N_Y: np.ascontiguousarray(n[..., 1]),
            CH_N_Z: np.ascontiguousarray(n[..., 2]),
        }


__all__ = ["MIN_VRAM_GB", "MarigoldNormalsPass"]
