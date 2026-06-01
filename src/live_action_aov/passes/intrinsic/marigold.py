# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Marigold-IID intrinsic passes — albedo / shading / residual / materials.

Backend: PRS-ETH Marigold Intrinsic Image Decomposition (diffusers
`MarigoldIntrinsicsPipeline`), an SD2-based (~1B) single-image model. Two
checkpoints, exposed as two passes sharing this module's loader/inference:

- **Lighting** (`marigold-iid-lighting-v1-1`, HyperSim): albedo + diffuse
  shading + non-diffuse residual, with `I = A*S + R`. → `albedo.*`,
  `irradiance.*` (shading), `residual.*`.
- **Appearance** (`marigold-iid-appearance-v1-1`, InteriorVerse): albedo +
  roughness + metallicity (PBR). → `albedo.*`, `material.roughness/metalness`.

Why Marigold (chosen over UniVidX for this tool): ~0.4 s/frame, ~3-4 GB VRAM,
~2 GB model — vs UniVidX's 14B/85 GB/minutes-per-frame — for comparable albedo
(validated head-to-head 2026-06, see docs/albedo-marigold.md). License is
OpenRAIL++-M (commercial use permitted, use-restricted) — fine for this tool.

Temporal: PER_FRAME (single-image model). A fixed per-frame seed gives strong
frame-to-frame stability; the flow-guided temporal smoother (`smooth: auto`)
cleans up any residual flicker. Outputs are LINEAR (albedo absolute; shading/
residual up-to-scale) so they drop straight into the linear sidecar.
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
from live_action_aov.io.channels import (
    CH_ALBEDO_B,
    CH_ALBEDO_G,
    CH_ALBEDO_R,
    CH_IRRADIANCE_B,
    CH_IRRADIANCE_G,
    CH_IRRADIANCE_R,
    CH_METALNESS,
    CH_RESIDUAL_B,
    CH_RESIDUAL_G,
    CH_RESIDUAL_R,
    CH_ROUGHNESS,
)

# Marigold is an SD2-derived ~1B model; ~3-4 GB at fp16 incl. VAE + activations.
MIN_VRAM_GB = 4.0

_MARIGOLD_LICENSE = License(
    spdx="OpenRAIL++-M",
    commercial_use=True,
    commercial_tool_resale=True,
    notes=(
        "Marigold-IID (PRS-ETH / ETH Zurich), SD2 backbone. CreativeML "
        "OpenRAIL++-M permits commercial use with behavioural-use restrictions "
        "(not a pure OSS/Apache license). Acceptable for this tool's personal / "
        "freelance use."
    ),
)


class _MarigoldIntrinsicBase(UtilityPass):
    """Shared loader + per-frame inference for the Marigold IID passes.

    Subclasses set `_MODEL_ID`, identity/contract attributes, and implement
    `_channels_from_prediction`.
    """

    _MODEL_ID: str = ""

    license = _MARIGOLD_LICENSE
    pass_type = PassType.RADIOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    input_colorspace = "srgb_display"

    @staticmethod
    def vram_estimate_gb_fn(w: int, h: int) -> float:
        del w, h  # processing_resolution-bounded; plate size barely matters
        return MIN_VRAM_GB

    DEFAULT_PARAMS: dict[str, Any] = {
        "num_inference_steps": 4,  # Marigold is reliable at 1-4 steps
        "ensemble_size": 1,  # >=3 improves precision at linear cost
        "seed": 42,  # fixed per frame -> temporal stability
        "precision": "fp16",  # "fp16" | "fp32" (fp16 only on cuda)
        "smooth": "auto",  # executor auto-wires the flow smoother
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._pipe: Any = None
        self._device: Any = None
        self._target_names: list[str] = []

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

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
            pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(self._MODEL_ID, **kwargs)
        except Exception:
            kwargs.pop("variant", None)
            pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(self._MODEL_ID, **kwargs)
        pipe = pipe.to(self._device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        self._target_names = list(pipe.target_properties["target_names"])

    # ------------------------------------------------------------------
    # Single-frame lifecycle (PER_FRAME)
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        if frames.ndim != 4 or frames.shape[0] != 1 or frames.shape[-1] != 3:
            raise ValueError(
                f"{type(self).__name__} preprocess expects (1, H, W, 3), got {frames.shape}"
            )
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        u8 = (np.clip(frames[0], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)  # sRGB-display
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
        # prediction: (n_targets, H, W, 3) float32 linear in [0, 1].
        return {"prediction": np.asarray(out.prediction), "plate_shape": tensor["plate_shape"]}

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        pred = tensor["prediction"]
        if pred.ndim != 4 or pred.shape[0] != len(self._target_names) or pred.shape[-1] != 3:
            raise ValueError(
                f"Unexpected Marigold prediction shape {pred.shape} for targets {self._target_names}"
            )
        plate_h, plate_w = tensor["plate_shape"]
        by_target = {name: pred[i] for i, name in enumerate(self._target_names)}  # each (H, W, 3)
        out = self._channels_from_prediction(by_target)
        # Safety: ensure every channel is (plate_h, plate_w) float32.
        return {k: _to_plate(v, plate_h, plate_w) for k, v in out.items()}

    # Subclasses map model targets -> sidecar channels.
    def _channels_from_prediction(self, by_target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        raise NotImplementedError


class MarigoldIntrinsicsLightingPass(_MarigoldIntrinsicBase):
    name = "marigold_iid_lighting"
    version = "0.1.0"
    _MODEL_ID = "prs-eth/marigold-iid-lighting-v1-1"

    produces_channels = [
        ChannelSpec(
            name=CH_ALBEDO_R, description="Albedo (base colour) R, lighting removed — linear"
        ),
        ChannelSpec(name=CH_ALBEDO_G, description="Albedo G — linear"),
        ChannelSpec(name=CH_ALBEDO_B, description="Albedo B — linear"),
        ChannelSpec(name=CH_IRRADIANCE_R, description="Diffuse shading R — linear, up-to-scale"),
        ChannelSpec(name=CH_IRRADIANCE_G, description="Diffuse shading G"),
        ChannelSpec(name=CH_IRRADIANCE_B, description="Diffuse shading B"),
        ChannelSpec(
            name=CH_RESIDUAL_R,
            description="Non-diffuse residual R (specular) — linear, up-to-scale",
        ),
        ChannelSpec(name=CH_RESIDUAL_G, description="Non-diffuse residual G"),
        ChannelSpec(name=CH_RESIDUAL_B, description="Non-diffuse residual B"),
    ]
    smoothable_channels = [
        CH_ALBEDO_R,
        CH_ALBEDO_G,
        CH_ALBEDO_B,
        CH_IRRADIANCE_R,
        CH_IRRADIANCE_G,
        CH_IRRADIANCE_B,
        CH_RESIDUAL_R,
        CH_RESIDUAL_G,
        CH_RESIDUAL_B,
    ]

    def _channels_from_prediction(self, by_target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        a, s, r = by_target["albedo"], by_target["shading"], by_target["residual"]
        return {
            CH_ALBEDO_R: a[..., 0],
            CH_ALBEDO_G: a[..., 1],
            CH_ALBEDO_B: a[..., 2],
            CH_IRRADIANCE_R: s[..., 0],
            CH_IRRADIANCE_G: s[..., 1],
            CH_IRRADIANCE_B: s[..., 2],
            CH_RESIDUAL_R: r[..., 0],
            CH_RESIDUAL_G: r[..., 1],
            CH_RESIDUAL_B: r[..., 2],
        }


class MarigoldIntrinsicsAppearancePass(_MarigoldIntrinsicBase):
    name = "marigold_iid_appearance"
    version = "0.1.0"
    _MODEL_ID = "prs-eth/marigold-iid-appearance-v1-1"

    produces_channels = [
        ChannelSpec(
            name=CH_ALBEDO_R, description="Albedo (base colour) R, lighting removed — linear"
        ),
        ChannelSpec(name=CH_ALBEDO_G, description="Albedo G — linear"),
        ChannelSpec(name=CH_ALBEDO_B, description="Albedo B — linear"),
        ChannelSpec(name=CH_ROUGHNESS, description="PBR roughness [0,1]"),
        ChannelSpec(name=CH_METALNESS, description="PBR metalness [0,1]"),
    ]
    smoothable_channels = [CH_ALBEDO_R, CH_ALBEDO_G, CH_ALBEDO_B, CH_ROUGHNESS, CH_METALNESS]

    def _channels_from_prediction(self, by_target: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Target names per upstream: albedo, roughness, metallicity. Material
        # maps are scalar — Marigold emits them as 3-channel grey; average to 1.
        a = by_target["albedo"]
        rough = by_target.get("roughness")
        metal = by_target.get("metallicity", by_target.get("metalness"))
        if rough is None or metal is None:
            raise ValueError(f"Appearance targets missing roughness/metallicity: {list(by_target)}")
        return {
            CH_ALBEDO_R: a[..., 0],
            CH_ALBEDO_G: a[..., 1],
            CH_ALBEDO_B: a[..., 2],
            CH_ROUGHNESS: rough.mean(axis=-1),
            CH_METALNESS: metal.mean(axis=-1),
        }


def _to_plate(arr: np.ndarray, plate_h: int, plate_w: int) -> np.ndarray:
    """Ensure a single-channel (H, W) float32 array at plate resolution."""
    a = np.asarray(arr, dtype=np.float32)
    if a.shape[:2] != (plate_h, plate_w):
        import cv2

        resized = cv2.resize(a, (plate_w, plate_h), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(resized, dtype=np.float32)
    return np.ascontiguousarray(a, dtype=np.float32)


__all__ = [
    "MIN_VRAM_GB",
    "MarigoldIntrinsicsAppearancePass",
    "MarigoldIntrinsicsLightingPass",
]
