# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""MegaFlow optical-flow pass (vendored, Apache-2.0).

MegaFlow (Zhang, Wang, Pollefeys, Xu — ECCV 2026): zero-shot large-
displacement flow from frozen DINOv2 ViT-L/14 features + a light iterative
refinement. Same channel contract as the RAFT pass (this is a subclass —
only the model, the preprocessing and the pair inference differ; F-B
confidence, upscale-with-vector-scaling, run_shot and the artifacts are
inherited).

Why it exists next to RAFT (bench on PF0071, 1024px, RTX 5090):
- consecutive frames (median ~1 px): both fine, RAFT 68 ms/pair vs
  MegaFlow 320 ms/pair, 1.1 GB vs 6 GB VRAM;
- 4x frame stride (~2.7 px): equal;
- 8x stride (~4.4 px, textureless water): RAFT's forward/backward error
  explodes (5.2 px) and it hallucinates flow on the water, MegaFlow stays
  at 0.67 px with a clean background.
So: RAFT stays the default for ordinary plates; MegaFlow is the choice for
fast action, retimes and textureless regions. Weights: HF Kristen-Z/MegaFlow
(Apache-2.0); the ResNet-34 stem is torchvision's (BSD-3, overwritten by the
checkpoint but downloaded at construction — prefetch covers it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import License
from live_action_aov.core.runtime_env import load_local_first
from live_action_aov.passes.flow.raft import RAFTPass, _linear_to_display

_HF_REPO = "Kristen-Z/MegaFlow"
_HF_FILE = "megaflow-flow.safetensors"


class MegaFlowPass(RAFTPass):
    name = "flow_megaflow"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "MegaFlow code + weights Apache-2.0 (ETH CVG, 2026); frozen DINOv2 "
            "ViT-L/14 backbone (Apache-2.0, Meta); torchvision ResNet-34 stem (BSD-3)."
        ),
    )

    DEFAULT_PARAMS: dict[str, Any] = {
        **RAFTPass.DEFAULT_PARAMS,
        "backend": "megaflow-flow",
        # Long edge of the inference image; both dims round to the ViT patch
        # (14). 952 = the authors' default; ~6 GB VRAM. 1400 ≈ 1080p-ish.
        "inference_resolution": 952,
        "num_flow_updates": 8,  # local regression refinement iterations
        "precision": "bf16",  # autocast dtype on CUDA: "bf16" | "fp16" | "fp32"
    }

    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from live_action_aov.vendored.megaflow.model.megaflow import MegaFlow

        ckpt = load_local_first(hf_hub_download, repo_id=_HF_REPO, filename=_HF_FILE)
        model = MegaFlow(fuse_cnn=True, use_temporal_attn=True, fix_width=True)
        model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=False)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prec = str(self.params.get("precision", "bf16"))
        self._dtype = (
            {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec, torch.float32)
            if self._device.type == "cuda"
            else torch.float32
        )
        self._model = model.to(self._device).eval()

    def preprocess(self, frames: np.ndarray) -> Any:
        """(2, H, W, 3) float32 display-referred [0,1] -> MegaFlow video
        tensor (1, 2, 3, h, w) in [0, 255], both dims multiples of 14."""
        import torch

        if frames.ndim != 4 or frames.shape[0] != 2:
            raise ValueError(f"MegaFlow PAIR preprocess expects (2, H, W, 3), got {frames.shape}")
        self._load_model()
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        target = int(self.params["inference_resolution"])
        ratio = min(1.0, target / max(plate_h, plate_w))
        inf_h = max(14, int(round(plate_h * ratio / 14)) * 14)
        inf_w = max(14, int(round(plate_w * ratio / 14)) * 14)
        disp = (
            _linear_to_display(frames[..., :3])
            if frames.max() > 1.0
            else np.clip(frames[..., :3], 0, 1)
        )
        t = torch.from_numpy(np.ascontiguousarray(disp)).permute(0, 3, 1, 2).float()
        t = torch.nn.functional.interpolate(
            t, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        return {
            "video": (t * 255.0)[None].to(self._device),  # (1, 2, 3, h, w)
            "plate_shape": (plate_h, plate_w),
            "inf_shape": (inf_h, inf_w),
        }

    def infer(self, tensor: Any) -> Any:
        import torch

        assert self._model is not None
        v = tensor["video"]
        iters = int(self.params["num_flow_updates"])
        use_ac = (
            self._device is not None
            and self._device.type == "cuda"
            and self._dtype != torch.float32
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=self._dtype, enabled=use_ac):
            fwd = self._model(v, num_reg_refine=iters)["flow_preds"][-1]  # (1, 1, 2, h, w)
            bwd = self._model(v.flip(1), num_reg_refine=iters)["flow_preds"][-1]
        return {
            "fwd": fwd[:, 0].float(),
            "bwd": bwd[:, 0].float(),
            "plate_shape": tensor["plate_shape"],
            "inf_shape": tensor["inf_shape"],
        }


__all__ = ["MegaFlowPass"]
