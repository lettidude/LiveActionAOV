# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""RAFT optical-flow pass (torchvision backend).

Backend: `torchvision.models.optical_flow.raft_large` (BSD-3, weights ship
with torchvision — no separate download step, no Hugging Face dependency).
RAFT is the v1 keystone intermediate: the smoother reads it; depth/normals
passes can consume its `parallax_estimate` artifact for per-shot backend
routing (v2a).

Outputs (spec §5.1, channels.py):
- `motion.x` / `motion.y`  — forward flow at frame f (f → f+1) in pixels at plate res
- `back.x` / `back.y`      — backward flow at frame f (f → f-1) in pixels
- `flow.confidence`        — F-B consistency in [0, 1]; 1 = perfectly consistent

Artifacts emitted to downstream consumers (FlowCache + smoother):
- `forward_flow` / `backward_flow` — per-frame `(2, H, W)` arrays
- `occlusion_mask`                 — per-frame `(H, W)`, 1 where occluded
- `parallax_estimate`              — per-shot scalar: median |forward flow| / W

Implementation notes (spec §11.3 traps):
- **Trap 1**: RAFT infers at a downscaled resolution; we scale flow vectors
  by the plate/inference ratio on upscale. Otherwise Nuke VectorBlur breaks.
- Inference resolution rounds to a multiple of 8 (RAFT internal stride).
- `input_colorspace = "srgb_display"` — we assume the executor hands us
  display-referred sRGB frames. For Phase 1's linear fixture we apply a
  cheap gamma-2.2 + clamp in `preprocess`; v2 wires through the proper
  DisplayTransform.
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
    CH_BACK_X,
    CH_BACK_Y,
    CH_BACKWARD_U,
    CH_BACKWARD_V,
    CH_FLOW_CONFIDENCE,
    CH_FORWARD_U,
    CH_FORWARD_V,
    CH_MOTION_X,
    CH_MOTION_Y,
)


class RAFTPass(UtilityPass):
    name = "flow"
    version = "0.1.0"
    license = License(
        spdx="BSD-3-Clause",
        commercial_use=True,
        commercial_tool_resale=True,
        notes="RAFT (Teed & Deng 2020). Weights ship with torchvision under BSD-3.",
    )
    pass_type = PassType.MOTION
    temporal_mode = TemporalMode.PAIR
    input_colorspace = "srgb_display"

    produces_channels = [
        # Canonical spec names (design §5.1).
        ChannelSpec(name=CH_MOTION_X, description="Forward flow x (px at plate res)"),
        ChannelSpec(name=CH_MOTION_Y, description="Forward flow y"),
        ChannelSpec(name=CH_BACK_X, description="Backward flow x"),
        ChannelSpec(name=CH_BACK_Y, description="Backward flow y"),
        ChannelSpec(name=CH_FLOW_CONFIDENCE, description="F-B consistency [0,1]"),
        # Nuke-native aliases (same data, VectorBlur-friendly naming).
        ChannelSpec(name=CH_FORWARD_U, description="Alias of motion.x for Nuke VectorBlur"),
        ChannelSpec(name=CH_FORWARD_V, description="Alias of motion.y for Nuke VectorBlur"),
        ChannelSpec(name=CH_BACKWARD_U, description="Alias of back.x for Nuke VectorBlur"),
        ChannelSpec(name=CH_BACKWARD_V, description="Alias of back.y for Nuke VectorBlur"),
    ]
    provides_artifacts = [
        "forward_flow",
        "backward_flow",
        "occlusion_mask",
        "parallax_estimate",
    ]

    DEFAULT_PARAMS: dict[str, Any] = {
        "backend": "raft_large",  # "raft_large" | "raft_small"
        "precision": "fp32",  # "fp32" | "fp16" (fp16 only if cuda)
        "fb_threshold_px": 1.0,
        "inference_resolution": 520,  # target max-side in px (multiple of 8)
        "num_flow_updates": 12,  # RAFT iters; raft_large default
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._transforms: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Filled by run_shot; emitted via emit_artifacts().
        self._forward: dict[int, np.ndarray] = {}
        self._backward: dict[int, np.ndarray] = {}
        self._occlusion: dict[int, np.ndarray] = {}
        self._parallax: float = 0.0

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from torchvision.models.optical_flow import (
            Raft_Large_Weights,
            Raft_Small_Weights,
            raft_large,
            raft_small,
        )

        backend = self.params["backend"]
        if backend == "raft_large":
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        elif backend == "raft_small":
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)
        else:
            raise ValueError(f"Unknown RAFT backend: {backend!r}")

        self._transforms = weights.transforms()
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
    # Single-pair lifecycle (still implemented so external tools can call
    # RAFTPass on a single pair without going through run_shot).
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Pair preprocess: frames is (2, H, W, 3) float32.

        Returns a dict suitable for `infer()`. We record plate shape +
        inference shape for the upscale + vector-scaling in `postprocess`.
        """
        import torch

        if frames.ndim != 4 or frames.shape[0] != 2:
            raise ValueError(f"RAFT PAIR preprocess expects (2, H, W, 3), got {frames.shape}")
        self._load_model()
        assert self._transforms is not None and self._device is not None

        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        inf_target = int(self.params["inference_resolution"])
        ratio = inf_target / max(plate_h, plate_w)
        # RAFT's correlation pyramid needs feature maps ≥ 16 on both axes,
        # which means inference input ≥ 128 px on each side (8× downsample).
        inf_h = max(128, int(round(plate_h * ratio / 8)) * 8)
        inf_w = max(128, int(round(plate_w * ratio / 8)) * 8)

        # Linear scene-referred → display-referred approximation (trap 4's
        # poor cousin; the proper DisplayTransform wires through in v2).
        disp = _linear_to_display(frames[..., :3])

        t = torch.from_numpy(disp).permute(0, 3, 1, 2).contiguous()  # (2, 3, H, W)
        t = torch.nn.functional.interpolate(
            t, size=(inf_h, inf_w), mode="bilinear", align_corners=False
        )
        img1 = t[0:1]
        img2 = t[1:2]
        img1, img2 = self._transforms(img1, img2)

        return {
            "img1": img1.to(self._device, dtype=self._dtype),
            "img2": img2.to(self._device, dtype=self._dtype),
            "plate_shape": (plate_h, plate_w),
            "inf_shape": (inf_h, inf_w),
        }

    def infer(self, tensor: Any) -> Any:
        import torch

        assert self._model is not None
        img1 = tensor["img1"]
        img2 = tensor["img2"]
        iters = int(self.params["num_flow_updates"])
        with torch.no_grad():
            fwd_list = self._model(img1, img2, num_flow_updates=iters)
            bwd_list = self._model(img2, img1, num_flow_updates=iters)
        return {
            "fwd": fwd_list[-1].float(),
            "bwd": bwd_list[-1].float(),
            "plate_shape": tensor["plate_shape"],
            "inf_shape": tensor["inf_shape"],
        }

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Upscale flow to plate res (scaling vectors), compute F-B conf.

        Returns the per-pair channel dict. For `run_shot` the executor uses
        this to populate frame-indexed outputs; direct callers get flow at
        the first frame of the pair (motion = f→f+1, back at f is the
        reverse flow from f+1→f, which conceptually belongs at frame f+1
        but we return it here so single-pair users can still inspect it).
        """
        fwd_plate, bwd_plate = _upscale_flow_to_plate(
            tensor["fwd"], tensor["bwd"], tensor["plate_shape"], tensor["inf_shape"]
        )
        conf = _fb_consistency(
            fwd_plate, bwd_plate, threshold_px=float(self.params["fb_threshold_px"])
        )
        fwd_np = fwd_plate[0].cpu().numpy().astype(np.float32)
        bwd_np = bwd_plate[0].cpu().numpy().astype(np.float32)
        conf_np = conf[0].cpu().numpy().astype(np.float32)
        motion_x, motion_y = fwd_np[0], fwd_np[1]
        back_x, back_y = bwd_np[0], bwd_np[1]
        return {
            # Spec §5.1 canonical names
            CH_MOTION_X: motion_x,
            CH_MOTION_Y: motion_y,
            CH_BACK_X: back_x,
            CH_BACK_Y: back_y,
            CH_FLOW_CONFIDENCE: conf_np,
            # Nuke-native aliases (same arrays — EXR zip compresses the
            # duplication to near-zero disk overhead).
            CH_FORWARD_U: motion_x,
            CH_FORWARD_V: motion_y,
            CH_BACKWARD_U: back_x,
            CH_BACKWARD_V: back_y,
        }

    # ------------------------------------------------------------------
    # Shot-level iteration: iterate pairs, assemble per-frame outputs
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        # Run RAFT on every consecutive pair. For pair (f, f+1) we get both
        # fwd (f → f+1) and bwd (f+1 → f). We store fwd at frame f and bwd
        # at frame f+1 so each frame's channels are semantically correct.
        pair_fwd: dict[int, np.ndarray] = {}  # fwd at frame f = flow f → f+1
        pair_bwd: dict[int, np.ndarray] = {}  # bwd at frame f+1 = flow f+1 → f
        pair_conf: dict[int, np.ndarray] = {}  # confidence of fwd at f

        plate_h: int | None = None
        plate_w: int | None = None
        for f in range(first, last):
            frame_a, _ = reader.read_frame(f)
            frame_b, _ = reader.read_frame(f + 1)
            pair = np.stack([frame_a, frame_b], axis=0)
            if plate_h is None:
                plate_h, plate_w = int(pair.shape[1]), int(pair.shape[2])
            model_in = self.preprocess(pair)
            model_out = self.infer(model_in)
            channels = self.postprocess(model_out)
            pair_fwd[f] = np.stack([channels[CH_MOTION_X], channels[CH_MOTION_Y]], axis=0)
            pair_bwd[f + 1] = np.stack([channels[CH_BACK_X], channels[CH_BACK_Y]], axis=0)
            pair_conf[f] = channels[CH_FLOW_CONFIDENCE]

        # Assemble per-frame channel dicts. Endpoints get zero-valued flow.
        all_flow_channels = (
            CH_MOTION_X,
            CH_MOTION_Y,
            CH_BACK_X,
            CH_BACK_Y,
            CH_FLOW_CONFIDENCE,
            CH_FORWARD_U,
            CH_FORWARD_V,
            CH_BACKWARD_U,
            CH_BACKWARD_V,
        )
        if plate_h is None or plate_w is None:
            # Single-frame shot — nothing to compute.
            zero = np.zeros((1, 1), dtype=np.float32)
            return {first: dict.fromkeys(all_flow_channels, zero)}

        zero2 = np.zeros((2, plate_h, plate_w), dtype=np.float32)
        zero1 = np.zeros((plate_h, plate_w), dtype=np.float32)

        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            fwd = pair_fwd.get(f, zero2)
            bwd = pair_bwd.get(f, zero2)
            conf = pair_conf.get(f, zero1)  # conf at endpoint = 0 (no forward)
            out[f] = {
                # Spec §5.1 canonical names
                CH_MOTION_X: fwd[0],
                CH_MOTION_Y: fwd[1],
                CH_BACK_X: bwd[0],
                CH_BACK_Y: bwd[1],
                CH_FLOW_CONFIDENCE: conf,
                # Nuke-native aliases (same arrays)
                CH_FORWARD_U: fwd[0],
                CH_FORWARD_V: fwd[1],
                CH_BACKWARD_U: bwd[0],
                CH_BACKWARD_V: bwd[1],
            }
            # Stash for artifact emission.
            self._forward[f] = fwd
            self._backward[f] = bwd
            self._occlusion[f] = (conf < 0.5).astype(np.float32)

        # Per-shot parallax: median |fwd| / plate_w across all pairs.
        if pair_fwd:
            mags = np.concatenate(
                [np.sqrt(v[0] ** 2 + v[1] ** 2).ravel() for v in pair_fwd.values()]
            )
            self._parallax = float(np.median(mags) / max(plate_w, 1))
        return out

    def emit_artifacts(self) -> dict[str, dict[int, np.ndarray]]:
        # Scalar parallax is broadcast to every frame as a (1,) array so it
        # matches the generic `{name: {frame: array}}` shape; consumers that
        # only want the scalar read any frame.
        parallax = np.asarray([self._parallax], dtype=np.float32)
        return {
            "forward_flow": dict(self._forward),
            "backward_flow": dict(self._backward),
            "occlusion_mask": dict(self._occlusion),
            "parallax_estimate": dict.fromkeys(self._forward, parallax),
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _linear_to_display(x: np.ndarray) -> np.ndarray:
    """Cheap stand-in for the DisplayTransform when Phase 1 feeds linear
    ACEScg-ish pixels to RAFT. Applies gamma 2.2 and clamps to [0,1]."""
    out = np.clip(x, 0.0, None)
    out = np.power(out, 1.0 / 2.2, where=out > 0)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _upscale_flow_to_plate(
    fwd: Any,
    bwd: Any,
    plate_shape: tuple[int, int],
    inf_shape: tuple[int, int],
) -> tuple[Any, Any]:
    """Bilinear upscale to plate resolution + scale vectors by the ratio
    (spec §11.3 trap 1)."""
    import torch

    plate_h, plate_w = plate_shape
    inf_h, inf_w = inf_shape
    sx = plate_w / max(inf_w, 1)
    sy = plate_h / max(inf_h, 1)
    fwd_up = torch.nn.functional.interpolate(
        fwd, size=(plate_h, plate_w), mode="bilinear", align_corners=False
    )
    bwd_up = torch.nn.functional.interpolate(
        bwd, size=(plate_h, plate_w), mode="bilinear", align_corners=False
    )
    fwd_up = fwd_up.clone()
    bwd_up = bwd_up.clone()
    fwd_up[:, 0] *= sx
    fwd_up[:, 1] *= sy
    bwd_up[:, 0] *= sx
    bwd_up[:, 1] *= sy
    return fwd_up, bwd_up


def _fb_consistency(fwd: Any, bwd: Any, threshold_px: float) -> Any:
    """Forward-backward consistency → [0,1] confidence.

    For pixel p: warped = p + fwd[p]; sample bwd at warped; err = fwd[p] +
    bwd[warped]. confidence = exp(-|err|^2 / threshold^2).
    """
    import torch

    b, _, h, w = fwd.shape
    y_coords = torch.arange(h, dtype=fwd.dtype, device=fwd.device)
    x_coords = torch.arange(w, dtype=fwd.dtype, device=fwd.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    warped_x = xx + fwd[:, 0]
    warped_y = yy + fwd[:, 1]
    grid = torch.stack(
        [warped_x / max(w - 1, 1) * 2 - 1, warped_y / max(h - 1, 1) * 2 - 1],
        dim=-1,
    )  # (B, H, W, 2)
    bwd_sampled = torch.nn.functional.grid_sample(
        bwd, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    err = fwd + bwd_sampled
    err_mag = torch.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2)
    conf = torch.exp(-(err_mag**2) / (threshold_px**2 + 1e-6))
    return conf


__all__ = ["RAFTPass"]
