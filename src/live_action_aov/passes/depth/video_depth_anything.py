# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Video Depth Anything pass (temporal-native, commercial-safe, spec §13.1 Phase 2).

Backend: the vendored VideoDepthAnything model from
github.com/DepthAnything/Video-Depth-Anything. This is a video-native
successor to DA-V2 — same DINOv2 + DPT backbone, plus a temporal attention
module injected into the DPT decoder, trained on video clips to produce
temporally consistent relative depth. Apache-2.0 for all three variants
(vits / vitb / vitl), no gating.

Why this pass over DepthCrafter
-------------------------------

DepthCrafter is a diffusion model on top of Stable Video Diffusion; its
weights inherit SVD's non-commercial licence. VDA is a plain feed-forward
ViT and runs seconds-per-clip instead of minutes, at the cost of slightly
weaker absolute depth quality. For our use case (scene-referred VFX
plates, internal pipeline, need temporal stability and commercial safety)
VDA wins on all axes.

Outputs (spec §5.1, channels.py):
- `Z`     — relative depth, per-clip normalised [0, 1] (1 = near, flipped
            to match Nuke PositionFromDepth convention).
- `Z_raw` — raw model output, un-normalised.

Temporal: VIDEO_CLIP. VideoDepthAnything.infer_video_depth handles its own
sliding-window inference (INFER_LEN=32 frames with OVERLAP=10, keyframe-
aligned via `compute_scale_and_shift` across windows), so there is no
`smooth` to wire — the pass is already temporally consistent.

Implementation notes
--------------------

- Weights ship as a single `.pth` on HuggingFace
  (`depth-anything/Video-Depth-Anything-{Small,Base,Large}`); we pull with
  `hf_hub_download` on first use and keep it in the HF cache.
- The model's internal `infer_video_depth` expects frames as uint8 in
  [0, 255]. The display transform upstream hands us float32 [0, 1]
  sRGB-display, so we convert with `* 255` before the call.
- Per-clip normalisation (spec §2.2 trap 5) runs once after the model
  returns, across the whole clip. The flip `1 - (d - d_min) / span` puts
  larger Z at nearer objects.
- ViT-14 patch size requires frame dims to be multiples of 14; the model's
  internal `Resize` handles that from the user-visible `input_size` (518
  default, matching DA-V2's training resolution).

The PER_FRAME versions of this model exist as the `depth_anything_v2`
pass — keep that around for sub-shot preview or single-image workflows;
use this pass for anything that ends up on a timeline.
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

# HF repo ids for the three variants.
_VARIANT_REPOS: dict[str, str] = {
    "vits": "depth-anything/Video-Depth-Anything-Small",
    "vitb": "depth-anything/Video-Depth-Anything-Base",
    "vitl": "depth-anything/Video-Depth-Anything-Large",
}

# Checkpoint filenames at each repo root.
_VARIANT_WEIGHTS: dict[str, str] = {
    "vits": "video_depth_anything_vits.pth",
    "vitb": "video_depth_anything_vitb.pth",
    "vitl": "video_depth_anything_vitl.pth",
}

# Arch configs mirroring the upstream `run.py`.
_VARIANT_CONFIGS: dict[str, dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class VideoDepthAnythingPass(UtilityPass):
    name = "video_depth_anything"
    version = "0.1.0"
    license = License(
        spdx="Apache-2.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Video Depth Anything is Apache-2.0 (Bytedance 2025). No "
            "license gating. Weights for all three encoders ship on HF "
            "under the `depth-anything/` org."
        ),
    )
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    # Model's internal INFER_LEN; surfaced here so the scheduler can
    # size its frame buffer ahead of time.
    temporal_window = 32
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_Z, description="Relative depth, per-clip normalised [0, 1] (1=near)"),
        ChannelSpec(name=CH_Z_RAW, description="Raw model output (un-normalised)"),
    ]
    # VIDEO_CLIP passes are already temporally coherent — nothing for the
    # smoother to touch.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "variant": "vits",  # "vits" | "vitb" | "vitl"
        "input_size": 518,  # ViT-14 training size
        "precision": "fp16",  # "fp16" | "fp32" — fp16 only if CUDA
        "depth_space": "relative",
        "metric": False,  # True uses the separate metric weights
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        variant = str(self.params["variant"]).lower()
        if variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unknown variant {variant!r}. Choose from {sorted(_VARIANT_CONFIGS)}."
            )
        self._model: Any = None
        self._device: Any = None
        self._fp32: bool = str(self.params["precision"]).lower() != "fp16"
        # Per-clip normalisation constants recorded by run_shot.
        self._norm_min: float = 0.0
        self._norm_max: float = 1.0
        self._frame_keys: list[int] = []

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import hf_hub_download

        from live_action_aov.vendored.video_depth_anything.video_depth import (
            VideoDepthAnything,
        )

        variant = str(self.params["variant"]).lower()
        cfg = _VARIANT_CONFIGS[variant]
        repo = _VARIANT_REPOS[variant]
        weights_name = _VARIANT_WEIGHTS[variant]
        if self.params.get("metric"):
            # The upstream repo publishes metric weights in separate repos
            # (`depth-anything/Metric-Video-Depth-Anything-{Small,Large}`).
            # Not wired yet — fail loudly instead of silently loading the
            # relative weights.
            raise NotImplementedError(
                "metric=True not yet wired — needs a separate HF repo id "
                "mapping and a different checkpoint filename."
            )

        ckpt = hf_hub_download(repo_id=repo, filename=weights_name)

        model = VideoDepthAnything(**cfg, metric=False)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self._device).eval()
        self._model = model

    # ------------------------------------------------------------------
    # Single-frame lifecycle (stubs — this pass is VIDEO_CLIP, driven by
    # `run_shot`, but the base class still declares these as abstract).
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        """Not used — VIDEO_CLIP drives everything through `run_shot`.

        Left as a permissive passthrough so ad-hoc callers that know the
        model's API can still drive a single window without going through
        `run_shot`.
        """
        return frames

    def infer(self, tensor: Any) -> Any:
        raise NotImplementedError(
            "VideoDepthAnythingPass is VIDEO_CLIP; drive it via run_shot, "
            "not preprocess/infer/postprocess."
        )

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError(
            "VideoDepthAnythingPass is VIDEO_CLIP; drive it via run_shot, "
            "not preprocess/infer/postprocess."
        )

    # ------------------------------------------------------------------
    # Shot-level iteration: one model call across the whole clip.
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        self._load_model()
        first, last = frame_range

        frames_list: list[np.ndarray] = []
        for f in range(first, last + 1):
            arr, _ = reader.read_frame(f)
            # Model expects uint8 [0, 255] (its internal transform does
            # `/255.0` then ImageNet-normalises). Our input is float32
            # [0, 1] sRGB-display coming out of the display transform.
            arr8 = np.clip(arr, 0.0, 1.0)
            arr8 = (arr8 * 255.0 + 0.5).astype(np.uint8)
            frames_list.append(arr8)
        frames = np.stack(frames_list, axis=0)  # (N, H, W, 3) uint8

        raw = self._infer_clip(frames)  # (N, H, W) float32

        # Per-clip normalisation (trap 5). Unlike DA-V2 — which outputs
        # a depth-like scalar (larger == farther) and needs a `1 - x`
        # flip — VideoDepthAnything outputs **disparity** (larger ==
        # nearer; the final `F.relu` in `VideoDepthAnything.forward`
        # is a tell). Straight min/max already satisfies the Nuke
        # PositionFromDepth convention "larger Z == nearer", so no
        # flip here. First run visually confirmed this: with the flip,
        # the foreground subject rendered as a black silhouette on a
        # bright sky.
        d_min = float(raw.min())
        d_max = float(raw.max())
        span = max(d_max - d_min, 1e-6)
        z_clip = ((raw - d_min) / span).astype(np.float32)

        self._norm_min = d_min
        self._norm_max = d_max
        self._frame_keys = list(range(first, last + 1))
        out: dict[int, dict[str, np.ndarray]] = {}
        for i, f in enumerate(self._frame_keys):
            out[f] = {
                CH_Z: z_clip[i],
                CH_Z_RAW: raw[i].astype(np.float32, copy=False),
            }
        return out

    def _infer_clip(self, frames_u8: np.ndarray) -> np.ndarray:
        """Drive the model on the full clip. Overridable in tests."""
        assert self._model is not None
        device_str = "cuda" if self._device.type == "cuda" else "cpu"
        # target_fps argument is only used internally for FPS bookkeeping
        # when saving MP4s; we pass a nominal value and drop the returned
        # fps since we write EXRs.
        depth, _fps = self._model.infer_video_depth(
            frames_u8,
            target_fps=24.0,
            input_size=int(self.params["input_size"]),
            device=device_str,
            fp32=self._fp32,
        )
        return depth.astype(np.float32, copy=False)

    def emit_artifacts(self) -> dict[str, dict[int, np.ndarray]]:
        if not self._frame_keys:
            return {}
        any_frame = self._frame_keys[0]
        return {
            "depth_norm_min": {any_frame: np.asarray([self._norm_min], dtype=np.float32)},
            "depth_norm_max": {any_frame: np.asarray([self._norm_max], dtype=np.float32)},
        }


__all__ = ["VideoDepthAnythingPass"]
