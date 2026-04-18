"""Robust Video Matting refiner (spec §13.1 Phase 3).

Backend: `PeterL1n/RobustVideoMatting` — a light recurrent network that
turns a hard mask + RGB plate into a soft alpha matte. MIT-licensed and
commercial-safe, which is why it's the default refiner for Phase 3 round 1.
The non-commercial MatAnyone 2 refiner (Round 2) will drop in behind the
same contract (same `requires_artifacts`, same emitted channels).

Pipeline:
1. SAM 3 runs upstream and publishes:
   - `sam3_hard_masks`: per-track `{label, frames, stack (T, H, W)}`
   - `sam3_instances` : ranked hero list with pre-assigned RGBA slots
2. The refiner's `ingest_artifacts` stashes both.
3. `run_shot` reads every plate frame, then for each hero slot:
   - Grabs the track's hard-mask stack
   - Calls `_refine_instance(plate_stack, hard_stack)` → soft alpha (T, H, W)
   - Writes the soft alpha into the channel corresponding to the hero's slot
     (`matte.r` / `matte.g` / `matte.b` / `matte.a`)
4. Frames where an instance is absent receive zeros (per-clip slot lock from
   the Phase 3 brainstorm — slots never change identity mid-shot, but a
   track may leave screen).

The refiner is deliberately the last-step packer (brainstorm §5): no
separate "packing pass", `matte.*` channels are emitted directly here.

Test plumbing mirrors the other VIDEO_CLIP passes: `_load_model` and
`_refine_instance` are override hooks so CI can inject a fake that turns a
hard mask into a "softened" alpha (e.g. a Gaussian blur) without importing
torch or pulling RVM weights.
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
    CH_MATTE_A,
    CH_MATTE_B,
    CH_MATTE_G,
    CH_MATTE_R,
)

# Canonical slot -> channel name mapping. Kept here (not in channels.py) so
# rank.py remains a pure-Python helper with no dependency on the writer's
# channel contract.
_SLOT_TO_CHANNEL: dict[str, str] = {
    "r": CH_MATTE_R,
    "g": CH_MATTE_G,
    "b": CH_MATTE_B,
    "a": CH_MATTE_A,
}


class RVMRefinerPass(UtilityPass):
    name = "rvm_refiner"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Robust Video Matting (PeterL1n/RobustVideoMatting) is MIT-"
            "licensed. No usage gate needed — both inference and outputs "
            "are commercial-safe. Sidecars stamp "
            "`utilityPass/matte/commercial = true` so downstream QC can "
            "distinguish them from MatAnyone 2 (NTU-S-Lab-1.0, NC) deliverables."
        ),
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=CH_MATTE_R, description="Hero matte slot R (soft alpha)"),
        ChannelSpec(name=CH_MATTE_G, description="Hero matte slot G (soft alpha)"),
        ChannelSpec(name=CH_MATTE_B, description="Hero matte slot B (soft alpha)"),
        ChannelSpec(name=CH_MATTE_A, description="Hero matte slot A (soft alpha)"),
    ]

    # Hard DAG dep: SAM 3 (or any future detector that emits the same
    # artifact schema) must run first.
    requires_artifacts = ["sam3_hard_masks", "sam3_instances"]
    provides_artifacts = ["matte_heroes"]

    # RVM's recurrent state already blends across frames; extra smoothing
    # would just soften edges. No smoothable channels.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "PeterL1n/RobustVideoMatting",
        "variant": "mobilenetv3",          # "mobilenetv3" | "resnet50"
        "inference_short_edge": 512,       # RVM is tolerant; keep it light
        "precision": "fp16",
        "downsample_ratio": 0.25,          # RVM README default for <=2K input
        "hard_mask_dilate": 3,             # grow the seed mask by N pixels
                                           # before multiplying the plate —
                                           # gives RVM a little context around
                                           # the hard boundary to refine.
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Populated by ingest_artifacts (shot-scoped; reset per shot because
        # the executor rebuilds the instance for every shot).
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []
        # Published by run_shot, read by emit_artifacts.
        self._refined: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Artifact ingestion — pull SAM 3's outputs before run_shot.
    # ------------------------------------------------------------------

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:  # type: ignore[override]
        hard = artifacts.get("sam3_hard_masks") or {}
        # `sam3_hard_masks` is shot-level, stored under a single frame key.
        # Unwrap to `dict[track_id, {"label", "frames", "stack"}]`.
        if hard:
            self._hard_masks = next(iter(hard.values())) or {}
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            self._heroes = list(next(iter(heroes.values())) or [])

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch

        # RVM is not on the HF transformers registry — it's usually loaded
        # via `torch.hub.load('PeterL1n/RobustVideoMatting', variant)`. We
        # prefer torch.hub so there's no extra dep beyond the torch stack.
        variant = str(self.params.get("variant", "mobilenetv3"))
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            variant,
            trust_repo=True,
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = (
            torch.float16
            if self.params.get("precision") == "fp16" and self._device.type == "cuda"
            else torch.float32
        )
        model.to(self._device, dtype=self._dtype).eval()
        self._model = model

    def _refine_instance(
        self,
        plate_stack: np.ndarray,
        hard_stack: np.ndarray,
    ) -> np.ndarray:
        """Turn a hard per-instance mask stack into a soft alpha stack.

        Input shapes:
            plate_stack: (T, H, W, 3) float32 sRGB in [0, 1]
            hard_stack:  (T, H, W)    float32 in [0, 1]
        Output:
            (T, H, W) float32 in [0, 1]

        ### Recipe: "plate-masking sandwich"

        RVM is a generic image-matting network; it doesn't accept a prior
        hard mask as a direct conditioning input. The workaround is to
        multiply the hard mask into the plate *before* inference, so the
        network only sees pixels inside (a slightly dilated version of)
        the seed region. Then the output alpha is bound by the same
        dilated mask so RVM can't hallucinate matte outside the seed.

        Steps:
        1. Upload hard mask to GPU, dilate by ``hard_mask_dilate`` pixels
           via a max-pool — this gives RVM a little context around the
           hard edge so its refinement has somewhere to grow from.
        2. Per-frame recurrent loop: mask the plate, forward through RVM
           (carrying ``r1..r4`` across frames — this is where temporal
           consistency comes from), clip the alpha to [0, 1], multiply
           back by the dilated mask.
        3. Stack the per-frame alphas into an (T, H, W) float32 array.

        Tests override this with a numpy-only fake so we don't need
        torch.hub weights on CI.
        """
        import torch
        import torch.nn.functional as F

        self._load_model()
        assert self._model is not None
        device = self._device
        dtype = self._dtype

        T, H, W, _ = plate_stack.shape
        dilate_px = max(int(self.params.get("hard_mask_dilate", 3)), 0)
        downsample_ratio = float(self.params.get("downsample_ratio", 0.25))

        # (T, 1, H, W) on-device copy of the hard mask, dilated via a
        # max-pool. Dilation on GPU avoids pulling scipy/cv2 into the
        # hot path. Kernel size 2k+1 keeps the mask centred.
        hard_t = (
            torch.from_numpy(hard_stack)
            .to(device=device, dtype=dtype)
            .unsqueeze(1)
        )
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            dilated_t = F.max_pool2d(
                hard_t, kernel_size=k, stride=1, padding=dilate_px
            )
        else:
            dilated_t = hard_t

        # Empty-mask fast path: if every frame's seed is all-zero, RVM
        # would just refine nothing. Skip the forward pass entirely.
        if float(dilated_t.sum().item()) == 0.0:
            return np.zeros((T, H, W), dtype=np.float32)

        r1 = r2 = r3 = r4 = None
        out_alpha = np.empty((T, H, W), dtype=np.float32)

        for t in range(T):
            # (1, 3, H, W) frame, masked by the (possibly dilated) seed.
            frame_np = plate_stack[t]
            frame_t = (
                torch.from_numpy(frame_np)
                .to(device=device, dtype=dtype)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            mask_t = dilated_t[t : t + 1]  # (1, 1, H, W)
            masked_frame = frame_t * mask_t

            with torch.no_grad():
                fgr, pha, r1, r2, r3, r4 = self._model(
                    masked_frame,
                    r1,
                    r2,
                    r3,
                    r4,
                    downsample_ratio=downsample_ratio,
                )
            # Bound the alpha by the dilated seed so RVM can't leak
            # matte into the background beyond our context band.
            pha = (pha * mask_t).clamp_(0.0, 1.0)
            out_alpha[t] = pha.squeeze(0).squeeze(0).float().cpu().numpy()

        return out_alpha

    # ------------------------------------------------------------------
    # Per-frame lifecycle is unused — refinement is shot-level. We still
    # implement the abstract methods because the ABC requires them.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        return {}

    # ------------------------------------------------------------------
    # Shot-level: loop heroes × frames, pack into matte.r/g/b/a
    # ------------------------------------------------------------------

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        n_frames = last - first + 1
        frames = np.stack(
            [reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0
        ).astype(np.float32, copy=False)
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])

        # Initialize all four matte channels with zeros — any slot that
        # doesn't receive a hero (e.g. clip has only 2 instances) stays
        # transparent.
        channel_stacks: dict[str, np.ndarray] = {
            ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
            for ch in _SLOT_TO_CHANNEL.values()
        }

        self._refined = []
        for hero in self._heroes:
            slot = str(hero.get("slot", ""))
            channel = _SLOT_TO_CHANNEL.get(slot)
            if channel is None:
                # Slot not in r/g/b/a — silently skip. (ranker guarantees
                # valid slots; this is belt-and-braces.)
                continue
            track_id = int(hero["track_id"])
            track = self._hard_masks.get(track_id)
            if track is None:
                # Ranker saw the track but the hard-mask artifact is
                # missing. Leave zeros; record for metadata so QC can
                # spot it.
                self._refined.append(
                    {**hero, "refined_frames": [], "missing_hard_mask": True}
                )
                continue

            # Build the dense (n_frames, H, W) hard-mask stack for this track.
            # The track's native stack only covers frames where the instance
            # was present; frames before/after get zeros so the refiner sees
            # a consistent-length video (required by RVM's recurrent loop).
            track_frames: list[int] = list(track.get("frames") or [])
            track_stack: np.ndarray = np.asarray(track.get("stack"), dtype=np.float32)
            if track_stack.ndim != 3 or track_stack.shape[0] != len(track_frames):
                raise ValueError(
                    f"sam3_hard_masks[{track_id}] stack has shape {track_stack.shape}, "
                    f"expected (T={len(track_frames)}, H, W)"
                )
            dense_hard = np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
            frame_to_local = {f: i for i, f in enumerate(range(first, last + 1))}
            for k, f in enumerate(track_frames):
                local = frame_to_local.get(int(f))
                if local is None:
                    continue
                dense_hard[local] = track_stack[k]

            soft = self._refine_instance(frames, dense_hard)
            if soft.shape != dense_hard.shape:
                raise ValueError(
                    f"_refine_instance({track_id}) returned shape {soft.shape}, "
                    f"expected {dense_hard.shape}"
                )
            soft = np.clip(soft.astype(np.float32, copy=False), 0.0, 1.0)

            # Zero out frames where the original hard mask was absent — the
            # refiner can hallucinate alpha in bordering frames (RVM bleeds
            # across its memory bank); the per-clip slot lock says "if the
            # instance isn't there, the slot is zero".
            present_mask = dense_hard.sum(axis=(1, 2)) > 0.0
            soft[~present_mask] = 0.0

            channel_stacks[channel] = soft
            refined_frames = sorted(int(f) for f in track_frames if int(f) in frame_to_local)
            self._refined.append(
                {
                    "track_id": track_id,
                    "slot": slot,
                    "label": hero.get("label", ""),
                    "score": float(hero.get("score", 0.0)),
                    "refined_frames": refined_frames,
                    "missing_hard_mask": False,
                }
            )

        # Emit per-frame channel dicts keyed by absolute frame index.
        out: dict[int, dict[str, np.ndarray]] = {}
        for i in range(n_frames):
            f = first + i
            out[f] = {
                CH_MATTE_R: channel_stacks[CH_MATTE_R][i],
                CH_MATTE_G: channel_stacks[CH_MATTE_G][i],
                CH_MATTE_B: channel_stacks[CH_MATTE_B][i],
                CH_MATTE_A: channel_stacks[CH_MATTE_A][i],
            }
        return out

    # ------------------------------------------------------------------
    # Artifact emission — expose the refined hero list so the executor
    # can stamp `utilityPass/matte/*` metadata onto every sidecar.
    # ------------------------------------------------------------------

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._refined:
            return {}
        # Shot-level artifact — stash under a sentinel frame key 0, same
        # trick the depth passes use for per-clip normalization scalars.
        return {"matte_heroes": {0: list(self._refined)}}


__all__ = ["RVMRefinerPass"]
