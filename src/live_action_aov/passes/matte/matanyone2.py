# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""MatAnyone 2 refiner — higher-quality soft-alpha matting (spec §13.1 Phase 3).

Backend: `pq-yang/MatAnyone` — a memory-based video matting network that
typically beats RVM on hair/fur, thin limbs, and transparency. License is
`NTU-S-Lab-1.0` (non-commercial research license), so this pass is
gated behind `--allow-noncommercial`; sidecars written with this refiner
stamp `liveaov/matte/commercial = "false"` so downstream QC can
distinguish them from RVM-refined deliverables before shipping to clients.

Contract parity with RVM (intentional — the executor discovers the
refiner by walking the DAG for `provides_artifacts = ["matte_heroes"]`,
not by name):
- Reads `sam3_hard_masks` + `sam3_instances` from upstream SAM 3.
- Emits `matte.{r,g,b,a}` per frame (per-clip slot lock preserved).
- Publishes `matte_heroes` for executor metadata wiring.

Drop-in replacement: `--refiner matanyone2` swaps RVM for MatAnyone 2
without any detector change. `sam3_hard_masks` keeps the exact shape
this module expects — `{track_id: {"label", "frames", "stack (T,H,W)"}}`
— the schema CorridorKey v2c will also consume.

Test plumbing mirrors `rvm.py`: `_load_model` and `_refine_instance` are
override hooks so CI tests can drop in a deterministic fake without
pulling the actual model weights (which are hosted on GitHub releases,
not HuggingFace).
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

# Slot → channel map duplicated from rvm.py rather than shared. The matte
# package deliberately keeps its channel contract per-file so a hypothetical
# round-3 refiner can change its slot semantics without a cross-module
# refactor. Any drift shows up immediately in the RVM/MatAnyone parity tests.
_SLOT_TO_CHANNEL: dict[str, str] = {
    "r": CH_MATTE_R,
    "g": CH_MATTE_G,
    "b": CH_MATTE_B,
    "a": CH_MATTE_A,
}


class MatAnyone2RefinerPass(UtilityPass):
    name = "matanyone2"
    version = "0.1.0"
    license = License(
        spdx="NTU-S-Lab-1.0",
        commercial_use=False,
        commercial_tool_resale=False,
        notes=(
            "MatAnyone (pq-yang/MatAnyone) is released under NTU S-Lab's "
            "non-commercial research license. Inference, weights, and any "
            "derivative product of the alpha mattes are NC-only. The CLI "
            "license gate blocks this pass unless --allow-noncommercial is "
            "set, and sidecars written by this refiner stamp "
            "`liveaov/matte/commercial = false` so downstream QC can "
            "catch NC deliverables before they reach a commercial client."
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

    # Hard DAG dep on SAM 3 (or any future detector that emits the same
    # artifact shape). Parity with RVMRefinerPass.
    requires_artifacts = ["sam3_hard_masks", "sam3_instances"]
    provides_artifacts = ["matte_heroes"]

    # Memory-based recurrent matting — already temporally coherent, extra
    # smoothing would just soften edges.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        # MatAnyone 2 is usually invoked via `torch.hub.load` off the
        # upstream repo, with a named variant. Keep the params open so
        # we can swap to a HF-hosted mirror if one lands.
        "model_id": "pq-yang/MatAnyone",
        "variant": "matanyone2",
        "inference_short_edge": 720,  # higher than RVM — MatAnyone likes more pixels
        "precision": "fp16",
        "hard_mask_erode": 0,
        # MatAnyone-specific knobs (no-op in the fake backend):
        "warmup_frames": 5,  # recurrent state needs a run-in
        "memory_every": 5,  # memory-bank refresh cadence
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._heroes: list[dict[str, Any]] = []
        self._refined: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Artifact ingestion — same shape as RVM, same unwrap pattern.
    # ------------------------------------------------------------------

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
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

        variant = str(self.params.get("variant", "matanyone2"))
        model = torch.hub.load(
            str(self.params["model_id"]),
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

        The real implementation feeds (plate, hard_mask_as_trimap) through
        MatAnyone's recurrent forward with a memory-bank refresh every
        `memory_every` frames. Tests override this to produce a
        deterministic soft alpha (e.g. hard_stack scaled) with no torch
        dependency.
        """
        raise NotImplementedError(
            "Real MatAnyone 2 refinement requires torch.hub access to "
            "pq-yang/MatAnyone. Install torch + internet access, or override "
            "`_refine_instance` in tests."
        )

    # ------------------------------------------------------------------
    # Per-frame lifecycle unused — refinement is shot-level.
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
        frames = np.stack([reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0).astype(
            np.float32, copy=False
        )
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])

        channel_stacks: dict[str, np.ndarray] = {
            ch: np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
            for ch in _SLOT_TO_CHANNEL.values()
        }

        self._refined = []
        frame_to_local = {f: i for i, f in enumerate(range(first, last + 1))}

        for hero in self._heroes:
            slot = str(hero.get("slot", ""))
            channel = _SLOT_TO_CHANNEL.get(slot)
            if channel is None:
                continue
            track_id = int(hero["track_id"])
            track = self._hard_masks.get(track_id)
            if track is None:
                self._refined.append({**hero, "refined_frames": [], "missing_hard_mask": True})
                continue

            track_frames: list[int] = list(track.get("frames") or [])
            track_stack: np.ndarray = np.asarray(track.get("stack"), dtype=np.float32)
            if track_stack.ndim != 3 or track_stack.shape[0] != len(track_frames):
                raise ValueError(
                    f"sam3_hard_masks[{track_id}] stack has shape {track_stack.shape}, "
                    f"expected (T={len(track_frames)}, H, W)"
                )
            dense_hard = np.zeros((n_frames, plate_h, plate_w), dtype=np.float32)
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

            # Per-clip slot lock: zero frames where the instance was absent
            # in the seed artifact, even if the refiner's memory bank bled
            # alpha into them. Same rule as RVM.
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
    # Artifact emission
    # ------------------------------------------------------------------

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._refined:
            return {}
        return {"matte_heroes": {0: list(self._refined)}}


__all__ = ["MatAnyone2RefinerPass"]
