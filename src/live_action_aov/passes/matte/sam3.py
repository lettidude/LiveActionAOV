"""SAM 3 concept detector + video tracker (spec §13.1 Phase 3).

Backend: `facebookresearch/sam3` on HuggingFace. Takes a list of concept
prompts (e.g. `["person", "vehicle", ...]`), detects every instance
matching any concept on a seed frame, then tracks each instance across
the clip with SAM 3's own memory bank.

License: `SAM-License-1.0` (Meta's custom license). Commercial use is
permitted with a prohibition on military / ITAR applications — we ship
with `commercial_use=True` and surface the carve-out in `notes`. The
CLI license gate does not block this pass; users who ship for defense
must consult the upstream license themselves.

Outputs (spec §5.1, channels.py):
- Dynamic `mask.<concept>` channels — **union of all instances** of that
  concept per frame. The ExrSidecarWriter appends them after the canonical
  channel order.

Artifacts:
- `sam3_hard_masks`: `dict[int, {"label": str, "stack": (T, H, W) float32}]`
  keyed by `track_id`. Raw per-instance hard masks across the clip. This
  is the same structure CorridorKey (v2c) will consume, so we ship it in
  the shape the spec §21.8 locks in — don't simplify later.
- `sam3_instances`: a list of `HeroSlot` objects from `rank.py`, sorted
  by slot (r, g, b, a). The refiner consumes this to drive per-slot
  refinement.
- `matte_concepts`: the list of concept names actually detected on this
  clip (so the executor can stamp the list into sidecar metadata).

Temporal mode: VIDEO_CLIP — SAM 3 carries state across frames internally.
Per-clip slot lock (brainstorm decision #3): slots never change mid-clip.

Test plumbing mirrors DepthCrafter: `_load_model` + `_detect_seed` +
`_track_instance` are subclass-override hooks so CI can inject deterministic
rectangles without downloading 2 GB of weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import MASK_PREFIX
from live_action_aov.passes.matte.rank import (
    HeroOverride,
    Instance,
    RankWeights,
    rank_and_assign,
)


@dataclass
class _DetectedInstance:
    """Internal scratch structure — one detected + tracked instance.

    Kept separate from `rank.Instance` because this one carries the full
    per-frame mask stack (heavy), while `rank.Instance` only carries the
    reduced scalars (light, serializable).
    """

    track_id: int
    label: str
    masks: dict[int, np.ndarray]   # frame_idx -> (H, W) float32 in [0, 1]


class SAM3MattePass(UtilityPass):
    name = "sam3_matte"
    version = "0.1.0"
    license = License(
        spdx="SAM-License-1.0",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "SAM 3 uses Meta's custom license. Commercial use is permitted "
            "but military / ITAR applications are prohibited. Consult the "
            "upstream LICENSE before any defense-adjacent deployment."
        ),
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    # Channels are dynamic (one `mask.<concept>` per detected concept). We
    # declare none statically — the writer accepts unknown channels as long
    # as they follow the `mask.*` / `matte.*` naming convention.
    produces_channels: list[Any] = []
    provides_artifacts = ["sam3_hard_masks", "sam3_instances", "matte_concepts"]

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_id": "facebook/sam3",
        "concepts": ["person", "vehicle", "tree", "building", "sky", "water", "animal"],
        "confidence_threshold": 0.4,
        "min_area_fraction": 0.005,    # drop instances smaller than 0.5% of plate
        "sample_frame": "middle",      # "first" | "middle" | "last" | int
        "redetect_stride": None,       # None = seed-and-track only (v1 default)
        # Ranking weights (RankWeights schema).
        "ranking": {
            "area": 0.4,
            "centrality": 0.2,
            "motion": 0.2,
            "duration": 0.2,
            "user_priority": 0.0,
        },
        "max_heroes": 4,
        "heroes": [],                  # user overrides: [{"track_id": 17, "slot": "r"}]
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Populated by `run_shot`; consumed by `emit_artifacts`.
        self._instances: list[_DetectedInstance] = []
        self._heroes: list[Any] = []   # list[rank.HeroSlot]
        self._concepts_found: list[str] = []
        self._plate_shape: tuple[int, int] = (0, 0)
        # Flow snapshot (optional) — used for motion-energy feature when
        # present, ignored otherwise. Populated via ingest_artifacts.
        self._forward_flow: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Artifact ingestion — consume flow if the flow pass ran first.
    # ------------------------------------------------------------------

    # SAM 3 doesn't *require* flow, but if the user put `flow` in their job
    # the ranker can use it. We declare the soft dep by reading it in
    # `ingest_artifacts` and tolerating absence.
    requires_artifacts: list[str] = []   # soft dep; no DAG enforcement

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:  # type: ignore[override]
        self._forward_flow = dict(artifacts.get("forward_flow") or {})

    # ------------------------------------------------------------------
    # Model lifecycle (tests override)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load BOTH the detector and the tracker.

        SAM 3 ships two models that live under the same HF repo:

        - `Sam3Model` / `Sam3Processor`  — single-image, concept-conditioned
          instance segmentation. Used by `_detect_seed`. `AutoProcessor` on
          `facebook/sam3` resolves to the *video* processor instead of this
          one (the checkpoint's `model_type` is `sam3_video`), so we import
          the image classes explicitly to force the correct path.
        - `Sam3TrackerVideoModel` / `Sam3TrackerVideoProcessor` — inference
          session API with `input_masks=` seeding and
          `propagate_in_video_iterator`. Used by `_track_instance`.

        Tracker runs in bf16 on CUDA (per the SAM 3 model card); the
        detector stays in float32 because it fires once per concept and
        is not the hot loop.
        """
        if self._model is not None:
            return
        import torch
        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerVideoModel,
            Sam3TrackerVideoProcessor,
        )

        repo = str(self.params["model_id"])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.float32

        self._det_processor = Sam3Processor.from_pretrained(repo)
        det_model = Sam3Model.from_pretrained(repo)
        det_model.to(self._device).eval()
        self._det_model = det_model

        self._trk_processor = Sam3TrackerVideoProcessor.from_pretrained(repo)
        trk_dtype = (
            torch.bfloat16 if self._device.type == "cuda" else torch.float32
        )
        trk_model = Sam3TrackerVideoModel.from_pretrained(repo, dtype=trk_dtype)
        trk_model.to(self._device).eval()
        self._trk_model = trk_model
        self._trk_dtype = trk_dtype

        # Sentinel that the other guard checks — any of the four attrs
        # set above would work; pick one that's definitely non-None.
        self._model = det_model

    # ------------------------------------------------------------------
    # Detection + tracking — split so tests can override either.
    # ------------------------------------------------------------------

    def _detect_seed(
        self,
        seed_frame: np.ndarray,
        concepts: list[str],
    ) -> list[tuple[int, str, np.ndarray]]:
        """Run SAM 3 on one frame, return a list of (track_id, label, mask).

        Single-image, open-vocabulary instance segmentation: for each
        concept in `concepts`, run one forward pass and collect every
        detected instance whose score exceeds `confidence_threshold`.
        Track IDs are assigned sequentially starting at 1 — the tracker
        uses them as SAM 3 `obj_ids` in the downstream session.

        SAM 3's classification head is single-concept per forward, so we
        loop over concepts. For the "person + vehicle + tree" style job
        this means ~3 forward passes on the seed frame — cheap.
        """
        import torch
        from PIL import Image

        self._load_model()
        assert self._det_model is not None
        processor = self._det_processor
        model = self._det_model
        device = self._device

        H, W = int(seed_frame.shape[0]), int(seed_frame.shape[1])
        # Processor expects PIL RGB uint8 (or a torch/numpy tensor the
        # image processor can rescale). PIL is the least-surprising path.
        arr_u8 = (np.clip(seed_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil = Image.fromarray(arr_u8, "RGB")

        threshold = float(self.params.get("confidence_threshold", 0.4))
        mask_threshold = 0.5
        area_floor = float(self.params["min_area_fraction"]) * H * W

        seeds: list[tuple[int, str, np.ndarray]] = []
        next_track_id = 1
        for concept in concepts:
            inputs = processor(
                images=pil, text=concept, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # target_sizes is (H, W) not (W, H) — the post-processor
            # asserts this ordering when upsampling the mask logits.
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(H, W)],
            )
            if not results:
                continue
            result = results[0]
            masks = result.get("masks")
            scores = result.get("scores")
            if masks is None or scores is None:
                continue
            n = int(masks.shape[0]) if hasattr(masks, "shape") else 0
            for i in range(n):
                m = masks[i]
                mask_np = (
                    m.float().cpu().numpy()
                    if hasattr(m, "float")
                    else np.asarray(m, dtype=np.float32)
                ).astype(np.float32)
                # Belt-and-braces area floor — the ranker will drop them
                # too, but skipping here saves a tracker session.
                if float(mask_np.sum()) < area_floor:
                    continue
                seeds.append((next_track_id, concept, mask_np))
                next_track_id += 1
        return seeds

    def _track_instance(
        self,
        frames: np.ndarray,
        seed_frame_idx: int,
        seed_mask: np.ndarray,
    ) -> np.ndarray:
        """Propagate `seed_mask` across all frames with SAM 3's tracker.

        Uses SAM 3's inference-session API: seed the mask at
        `seed_frame_idx` via `add_inputs_to_inference_session(input_masks=)`,
        then propagate forward + backward with
        `propagate_in_video_iterator`. Each yielded step carries the logits
        for its frame, which `post_process_masks` upsamples + binarizes.

        Input: `frames` is (N, H, W, 3) float32 in [0, 1]; `seed_frame_idx`
        is the local index within `frames`; `seed_mask` is (H, W) float32
        in [0, 1] at plate resolution. Output: (N, H, W) float32 stack of
        hard-ish masks.
        """
        import torch
        from PIL import Image

        self._load_model()
        assert self._trk_model is not None
        processor = self._trk_processor
        model = self._trk_model
        device = self._device
        dtype = self._trk_dtype

        N = int(frames.shape[0])
        H, W = int(frames.shape[1]), int(frames.shape[2])

        # Processor accepts a list of PIL images (or a 4D torch tensor);
        # PIL matches the detector path and dodges dtype surprises.
        frame_pils = [
            Image.fromarray(
                (np.clip(frames[i], 0.0, 1.0) * 255.0).astype(np.uint8),
                "RGB",
            )
            for i in range(N)
        ]

        session = processor.init_video_session(
            video=frame_pils,
            inference_device=device,
            dtype=dtype,
        )

        # SAM 3 wants a bool / {0,1} mask. Threshold at 0.5.
        seed_bool = (seed_mask > 0.5).astype(np.uint8)
        processor.add_inputs_to_inference_session(
            session,
            frame_idx=int(seed_frame_idx),
            obj_ids=1,               # single-object track; we repeat per instance
            input_masks=seed_bool,
            original_size=(H, W),
        )

        out = np.zeros((N, H, W), dtype=np.float32)

        def _consume(step_iter: Any) -> None:
            """Drain a propagate_in_video_iterator into `out`.

            Each `step` is a Sam3TrackerVideoSegmentationOutput. We try the
            two field names transformers has used historically
            (`frame_idx` / `frame_index`) and skip the step if neither is
            present — defensive against minor API drift between versions.
            """
            for step in step_iter:
                frame_idx = getattr(step, "frame_idx", None)
                if frame_idx is None:
                    frame_idx = getattr(step, "frame_index", None)
                pred_masks = getattr(step, "pred_masks", None)
                if frame_idx is None or pred_masks is None:
                    continue
                # Step yields pred_masks of shape (batch, obj_ids, h, w) — 4D.
                # post_process_masks expects a 5D input (list-of-batches
                # style: batch, frames_per_batch, obj_ids, h, w) and upsamples
                # the trailing 2D to original_sizes. Add the leading dim here.
                if pred_masks.ndim == 4:
                    pred_masks_5d = pred_masks.unsqueeze(0)
                else:
                    pred_masks_5d = pred_masks
                post = processor.post_process_masks(
                    pred_masks_5d,
                    original_sizes=[(H, W)],
                    mask_threshold=0.0,
                    binarize=True,
                )
                # post_process_masks returns list-per-batch; we have one batch.
                if not post:
                    continue
                m = post[0]
                if hasattr(m, "float"):
                    arr = m.float().cpu().numpy()
                else:
                    arr = np.asarray(m, dtype=np.float32)
                # Collapse leading non-spatial dims — post is either
                # (frames_in_batch, obj_ids, H, W) or (obj_ids, H, W).
                # We asked for obj_ids=[1], so the first slice along any
                # non-spatial dim is the single object.
                while arr.ndim > 2:
                    arr = arr[0]
                out[int(frame_idx)] = arr.astype(np.float32, copy=False)

        # Forward from seed to end; then backward from seed to start. The
        # seed frame itself is emitted by forward propagation.
        _consume(
            model.propagate_in_video_iterator(
                session, start_frame_idx=int(seed_frame_idx)
            )
        )
        _consume(
            model.propagate_in_video_iterator(
                session,
                start_frame_idx=int(seed_frame_idx),
                reverse=True,
            )
        )

        return out

    # ------------------------------------------------------------------
    # Per-window lifecycle — SAM 3 is VIDEO_CLIP-native, so `preprocess` /
    # `infer` / `postprocess` are convenience paths that delegate to the
    # shot-level path for a single window.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"SAM3MattePass preprocess expects (N, H, W, 3), got {frames.shape}"
            )
        self._load_model()
        return {"video": frames.astype(np.float32, copy=False)}

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        # Single-window call path is rarely used for VIDEO_CLIP; run_shot is
        # the canonical entry. Return an empty dict so per-frame iterators
        # don't crash.
        return {}

    # ------------------------------------------------------------------
    # Shot-level: detect → track → union → stats → rank
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
        )
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        self._plate_shape = (plate_h, plate_w)

        seed_local = _pick_seed_frame(n_frames, self.params["sample_frame"])
        seed_rgb = frames[seed_local]

        concepts = [str(c) for c in self.params["concepts"]]
        seeds = self._detect_seed(seed_rgb, concepts)

        # Track each seed across the clip; accumulate _DetectedInstance.
        self._instances = []
        for track_id, label, seed_mask in seeds:
            if seed_mask.shape != (plate_h, plate_w):
                raise ValueError(
                    f"Seed mask for track {track_id} has shape {seed_mask.shape}, "
                    f"expected plate shape {(plate_h, plate_w)}"
                )
            stack = self._track_instance(frames, seed_local, seed_mask)
            if stack.ndim != 3 or stack.shape[1:] != (plate_h, plate_w):
                raise ValueError(
                    f"Track stack for {track_id} has shape {stack.shape}, "
                    f"expected ({n_frames}, {plate_h}, {plate_w})"
                )
            masks = {first + k: stack[k].astype(np.float32, copy=False)
                     for k in range(stack.shape[0])}
            # Drop instances that fall below the area floor everywhere.
            area_floor = float(self.params["min_area_fraction"]) * plate_h * plate_w
            if all(m.sum() < area_floor for m in masks.values()):
                continue
            self._instances.append(_DetectedInstance(track_id, label, masks))

        # Union by concept into per-frame channels.
        per_frame: dict[int, dict[str, np.ndarray]] = {
            first + i: {} for i in range(n_frames)
        }
        concepts_found: set[str] = set()
        for concept in concepts:
            # OR across all instances of this concept, per frame.
            any_nonzero = False
            for f in range(first, last + 1):
                union = np.zeros((plate_h, plate_w), dtype=np.float32)
                for inst in self._instances:
                    if inst.label != concept:
                        continue
                    m = inst.masks.get(f)
                    if m is not None:
                        np.maximum(union, m, out=union)
                if union.any():
                    any_nonzero = True
                    per_frame[f][f"{MASK_PREFIX}{concept}"] = union
            if any_nonzero:
                concepts_found.add(concept)
        # If a concept produced no mask on any frame, emit nothing for it
        # (we don't want zero-valued `mask.sky` layers when there's no sky).
        self._concepts_found = sorted(concepts_found)

        # Build the light Instance objects and rank them.
        rank_weights = RankWeights(**self.params["ranking"])
        overrides = [
            HeroOverride(track_id=int(h["track_id"]), slot=h["slot"])
            for h in (self.params.get("heroes") or [])
            if "track_id" in h and "slot" in h
        ]
        ranked_input = [self._to_rank_instance(inst, n_frames, plate_h, plate_w, first)
                        for inst in self._instances]
        self._heroes = rank_and_assign(
            ranked_input,
            rank_weights,
            n_clip_frames=n_frames,
            max_heroes=int(self.params["max_heroes"]),
            overrides=overrides,
        )
        return per_frame

    def _to_rank_instance(
        self,
        inst: _DetectedInstance,
        n_frames: int,
        plate_h: int,
        plate_w: int,
        first_frame: int,
    ) -> Instance:
        """Reduce per-frame masks (+ optional flow) into rank-friendly scalars."""
        plate_area = float(plate_h * plate_w)
        plate_diag = math.hypot(plate_h, plate_w)

        # Area fraction: mean over frames present.
        area_fractions: list[float] = []
        centralities: list[float] = []
        motion_energies: list[float] = []
        for f, mask in inst.masks.items():
            s = float(mask.sum())
            if s <= 0.0:
                continue
            area_fractions.append(s / plate_area)
            # Centroid -> distance from center -> 1 - (d / half_diag) clipped.
            ys, xs = np.where(mask > 0.5)
            if ys.size == 0:
                ys, xs = np.where(mask > 0)
            if ys.size:
                cy = float(ys.mean())
                cx = float(xs.mean())
                dx = cx - plate_w / 2.0
                dy = cy - plate_h / 2.0
                d = math.hypot(dx, dy)
                centralities.append(max(0.0, 1.0 - d / (plate_diag / 2.0)))
            # Motion energy from forward_flow[f] if available.
            fwd = self._forward_flow.get(f)
            if fwd is not None and fwd.shape[-2:] == (plate_h, plate_w):
                mag = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2)
                masked = mag[mask > 0.5]
                if masked.size:
                    motion_energies.append(float(masked.mean()) / max(plate_diag, 1.0))

        return Instance(
            track_id=inst.track_id,
            label=inst.label,
            frames=sorted(f for f, m in inst.masks.items() if float(m.sum()) > 0.0),
            area_fraction=_mean_or_zero(area_fractions),
            centrality=_mean_or_zero(centralities),
            motion_energy=_mean_or_zero(motion_energies),
            user_priority=0.0,
        )

    # ------------------------------------------------------------------
    # Artifact emission
    # ------------------------------------------------------------------

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._instances and not self._heroes:
            return {}
        # sam3_hard_masks — stash as a single dict value under frame key 0
        # (the artifact is shot-level, not per-frame). Value is
        # {track_id: {"label": str, "stack": (T, H, W)}} — the shape
        # CorridorKey will consume (design §21.8).
        n_frames = 0
        plate_h, plate_w = self._plate_shape
        if self._instances:
            any_inst = self._instances[0]
            if any_inst.masks:
                n_frames = len(any_inst.masks)
        hard_masks: dict[int, dict[str, Any]] = {}
        for inst in self._instances:
            frames_sorted = sorted(inst.masks)
            if not frames_sorted:
                continue
            stack = np.stack([inst.masks[f] for f in frames_sorted], axis=0)
            hard_masks[inst.track_id] = {
                "label": inst.label,
                "frames": frames_sorted,
                "stack": stack.astype(np.float32, copy=False),
            }

        # sam3_instances — the ranked+slotted hero list, as serializable dicts
        # so downstream consumers don't have to import rank.py just to unpack.
        hero_dicts: list[dict[str, Any]] = []
        for h in self._heroes:
            hero_dicts.append(
                {
                    "track_id": h.track_id,
                    "slot": h.slot,
                    "label": h.label,
                    "score": h.score,
                    # Reference the frames where the instance is present so
                    # the refiner can skip empty frames cheaply.
                    "frames": list(h.instance.frames),
                }
            )

        # Pick any frame key (artifact is shot-level).
        any_frame = 0
        for inst in self._instances:
            if inst.masks:
                any_frame = next(iter(inst.masks))
                break

        return {
            "sam3_hard_masks": {any_frame: hard_masks},
            "sam3_instances": {any_frame: hero_dicts},
            "matte_concepts": {any_frame: list(self._concepts_found)},
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _pick_seed_frame(n_frames: int, sample_frame: Any) -> int:
    """Resolve the `sample_frame` param to a local index in [0, n_frames)."""
    if isinstance(sample_frame, int):
        return max(0, min(n_frames - 1, sample_frame))
    s = str(sample_frame).lower()
    if s == "first":
        return 0
    if s == "last":
        return n_frames - 1
    # "middle" and any unknown value default to middle.
    return n_frames // 2


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


__all__ = ["SAM3MattePass"]
