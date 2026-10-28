# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

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
- `sam3_hard_masks`: `dict[int, {"label": str, "stack": (T, H, W) uint8}]`
  keyed by `track_id`. Raw per-instance hard masks (0/1) across the clip;
  uint8, not float32 — mattes are binary, and holding every instance's
  whole-clip stack at once made float32 the dominant host-RAM cost.
  Consumers cast to float32 one instance at a time. This
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

import logging
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
from live_action_aov.core.runtime_env import load_local_first
from live_action_aov.io.channels import MASK_PREFIX
from live_action_aov.passes.matte.rank import (
    HeroOverride,
    Instance,
    RankWeights,
    rank_and_assign,
)

_log = logging.getLogger(__name__)


def _available_ram_bytes() -> int | None:
    """Free host RAM in bytes, or ``None`` if it can't be measured.

    Uses ``psutil`` when present (a transitive dep already in the env). The
    pre-flight check degrades to estimate-only logging if it's missing, so
    this is never a hard dependency.
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _mask_ram_message(n_inst: int, n_frames: int, h: int, w: int, peak_bytes: int) -> str:
    """Actionable OOM guidance — what's needed, and the levers to fit it."""
    need_gib = peak_bytes / 2**30
    avail = _available_ram_bytes()
    avail_str = f"{avail / 2**30:.1f} GiB free" if avail is not None else "free RAM unknown"
    return (
        f"SAM3 matte needs ~{need_gib:.1f} GiB of host RAM to hold {n_inst} "
        f"instance(s) of {w}x{h} masks over {n_frames} frames ({avail_str}). "
        "Reduce the number of detected instances (tighten the concepts list / "
        "raise min_area_fraction / set max_heroes), run the matte at a lower "
        "proxy resolution, or process the shot in a shorter frame range."
    )


def _warn_unmatched_concepts(concepts: list[str], found_labels: set[str]) -> None:
    """Loud warning for every USER-typed concept that seeded no track.

    SAM 3 only finds what it's prompted for, so a typo ('peron') or an
    off-frame subject silently produces no mask.<name> channel at all —
    the delivery just misses a matte with zero errors. Field case
    2026-08-12: 'peron, rifle, car, greenscreen' shipped without the
    person matte."""
    for concept in concepts:
        if concept not in found_labels:
            _log.warning(
                "SAM3 matte: concept '%s' matched 0 objects on the seed frame - "
                "no track and no mask.%s channel will be produced. Check the "
                "spelling in 'SAM 3 detects', or the subject may not be visible "
                "on the seed frame.",
                concept,
                concept,
            )


def _wrap_if_gated_repo(repo: str, exc: BaseException) -> RuntimeError | None:
    """Translate a Hugging Face gated-repo / 401 error into actionable guidance.

    Returns a ``RuntimeError`` with setup steps if ``exc`` is recognisable
    as an HF gated-repo failure, or ``None`` if it's some other exception
    (network blip, bad model_id, torch issue) that should propagate unchanged.

    Why both a typed check and a string match: ``transformers.from_pretrained``
    catches ``huggingface_hub.errors.GatedRepoError`` and re-raises it as
    ``OSError``, so the typed isinstance check rarely fires through the
    transformers path. The substring fallback is what handles real-world
    cases. The typed check still helps for direct ``huggingface_hub`` calls.
    """
    is_gated = False
    try:
        from huggingface_hub.errors import GatedRepoError

        if isinstance(exc, GatedRepoError):
            is_gated = True
    except ImportError:
        pass
    if not is_gated:
        msg = str(exc).lower()
        if (
            "gated repo" in msg
            or "401 client error" in msg
            or ("access" in msg and "restricted" in msg)
        ):
            is_gated = True
    if not is_gated:
        return None
    return RuntimeError(
        f"Hugging Face model '{repo}' is gated — you need to request access "
        "and authenticate before it can download.\n\n"
        f"  1. Request access at https://huggingface.co/{repo}\n"
        "  2. Create a token at https://huggingface.co/settings/tokens "
        "('Read' scope is sufficient).\n"
        "  3. Authenticate locally:\n"
        "         uv run hf auth login\n"
        "     (or `uv run huggingface-cli login` on older huggingface_hub).\n"
        "     Paste your token when prompted.\n\n"
        "Full instructions: docs/install.md → "
        "'Hugging Face authentication for gated models'."
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
    masks: dict[int, np.ndarray]  # frame_idx -> (H, W) float32 in [0, 1]


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
        "min_area_fraction": 0.005,  # drop instances smaller than 0.5% of plate
        "sample_frame": "middle",  # "first" | "middle" | "last" | int
        # Re-run the detector every N frames and start a track for anything
        # not already covered (objects entering after the seed frame, or lost
        # by the tracker). None/0 = seed-and-track only (pre-0.9 behaviour).
        "redetect_stride": 12,
        "redetect_new_coverage": 0.5,  # det. less covered than this = new object
        "max_redetect_tracks": 8,
        # A new detection overlapping a same-label track by at least this
        # much is a lost PART of it (re-attached), not a new object.
        "redetect_merge_coverage": 0.1,
        # Track all objects that share a seed frame in ONE tracker session
        # (video encoded once, not once per object) and give each pixel to
        # ONE object: the per-object sessions let masks overlap freely
        # (person vs greenscreen at the hair, car vs a person in its window),
        # which is where the refiners' edges broke.
        "joint_tracking": True,
        "exclusive_masks": True,
        # Inside a joint session, user clicks out-rank detector objects on
        # contested pixels (added to the click objects' mask logits).
        "click_priority_bias": 4.0,
        # Two tracks this similar (clip IoU) are the same object tracked
        # twice (e.g. a click on a person the detector also found): keep one.
        "duplicate_iou": 0.85,
        # SAM leaves a thin unowned line (~3-8 px at 2K) where two objects
        # touch: neither logit is positive on the contact pixels. Pixels in a
        # gap this narrow, between two DIFFERENT objects, go to the nearest
        # one (closes gaps up to this wide). "auto" = 0.4% of the long edge
        # (4 px at 1024, 8 px at 2K); 0 = off.
        "seam_close_px": "auto",
        # Ranking weights (RankWeights schema).
        "ranking": {
            "area": 0.4,
            "centrality": 0.2,
            "motion": 0.2,
            "duration": 0.2,
            "user_priority": 0.0,
        },
        "max_heroes": 4,
        "heroes": [],  # user overrides: [{"track_id": 17, "slot": "r"}]
        # Interactive click-to-mask prompts (GUI viewport, v0.3). Each entry:
        # {"name": str, "seed_frame": int (plate numbering), "points":
        # [[x, y, label], ...] (label 1=fg, 0=bg), "box": [x1, y1, x2, y2] |
        # None, "ref_size": [w, h] (resolution the clicks were captured at —
        # we rescale to the frames the pass actually sees, so proxy mode
        # doesn't shift the clicks)}. ADDITIVE to text concepts: empty list =
        # concept detection only.
        "prompt_instances": [],
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        # Whether the concepts list was TYPED by the user (GUI field / YAML)
        # rather than our built-in defaults — user concepts that match
        # nothing get a loud warning (typos: 'peron'); silent defaults
        # matching nothing is normal and stays quiet.
        self._user_concepts = bool((params or {}).get("concepts"))
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._model: Any = None
        self._device: Any = None
        self._dtype: Any = None
        # Populated by `run_shot`; consumed by `emit_artifacts`.
        self._instances: list[_DetectedInstance] = []
        self._priority: dict[int, int] = {}  # track_id -> 1 = user click, 0 = detector
        # track_id -> depth rank, 0 = front. Published in sam3_hard_masks so
        # the matte layering post knows whose edge defines a shared seam.
        self._layer: dict[int, int] = {}
        self._heroes: list[Any] = []  # list[rank.HeroSlot]
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
    requires_artifacts: list[str] = []  # soft dep; no DAG enforcement

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
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

        # SAM 3 (and other Meta releases) are gated on the HF Hub. The raw
        # `OSError: You are trying to access a gated repo.` PyTorch surfaces
        # is unhelpful — users don't know they need to (a) request access
        # and (b) `hf auth login` locally. Wrap the four `from_pretrained`
        # calls and translate gated-repo / 401 errors into actionable
        # guidance pointing at `docs/install.md`.
        try:
            self._det_processor = load_local_first(Sam3Processor.from_pretrained, repo)
            det_model = load_local_first(Sam3Model.from_pretrained, repo)
            det_model.to(self._device).eval()
            self._det_model = det_model

            self._trk_processor = load_local_first(Sam3TrackerVideoProcessor.from_pretrained, repo)
            trk_dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
            trk_model = load_local_first(
                Sam3TrackerVideoModel.from_pretrained, repo, dtype=trk_dtype
            )
            trk_model.to(self._device).eval()
            self._trk_model = trk_model
            self._trk_dtype = trk_dtype
        except Exception as exc:
            wrapped = _wrap_if_gated_repo(repo, exc)
            if wrapped is not None:
                raise wrapped from exc
            raise

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
            inputs = processor(images=pil, text=concept, return_tensors="pt").to(device)
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
        seed_mask: np.ndarray | None,
        *,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        box: list[float] | None = None,
    ) -> np.ndarray:
        """Propagate one object across all frames with SAM 3's tracker.

        Uses SAM 3's inference-session API: seed the object at
        `seed_frame_idx` via `add_inputs_to_inference_session` — either from
        a detector mask (`input_masks=`, the text-concept path) or from
        interactive click prompts (`input_points`/`input_labels`/
        `input_boxes`, the click-to-mask path) — then propagate forward +
        backward with `propagate_in_video_iterator`. Each yielded step
        carries the logits for its frame, which `post_process_masks`
        upsamples + binarizes.

        Input: `frames` is (N, H, W, 3) float32 in [0, 1]; `seed_frame_idx`
        is the local index within `frames`; `seed_mask` is (H, W) float32
        in [0, 1] at plate resolution, or None when seeding from prompts.
        `points` is [[x, y], ...] and `labels` [1|0, ...] (1=fg, 0=bg);
        `box` is [x1, y1, x2, y2] — all in pixels of `frames`. Output:
        (N, H, W) float32 stack of hard-ish masks.
        """
        if seed_mask is None and not points and box is None:
            raise ValueError("_track_instance needs a seed_mask, points, or a box")
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

        if seed_mask is not None:
            # SAM 3 wants a bool / {0,1} mask. Threshold at 0.5.
            seed_bool = (seed_mask > 0.5).astype(np.uint8)
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=int(seed_frame_idx),
                obj_ids=1,  # single-object track; we repeat per instance
                input_masks=seed_bool,
                original_size=(H, W),
            )
        else:
            # Click-to-mask path: seed from interactive prompts. Nesting per
            # the modeling docstrings — points (batch, point_batch, n, 2),
            # labels (batch, point_batch, n), boxes (batch, point_batch, 4):
            # one batch, one object ⇒ wrap twice.
            prompt_kwargs: dict[str, Any] = {}
            if points:
                prompt_kwargs["input_points"] = [[points]]
                prompt_kwargs["input_labels"] = [[labels or [1] * len(points)]]
            if box is not None:
                prompt_kwargs["input_boxes"] = [[box]]
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=int(seed_frame_idx),
                obj_ids=1,
                original_size=(H, W),
                **prompt_kwargs,
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
        _consume(model.propagate_in_video_iterator(session, start_frame_idx=int(seed_frame_idx)))
        _consume(
            model.propagate_in_video_iterator(
                session,
                start_frame_idx=int(seed_frame_idx),
                reverse=True,
            )
        )

        return out

    def _track_objects(
        self, frames: np.ndarray, objs: list[dict[str, Any]], first: int, area_floor: float
    ) -> None:
        """Track `objs` (grouped by seed frame) and append the survivors to
        `self._instances`. Subclasses overriding `_track_instance` (test
        fakes, or joint_tracking=False) keep the old one-session-per-object
        path."""
        n_frames, plate_h, plate_w = (
            int(frames.shape[0]),
            int(frames.shape[1]),
            int(frames.shape[2]),
        )
        legacy = (type(self)._track_instance is not SAM3MattePass._track_instance) or not bool(
            self.params.get("joint_tracking", True)
        )
        groups: dict[int, list[dict[str, Any]]] = {}
        for o in objs:
            groups.setdefault(int(o["seed_local"]), []).append(o)
        for seed_l, group in groups.items():
            if legacy:
                stacks: dict[int, np.ndarray] = {}
                for o in group:
                    if o.get("mask") is not None:
                        stacks[o["obj_id"]] = self._track_instance(frames, seed_l, o["mask"])
                    else:
                        stacks[o["obj_id"]] = self._track_instance(
                            frames,
                            seed_l,
                            None,
                            points=o.get("points"),
                            labels=o.get("labels"),
                            box=o.get("box"),
                        )
            else:
                stacks = self._track_group(frames, group)
            for o in group:
                stack = stacks.get(o["obj_id"])
                if stack is None:
                    continue
                if stack.ndim != 3 or stack.shape[1:] != (plate_h, plate_w):
                    raise ValueError(
                        f"Track stack for {o['label']!r} ({o['obj_id']}) has shape {stack.shape}, "
                        f"expected ({n_frames}, {plate_h}, {plate_w})"
                    )
                # uint8 (0/1), not float32: the whole-clip stack of every
                # instance is held at once, so dtype scales peak host RAM.
                masks = {
                    first + k: (stack[k] > 0.5).astype(np.uint8) for k in range(stack.shape[0])
                }
                if all(m.sum() < area_floor for m in masks.values()):
                    continue
                self._instances.append(_DetectedInstance(int(o["obj_id"]), str(o["label"]), masks))
                self._priority[int(o["obj_id"])] = int(o.get("priority", 0))

    def _track_group(
        self, frames: np.ndarray, group: list[dict[str, Any]]
    ) -> dict[int, np.ndarray]:
        """Track every object of `group` (same seed frame) in ONE tracker
        session and give each pixel to one object (highest mask logit; user
        clicks get `click_priority_bias`). Returns obj_id -> (N, H, W) uint8."""
        import torch
        from PIL import Image

        self._load_model()
        assert self._trk_model is not None
        processor, model = self._trk_processor, self._trk_model
        N, H, W = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        seed = int(group[0]["seed_local"])
        frame_pils = [
            Image.fromarray((np.clip(frames[i], 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
            for i in range(N)
        ]
        session = processor.init_video_session(
            video=frame_pils, inference_device=self._device, dtype=self._trk_dtype
        )
        mask_objs = [o for o in group if o.get("mask") is not None]
        if mask_objs:
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=seed,
                obj_ids=[int(o["obj_id"]) for o in mask_objs],
                input_masks=[(o["mask"] > 0.5).astype(np.uint8) for o in mask_objs],
                original_size=(H, W),
            )
        for o in group:
            if o.get("mask") is not None:
                continue
            kw: dict[str, Any] = {}
            if o.get("points"):
                kw["input_points"] = [[o["points"]]]
                kw["input_labels"] = [[o.get("labels") or [1] * len(o["points"])]]
            if o.get("box") is not None:
                kw["input_boxes"] = [[o["box"]]]
            processor.add_inputs_to_inference_session(
                session, frame_idx=seed, obj_ids=int(o["obj_id"]), original_size=(H, W), **kw
            )
        # Each add call REPLACES the session's "objects with new inputs"
        # list; without this only the last-added object would be conditioned
        # on the seed frame and the others would track from nothing.
        session.obj_with_new_inputs = [int(o["obj_id"]) for o in group]

        out = {int(o["obj_id"]): np.zeros((N, H, W), dtype=np.uint8) for o in group}
        bias_of = {
            int(o["obj_id"]): float(self.params.get("click_priority_bias", 4.0))
            if o.get("priority", 0)
            else 0.0
            for o in group
        }
        exclusive = bool(self.params.get("exclusive_masks", True))

        def _consume(step_iter: Any) -> None:
            for step in step_iter:
                fi = getattr(step, "frame_idx", None)
                if fi is None:
                    fi = getattr(step, "frame_index", None)
                pm = getattr(step, "pred_masks", None)
                ids = list(getattr(step, "object_ids", None) or [])
                if fi is None or pm is None or not ids:
                    continue
                post = processor.post_process_masks(
                    pm.unsqueeze(0) if pm.ndim == 4 else pm,
                    original_sizes=[(H, W)],
                    binarize=False,
                )
                if not post:
                    continue
                logits = post[0]
                if not hasattr(logits, "reshape"):
                    logits = torch.as_tensor(np.asarray(logits))
                logits = logits.reshape(len(ids), H, W).float()
                present = logits > 0.0
                if exclusive and len(ids) > 1:
                    bias = torch.tensor(
                        [bias_of.get(int(i), 0.0) for i in ids], device=logits.device
                    )
                    winner = torch.argmax(logits + bias[:, None, None], dim=0)
                    for k in range(len(ids)):
                        present[k] &= winner == k
                pres = present.cpu().numpy()
                for k, oid in enumerate(ids):
                    if int(oid) in out:
                        out[int(oid)][int(fi)] = pres[k]

        _consume(model.propagate_in_video_iterator(session, start_frame_idx=seed))
        _consume(model.propagate_in_video_iterator(session, start_frame_idx=seed, reverse=True))
        return out

    def _redetect(
        self,
        frames: np.ndarray,
        concepts: list[str],
        stride: int,
        seed_local: int,
        first: int,
        area_floor: float,
    ) -> None:
        """Run the detector every `stride` frames; anything mostly NOT
        covered by an existing track becomes a new track seeded there."""
        n = int(frames.shape[0])
        budget = int(self.params.get("max_redetect_tracks", 8))
        thr = float(self.params.get("redetect_new_coverage", 0.5))
        next_id = 2001  # own id block: concepts 1.., clicks 1001..
        import cv2

        r = max(1, round(0.015 * max(frames.shape[1], frames.shape[2])))
        touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        for k in range(0, n, max(1, stride)):
            if budget <= 0:
                break
            if k == seed_local:
                continue
            f = first + k
            covered = np.zeros(frames.shape[1:3], dtype=bool)
            for inst in self._instances:
                m = inst.masks.get(f)
                if m is not None:
                    covered |= m > 0
            new: list[dict[str, Any]] = []
            for _tid, label, m in self._detect_seed(frames[k], concepts):
                mb = m > 0.5
                area = float(mb.sum())
                if area < area_floor:
                    continue
                if float((mb & covered).sum()) / area >= thr:
                    continue
                # Partly covered by a track of the SAME label = a piece the
                # tracker lost (PF0048: the arm holding the rifle dropped out
                # of the woman's track). Re-attach it to that track instead
                # of starting a second "person".
                # "Touching" = overlap with the track grown by ~1.5% of the
                # long edge: a limb detected alone is adjacent, not inside.
                merge_into = None
                best = float(self.params.get("redetect_merge_coverage", 0.1))
                for inst in self._instances:
                    im = inst.masks.get(f)
                    if inst.label != label or im is None or not im.any():
                        continue
                    grown = cv2.dilate(im.astype(np.uint8), touch_kernel) > 0
                    cov = float((mb & grown).sum()) / area
                    if cov >= best:
                        best, merge_into = cov, inst.track_id
                new.append(
                    {
                        "obj_id": next_id,
                        "label": label,
                        "seed_local": k,
                        "mask": m,
                        "priority": 0,
                        "merge_into": merge_into,
                    }
                )
                next_id += 1
                covered |= mb  # one new track per thing on this frame
                if len(new) >= budget:
                    break
            if not new:
                continue
            budget -= len(new)
            before = {i.track_id for i in self._instances}
            self._track_objects(frames, new, first, area_floor)
            by_id = {i.track_id: i for i in self._instances}
            started, extended = [], []
            for o in new:
                got = by_id.get(int(o["obj_id"]))
                if got is None or got.track_id in before:
                    continue
                host = by_id.get(o["merge_into"]) if o["merge_into"] is not None else None
                if host is None:
                    started.append(o["label"])
                    continue
                for fr, mm in got.masks.items():
                    hm = host.masks.get(fr)
                    host.masks[fr] = mm if hm is None else np.maximum(hm, mm)
                self._instances.remove(got)
                self._priority.pop(got.track_id, None)
                extended.append(f"{o['label']} (track {host.track_id})")
            if started or extended:
                _log.info(
                    "SAM3 matte: re-detection on frame %d - new: %s; re-attached lost parts to: %s",
                    f,
                    ", ".join(started) or "none",
                    ", ".join(extended) or "none",
                )

    # ------------------------------------------------------------------
    # Per-window lifecycle — SAM 3 is VIDEO_CLIP-native, so `preprocess` /
    # `infer` / `postprocess` are convenience paths that delegate to the
    # shot-level path for a single window.
    # ------------------------------------------------------------------

    def preprocess(self, frames: np.ndarray) -> Any:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"SAM3MattePass preprocess expects (N, H, W, 3), got {frames.shape}")
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
        frames = np.stack([reader.read_frame(f)[0] for f in range(first, last + 1)], axis=0)
        plate_h, plate_w = int(frames.shape[1]), int(frames.shape[2])
        self._plate_shape = (plate_h, plate_w)

        seed_local = _pick_seed_frame(n_frames, self.params["sample_frame"])
        seed_rgb = frames[seed_local]

        concepts = [str(c) for c in self.params["concepts"]]
        seeds = self._detect_seed(seed_rgb, concepts)

        # Pre-flight host-RAM check (fail fast, before the minutes-long track).
        # Peak host RAM is dominated by holding every instance's whole-clip mask
        # stack at once (here in _instances, then again in emit_artifacts) — so
        # ~2x the uint8 footprint. We know the instance count now, right after
        # detection, so abort at second 0 with a clear reason instead of dying
        # on a raw MemoryError after the GPU has churned for minutes.
        n_clicks = sum(
            1
            for s in (self.params.get("prompt_instances") or [])
            if (s.get("points") or s.get("box"))
        )
        n_inst_est = len(seeds) + n_clicks
        per_inst_bytes = n_frames * plate_h * plate_w  # uint8 = 1 byte/px
        peak_bytes = 2 * n_inst_est * per_inst_bytes
        if n_inst_est:
            _log.info(
                "SAM3 matte: %d instance(s), %dx%d, %d frames -> ~%.1f GiB peak host RAM",
                n_inst_est,
                plate_w,
                plate_h,
                n_frames,
                peak_bytes / 2**30,
            )
        avail = _available_ram_bytes()
        if avail is not None and peak_bytes > 0.9 * avail:
            raise MemoryError(_mask_ram_message(n_inst_est, n_frames, plate_h, plate_w, peak_bytes))

        # --- Everything to track: concept seeds + user clicks -------------
        area_floor = float(self.params["min_area_fraction"]) * plate_h * plate_w
        objs: list[dict[str, Any]] = []
        for track_id, label, seed_mask in seeds:
            if seed_mask.shape != (plate_h, plate_w):
                raise ValueError(
                    f"Seed mask for track {track_id} has shape {seed_mask.shape}, "
                    f"expected plate shape {(plate_h, plate_w)}"
                )
            objs.append(
                {
                    "obj_id": track_id,
                    "label": label,
                    "seed_local": seed_local,
                    "mask": seed_mask,
                    "priority": 0,
                }
            )

        # Interactive click prompts (click-to-mask). Coordinates are rescaled
        # from the resolution they were captured at (`ref_size`) to the frames
        # this pass sees, so proxy mode doesn't shift the clicks. Track ids
        # live in a 1000+ block so they never collide with detector ids.
        next_click_id = 1001
        for idx, spec in enumerate(self.params.get("prompt_instances") or []):
            name = str(spec.get("name") or f"click_{idx + 1}")
            seed_plate = int(spec.get("seed_frame", first))
            seed_local_i = min(max(seed_plate - first, 0), n_frames - 1)
            ref = spec.get("ref_size") or None
            sx = plate_w / float(ref[0]) if ref and float(ref[0]) > 0 else 1.0
            sy = plate_h / float(ref[1]) if ref and float(ref[1]) > 0 else 1.0
            raw_pts = spec.get("points") or []
            pts = [[float(p[0]) * sx, float(p[1]) * sy] for p in raw_pts]
            lbls = [int(p[2]) for p in raw_pts]
            raw_box = spec.get("box")
            box = (
                [
                    float(raw_box[0]) * sx,
                    float(raw_box[1]) * sy,
                    float(raw_box[2]) * sx,
                    float(raw_box[3]) * sy,
                ]
                if raw_box
                else None
            )
            if not pts and box is None:
                continue
            objs.append(
                {
                    "obj_id": next_click_id,
                    "label": name,
                    "seed_local": seed_local_i,
                    "points": pts or None,
                    "labels": lbls or None,
                    "box": box,
                    "priority": 1,
                }
            )
            next_click_id += 1

        self._instances = []
        self._priority = {}
        self._track_objects(frames, objs, first, area_floor)

        # --- Re-detection: objects that enter later / the tracker lost ---
        stride = self.params.get("redetect_stride")
        if stride and concepts:
            self._redetect(frames, concepts, int(stride), seed_local, first, area_floor)
        if self._user_concepts:
            _warn_unmatched_concepts(concepts, {i.label for i in self._instances})

        # --- One owner per pixel -------------------------------------------
        if bool(self.params.get("exclusive_masks", True)) and len(self._instances) > 1:
            self._instances = _dedupe_instances(
                self._instances, self._priority, float(self.params.get("duplicate_iou", 0.85))
            )
            _make_exclusive(self._instances, self._priority)
            gap = self.params.get("seam_close_px", "auto")
            gap_px = (
                max(2, round(0.004 * max(plate_h, plate_w))) if str(gap) == "auto" else int(gap)
            )
            if gap_px > 0:
                _close_seams(self._instances, gap_px)
            for inst in [i for i in self._instances if not any(m.any() for m in i.masks.values())]:
                _log.info(
                    "SAM3 matte: '%s' (track %d) lies entirely under higher-priority objects - dropped.",
                    inst.label,
                    inst.track_id,
                )
            self._instances = [i for i in self._instances if any(m.any() for m in i.masks.values())]
        self._layer = {
            inst.track_id: rank
            for rank, inst in enumerate(_layer_order(self._instances, self._priority))
        }
        click_labels = list(
            dict.fromkeys(i.label for i in self._instances if self._priority.get(i.track_id) == 1)
        )

        # Union by label into per-frame channels: text concepts first, then
        # click names (deduped — a click named like a concept unions with it).
        per_frame: dict[int, dict[str, np.ndarray]] = {first + i: {} for i in range(n_frames)}
        concepts_found: set[str] = set()
        for concept in dict.fromkeys([*concepts, *click_labels]):
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
        ranked_input = [
            self._to_rank_instance(inst, n_frames, plate_h, plate_w, first)
            for inst in self._instances
        ]
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
        plate_h, plate_w = self._plate_shape
        hard_masks: dict[int, dict[str, Any]] = {}
        try:
            for inst in self._instances:
                frames_sorted = sorted(inst.masks)
                if not frames_sorted:
                    continue
                # Keep the stack uint8 (0/1) — consumers cast per-instance. See
                # the storage note in run_shot; this is the second of the two
                # whole-clip materializations, and the one that peaked ~70 GB in
                # float32.
                stack = np.stack([inst.masks[f] for f in frames_sorted], axis=0)
                hard_masks[inst.track_id] = {
                    "label": inst.label,
                    "frames": frames_sorted,
                    "stack": stack,
                    # Depth rank (0 = front): who owns a shared seam.
                    "layer": self._layer.get(inst.track_id, len(hard_masks)),
                }
        except MemoryError as exc:
            # The pre-flight in run_shot should catch this first, but if the
            # estimate was optimistic, surface the same actionable guidance
            # instead of a raw traceback.
            n = len(self._instances)
            n_frames = max((len(i.masks) for i in self._instances), default=0)
            peak = 2 * n * n_frames * plate_h * plate_w
            raise MemoryError(_mask_ram_message(n, n_frames, plate_h, plate_w, peak)) from exc

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


def _clip_iou(a: _DetectedInstance, b: _DetectedInstance, step: int = 4) -> float:
    inter = union = 0.0
    for f in sorted(set(a.masks) | set(b.masks))[::step]:
        ma, mb = a.masks.get(f), b.masks.get(f)
        if ma is None or mb is None:
            only = ma if ma is not None else mb
            union += float(only.sum()) if only is not None else 0.0
            continue
        x, y = ma > 0, mb > 0
        inter += float((x & y).sum())
        union += float((x | y).sum())
    return inter / union if union > 0 else 0.0


def _dedupe_instances(
    instances: list[_DetectedInstance], priority: dict[int, int], iou_thr: float
) -> list[_DetectedInstance]:
    """Drop tracks that are the same object tracked twice (clip IoU above
    `iou_thr`): the user's click wins over the detector, else the first."""
    keep: list[_DetectedInstance] = []
    for inst in sorted(instances, key=lambda i: (-priority.get(i.track_id, 0), i.track_id)):
        dup = next((k for k in keep if _clip_iou(inst, k) > iou_thr), None)
        if dup is not None:
            _log.info(
                "SAM3 matte: '%s' (track %d) duplicates '%s' (track %d) - kept one.",
                inst.label,
                inst.track_id,
                dup.label,
                dup.track_id,
            )
            continue
        keep.append(inst)
    order = {i.track_id: n for n, i in enumerate(instances)}
    return sorted(keep, key=lambda i: order[i.track_id])


def _layer_order(
    instances: list[_DetectedInstance], priority: dict[int, int]
) -> list[_DetectedInstance]:
    """Front-to-back order: user clicks first, then the smaller object
    (clip-mean area: what is held or in front - a rifle over the person, a
    person in front of a car), then the lower track id. Clip-mean (not
    per-frame) area keeps the order stable over time, so seams don't flicker."""

    def mean_area(i: _DetectedInstance) -> float:
        areas = [float(m.sum()) for m in i.masks.values() if m.any()]
        return sum(areas) / len(areas) if areas else 0.0

    return sorted(instances, key=lambda i: (-priority.get(i.track_id, 0), mean_area(i), i.track_id))


def _close_seams(instances: list[_DetectedInstance], gap_px: int) -> None:
    """Fill the thin unowned line SAM leaves where two objects touch, in
    place. A free pixel within `gap_px` of at least two DIFFERENT objects
    goes to the nearest one. Concavities of a single object (the gap
    between an arm and the body) are left alone - only contact lines."""
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gap_px + 1, 2 * gap_px + 1))
    frames = sorted({f for i in instances for f in i.masks})
    for f in frames:
        present = [i for i in instances if f in i.masks and i.masks[f].any()]
        if len(present) < 2:
            continue
        bins = [(i.masks[f] > 0).astype(np.uint8) for i in present]
        taken = np.zeros_like(bins[0], dtype=bool)
        near = np.zeros(bins[0].shape, dtype=np.uint8)
        for b in bins:
            taken |= b > 0
            near += (cv2.dilate(b, k) > 0).astype(np.uint8)
        gap = (near >= 2) & ~taken
        if not gap.any():
            continue
        dist = np.stack([cv2.distanceTransform(1 - b, cv2.DIST_L2, 3) for b in bins], axis=0)
        owner = np.argmin(dist, axis=0)
        ys, xs = np.nonzero(gap)
        for j, inst in enumerate(present):
            sel = owner[ys, xs] == j
            if sel.any():
                m = (inst.masks[f] > 0).astype(np.uint8)
                m[ys[sel], xs[sel]] = 1
                inst.masks[f] = m


def _make_exclusive(instances: list[_DetectedInstance], priority: dict[int, int]) -> None:
    """Give every contested pixel to ONE instance, in place, in
    `_layer_order` (the front object owns the overlap)."""
    order = _layer_order(instances, priority)
    frames = sorted({f for i in instances for f in i.masks})
    for f in frames:
        present = [i for i in order if f in i.masks and i.masks[f].any()]
        if len(present) < 2:
            continue
        taken = np.zeros(present[0].masks[f].shape, dtype=bool)
        for inst in present:
            m = inst.masks[f] > 0
            inst.masks[f] = (m & ~taken).astype(np.uint8)
            taken |= m


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
