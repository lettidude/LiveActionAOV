# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

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
from collections.abc import Callable
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
from live_action_aov.io.channels import CH_Z_RAW, mask_channel
from live_action_aov.passes.matte.boxmask import (
    BoxMask,
    LazyStack,
    MaskDict,
    boxes_touch,
    inter_area,
    paint_max,
    union_box,
)
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
    # frame_idx -> (H, W) 0/1 mask, stored as its bounding-box crop
    # (BoxMask): a crowd of tiny people at 4K would otherwise hold ~8 MB per
    # person per frame. Plain arrays assigned in are compressed.
    masks: dict[int, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.masks, MaskDict):
            self.masks = MaskDict(self.masks)


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
        # Small subjects (a crowd on a beach). SAM 3 sees ~1008 px, so a 4K
        # plate reaches it 4x smaller and people under ~100 px vanish. "auto"
        # ALSO runs the detector on 2x2 overlapping tiles of plates 1800 px
        # and up, "max" on 3x3 (each tile reaches the model zoomed in); "off"
        # = full frame only. Tiles on, the size floor is `min_object_px` (box
        # height in plate px) instead of `min_area_fraction`, which dropped
        # anyone shorter than ~1/5 of the frame (ALEXA 65 beach plate: 0 of
        # ~180 people; 3x3 tiles find ~90% of those 12-96 px tall).
        "small_objects": "auto",
        "tile_overlap": 0.2,
        "min_object_px": 10,
        # Re-detection budget when tiles are on (a crowd keeps arriving).
        "small_redetect_tracks": 64,
        # Objects post-processed together in a joint tracker session (bounds
        # GPU memory with crowds).
        "tracker_chunk": 24,
        # At most this many objects per tracker session: the tracker attends
        # over every object of a session at once, so VRAM grows with the
        # count (126 people, 100 frames at a 1K proxy: 17.4 GB in one
        # session). Bigger groups are split by position (neighbours stay
        # together and still compete for pixels).
        "tracker_max_objects": 32,
        # Sessions with more objects than this keep the tracker's memory bank
        # in host RAM (same speed; 126 people at a 1K proxy: 13 GB VRAM peak
        # instead of 30).
        "tracker_state_cpu_above": 8,
        # Detection masks are upsampled at most to this long edge, then
        # scaled per box (VRAM: a 4K frame's candidates at once took ~9 GB).
        "detect_post_px": 2048,
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
        # ...but only a PIECE: at most this fraction of the track's own area.
        # A whole new person standing next to another (a crowd) is not a
        # lost arm of them (beach plate: dozens of people merged in pairs).
        "redetect_merge_max_ratio": 0.6,
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
        # Front-to-back order (who owns a contested pixel, who is held out in
        # the layering post): "depth" = Depth Anything V2 Small (Apache-2.0)
        # on a few frames, nearer first; objects at about the same depth (a
        # rifle in the hands) fall back to "area" = the smaller object in
        # front. Area alone put thin background shelves IN FRONT of the
        # person behind whom they stand (field report 2026-10-28).
        "layer_order": "depth",
        "depth_order_frames": 5,
        # Parts of an object ("hair", "jacket" inside "person"): a track with a
        # part-like label (`part_labels`) whose mask lies at least this much
        # inside a bigger object of another label is that object's PART. The
        # whole keeps its full mask (mask.person = the full body), parts share
        # its pixels only among themselves, and each part stays inside the
        # whole. 0 = off (everything exclusive).
        "part_containment": 0.7,
        # "auto" = clothing / body / vehicle-part words (see _PART_WORDS), or
        # a list of labels. A person inside a car is NOT a part of the car.
        "part_labels": "auto",
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
        self._nearness: dict[int, float] = {}
        # part track_id -> whole track_id ("hair" -> "person").
        self._parent: dict[int, int] = {}
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

    def _tile_grid(self, h: int, w: int) -> int:
        """Detector tiles per side for this frame size (1 = full frame only)."""
        mode = str(self.params.get("small_objects", "auto")).lower()
        if mode == "max":
            return 3
        if mode == "auto":
            # Tiles reach the model zoomed in only when the frame is bigger
            # than its ~1008 px input.
            return 2 if max(h, w) >= 1200 else 1
        return 1

    def _detect_frame(
        self, frames: np.ndarray, k: int, concepts: list[str]
    ) -> list[tuple[int, str, Any]]:
        """Detect on local frame `k`. In proxy mode the detector reads the
        ORIGINAL plate (same colour transform) and the masks are scaled to
        the proxy: SAM 3's tracker works at ~1008 px either way, but a 1K
        proxy hides people the 4K plate shows (beach plate, 24-48 px tall:
        6/31 found on the proxy, 31/31 on the plate)."""
        h, w = int(frames.shape[1]), int(frames.shape[2])
        reader = getattr(self, "_reader", None)
        full = getattr(reader, "read_frame_full_res", None)
        if full is not None and str(self.params.get("small_objects", "auto")).lower() != "off":
            img = np.asarray(full(self._first + k)[0], dtype=np.float32)
            if img.shape[:2] != (h, w):
                self._det_hw = (int(img.shape[0]), int(img.shape[1]))
                return [
                    (t, lb, BoxMask.from_dense(m).resized((h, w)))
                    for t, lb, m in self._detect_seed(img, concepts)
                ]
        return [
            (t, lb, BoxMask.from_dense(m)) for t, lb, m in self._detect_seed(frames[k], concepts)
        ]

    def _size_ok(self, m: BoxMask, h: int, w: int) -> bool:
        """The size floor: box height with tiles on, area fraction without.

        Decided by the frames DETECTION ran on (the original plate in proxy
        mode), with `min_object_px` scaled to this (h, w): the 1K-proxy
        beach tracked 131 people and then dropped every one of them on the
        area floor, because the proxy itself is too small for tiles."""
        dh, dw = getattr(self, "_det_hw", None) or (h, w)
        if self._tile_grid(dh, dw) > 1:
            px = max(2.0, float(self.params.get("min_object_px", 10)) * h / float(dh))
            return m.height >= px and m.sum() >= max(1, int(px * px) // 4)
        return m.sum() >= float(self.params["min_area_fraction"]) * h * w

    def _full_hw(self) -> tuple[int, int]:
        """Size of the original plate (the proxy's source), if known."""
        reader = getattr(self, "_reader", None)
        full = getattr(getattr(reader, "_base", None), "full_res_reader", None)
        if full is None:
            return self._plate_shape
        w, h = full().resolution()
        return int(h), int(w)

    def _zoom_on_plate(
        self,
        frames: np.ndarray,
        k: int,
        pts: list[list[float]],
        lbls: list[int],
        box: list[float] | None,
    ) -> BoxMask | None:
        """`_zoom_click_seed` on the original plate in proxy mode (clicks and
        mask scaled), on the frames themselves otherwise."""
        h, w = int(frames.shape[1]), int(frames.shape[2])
        full = getattr(getattr(self, "_reader", None), "read_frame_full_res", None)
        fh, fw = self._full_hw()
        if full is None or (fh, fw) == (h, w):
            return self._zoom_click_seed(frames[k], pts, lbls, box)
        sx, sy = fw / w, fh / h
        img = np.asarray(full(self._first + k)[0], dtype=np.float32)
        z = self._zoom_click_seed(
            img,
            [[p[0] * sx, p[1] * sy] for p in pts],
            lbls,
            None if box is None else [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
        )
        return None if z is None else z.resized((h, w))

    def _zoom_click_seed(
        self,
        frame: np.ndarray,
        points: list[list[float]],
        labels: list[int],
        box: list[float] | None,
    ) -> BoxMask | None:
        """Segment a clicked object inside a zoomed window around the click.

        None when there is nothing to gain (a big object) or the object runs
        out of the window: the click then seeds the tracker as usual."""
        H, W = int(frame.shape[0]), int(frame.shape[1])
        win = _click_window(points, labels, box, H, W)
        if win is None:
            return None
        import torch
        from PIL import Image

        y0, x0, y1, x1 = win
        h, w = y1 - y0, x1 - x0
        self._load_model()
        processor, model = self._trk_processor, self._trk_model
        pil = Image.fromarray(
            (np.clip(frame[y0:y1, x0:x1], 0.0, 1.0) * 255.0).astype(np.uint8), "RGB"
        )
        session = processor.init_video_session(
            video=[pil], inference_device=self._device, dtype=self._trk_dtype
        )
        kw: dict[str, Any] = {}
        if points:
            kw["input_points"] = [[[[p[0] - x0, p[1] - y0] for p in points]]]
            kw["input_labels"] = [[labels or [1] * len(points)]]
        if box is not None:
            kw["input_boxes"] = [[[box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]]]
        processor.add_inputs_to_inference_session(
            session, frame_idx=0, obj_ids=1, original_size=(h, w), **kw
        )
        m = None
        with torch.no_grad():
            for step in model.propagate_in_video_iterator(session, start_frame_idx=0):
                pm = step.pred_masks
                post = processor.post_process_masks(
                    pm.unsqueeze(0) if pm.ndim == 4 else pm,
                    original_sizes=[(h, w)],
                    binarize=False,
                )
                m = (post[0].reshape(-1, h, w)[0].float() > 0.0).cpu().numpy()
                break
        if m is None or not m.any():
            return None
        # Touching a window side that is not the frame edge: the object is
        # bigger than the window - let the full-frame click handle it.
        if (
            (y0 > 0 and m[0].any())
            or (y1 < H and m[-1].any())
            or (x0 > 0 and m[:, 0].any())
            or (x1 < W and m[:, -1].any())
        ):
            return None
        return BoxMask.from_crop((H, W), y0, x0, m.astype(np.uint8))

    def _detect_seed(
        self,
        seed_frame: np.ndarray,
        concepts: list[str],
    ) -> list[tuple[int, str, Any]]:
        """Run SAM 3 on one frame (and its tiles, see `small_objects`),
        return a list of (track_id, label, BoxMask).

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
        threshold = float(self.params.get("confidence_threshold", 0.4))
        mask_threshold = 0.5

        # Windows: the full frame, plus overlapping tiles for small subjects.
        g = self._tile_grid(H, W)
        windows = [(0, 0, H, W)]
        if g > 1:
            ov = float(self.params.get("tile_overlap", 0.2))
            th, tw = H / (g - (g - 1) * ov), W / (g - (g - 1) * ov)
            for iy in range(g):
                for ix in range(g):
                    y0, x0 = int(iy * th * (1 - ov)), int(ix * tw * (1 - ov))
                    windows.append((y0, x0, min(H, int(y0 + th)), min(W, int(x0 + tw))))

        def _run(concept: str, y0: int, x0: int, y1: int, x1: int) -> list[tuple[float, Any]]:
            # Processor expects PIL RGB uint8 (or a torch/numpy tensor the
            # image processor can rescale). PIL is the least-surprising path.
            crop = seed_frame[y0:y1, x0:x1]
            pil = Image.fromarray((np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8), "RGB")
            inputs = processor(images=pil, text=concept, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # target_sizes is (H, W) not (W, H) — the post-processor
            # asserts this ordering when upsampling the mask logits.
            # Capped at `detect_post_px`: SAM's masks come from ~288 px
            # logits, and upsampling every candidate of a 4K frame at once
            # took ~9 GB of VRAM; each kept mask is scaled up in its box.
            wh, ww = y1 - y0, x1 - x0
            cap = int(self.params.get("detect_post_px", 2048))
            sc = min(1.0, cap / float(max(wh, ww)))
            ph, pw = max(1, round(wh * sc)), max(1, round(ww * sc))
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(ph, pw)],
            )
            if not results:
                return []
            masks, scores = results[0].get("masks"), results[0].get("scores")
            if masks is None or scores is None:
                return []
            tile = (y0, x0, y1, x1) != (0, 0, H, W)
            out: list[tuple[float, Any]] = []
            for i in range(int(masks.shape[0]) if hasattr(masks, "shape") else 0):
                m = masks[i]
                mask_np = m.float().cpu().numpy() if hasattr(m, "float") else np.asarray(m)
                local = BoxMask.from_dense(np.asarray(mask_np) > 0.5)
                if not local.crop.size:
                    continue
                if (ph, pw) != (wh, ww):
                    local = local.resized((wh, ww))
                bm = BoxMask((H, W), y0 + local.y0, x0 + local.x0, local.crop)
                if tile:
                    # Cut by an INNER tile border: the overlapping neighbour
                    # tile (or the full frame) has it whole.
                    by0, bx0, by1, bx1 = bm.box
                    if (
                        (by0 <= y0 + 1 and y0 > 0)
                        or (bx0 <= x0 + 1 and x0 > 0)
                        or (by1 >= y1 - 1 and y1 < H)
                        or (bx1 >= x1 - 1 and x1 < W)
                    ):
                        continue
                score = float(scores[i])
                out.append((score, bm))
            return out

        seeds: list[tuple[int, str, Any]] = []
        next_track_id = 1
        for concept in concepts:
            full = _run(concept, *windows[0])
            tiled = [d for win in windows[1:] for d in _run(concept, *win)]
            for _score, bm in _merge_detections(full, tiled):
                # Belt-and-braces size floor — the ranker will drop them
                # too, but skipping here saves a tracker session.
                if not self._size_ok(bm, H, W):
                    continue
                seeds.append((next_track_id, concept, bm))
                next_track_id += 1
        if g > 1:
            _log.info(
                "SAM3 matte: detection on the full frame + %dx%d tiles: %d object(s).",
                g,
                g,
                len(seeds),
            )
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
        self,
        frames: np.ndarray,
        objs: list[dict[str, Any]],
        first: int,
        area_floor: float | None = None,
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
                stacks: dict[int, Any] = {}
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
                stacks = {}
                for part in _session_chunks(group, int(self.params.get("tracker_max_objects", 32))):
                    stacks.update(self._track_group(frames, part))
            for o in group:
                stack = stacks.get(o["obj_id"])
                if stack is None:
                    continue
                if isinstance(stack, np.ndarray):
                    ok = stack.ndim == 3 and stack.shape[1:] == (plate_h, plate_w)
                    shape = stack.shape
                else:  # list of BoxMasks (joint session)
                    ok = len(stack) == n_frames and all(
                        m.shape == (plate_h, plate_w) for m in stack
                    )
                    shape = (len(stack), *(stack[0].shape if len(stack) else ()))
                if not ok:
                    raise ValueError(
                        f"Track stack for {o['label']!r} ({o['obj_id']}) has shape {shape}, "
                        f"expected ({n_frames}, {plate_h}, {plate_w})"
                    )
                # Stored as box crops (MaskDict): the whole-clip masks of
                # every instance are held at once, so this bounds host RAM.
                masks = MaskDict((first + k, stack[k]) for k in range(len(stack)))
                if area_floor is None:
                    if not any(self._size_ok(m, plate_h, plate_w) for m in masks.values()):
                        continue
                elif all(m.sum() < area_floor for m in masks.values()):
                    continue
                self._instances.append(_DetectedInstance(int(o["obj_id"]), str(o["label"]), masks))
                self._priority[int(o["obj_id"])] = int(o.get("priority", 0))

    def _track_group(self, frames: np.ndarray, group: list[dict[str, Any]]) -> dict[int, list[Any]]:
        """Track every object of `group` (same seed frame) in ONE tracker
        session and give each pixel to one object (highest mask logit; user
        clicks get `click_priority_bias`). Returns obj_id -> N BoxMasks."""
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
        chunk = max(1, int(self.params.get("tracker_chunk", 24)))
        session = processor.init_video_session(
            video=frame_pils,
            inference_device=self._device,
            # A crowd's per-object memory bank lives in host RAM.
            inference_state_device=(
                "cpu" if len(group) > int(self.params.get("tracker_state_cpu_above", 8)) else None
            ),
            dtype=self._trk_dtype,
        )
        mask_objs = [o for o in group if o.get("mask") is not None]
        if mask_objs:
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=seed,
                obj_ids=[int(o["obj_id"]) for o in mask_objs],
                input_masks=[(np.asarray(o["mask"]) > 0.5).astype(np.uint8) for o in mask_objs],
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

        empty = BoxMask((H, W))
        out: dict[int, list[Any]] = {int(o["obj_id"]): [empty] * N for o in group}
        bias_of = {
            int(o["obj_id"]): float(self.params.get("click_priority_bias", 4.0))
            if o.get("priority", 0)
            else 0.0
            for o in group
        }
        exclusive = bool(self.params.get("exclusive_masks", True))
        contain = float(self.params.get("part_containment", 0.7) or 0.0)
        is_part = _part_label_test(self.params.get("part_labels", "auto"))
        parent = _seed_parents(mask_objs, contain, is_part) if contain > 0 else {}
        self._parent.update(parent)

        def _up(pm: Any, sl: slice, n: int) -> Any:
            """Logits of objects `sl` upsampled to the plate: (n, H, W)."""
            post = processor.post_process_masks(
                pm[sl].unsqueeze(0) if pm.ndim == 4 else pm[sl],
                original_sizes=[(H, W)],
                binarize=False,
            )
            lg = post[0] if post else torch.zeros((n, H, W))
            if not hasattr(lg, "reshape"):
                lg = torch.as_tensor(np.asarray(lg))
            return lg.reshape(n, H, W).float()

        def _store(fi: int, oid: int, pres: Any) -> None:
            """One object's (H, W) bool tensor -> its BoxMask, cropped on device."""
            rows = torch.nonzero(pres.any(dim=1)).flatten()
            if rows.numel() == 0:
                return
            cols = torch.nonzero(pres.any(dim=0)).flatten()
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            crop = pres[y0:y1, x0:x1].to(torch.uint8).cpu().numpy()
            out[oid][fi] = BoxMask((H, W), y0, x0, crop)

        def _consume(step_iter: Any) -> None:
            for step in step_iter:
                fi = getattr(step, "frame_idx", None)
                if fi is None:
                    fi = getattr(step, "frame_index", None)
                pm = getattr(step, "pred_masks", None)
                ids = list(getattr(step, "object_ids", None) or [])
                if fi is None or pm is None or not ids:
                    continue
                n_obj = len(ids)
                if n_obj > chunk and not parent:
                    # Crowd: never hold every object's plate-size logits at
                    # once. Pass 1 finds each pixel's winner chunk by chunk,
                    # pass 2 re-upsamples and keeps winners only.
                    bias_all = torch.tensor([bias_of.get(int(i), 0.0) for i in ids])
                    win = None
                    if exclusive:
                        best = None
                        for c0 in range(0, n_obj, chunk):
                            c1 = min(n_obj, c0 + chunk)
                            lg = _up(pm, slice(c0, c1), c1 - c0)
                            lg += bias_all[c0:c1].to(lg.device)[:, None, None]
                            v, k = lg.max(dim=0)
                            if best is None:
                                best, win = v, k
                            else:
                                assert win is not None
                                upd = v > best
                                best = torch.where(upd, v, best)
                                win = torch.where(upd, k + c0, win)
                            del lg, v, k
                    for c0 in range(0, n_obj, chunk):
                        c1 = min(n_obj, c0 + chunk)
                        lg = _up(pm, slice(c0, c1), c1 - c0)
                        for j in range(c1 - c0):
                            pres = lg[j] > 0.0
                            if win is not None:
                                pres &= win == (c0 + j)
                            if int(ids[c0 + j]) in out:
                                _store(int(fi), int(ids[c0 + j]), pres)
                        del lg
                    continue
                logits = _up(pm, slice(0, n_obj), n_obj)
                present = logits > 0.0
                if exclusive and len(ids) > 1:
                    bias = torch.tensor(
                        [bias_of.get(int(i), 0.0) for i in ids], device=logits.device
                    )
                    eff = logits + bias[:, None, None]
                    idx = {int(i): k for k, i in enumerate(ids)}
                    kids: dict[int, list[int]] = {}
                    for c, p in parent.items():
                        if c in idx and p in idx:
                            kids.setdefault(idx[p], []).append(idx[c])
                    if not kids:
                        winner = torch.argmax(eff, dim=0)
                        for k in range(len(ids)):
                            present[k] &= winner == k
                    else:
                        # Wholes compete with everything else, each as the
                        # union of itself and its parts; parts then split
                        # their whole's pixels among themselves.
                        child = {c for cs in kids.values() for c in cs}
                        roots = [k for k in range(len(ids)) if k not in child]
                        root_eff = eff.clone()
                        for p, cs in kids.items():
                            root_eff[p] = torch.maximum(eff[p], eff[cs].max(dim=0).values)
                        win = torch.argmax(root_eff[roots], dim=0)
                        for j, r in enumerate(roots):
                            present[r] = (root_eff[r] > 0.0) & (win == j)
                        for p, cs in kids.items():
                            wk = torch.argmax(eff[cs], dim=0)
                            for j, c in enumerate(cs):
                                present[c] = (logits[c] > 0.0) & (wk == j) & present[p]
                for k, oid in enumerate(ids):
                    if int(oid) in out:
                        _store(int(fi), int(oid), present[k])

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
        area_floor: float | None = None,
    ) -> None:
        """Run the detector every `stride` frames; anything mostly NOT
        covered by an existing track becomes a new track seeded there."""
        n = int(frames.shape[0])
        H, W = int(frames.shape[1]), int(frames.shape[2])
        tiles_on = self._tile_grid(H, W) > 1
        budget = int(
            self.params.get("small_redetect_tracks", 64)
            if tiles_on
            else self.params.get("max_redetect_tracks", 8)
        )
        thr = float(self.params.get("redetect_new_coverage", 0.5))
        next_id = 2001  # own id block: concepts 1.., clicks 1001..
        import cv2

        r = max(1, round(0.015 * max(H, W)))
        touch_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        for k in range(0, n, max(1, stride)):
            if budget <= 0:
                break
            if k == seed_local:
                continue
            f = first + k
            covered = np.zeros((H, W), dtype=np.uint8)
            for inst in self._instances:
                m = inst.masks.get(f)
                if m is not None:
                    paint_max(covered, BoxMask.from_dense(m))
            new: list[dict[str, Any]] = []
            for _tid, label, m in self._detect_frame(frames, k, concepts):
                mb = BoxMask.from_dense(m)
                area = float(mb.sum())
                if area <= 0 or (area_floor is not None and area < area_floor):
                    continue
                by0, bx0, by1, bx1 = mb.box
                if float((mb.crop & covered[by0:by1, bx0:bx1]).sum()) / area >= thr:
                    continue
                # Partly covered by a track of the SAME label = a piece the
                # tracker lost (PF0048: the arm holding the rifle dropped out
                # of the woman's track). Re-attach it to that track instead
                # of starting a second "person".
                # "Touching" = overlap with the track grown by ~1.5% of the
                # long edge: a limb detected alone is adjacent, not inside.
                merge_into = None
                best = float(self.params.get("redetect_merge_coverage", 0.1))
                max_ratio = float(self.params.get("redetect_merge_max_ratio", 0.6))
                for inst in self._instances:
                    im = inst.masks.get(f)
                    if inst.label != label or im is None or not im.any():
                        continue
                    im = BoxMask.from_dense(im)
                    if area > max_ratio * im.sum():
                        continue
                    if not boxes_touch(im, mb, margin=r + 1):
                        continue
                    wy0, wx0, wy1, wx1 = union_box([im, mb], r + 1, (H, W)) or (0, 0, H, W)
                    grown = cv2.dilate(im.window(wy0, wx0, wy1, wx1), touch_kernel) > 0
                    cov = float((mb.window(wy0, wx0, wy1, wx1).astype(bool) & grown).sum()) / area
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
                paint_max(covered, mb)  # one new track per thing on this frame
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
                    host.masks[fr] = mm if hm is None else _box_or(hm, mm)
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

    def _depth_nearness(self, frames: np.ndarray, first: int) -> dict[int, float]:
        """Per-track nearness in [0, 1] (1 = nearest), the median of Depth
        Anything V2 Small's per-frame-normalised disparity inside the track's
        eroded mask, over a few frames. Empty dict when there is nothing to
        order or the depth model is unavailable (area order then)."""
        if len(self._instances) < 2:
            return {}
        try:
            import cv2
            import torch

            from live_action_aov.passes.depth.depth_anything_v2 import DepthAnythingV2Pass

            depth = DepthAnythingV2Pass({"variant": "small"})
            depth._load_model()
        except Exception as exc:  # weights missing offline, no GPU, ...
            _log.warning("SAM3 matte: depth order unavailable (%s) - using area order.", exc)
            return {}
        all_f = sorted({f for i in self._instances for f in i.masks})
        n = max(int(self.params.get("depth_order_frames", 5)), 1)
        pick = [all_f[int(k)] for k in np.linspace(0, len(all_f) - 1, min(n, len(all_f)))]
        k3 = np.ones((7, 7), np.uint8)
        vals: dict[int, list[float]] = {}
        for f in dict.fromkeys(pick):
            img = np.clip(frames[f - first], 0.0, 1.0)[None].astype(np.float32)
            # Raw DA V2 output is relative disparity: larger = nearer.
            disp = depth.postprocess(depth.infer(depth.preprocess(img)))[CH_Z_RAW]
            lo, hi = np.percentile(disp, [1, 99])
            disp = np.clip((disp - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            for inst in self._instances:
                m = inst.masks.get(f)
                if m is None or not m.any():
                    continue
                bm = BoxMask.from_dense(m)
                y0, x0, y1, x1 = bm.box
                core = cv2.erode(np.pad(bm.crop, 3), k3)[3:-3, 3:-3]
                sel = core > 0 if core.any() else bm.crop > 0
                vals.setdefault(inst.track_id, []).append(float(np.median(disp[y0:y1, x0:x1][sel])))
        del depth
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        near = {t: float(np.mean(v)) for t, v in vals.items()}
        ranked = [
            f"{i.label}#{i.track_id}={near[i.track_id]:.2f}"
            for i in sorted(self._instances, key=lambda i: -near.get(i.track_id, 0.0))
            if i.track_id in near
        ]
        more = f" ... (+{len(ranked) - 12} more)" if len(ranked) > 12 else ""
        _log.info("SAM3 matte: depth order (near -> far): %s%s", ", ".join(ranked[:12]), more)
        return near

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

        concepts = [str(c) for c in self.params["concepts"]]
        # Box crops (test fakes and older overrides return dense masks).
        self._reader, self._first = reader, first
        self._plate_shape = (plate_h, plate_w)
        on_plate = callable(getattr(reader, "read_frame_full_res", None)) and (
            str(self.params.get("small_objects", "auto")).lower() != "off"
        )
        self._det_hw = self._full_hw() if on_plate else (plate_h, plate_w)
        seeds = self._detect_frame(frames, seed_local, concepts)

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
        # Masks are kept as box crops (uint8): budget each object at 4x its
        # seed box (it moves / turns), clicks at a quarter of the plate.
        plate_px = plate_h * plate_w
        per_frame_bytes = (
            sum(
                min(plate_px, 4 * int(np.prod(np.subtract(m.box[2:], m.box[:2]))))
                for _t, _l, m in seeds
            )
            + n_clicks * plate_px // 4
        )
        peak_bytes = n_frames * per_frame_bytes
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
        # Size floor: `_size_ok` (box height with tiles on, area fraction off).
        area_floor = None
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
            obj: dict[str, Any] = {
                "obj_id": next_click_id,
                "label": name,
                "seed_local": seed_local_i,
                "points": pts or None,
                "labels": lbls or None,
                "box": box,
                "priority": 1,
            }
            # A click on a small subject: segment it inside a zoomed window
            # around the click and seed the tracker with that mask (on the
            # full frame SAM sees a 40 px person at 4K as 10 px).
            if self._tile_grid(plate_h, plate_w) > 1 or self._tile_grid(*self._full_hw()) > 1:
                zoomed = self._zoom_on_plate(frames, seed_local_i, pts, lbls, box)
                if zoomed is not None:
                    obj.update(mask=zoomed, points=None, labels=None, box=None)
                    _log.info(
                        "SAM3 matte: click '%s' segmented in a zoomed window (%d px tall).",
                        name,
                        zoomed.height,
                    )
            objs.append(obj)
            next_click_id += 1

        self._instances = []
        self._priority = {}
        self._parent = {}
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
            contain = float(self.params.get("part_containment", 0.7) or 0.0)
            if contain > 0:
                is_part = _part_label_test(self.params.get("part_labels", "auto"))
                self._parent.update(_clip_parents(self._instances, self._parent, contain, is_part))
            live = {i.track_id for i in self._instances}
            self._parent = {c: p for c, p in self._parent.items() if c in live and p in live}
            for c, p in self._parent.items():
                _log.info("SAM3 matte: track %d is a part of track %d.", c, p)
            self._nearness = (
                self._depth_nearness(frames, first)
                if str(self.params.get("layer_order", "depth")) == "depth"
                else {}
            )
            _make_exclusive(self._instances, self._priority, self._parent, self._nearness)
            gap = self.params.get("seam_close_px", "auto")
            gap_px = (
                max(2, round(0.004 * max(plate_h, plate_w))) if str(gap) == "auto" else int(gap)
            )
            if gap_px > 0:
                # A whole and its parts overlap by design: close seams between
                # wholes/independent objects, then between sibling parts.
                _close_seams([i for i in self._instances if i.track_id not in self._parent], gap_px)
                for p in set(self._parent.values()):
                    sibs = [i for i in self._instances if self._parent.get(i.track_id) == p]
                    if len(sibs) > 1:
                        _close_seams(sibs, gap_px)
            _union_parts(self._instances, self._parent)
            for inst in [i for i in self._instances if not any(m.any() for m in i.masks.values())]:
                _log.info(
                    "SAM3 matte: '%s' (track %d) lies entirely under higher-priority objects - dropped.",
                    inst.label,
                    inst.track_id,
                )
            self._instances = [i for i in self._instances if any(m.any() for m in i.masks.values())]
            live = {i.track_id for i in self._instances}
            self._parent = {c: p for c, p in self._parent.items() if c in live and p in live}
        self._layer = {
            inst.track_id: rank
            for rank, inst in enumerate(
                _layer_order(self._instances, self._priority, self._nearness)
            )
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
                        paint_max(union, BoxMask.from_dense(m), 1.0)
                if union.any():
                    any_nonzero = True
                    per_frame[f][mask_channel(concept)] = union
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
            bm = BoxMask.from_dense(mask)
            s = float(bm.sum())
            if s <= 0.0:
                continue
            area_fractions.append(s / plate_area)
            # Centroid -> distance from center -> 1 - (d / half_diag) clipped.
            ys, xs = np.nonzero(bm.crop)
            if ys.size:
                cy = float(ys.mean()) + bm.y0
                cx = float(xs.mean()) + bm.x0
                dx = cx - plate_w / 2.0
                dy = cy - plate_h / 2.0
                d = math.hypot(dx, dy)
                centralities.append(max(0.0, 1.0 - d / (plate_diag / 2.0)))
            # Motion energy from forward_flow[f] if available.
            fwd = self._forward_flow.get(f)
            if fwd is not None and fwd.shape[-2:] == (plate_h, plate_w):
                y0, x0, y1, x1 = bm.box
                mag = np.sqrt(fwd[0, y0:y1, x0:x1] ** 2 + fwd[1, y0:y1, x0:x1] ** 2)
                masked = mag[bm.crop > 0]
                if masked.size:
                    motion_energies.append(float(masked.mean()) / max(plate_diag, 1.0))

        return Instance(
            track_id=inst.track_id,
            label=inst.label,
            frames=sorted(f for f, m in inst.masks.items() if m.any()),
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
                # A LazyStack over the instance's own box crops: no second
                # copy. Consumers get dense (T, H, W) uint8 via np.asarray()
                # one track at a time, or the crops via `stack.box(k)`.
                stack = LazyStack([inst.masks[f] for f in frames_sorted], (plate_h, plate_w))
                hard_masks[inst.track_id] = {
                    "label": inst.label,
                    "frames": frames_sorted,
                    "stack": stack,
                    # Depth rank (0 = front): who owns a shared seam.
                    "layer": self._layer.get(inst.track_id, len(hard_masks)),
                    # Whole this track is a part of (None = independent).
                    "parent": self._parent.get(inst.track_id),
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
        ba, bb = BoxMask.from_dense(ma), BoxMask.from_dense(mb)
        i = inter_area(ba, bb)
        inter += i
        union += ba.sum() + bb.sum() - i
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


#: Nearness (0 = far, 1 = near) is compared in steps this wide: closer
#: than this is "the same depth" and the area rule decides.
_NEARNESS_STEP = 0.08


def _layer_order(
    instances: list[_DetectedInstance],
    priority: dict[int, int],
    nearness: dict[int, float] | None = None,
) -> list[_DetectedInstance]:
    """Front-to-back order: user clicks first, then the NEARER object (when
    `nearness` is known: per-object median monocular depth), then the
    smaller object (clip-mean area: what is held or in front - a rifle over
    the person), then the lower track id. Clip-level values keep the order
    stable over time, so seams don't flicker."""
    near = nearness or {}

    def mean_area(i: _DetectedInstance) -> float:
        areas = [float(m.sum()) for m in i.masks.values() if m.any()]
        return sum(areas) / len(areas) if areas else 0.0

    return sorted(
        instances,
        key=lambda i: (
            -priority.get(i.track_id, 0),
            -round(near.get(i.track_id, 0.0) / _NEARNESS_STEP),
            mean_area(i),
            i.track_id,
        ),
    )


def _close_seams(instances: list[_DetectedInstance], gap_px: int) -> None:
    """Fill the thin unowned line SAM leaves where two objects touch, in
    place. A free pixel within reach of at least two DIFFERENT objects goes
    to the nearest one. Concavities of a single object (the gap between an
    arm and the body) are left alone - only contact lines.

    Each object reaches `gap_px` at most, and 6% of its own box height: in
    a crowd, 16 px of sand between two 80 px people at 4K is real ground,
    not a contact line (it came out as triangular sand shards)."""
    import cv2

    kernels: dict[int, np.ndarray] = {}

    def kernel(g: int) -> np.ndarray:
        if g not in kernels:
            kernels[g] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        return kernels[g]

    frames = sorted({f for i in instances for f in i.masks})
    for f in frames:
        present = [i for i in instances if f in i.masks and i.masks[f].any()]
        if len(present) < 2:
            continue
        # Only objects whose boxes come within the gap can share a seam:
        # work cluster by cluster, inside the cluster's window (a crowd of
        # 200 people is 200 small windows, not 200 full-frame transforms).
        boxes = [BoxMask.from_dense(i.masks[f]) for i in present]
        for cl in _touch_clusters(boxes, 2 * gap_px + 1):
            if len(cl) < 2:
                continue
            shape = boxes[cl[0]].shape
            wy0, wx0, wy1, wx1 = union_box([boxes[j] for j in cl], gap_px + 1, shape) or (
                0,
                0,
                *shape,
            )
            bins = [boxes[j].window(wy0, wx0, wy1, wx1) for j in cl]
            taken = np.zeros(bins[0].shape, dtype=bool)
            near = np.zeros(bins[0].shape, dtype=np.uint8)
            for j, b in zip(cl, bins, strict=True):
                g = min(gap_px, max(1, round(0.06 * boxes[j].height)))
                taken |= b > 0
                near += (cv2.dilate(b, kernel(g)) > 0).astype(np.uint8)
            gap = (near >= 2) & ~taken
            if not gap.any():
                continue
            dist = np.stack([cv2.distanceTransform(1 - b, cv2.DIST_L2, 3) for b in bins], axis=0)
            owner = np.argmin(dist, axis=0)
            ys, xs = np.nonzero(gap)
            for j, b in zip(cl, bins, strict=True):
                sel = owner[ys, xs] == cl.index(j)
                if sel.any():
                    b = b.copy()
                    b[ys[sel], xs[sel]] = 1
                    present[j].masks[f] = BoxMask.from_crop(shape, wy0, wx0, b)


def _make_exclusive(
    instances: list[_DetectedInstance],
    priority: dict[int, int],
    parent: dict[int, int] | None = None,
    nearness: dict[int, float] | None = None,
) -> None:
    """Give every contested pixel to ONE instance, in place, in
    `_layer_order` (the front object owns the overlap). A whole and its own
    parts (`parent`: part -> whole) do not contest each other."""
    parent = parent or {}
    order = _layer_order(instances, priority, nearness)
    frames = sorted({f for i in instances for f in i.masks})
    for f in frames:
        present = [i for i in order if f in i.masks and i.masks[f].any()]
        if len(present) < 2:
            continue
        shape = BoxMask.from_dense(present[0].masks[f]).shape
        if not parent:
            taken = np.zeros(shape, dtype=np.uint8)
            for inst in present:
                bm = BoxMask.from_dense(inst.masks[f])
                y0, x0, y1, x1 = bm.box
                inst.masks[f] = BoxMask.from_crop(
                    shape, y0, x0, bm.crop & (1 - taken[y0:y1, x0:x1])
                )
                np.maximum(taken[y0:y1, x0:x1], bm.crop, out=taken[y0:y1, x0:x1])
            continue
        claimed: list[tuple[int, BoxMask]] = []
        for inst in present:
            bm = BoxMask.from_dense(inst.masks[f])
            y0, x0, y1, x1 = bm.box
            blocked = np.zeros(bm.crop.shape, dtype=np.uint8)
            for tid, om in claimed:
                if parent.get(inst.track_id) == tid or parent.get(tid) == inst.track_id:
                    continue
                if boxes_touch(om, bm):
                    blocked |= om.window(y0, x0, y1, x1)
            inst.masks[f] = BoxMask.from_crop(shape, y0, x0, bm.crop & (1 - blocked))
            claimed.append((inst.track_id, bm))


#: Labels that name a PART of something else. A label is part-like when any
#: of its words is here ("leather jacket", "front wheel").
_PART_WORDS = frozenset(
    [
        "hair",
        "head",
        "face",
        "eye",
        "eyes",
        "mouth",
        "nose",
        "ear",
        "ears",
        "beard",
        "neck",
        "hand",
        "hands",
        "arm",
        "arms",
        "leg",
        "legs",
        "foot",
        "feet",
        "finger",
        "fingers",
        "skin",
        "torso",
        "jacket",
        "coat",
        "shirt",
        "t-shirt",
        "tshirt",
        "top",
        "sweater",
        "hoodie",
        "vest",
        "dress",
        "skirt",
        "trousers",
        "pants",
        "jeans",
        "shorts",
        "shoe",
        "shoes",
        "boot",
        "boots",
        "sneaker",
        "sneakers",
        "sock",
        "socks",
        "glove",
        "gloves",
        "hat",
        "cap",
        "helmet",
        "hood",
        "scarf",
        "tie",
        "belt",
        "bag",
        "backpack",
        "glasses",
        "sunglasses",
        "goggles",
        "headphones",
        "earphones",
        "mask",
        "sleeve",
        "sleeves",
        "collar",
        "pocket",
        "strap",
        "watch",
        "bracelet",
        "necklace",
        "earring",
        "wheel",
        "wheels",
        "tire",
        "tires",
        "tyre",
        "tyres",
        "window",
        "windows",
        "windshield",
        "windscreen",
        "door",
        "doors",
        "headlight",
        "headlights",
        "taillight",
        "bumper",
        "hood",
        "mirror",
        "roof",
        "licence",
        "license",
        "plate",
        "grille",
        "handle",
    ]
)


def _part_label_test(spec: Any) -> Callable[[str], bool]:
    if isinstance(spec, (list, tuple, set)):
        allowed = {str(s).strip().lower() for s in spec}
        return lambda label: label.strip().lower() in allowed
    return lambda label: any(
        w in _PART_WORDS for w in label.lower().replace("-", " ").replace("_", " ").split()
    )


def _seed_parents(
    objs: list[dict[str, Any]], contain: float, is_part: Callable[[str], bool]
) -> dict[int, int]:
    """Part -> whole on the seed frame: an object whose detection mask lies at
    least `contain` inside a bigger object of another label is its part
    (the smallest such whole). Parts of parts hang on the outermost whole."""
    ms = {
        int(o["obj_id"]): (BoxMask.from_dense(o["mask"]), str(o.get("label", "")))
        for o in objs
        if o.get("mask") is not None
    }
    areas = {k: int(m.sum()) for k, (m, _) in ms.items()}
    parent: dict[int, int] = {}
    for p, (mp, lp) in ms.items():
        if areas[p] == 0 or not is_part(lp):
            continue
        best = None
        for w, (mw, lw) in ms.items():
            if w == p or lw == lp or areas[w] <= areas[p] * 1.1:
                continue
            if float(inter_area(mp, mw)) / areas[p] >= contain and (
                best is None or areas[w] < areas[best]
            ):
                best = w
        if best is not None:
            parent[p] = best
    return _flatten_parents(parent)


def _clip_parents(
    instances: list[_DetectedInstance],
    known: dict[int, int],
    contain: float,
    is_part: Callable[[str], bool],
) -> dict[int, int]:
    """Same test over the clip (every 4th frame) for tracks that did not
    share a seed frame (re-detected later, other tracker session)."""
    found = dict(known)
    involved = set(known) | set(known.values())
    by_id = {i.track_id: i for i in instances}

    def family(w: _DetectedInstance, f: int) -> np.ndarray | None:
        # A whole's region includes its known parts: in a joint session the
        # parts OWN their pixels, so a hair track re-detected later sits on
        # the first hair track, not on the person's own mask (field report:
        # PF0100-02 hair split 50/50 with the person).
        m = w.masks.get(f)
        out = None if m is None else m > 0
        for c, pw in known.items():
            cm = by_id[c].masks.get(f) if pw == w.track_id and c in by_id else None
            if cm is not None:
                out = cm > 0 if out is None else out | (cm > 0)
        return out

    for p in instances:
        if p.track_id in involved or not is_part(p.label):
            continue
        fr = sorted(p.masks)[::4]
        ap = sum(int(p.masks[f].sum()) for f in fr)
        if ap == 0:
            continue
        best, best_area = None, 0
        for w in instances:
            if w.track_id == p.track_id or w.label == p.label or w.track_id in known:
                continue
            if not any(
                f in w.masks
                and boxes_touch(BoxMask.from_dense(p.masks[f]), BoxMask.from_dense(w.masks[f]))
                for f in fr
            ) and not any(pw == w.track_id for pw in known.values()):
                continue
            fams = {f: family(w, f) for f in fr}
            aw = sum(int(m.sum()) for m in fams.values() if m is not None)
            if aw <= ap * 1.1:
                continue
            inter = sum(int(((p.masks[f] > 0) & m).sum()) for f, m in fams.items() if m is not None)
            if inter / ap >= contain and (best is None or aw < best_area):
                best, best_area = w.track_id, aw
        if best is not None:
            found[p.track_id] = best
    return _flatten_parents(found)


def _click_window(
    points: list[list[float]], labels: list[int], box: list[float] | None, h: int, w: int
) -> tuple[int, int, int, int] | None:
    """Square window (y0, x0, y1, x1) to segment a click in, zoomed in: 3x
    the click spread (or 1.6x the box), at least 1/8 of the long edge, at
    most half of it. None for a big object (zooming gains nothing)."""
    long_edge = max(h, w)
    if box is not None:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        size = 1.6 * max(box[2] - box[0], box[3] - box[1])
    else:
        fg = [p for p, lb in zip(points, labels or [1] * len(points), strict=False) if lb == 1]
        fg = fg or points
        if not fg:
            return None
        xs, ys = [p[0] for p in fg], [p[1] for p in fg]
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        size = 3.0 * max(max(xs) - min(xs), max(ys) - min(ys))
    if size > 0.5 * long_edge:
        return None
    s = int(max(size, long_edge / 8.0))
    sh, sw = min(s, h), min(s, w)
    y0 = int(min(max(cy - sh / 2.0, 0), h - sh))
    x0 = int(min(max(cx - sw / 2.0, 0), w - sw))
    return y0, x0, y0 + sh, x0 + sw


def _session_chunks(group: list[dict[str, Any]], max_n: int) -> list[list[dict[str, Any]]]:
    """Split a tracker group into sessions of at most `max_n` objects,
    ordered left to right by seed position so neighbours share a session.
    Clicks (no mask) go with the first chunk."""
    if max_n <= 0 or len(group) <= max_n:
        return [group]

    def cx(o: dict[str, Any]) -> float:
        m = o.get("mask")
        if m is None:
            return -1.0
        y0, x0, y1, x1 = BoxMask.from_dense(m).box
        return (x0 + x1) / 2.0

    ordered = sorted(group, key=cx)
    return [ordered[i : i + max_n] for i in range(0, len(ordered), max_n)]


def _box_or(a: Any, b: Any) -> BoxMask:
    """Union of two masks without going full-frame."""
    ba, bb = BoxMask.from_dense(a), BoxMask.from_dense(b)
    win = union_box([ba, bb], 0, ba.shape)
    if win is None:
        return ba
    y0, x0, y1, x1 = win
    return BoxMask.from_crop(ba.shape, y0, x0, ba.window(*win) | bb.window(*win))


def _touch_clusters(boxes: list[BoxMask], margin: int) -> list[list[int]]:
    """Groups of masks whose boxes come within `margin` px (union-find)."""
    root = list(range(len(boxes)))

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_touch(boxes[i], boxes[j], margin):
                root[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _merge_detections(
    full: list[tuple[float, BoxMask]], tiled: list[tuple[float, BoxMask]]
) -> list[tuple[float, BoxMask]]:
    """One detection per object from the full frame + its tiles.

    A full-frame mask holding two or more tile detections is a GROUP (SAM 3
    sometimes returns several distant people as one instance): drop it,
    keep the individuals. Then greedy by score: a detection overlapping an
    already kept one (IoU >= 0.5, or 80% of the smaller inside the other)
    is the same object seen twice. Without tiles: the full frame as is."""
    if not tiled:
        return list(full)
    groups = set()
    for k, (_s, fm) in enumerate(full):
        fa = fm.sum()
        inside = [
            t for _ts, t in tiled if t.sum() < 0.6 * fa and inter_area(t, fm) >= 0.8 * t.sum()
        ]
        if len(inside) >= 2:
            groups.add(k)
    cands = [d for k, d in enumerate(full) if k not in groups] + list(tiled)
    kept: list[tuple[float, BoxMask]] = []
    for score, m in sorted(cands, key=lambda d: -d[0]):
        a = m.sum()
        dup = False
        for _ks, km in kept:
            if not boxes_touch(m, km):
                continue
            i = inter_area(m, km)
            ka = km.sum()
            if i / max(a + ka - i, 1) >= 0.5 or i >= 0.8 * min(a, ka):
                dup = True
                break
        if not dup:
            kept.append((score, m))
    return kept


def _flatten_parents(parent: dict[int, int]) -> dict[int, int]:
    """One level only: a part of a part hangs on the outermost whole."""
    out: dict[int, int] = {}
    for c, p in parent.items():
        seen = {c}
        while p in parent and p not in seen:
            seen.add(p)
            p = parent[p]
        out[c] = p
    return out


def _union_parts(instances: list[_DetectedInstance], parent: dict[int, int]) -> None:
    """A whole covers its parts (in place): mask.person is the full body
    even where SAM gave a strand of hair only to the hair track."""
    by_id = {i.track_id: i for i in instances}
    for c, p in parent.items():
        ci, pi = by_id.get(c), by_id.get(p)
        if ci is None or pi is None:
            continue
        for f, m in ci.masks.items():
            if f in pi.masks:
                pi.masks[f] = ((pi.masks[f] > 0) | (m > 0)).astype(np.uint8)
            elif m.any():
                pi.masks[f] = (m > 0).astype(np.uint8)


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
