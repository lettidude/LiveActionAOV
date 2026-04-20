# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Hero ranking + per-clip slot assignment (spec §13.1 Phase 3).

Pure Python, no model dependencies. Both the detector pass (to build the
canonical slotted list) and user-facing tools (to validate overrides) import
from here. Tests can exercise the ranker in isolation — no torch, no OIIO.

Design decisions (from Phase 3 brainstorm):
- **Per-clip lock**: slots are assigned once over the whole clip, not
  per-frame. `matte.r` refers to the same track for the entire shot. A
  track may exit screen mid-shot — the refiner writes zeros for frames
  where the instance is absent, but the slot stays reserved.
- **Deterministic**: same instances + weights → same slot assignment.
  Ties broken by (highest area, lowest track_id) for stability.
- **Overrides first**: user-forced `(concept, track_id, slot)` entries
  claim their slot before the scored ranking fills the rest.
- **Decoupled from inputs**: the scoring function takes pre-reduced
  scalars (area_fraction, centrality, motion_energy, …). Computing those
  from masks + flow is the caller's job. Keeps rank.py free of numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

# Canonical RGBA channel order. Enumerated here so tests and callers both
# reference a single source of truth.
SLOT_ORDER: tuple[str, ...] = ("r", "g", "b", "a")
SlotName = Literal["r", "g", "b", "a"]


class RankWeights(BaseModel):
    """Scoring weights for hero-ranking. Each weight is the coefficient on
    a [0, 1]-normalized feature; the sum need not equal 1.

    Defaults reproduce spec §3.3 (Phase 3 YAML).
    """

    model_config = ConfigDict(extra="forbid")

    area: float = 0.4
    centrality: float = 0.2
    motion: float = 0.2
    duration: float = 0.2
    user_priority: float = 0.0


@dataclass
class Instance:
    """One tracked object across the clip.

    All scalar fields are already [0, 1]-normalized (or near to it) so the
    ranker can just dot them with `RankWeights`. See `SAM3MattePass` for the
    canonical reduction: area = mean(mask)/plate_area,
    centrality = 1 - (distance_of_centroid_from_plate_center / plate_diag),
    motion_energy = mean(|flow| over mask) / plate_diag,
    duration = len(frames)/n_clip_frames.
    """

    track_id: int
    label: str
    frames: list[int]
    area_fraction: float
    centrality: float
    motion_energy: float
    user_priority: float = 0.0


@dataclass(frozen=True)
class HeroOverride:
    """User override declaring which track should live in which slot.

    `track_id` or `(label, nth)` — the detector pass resolves the latter
    form before handing overrides to `rank_and_assign`. Only the explicit
    `track_id` form reaches this module.
    """

    track_id: int
    slot: SlotName


@dataclass
class HeroSlot:
    """A ranked, slot-assigned hero. The refiner consumes a list of these."""

    track_id: int
    slot: SlotName
    label: str
    score: float
    instance: Instance = field(repr=False)


def score_instance(inst: Instance, weights: RankWeights, n_clip_frames: int) -> float:
    """Weighted sum of the pre-normalized features.

    `n_clip_frames` is the total number of frames in the clip so duration
    can be expressed as a fraction. With a zero-length clip we short-circuit
    to 0 rather than dividing by zero (shouldn't happen in real jobs).
    """
    if n_clip_frames <= 0:
        return 0.0
    duration = len(inst.frames) / n_clip_frames
    return (
        weights.area * inst.area_fraction
        + weights.centrality * inst.centrality
        + weights.motion * inst.motion_energy
        + weights.duration * duration
        + weights.user_priority * inst.user_priority
    )


def rank_and_assign(
    instances: Sequence[Instance],
    weights: RankWeights,
    n_clip_frames: int,
    *,
    max_heroes: int = 4,
    overrides: Sequence[HeroOverride] = (),
    slots: Sequence[str] = SLOT_ORDER,
) -> list[HeroSlot]:
    """Assign RGBA slots to the top `max_heroes` instances.

    Order of operations:
    1. Every instance is scored with `score_instance`.
    2. User overrides claim their named slots first. A track_id in
       `overrides` that doesn't appear in `instances` is silently skipped
       (caller should validate earlier; we don't raise so test fixtures
       stay light).
    3. Remaining slots are filled in descending score order. Ties are
       broken by (larger area_fraction first, then smaller track_id).

    Returns a list sorted by slot order (r, g, b, a) — NOT by score —
    so the refiner can index directly by slot.
    """
    if max_heroes > len(slots):
        raise ValueError(f"max_heroes={max_heroes} exceeds available slots {list(slots)}")

    by_id: dict[int, Instance] = {inst.track_id: inst for inst in instances}
    scored: list[tuple[float, Instance]] = [
        (score_instance(inst, weights, n_clip_frames), inst) for inst in instances
    ]
    # Deterministic sort: descending score, then descending area, then
    # ascending track_id. Python's sort is stable, so compose in reverse
    # order of precedence.
    scored.sort(key=lambda p: p[1].track_id)  # tertiary
    scored.sort(key=lambda p: p[1].area_fraction, reverse=True)  # secondary
    scored.sort(key=lambda p: p[0], reverse=True)  # primary

    assigned: dict[SlotName, HeroSlot] = {}
    slot_list: list[SlotName] = [cast("SlotName", s) for s in slots[:max_heroes]]

    # 1) Apply overrides.
    for ov in overrides:
        if ov.slot not in slot_list:
            continue
        inst = by_id.get(ov.track_id)
        if inst is None:
            continue
        score = score_instance(inst, weights, n_clip_frames)
        assigned[ov.slot] = HeroSlot(
            track_id=inst.track_id,
            slot=ov.slot,
            label=inst.label,
            score=score,
            instance=inst,
        )

    # 2) Fill remaining slots in score order, skipping any track_id already
    # claimed by an override.
    claimed_ids = {h.track_id for h in assigned.values()}
    free_slots = [s for s in slot_list if s not in assigned]
    for score, inst in scored:
        if not free_slots:
            break
        if inst.track_id in claimed_ids:
            continue
        slot = free_slots.pop(0)
        assigned[slot] = HeroSlot(
            track_id=inst.track_id,
            slot=slot,
            label=inst.label,
            score=score,
            instance=inst,
        )
        claimed_ids.add(inst.track_id)

    # Return in canonical slot order, dropping any slots nothing claimed.
    return [assigned[s] for s in slot_list if s in assigned]


__all__ = [
    "SLOT_ORDER",
    "HeroOverride",
    "HeroSlot",
    "Instance",
    "RankWeights",
    "SlotName",
    "rank_and_assign",
    "score_instance",
]
