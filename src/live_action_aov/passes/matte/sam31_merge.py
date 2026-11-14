# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""SAM 3.1 small-objects mode: several seed sessions, one track per object.

SAM 3.1 Object Multiplex detects on the frame where the text prompt is
given and tracks from there. With one seed frame it misses whatever it
does not confirm on that frame (an object that enters late or leaves
early, a small figure hidden at that moment) and can split one object in
two IDs when it loses and re-finds it. `add_prompt` starts the session
over, so seeding several frames means several sessions (`seed_frames`),
each propagated both ways, merged here into one track per object:

1. **Link.** Tracks of DIFFERENT sessions are the same object when their
   masks agree (mean IoU >= `link_iou` over >= 3 shared frames). Clusters
   grow best pair first and never join two tracks that coexist with a low
   IoU (< `conflict_iou` over >= 3 shared frames = two objects).
2. **Stitch.** A cluster that ends and another that starts at most
   `stitch_gap` frames later where the first left off (IoU of the dilated
   last / first masks >= `stitch_iou`) are one object lost and re-found.
3. **Duplicates.** Two clusters where the smaller lies inside the bigger
   (>= `dup_containment` of its area) AND is at least `dup_size_ratio` of
   its size, on >= `dup_frames` of their shared frames, are one object
   tracked twice. The size-ratio guard is what keeps PARTS apart: without
   it a hand, a bag or a body found as its own object merges into its
   whole, the cluster grows, and the next merge swallows more (tried
   without it, the merge snowballed and collapsed separate objects into a
   few). Parts stay separate objects and the matte pass's own
   part / whole logic (`part_containment`) handles them.
4. **Assembly.** Per frame, an object takes the mask of its highest
   priority member (primary session first), or a bigger member mask that
   contains it, plus the other members of that session on that frame.

Before merging, each raw track loses its **glitch frames**: a frame whose
coverage is over `glitch_factor` x the median of the frames before it and
of those after it (`glitch_window` each side; 0 = the whole track's) and more than
`glitch_min_jump` of the frame above it (a small object suddenly read as a
scattered mask over much of the frame). Measured against its neighbours, a
person who walks up to the camera is not a glitch. Those frames count as
not seen.

The comparisons run on a copy at most `work_width` px wide, kept as
bounding-box crops, so a crowd of small objects stays cheap in RAM and
time; the full-resolution masks are loaded only to build the result. A
mask too thin to survive the shrink keeps the pixels it touches there, so
the frame still counts as seen.
Everything here is numpy + OpenCV, with no model: tests run on synthetic
masks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

# Two tracks are compared only when they share at least this many frames.
MIN_SHARED = 3


@dataclass(frozen=True)
class MergeConfig:
    link_iou: float = 0.5
    conflict_iou: float = 0.3
    stitch_iou: float = 0.3
    stitch_gap: int = 8
    glitch_factor: float = 6.0  # 0 = no glitch filter
    glitch_min_jump: float = 0.05
    glitch_window: int = 0  # neighbouring frames either side; 0 = whole-track median
    dup_containment: float = 0.6
    dup_frames: float = 0.6
    dup_size_ratio: float = 0.5
    work_width: int = 480


def seed_frames(n: int, primary: int, every: int) -> list[int]:
    """Seed frames for `n` frames: `primary` first (its session keeps the
    single-seed IDs and order), then the first and last frame, then every
    `every` frames unless closer than `every // 2` to a seed already taken.
    `every` <= 0 = first, primary and last only."""
    if n <= 0:
        return []
    primary = min(max(int(primary), 0), n - 1)
    seeds = [primary]
    extra = [0, n - 1] + (list(range(every, n - 1, every)) if every > 0 else [])
    for s in extra:
        if s in seeds:
            continue
        if s in (0, n - 1) or all(abs(s - t) >= max(1, every // 2) for t in seeds):
            seeds.append(s)
    return seeds


def glitch_frames(cov: np.ndarray, factor: float, min_jump: float, window: int = 0) -> list[int]:
    """Frames where a track's coverage jumps far above the frames around it:
    with `window` > 0, above the median of the `window` present frames
    before it AND of those after it (one side at the clip's ends), so a
    subject that grows and stays grown is not a glitch; 0 = above the
    whole track's median."""
    if factor <= 0:
        return []
    present = np.flatnonzero(cov > 0)
    if len(present) < 2:
        return []

    def jumps(c: float, ref: np.ndarray) -> bool:
        med = float(np.median(cov[ref]))
        return c > factor * med and c > med + min_jump

    out = []
    for i, f in enumerate(present):
        c = float(cov[f])
        if window > 0:
            sides = [present[max(0, i - window) : i], present[i + 1 : i + 1 + window]]
            bad = all(jumps(c, side) for side in sides if len(side))
        else:
            bad = jumps(c, present)
        if bad:
            out.append(int(f))
    return out


# --- bounding-box crops ------------------------------------------------------


@dataclass
class Crop:
    """A binary mask stored as its bounding box: `m` sits at (y0, x0)."""

    y0: int
    x0: int
    m: np.ndarray  # bool
    area: int = 0

    def __post_init__(self) -> None:
        self.area = int(self.m.sum())

    @property
    def y1(self) -> int:
        return self.y0 + int(self.m.shape[0])

    @property
    def x1(self) -> int:
        return self.x0 + int(self.m.shape[1])


def crop_of(mask: np.ndarray) -> Crop | None:
    m = np.asarray(mask) > 0
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(m.any(0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return Crop(y0, x0, m[y0:y1, x0:x1].copy())


def intersection(a: Crop, b: Crop) -> int:
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if y0 >= y1 or x0 >= x1:
        return 0
    pa = a.m[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0]
    pb = b.m[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0]
    return int(np.logical_and(pa, pb).sum())


def iou(a: Crop, b: Crop) -> float:
    i = intersection(a, b)
    u = a.area + b.area - i
    return i / u if u else 0.0


def union(crops: Sequence[Crop]) -> Crop:
    if len(crops) == 1:
        return crops[0]
    y0, x0 = min(c.y0 for c in crops), min(c.x0 for c in crops)
    y1, x1 = max(c.y1 for c in crops), max(c.x1 for c in crops)
    out = np.zeros((y1 - y0, x1 - x0), bool)
    for c in crops:
        out[c.y0 - y0 : c.y1 - y0, c.x0 - x0 : c.x1 - x0] |= c.m
    return Crop(y0, x0, out)


def dilate(c: Crop, r: int) -> Crop:
    import cv2

    if r <= 0:
        return c
    pad = np.pad(c.m.astype(np.uint8), r)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return Crop(c.y0 - r, c.x0 - r, cv2.dilate(pad, ker) > 0)


# --- tracks --------------------------------------------------------------------


@dataclass
class RawTrack:
    """One object of one seed session, as the comparisons see it."""

    label: str
    session: int  # 0 = the primary seed
    seed: int
    obj_id: int
    load: Callable[[], np.ndarray] | None  # the full (T, H, W) uint8 stack
    small: dict[int, Crop]  # frame -> mask on the work copy (present frames only)
    glitches: list[int] = field(default_factory=list)


def make_track(
    stack: np.ndarray,
    label: str,
    session: int,
    seed: int,
    obj_id: int,
    load: Callable[[], np.ndarray],
    cfg: MergeConfig,
) -> RawTrack | None:
    """The comparison copy of one raw track (None if it is empty after
    the glitch filter). `load` gives the full stack back for assembly."""
    import cv2

    n, h, w = stack.shape
    cov = stack.reshape(n, -1).astype(bool).mean(1)
    bad = glitch_frames(cov, cfg.glitch_factor, cfg.glitch_min_jump, cfg.glitch_window)
    s = min(1.0, cfg.work_width / float(w)) if cfg.work_width > 0 else 1.0
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    small: dict[int, Crop] = {}
    for f in np.flatnonzero(cov > 0):
        if int(f) in bad:
            continue
        m = stack[f] > 0
        if s < 1.0:
            cov_s = cv2.resize(m.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
            m = cov_s >= 0.5
            if not m.any():  # too thin for the shrink: keep the pixels it touches
                m = cov_s > 0
        c = crop_of(m)
        if c is not None:
            small[int(f)] = c
    if not small:
        return None
    return RawTrack(label, session, seed, obj_id, load, small, bad)


@dataclass
class MergeReport:
    raw_tracks: int = 0
    objects: int = 0
    linked: int = 0
    stitched: int = 0
    duplicates: int = 0
    glitch_frames: int = 0
    merges: list[tuple[str, float, tuple[int, int], tuple[int, int]]] = field(default_factory=list)


def merge_tracks(
    tracks: Sequence[RawTrack], cfg: MergeConfig
) -> tuple[list[list[int]], MergeReport]:
    """Clusters of `tracks` indices, one per object, in output order: per
    label (first-seen order), clusters holding a primary-session track
    first (by its id), then by first frame."""
    rep = MergeReport(raw_tracks=len(tracks), glitch_frames=sum(len(t.glitches) for t in tracks))
    labels: list[str] = []
    for t in tracks:
        if t.label not in labels:
            labels.append(t.label)
    out: list[list[int]] = []
    for lb in labels:
        out += _merge_label([i for i, t in enumerate(tracks) if t.label == lb], tracks, cfg, rep)
    rep.objects = len(out)
    return out, rep


def _merge_label(
    idx: list[int], tracks: Sequence[RawTrack], cfg: MergeConfig, rep: MergeReport
) -> list[list[int]]:
    # Pairwise (mean IoU, shared frames) of the raw tracks.
    pair: dict[tuple[int, int], tuple[float, int]] = {}
    for a_i, i in enumerate(idx):
        for j in idx[a_i + 1 :]:
            shared = tracks[i].small.keys() & tracks[j].small.keys()
            if shared:
                ious = [iou(tracks[i].small[f], tracks[j].small[f]) for f in shared]
                pair[i, j] = (float(np.mean(ious)), len(shared))

    def stats(i: int, j: int) -> tuple[float, int]:
        return pair.get((min(i, j), max(i, j)), (0.0, 0))

    def conflict(i: int, j: int) -> bool:
        m, k = stats(i, j)
        return k >= MIN_SHARED and m < cfg.conflict_iou

    clusters: dict[int, set[int]] = {i: {i} for i in idx}
    of = {i: i for i in idx}

    def join(a: int, b: int, why: str, score: float, force: bool = False) -> bool:
        ca, cb = of[a], of[b]
        if ca == cb:
            return False
        if not force and any(conflict(x, y) for x in clusters[ca] for y in clusters[cb]):
            return False
        clusters[ca] |= clusters.pop(cb)
        for x in clusters[ca]:
            of[x] = ca
        ta, tb = tracks[a], tracks[b]
        rep.merges.append((why, round(score, 3), (ta.seed, ta.obj_id), (tb.seed, tb.obj_id)))
        return True

    # 1. The same object seen by different seed sessions.
    for (i, j), (m, k) in sorted(pair.items(), key=lambda kv: -kv[1][0]):
        if m < cfg.link_iou:
            break
        if k >= MIN_SHARED and tracks[i].session != tracks[j].session and join(i, j, "link", m):
            rep.linked += 1

    def frames_of(c: int) -> set[int]:
        return set().union(*(tracks[t].small.keys() for t in clusters[c]))

    def at(c: int, f: int) -> Crop | None:
        ms = [tracks[t].small[f] for t in clusters[c] if f in tracks[t].small]
        return union(ms) if ms else None

    # 2. Fragments: one cluster ends, another starts just after, in its place.
    r = max(1, cfg.work_width // 60)
    while cfg.stitch_gap > 0:
        spans = {c: (min(fr), max(fr)) for c in clusters if (fr := frames_of(c))}
        cand = []
        for a, (_a0, a1) in spans.items():
            for b, (b0, _b1) in spans.items():
                if a != b and 0 < b0 - a1 <= cfg.stitch_gap:
                    ea, sb = at(a, a1), at(b, b0)
                    if ea is not None and sb is not None:
                        s = iou(dilate(ea, r), dilate(sb, r))
                        if s >= cfg.stitch_iou:
                            cand.append((s, a, b))
        done = False
        for s, a, b in sorted(cand, reverse=True):
            if join(next(iter(clusters[a])), next(iter(clusters[b])), "stitch", s):
                rep.stitched += 1
                done = True
                break
        if not done:
            break

    # 3. Duplicates: comparable size, one inside the other, on most shared frames.
    def contained(a: int, b: int) -> float:
        shared = frames_of(a) & frames_of(b)
        if len(shared) < MIN_SHARED:
            return 0.0
        hit = 0
        for f in shared:
            ma, mb = at(a, f), at(b, f)
            if ma is None or mb is None:
                continue
            lo, hi = min(ma.area, mb.area), max(ma.area, mb.area)
            if lo and lo / hi >= cfg.dup_size_ratio:
                if intersection(ma, mb) / lo >= cfg.dup_containment:
                    hit += 1
        return hit / len(shared)

    if cfg.dup_frames > 0:
        score = {(a, b): contained(a, b) for a in clusters for b in clusters if a < b}
        while score:
            (a, b), s = max(score.items(), key=lambda kv: kv[1])
            if s < cfg.dup_frames:
                break
            join(next(iter(clusters[a])), next(iter(clusters[b])), "duplicate", s, force=True)
            rep.duplicates += 1
            keep = of[a]
            score = {k: v for k, v in score.items() if a not in k and b not in k}
            for c in clusters:
                if c != keep:
                    score[min(c, keep), max(c, keep)] = contained(c, keep)

    def order(c: int) -> tuple[int, int]:
        prim = [tracks[t].obj_id for t in clusters[c] if tracks[t].session == 0]
        return (0, min(prim)) if prim else (1, min(frames_of(c)))

    return [
        sorted(clusters[c], key=lambda t: (tracks[t].session, tracks[t].obj_id))
        for c in sorted(clusters, key=order)
    ]


def _load(t: RawTrack) -> np.ndarray:
    if t.load is None:
        raise ValueError("this track keeps no full-resolution masks to assemble")
    return t.load()


def best_member(tracks: Sequence[RawTrack], own: Sequence[int], f: int, cfg: MergeConfig) -> int:
    """Of a cluster's members present on frame `f` (`own`, priority order),
    the one whose mask the object takes there: the first, or a bigger
    member that contains it (its whole)."""
    best = own[0]
    for t in own[1:]:
        mb, mt = tracks[best].small[f], tracks[t].small[f]
        if mt.area > 1.5 * mb.area and mb.area:
            if intersection(mb, mt) / mb.area >= cfg.dup_containment:
                best = t
    return best


def assemble(tracks: Sequence[RawTrack], members: Sequence[int], cfg: MergeConfig) -> np.ndarray:
    """One full-resolution (T, H, W) uint8 stack for a cluster (members in
    priority order, as `merge_tracks` returns them)."""
    full = {t: _load(tracks[t]) for t in members}
    first = full[members[0]]
    out = np.zeros(first.shape, np.uint8)
    for f in range(first.shape[0]):
        own = [t for t in members if f in tracks[t].small]
        if not own:
            continue
        best = best_member(tracks, own, f, cfg)
        for t in own:
            if tracks[t].session == tracks[best].session:
                out[f] |= (full[t][f] > 0).astype(np.uint8)
    return out


__all__ = [
    "MIN_SHARED",
    "Crop",
    "MergeConfig",
    "MergeReport",
    "RawTrack",
    "assemble",
    "best_member",
    "glitch_frames",
    "make_track",
    "merge_tracks",
    "seed_frames",
]
