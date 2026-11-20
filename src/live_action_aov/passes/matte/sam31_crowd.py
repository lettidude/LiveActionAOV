# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""SAM 3.1 crowd mode (`sam31_tiles` "on"): track clean-up between the
sessions and the output IDs. Pure functions on `sam31_bridge.Sam31Track`
(box-crop masks), no model.

- `absorb_tile_tracks`: a tile track that lies mostly inside a whole-frame
  track of the same label is a piece of that object cut by the tile (a big
  person split at a tile border): it is absorbed into the whole-frame track
  instead of becoming its own ID.
- `swap_guard`: a track whose mask jumps far AND changes size a lot from
  one frame to the next has most likely moved onto a neighbour (two small
  heads swapping identities). A short excursion that comes back is dropped;
  otherwise the track is split there, and split pieces that continue each
  other (a swap between two tracks) are joined back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- tile absorb -----------------------------------------------------------


@dataclass
class AbsorbReport:
    tile_tracks: int = 0
    absorbed: int = 0
    # (absorbed track index, whole-frame track index) into the input list
    pairs: list[tuple[int, int]] = field(default_factory=list)


def _geom(t: Any) -> tuple[np.ndarray, np.ndarray]:
    """(boxes (n, 4) y0 x0 y1 x1, areas (n,)) of a track; empty frames 0."""
    n = len(t.masks)
    boxes = np.zeros((n, 4), np.int64)
    areas = np.zeros(n, np.int64)
    for k, m in enumerate(t.masks):
        if m.crop.size:
            boxes[k] = m.box
            areas[k] = int(m.crop.sum())
    return boxes, areas


def _touches_cut(
    box: np.ndarray, win: tuple[int, int, int, int], hw: tuple[int, int], tol: int
) -> bool:
    """The box reaches an edge of its tile that is inside the frame (a cut
    made by the tile, not the frame's own edge)."""
    y0, x0, y1, x1 = (int(v) for v in box)
    wy0, wx0, wy1, wx1 = win
    h, w = hw
    return bool(
        (wy0 > 0 and y0 <= wy0 + tol)
        or (wx0 > 0 and x0 <= wx0 + tol)
        or (wy1 < h and y1 >= wy1 - tol)
        or (wx1 < w and x1 >= wx1 - tol)
    )


def absorb_tile_tracks(
    tracks: list[Any],
    tiles: list[tuple[int, int, int, int]],
    plate_hw: tuple[int, int],
    *,
    contain: float = 0.7,
    frames: float = 0.5,
    cut_frames: float = 0.25,
    size_ratio: float = 0.5,
) -> tuple[list[Any], AbsorbReport]:
    """Tile tracks (`src` > 0) absorbed into whole-frame tracks (`src` 0).

    A tile track T goes into the whole-frame track F of the same label when
    on at least `frames` of T's frames at least `contain` of T lies inside
    F, AND T is a piece of F: it reaches a cut edge of its tile on at least
    `cut_frames` of its frames, or it is on median at least `size_ratio` of
    F's size (F seen whole in one tile). A small person fully inside a tile
    next to (or inside a group mask of) a big one is neither: kept.
    `tiles` are (y0, x0, y1, x1) on the plate (`plate_hw`); the tracks' masks
    are on the pass's frames. Returns (kept tracks in input order, report)."""
    rep = AbsorbReport()
    if contain <= 0 or not tracks or not tiles:
        return list(tracks), rep
    from live_action_aov.passes.matte.boxmask import inter_area

    h, w = tracks[0].masks[0].shape
    sy, sx = h / float(plate_hw[0]), w / float(plate_hw[1])
    wins = [
        (round(y0 * sy), round(x0 * sx), round(y1 * sy), round(x1 * sx)) for y0, x0, y1, x1 in tiles
    ]
    geo = [_geom(t) for t in tracks]
    whole = [i for i, t in enumerate(tracks) if t.src == 0]
    drop: set[int] = set()
    for i, t in enumerate(tracks):
        if t.src <= 0 or t.src > len(wins):
            continue
        rep.tile_tracks += 1
        boxes, areas = geo[i]
        pres = np.flatnonzero(areas > 0)
        if pres.size == 0:
            continue
        win = wins[t.src - 1]
        tol = max(2, round(0.01 * min(win[2] - win[0], win[3] - win[1])))
        cut = sum(_touches_cut(boxes[k], win, (h, w), tol) for k in pres) >= cut_frames * pres.size
        best, best_hits = -1, 0
        for j in whole:
            f = tracks[j]
            if f.label != t.label:
                continue
            fb, fa = geo[j]
            both = pres[fa[pres] > 0]
            if both.size == 0:
                continue
            bt, bf = boxes[both], fb[both]
            ov = (np.maximum(bt[:, 0], bf[:, 0]) < np.minimum(bt[:, 2], bf[:, 2])) & (
                np.maximum(bt[:, 1], bf[:, 1]) < np.minimum(bt[:, 3], bf[:, 3])
            )
            if ov.sum() < frames * pres.size:
                continue
            hits, ratios = 0, []
            for k in both[ov]:
                if inter_area(t.masks[k], f.masks[k]) >= contain * areas[k]:
                    hits += 1
                ratios.append(areas[k] / max(1, fa[k]))
            if hits < frames * pres.size:
                continue
            if not cut and float(np.median(ratios)) < size_ratio:
                continue
            if hits > best_hits:
                best, best_hits = j, hits
        if best >= 0:
            drop.add(i)
            rep.pairs.append((i, best))
    rep.absorbed = len(drop)
    return [t for i, t in enumerate(tracks) if i not in drop], rep


# --- swap guard -------------------------------------------------------------


@dataclass
class SwapReport:
    tracks: int = 0
    events: int = 0  # frame pairs flagged (jump + size change)
    dropped_stretches: int = 0
    dropped_frames: int = 0
    splits: int = 0
    rejoined: int = 0
    # (track index, local frame where the flagged reading starts, action)
    log: list[tuple[int, int, str]] = field(default_factory=list)


def _stats(m: Any) -> tuple[float, float, float, float]:
    """(cy, cx, area, half the box's longer side) of a BoxMask; area 0 when
    empty."""
    if not m.crop.size:
        return 0.0, 0.0, 0.0, 0.0
    ys, xs = np.nonzero(m.crop)
    if ys.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    half = 0.5 * float(max(m.crop.shape))
    return m.y0 + float(ys.mean()), m.x0 + float(xs.mean()), float(ys.size), half


def _jumps(a: tuple[float, ...], b: tuple[float, ...], jump: float, ratio: float) -> bool:
    """Two readings of one ID that cannot be the same object one frame apart:
    the centroid moves more than `jump` x half the longer side of the bigger
    box (a part of an object, e.g. a person half hidden, keeps its centroid
    inside the whole's box: never flagged at 1.0) AND the area changes more
    than `ratio` times."""
    d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
    reach = max(1.0, a[3], b[3])
    r = max(a[2], b[2]) / max(1.0, min(a[2], b[2]))
    return d > jump * reach and r > ratio


def swap_guard(
    tracks: list[Any],
    *,
    jump: float = 1.0,
    area_ratio: float = 3.0,
    max_stretch: int = 8,
    max_gap: int = 2,
) -> tuple[list[Any], SwapReport]:
    """Guard against identity swaps between neighbours (see the module doc).

    Consecutive readings of a track (at most `max_gap` frames apart) that
    `_jumps` flags start a suspect stretch. If the track jumps back to where
    it was (same test, not flagged against the reading before the stretch)
    within `max_stretch` frames, the stretch is dropped. Otherwise the track
    is split there. Pieces that start right where another track's piece
    stopped (both from a split, the two readings not flagged) are joined:
    two neighbours that swapped get their own identities back.
    Returns (tracks: unchanged ones as is, split ones as their pieces, in
    input order), report)."""
    from live_action_aov.passes.matte.boxmask import BoxMask

    rep = SwapReport(tracks=len(tracks))
    if jump <= 0 or area_ratio <= 0:
        return list(tracks), rep
    # pieces: (track index, sorted frame list, ends_by_split, starts_by_split)
    pieces: list[list[Any]] = []
    stats: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for ti, t in enumerate(tracks):
        st = {k: _stats(m) for k, m in enumerate(t.masks) if m.crop.size}
        st = {k: v for k, v in st.items() if v[2] > 0}
        stats[ti] = st
        pres = sorted(st)
        cur: list[int] = []
        starts_split = False
        dropped: set[int] = set()
        i = 0
        while i < len(pres):
            k = pres[i]
            if cur and k - cur[-1] <= max_gap and _jumps(st[cur[-1]], st[k], jump, area_ratio):
                rep.events += 1
                anchor = st[cur[-1]]
                back = next(
                    (
                        j
                        for j in range(i + 1, len(pres))
                        if pres[j] - k <= max_stretch
                        and not _jumps(anchor, st[pres[j]], jump, area_ratio)
                    ),
                    None,
                )
                if back is not None:
                    gone = pres[i:back]
                    dropped.update(gone)
                    rep.dropped_stretches += 1
                    rep.dropped_frames += len(gone)
                    rep.log.append((ti, k, f"dropped {len(gone)} frame(s)"))
                    i = back
                    continue
                pieces.append([ti, cur, True, starts_split])
                rep.splits += 1
                rep.log.append((ti, k, "split"))
                cur, starts_split = [], True
            cur.append(k)
            i += 1
        if cur:
            pieces.append([ti, cur, False, starts_split])
        elif not pres:
            pieces.append([ti, [], False, False])
    # Join pieces across tracks: a piece ended by a split + a piece of
    # ANOTHER track started by a split right after it, compatible readings.
    joined: dict[int, int] = {}  # piece index -> the piece it continues
    used_tail: set[int] = set()
    for qi, q in enumerate(pieces):
        if not q[3] or not q[1]:
            continue
        s = q[1][0]
        best, best_d = None, None
        for pi, p in enumerate(pieces):
            if pi in used_tail or not p[2] or not p[1] or p[0] == q[0]:
                continue
            e = p[1][-1]
            if not (0 < s - e <= max_gap):
                continue
            a, b = stats[p[0]][e], stats[q[0]][s]
            if _jumps(a, b, jump, area_ratio):
                continue
            d = float(np.hypot(a[0] - b[0], a[1] - b[1]))
            if best_d is None or d < best_d:
                best, best_d = pi, d
        if best is not None:
            joined[qi] = best
            used_tail.add(best)
            rep.rejoined += 1
    # Chains: follow each head piece through its continuations.
    nxt = {v: k for k, v in joined.items()}
    out: list[Any] = []
    split_tracks = {p[0] for p in pieces if p[2] or p[3]} | {
        ti for ti, _k, why in rep.log if why.startswith("dropped")
    }
    emitted: set[int] = set()
    order = sorted(range(len(pieces)), key=lambda i: (pieces[i][0], pieces[i][1][:1]))
    for pi in order:
        if pi in emitted or pi in joined:
            continue
        ti = pieces[pi][0]
        t = tracks[ti]
        if ti not in split_tracks:
            out.append(t)
            emitted.add(pi)
            continue
        chain = [pi]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        n = len(t.masks)
        shape = t.masks[0].shape
        masks = [BoxMask(shape) for _ in range(n)]
        for c in chain:
            src = tracks[pieces[c][0]]
            for k in pieces[c][1]:
                masks[k] = src.masks[k]
            emitted.add(c)
        out.append(type(t)(t.label, t.obj_id, t.src, t.seed, masks, getattr(t, "session", 0)))
    return out, rep


__all__ = ["AbsorbReport", "SwapReport", "absorb_tile_tracks", "swap_guard"]
