# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Holdout combine: an object's soft matte with its neighbours held out.

`holdout.<object> = A x (1 - B')`, per frame, where

- A = the object's soft matte: `mask_object.<object>` (refiner
  `object_channels` on), else its hero slot `matte.<r|g|b|a>`;
- B = the union of the OTHER objects' SAM masks (the object's own parts
  and its whole are never held out of it), closed by `close_px` (fills the
  cracks between fingers and strands) but never over the object's own SAM
  mask, grown by `grow_px` and feathered by `feather_px` -> B'.

Then a band-limited grey opening (`open_px` > 0): a grey-scale opening of
the result (erode then dilate, a disc of `open_px`) removes soft strips and
specks narrower than the disc that a matte leaves along a neighbour, but
ONLY within `open_band_px` of the neighbours and never within `protect_px`
of the object's own SAM mask (its hair and fingers are its edge, not a
strip). Outside that band the matte is untouched.

Every distance is a distance transform, so wide settings stay cheap.

Opt-in: add `holdout` to the job's post list. It writes new channels
(`holdout.<object>`) next to the mattes it reads and changes nothing else.
It runs after the smoothers and the matte layering, on the final mattes.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from live_action_aov.io.channels import object_channel
from live_action_aov.passes.matte.boxmask import stack_box

_log = logging.getLogger(__name__)

HOLDOUT_LAYER = "holdout"
_SLOTS = ("r", "g", "b", "a")


def _dist_outside(mask: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to `mask` (0 inside)."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    if not m.any():
        return np.full(m.shape, np.inf, np.float32)
    return cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def grow(mask: np.ndarray, px: float) -> np.ndarray:
    """`mask` grown by a disc of `px` (distance transform). Bool."""
    m = np.asarray(mask) > 0
    if px <= 0 or not m.any():
        return m
    return _dist_outside(m) <= float(px)


def shrink(mask: np.ndarray, px: float) -> np.ndarray:
    """`mask` shrunk by a disc of `px` (distance transform). Bool."""
    m = np.asarray(mask) > 0
    if px <= 0 or not m.any():
        return m
    return ~grow(~m, px)


def close(mask: np.ndarray, px: float) -> np.ndarray:
    """Morphological closing by a disc of `px`: gaps narrower than 2 x px
    filled, outline otherwise kept. Bool."""
    if px <= 0:
        return np.asarray(mask) > 0
    return shrink(grow(mask, px), px)


def grey_open(a: np.ndarray, px: int) -> np.ndarray:
    """Grey-scale opening of `a` by a disc of radius `px`."""
    import cv2

    if px <= 0:
        return np.asarray(a, np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.morphologyEx(np.asarray(a, np.float32), cv2.MORPH_OPEN, k)


def holdout_combine(
    a: np.ndarray,
    b: np.ndarray,
    *,
    close_px: float = 0.0,
    grow_px: float = 0.0,
    feather_px: float = 0.0,
    protect: np.ndarray | None = None,
    protect_px: float = 0.0,
    open_px: int = 0,
    open_band_px: float = 0.0,
) -> np.ndarray:
    """`a x (1 - B')` with B' = `b` closed / grown / feathered, then the
    band-limited grey opening (see the module doc). `a`: (h, w) soft matte;
    `b`: (h, w) neighbour mask; `protect`: (h, w) the object's own mask
    (None = no protect zone). Returns float32 (h, w) in [0, 1]."""
    import cv2

    a = np.clip(np.asarray(a, np.float32), 0.0, 1.0)
    bm = np.asarray(b) > 0
    if not bm.any():
        return a.copy()
    closed = close(bm, close_px)
    if protect is not None and np.asarray(protect).any():
        # The closing fills cracks of the NEIGHBOUR, never the object's own
        # pixels (fingers spread over a face: the gaps between them close
        # over the fingers otherwise). Only `grow_px` crosses into it: the seam.
        closed = (closed & ~(np.asarray(protect) > 0)) | bm
    hard = grow(closed, grow_px).astype(np.float32)
    soft = cv2.GaussianBlur(hard, (0, 0), float(feather_px)) if feather_px > 0 else hard
    out = a * (1.0 - np.clip(soft, 0.0, 1.0))
    if open_px > 0 and open_band_px > 0:
        band = _dist_outside(bm) <= float(open_band_px)
        if protect is not None and np.asarray(protect).any():
            band &= _dist_outside(np.asarray(protect) > 0) > float(protect_px)
        if band.any():
            out = np.where(band, np.minimum(out, grey_open(out, int(open_px))), out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


class HoldoutPost:
    """Post-processor: `holdout.<object>` per object (module doc)."""

    name = "holdout"
    algorithm = "holdout_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "enabled": True,
        # Which objects get a holdout channel: "heroes" (the matte.rgba
        # slots) or "all" (every object with a `mask_object.*` channel).
        "objects": "heroes",
        # Plate px. Generic defaults: a light hold-out with no opening;
        # widen them per shot.
        "close_px": 3.0,
        "grow_px": 1.0,
        "feather_px": 1.5,
        "protect_px": 8.0,
        "open_px": 0,  # 0 = no grey opening
        "open_band_px": 0.0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(self.DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self._tracks: dict[int, dict[str, Any]] = {}
        self._slot_track: dict[str, int] = {}

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        if hard:
            self._tracks = {int(k): v for k, v in (next(iter(hard.values())) or {}).items()}
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            for h in next(iter(heroes.values())) or []:
                slot = str(h.get("slot", ""))
                if slot in _SLOTS:
                    self._slot_track[slot] = int(h["track_id"])

    def _family(self, tid: int) -> set[int]:
        """The object itself, its whole and its parts: never held out of it."""
        fam = {tid}
        parent = self._tracks.get(tid, {}).get("parent")
        root = int(parent) if parent is not None and int(parent) in self._tracks else tid
        fam.add(root)
        for t, tr in self._tracks.items():
            p = tr.get("parent")
            if p is not None and int(p) == root:
                fam.add(int(t))
        return fam

    def _sources(self, chans: dict[str, Any]) -> list[tuple[int, str, str]]:
        """(track id, input channel, output name) per object to hold out."""
        out: list[tuple[int, str, str]] = []
        seen: set[int] = set()
        for tid, tr in sorted(self._tracks.items()):
            name = f"{tr.get('label') or 'object'}_{tid}"
            ch = object_channel(name)
            if ch in chans and (
                str(self.params.get("objects", "heroes")) == "all"
                or tid in self._slot_track.values()
            ):
                out.append((tid, ch, ch.partition(".")[2]))
                seen.add(tid)
        for slot, tid in self._slot_track.items():
            ch = f"matte.{slot}"
            if tid in seen or ch not in chans or tid not in self._tracks:
                continue
            name = f"{self._tracks[tid].get('label') or 'object'}_{tid}"
            out.append((tid, ch, object_channel(name).partition(".")[2]))
        return out

    def apply(
        self,
        per_frame_channels: dict[int, dict[str, np.ndarray]],
        flow_cache: Any,
        shot_id: str,
        **_kwargs: Any,
    ) -> dict[int, dict[str, np.ndarray]]:
        del flow_cache, shot_id
        if not bool(self.params.get("enabled", True)) or len(self._tracks) < 2:
            return per_frame_channels
        frames_of = {
            tid: {int(f): k for k, f in enumerate(tr.get("frames") or [])}
            for tid, tr in self._tracks.items()
        }

        def hard(tid: int, f: int, shape: tuple[int, int]) -> np.ndarray | None:
            k = frames_of.get(tid, {}).get(f)
            if k is None:
                return None
            bm = stack_box(self._tracks[tid].get("stack"), k)
            if not bm.crop.size or not bm.crop.any():
                return None
            m: np.ndarray = np.zeros(bm.shape, np.uint8)
            y0, x0, y1, x1 = bm.box
            m[y0:y1, x0:x1] = bm.crop
            if m.shape != shape:
                import cv2

                m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            return m

        p = self.params
        written = 0
        for f, chans in per_frame_channels.items():
            for tid, src, name in self._sources(chans):
                a = np.asarray(chans[src], np.float32)
                shape = (int(a.shape[0]), int(a.shape[1]))
                fam = self._family(tid)
                b = np.zeros(shape, np.uint8)
                for other in self._tracks:
                    if other in fam:
                        continue
                    m = hard(other, int(f), shape)
                    if m is not None:
                        b |= m
                own = hard(tid, int(f), shape)
                chans[f"{HOLDOUT_LAYER}.{name}"] = holdout_combine(
                    a,
                    b,
                    close_px=float(p["close_px"]),
                    grow_px=float(p["grow_px"]),
                    feather_px=float(p["feather_px"]),
                    protect=own,
                    protect_px=float(p["protect_px"]),
                    open_px=int(p["open_px"]),
                    open_band_px=float(p["open_band_px"]),
                )
                written += 1
        if written:
            _log.info("holdout: %d object-frame(s) written to the holdout layer", written)
        return per_frame_channels


__all__ = [
    "HOLDOUT_LAYER",
    "HoldoutPost",
    "close",
    "grey_open",
    "grow",
    "holdout_combine",
    "shrink",
]
