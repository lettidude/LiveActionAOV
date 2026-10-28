# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Matte layering: combine per-object soft mattes like a comp, front to back.

Why: every object is refined ON ITS OWN (its own engine, its own edge band).
Where two objects touch - a person in front of a car - both refiners see the
contact line as an edge and BOTH alphas fall to ~0.5 there. Added together in
comp that is a dark line along the seam (field report, PF0052 garage: the
alpha sum in the person/car contact zone averaged 0.74).

A compositor would roto the car THROUGH the person and hold it out with the
person's matte. This post does exactly that, per frame:

1. Objects are taken front to back (`layer` rank published by SAM 3: user
   clicks first, then the smaller object - what is held or in front).
2. The front object keeps its own refined alpha: its edge (hair, fingers)
   defines the seam.
3. A back object is FILLED near the contact - inside its own hard mask where
   it is closer to the front object than to real background, and under the
   front object where that is closer to the back object than to background -
   then held out: `alpha_back = max(alpha_back, fill) * (1 - cover)`, where
   `cover` is the union of everything in front.

Result: along a contact the two alphas sum to exactly 1 (front edge + its
complement), the sum never exceeds 1 anywhere, and edges against real
background are untouched. Two tracks with the same label (two pieces of one
car) get the same fill across their shared seam.

Runs after the temporal smoother (it is appended last), reads the hard masks
and depth ranks from the `sam3_hard_masks` artifact and the hero slots from
`sam3_instances`. Applies to every `mask*.<label>` and `matte*.r/g/b/a` layer
group separately (compare-mode suffixed layers are layered among themselves).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)

_SLOTS = ("r", "g", "b", "a")


class MatteLayeringPost:
    name = "matte_layering"
    algorithm = "matte_layering_v1"

    DEFAULT_PARAMS: dict[str, Any] = {
        "enabled": True,
        # How far from a contact line the back object is filled under the
        # front one. Must cover the front object's soft edge (hair over the
        # car). "auto" = 1% of the long edge (10 px at 1024, 20 px at 2K).
        "seam_px": "auto",
        # Fill an unowned contact pixel behind the front object only where
        # the front alpha is at least this (its hair/edge is really there).
        "gap_cover_min": 0.1,
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
            self._tracks = dict(next(iter(hard.values())) or {})
        heroes = artifacts.get("sam3_instances") or {}
        if heroes:
            for h in next(iter(heroes.values())) or []:
                slot = str(h.get("slot", ""))
                if slot in _SLOTS:
                    self._slot_track[slot] = int(h["track_id"])

    # ------------------------------------------------------------------

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
        # Per-frame lookup of each track's hard mask.
        dense: dict[int, dict[int, np.ndarray]] = {}
        for tid, tr in self._tracks.items():
            st = np.asarray(tr.get("stack"))
            dense[int(tid)] = {int(f): st[k] for k, f in enumerate(tr.get("frames") or [])}
        rank = {
            int(tid): int(tr.get("layer", i)) for i, (tid, tr) in enumerate(self._tracks.items())
        }
        label_tracks: dict[str, list[int]] = {}
        for tid, tr in self._tracks.items():
            label_tracks.setdefault(str(tr.get("label", "")), []).append(int(tid))

        touched = 0
        for f, chans in per_frame_channels.items():
            groups: dict[str, list[tuple[str, list[int]]]] = {}
            for name in chans:
                layer, _, chan = name.partition(".")
                if layer.startswith("mask") and chan in label_tracks:
                    groups.setdefault(layer, []).append((name, label_tracks[chan]))
                elif layer.startswith("matte") and chan in self._slot_track:
                    groups.setdefault(layer, []).append((name, [self._slot_track[chan]]))
            for members in groups.values():
                if len(members) < 2 and not any(len(t) > 1 for _, t in members):
                    continue
                if self._layer_frame(chans, members, dense, rank, int(f)):
                    touched += 1
        if touched:
            _log.info("matte layering: resolved shared seams on %d frame-groups", touched)
        return per_frame_channels

    # ------------------------------------------------------------------

    def _layer_frame(
        self,
        chans: dict[str, np.ndarray],
        members: list[tuple[str, list[int]]],
        dense: dict[int, dict[int, np.ndarray]],
        rank: dict[int, int],
        f: int,
    ) -> bool:
        import cv2

        shape = np.asarray(chans[members[0][0]]).shape[:2]
        h, w = int(shape[0]), int(shape[1])
        sp = self.params.get("seam_px", "auto")
        seam = max(3, round(0.01 * max(h, w))) if str(sp) == "auto" else int(sp)
        # Only a WIDE empty area is real background. Thin unowned cracks and
        # pinholes between strands of hair (SAM gives them to nobody) are
        # contact, not background: an opening removes them from `bg`.
        crack = max(2, round(0.004 * max(h, w)))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * crack + 1, 2 * crack + 1))

        def bg_dist(occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            bg = cv2.morphologyEx((1 - occupied).astype(np.uint8), cv2.MORPH_OPEN, k_open)
            return bg, cv2.distanceTransform((1 - bg).astype(np.uint8), cv2.DIST_L2, 3)

        def hard_of(tid: int) -> np.ndarray | None:
            m = dense.get(tid, {}).get(f)
            if m is None:
                return None
            b: np.ndarray = (np.asarray(m) > 0.5).astype(np.uint8)
            if b.shape != (h, w):
                b = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST)
            return b if b.any() else None

        objs: list[tuple[int, str, np.ndarray, list[np.ndarray]]] = []
        for name, tids in members:
            parts_present: list[tuple[int, np.ndarray]] = []
            for t in tids:
                hm = hard_of(t)
                if hm is not None:
                    parts_present.append((t, hm))
            if not parts_present:
                continue
            union: np.ndarray = np.zeros((h, w), np.uint8)
            for _, m in parts_present:
                union = union | m
            objs.append(
                (
                    min(rank.get(t, 1 << 20) for t, _ in parts_present),
                    name,
                    union,
                    [m for _, m in parts_present],
                )
            )
        if not objs:
            return False
        objs.sort(key=lambda o: o[0])

        def dist_to(mask: np.ndarray) -> np.ndarray:
            # Distance of every pixel to the nearest pixel OF `mask`.
            return cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)

        cover = np.zeros((h, w), np.float32)
        front = np.zeros((h, w), np.uint8)
        changed = False
        for _, name, hard, parts in objs:
            a = np.clip(np.asarray(chans[name], dtype=np.float32), 0.0, 1.0)
            fill = np.zeros((h, w), bool)
            # Seams between pieces of the SAME object (two "car" tracks).
            if len(parts) > 1:
                _, d_bg = bg_dist(hard)
                for m in parts:
                    others = hard & (1 - m)
                    if not others.any():
                        continue
                    d_o = dist_to(others)
                    fill |= (hard > 0) & (d_o <= seam) & (d_o < d_bg)
            # Seams against everything in front of this object.
            if front.any():
                union = hard | front
                bg, d_bg = bg_dist(union)
                d_front = dist_to(front)
                if float(d_front[hard > 0].min()) <= seam:
                    d_self = dist_to(hard)
                    fill |= (hard > 0) & (d_front <= seam) & (d_front < d_bg)
                    fill |= (front > 0) & (d_self <= seam) & (d_self < d_bg)
                    # Pixels NEITHER hard mask owns, between the two objects,
                    # where the front object's soft edge says something of
                    # it is there (hair over the car): what shows through is
                    # this object, not background. Real background gaps have
                    # no front alpha and stay empty.
                    gap = (union == 0) & (d_front <= seam) & (d_self <= seam)
                    fill |= gap & (bg == 0)  # cracks/pinholes in the contact
                    fill |= gap & (cover > float(self.params.get("gap_cover_min", 0.1)))
            if fill.any():
                a = np.maximum(a, fill.astype(np.float32))
            if cover.any():
                a = a * (1.0 - cover)
            out = a.astype(np.float32, copy=False)
            if not np.array_equal(out, chans[name]):
                changed = True
            chans[name] = out
            cover = np.minimum(cover + out, 1.0)
            front |= hard
        return changed


__all__ = ["MatteLayeringPost"]
