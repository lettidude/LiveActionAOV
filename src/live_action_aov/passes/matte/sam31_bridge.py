# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""SAM 3.1 Object Multiplex as the matte pass's detector + tracker.

SAM 3.1 (`facebook/sam3.1`, gated) tracks every instance of a concept
jointly in one session, so it keeps identities apart where SAM 3 can swap
two similar objects. It ships as Meta's
`sam3` package for Python 3.12, not through transformers, so it runs in
its own environment: this module writes the frames as JPEGs, runs
`sam31_worker.py` with that environment's interpreter, and reads back one
(T, H, W) uint8 mask stack per object. LiveActionAOV's own post-processing
(one owner per pixel, seams, depth layer order, hero ranking) is unchanged.

The interpreter: the `sam31_python` param, else the `LAOV_SAM31_PYTHON`
environment variable, else the folder `liveaov setup-sam31` installs into
(`default_dir()`). Setup: docs/install.md, "SAM 3.1".
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.work_dir import make_work_dir

_log = logging.getLogger(__name__)

WORKER = Path(__file__).with_name("sam31_worker.py")
ENV_PYTHON = "LAOV_SAM31_PYTHON"


@dataclass
class Sam31Object:
    label: str
    obj_id: int
    stack: np.ndarray  # (T, H, W) uint8


def default_dir() -> Path:
    """Where `liveaov setup-sam31` puts the environment when no folder is
    given: per user, outside the install, so an update never removes it."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "LiveActionAOV" / "sam31"


def env_python(root: Path) -> Path:
    """The interpreter of the environment inside a SAM 3.1 folder."""
    sub = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    return Path(root, "env", *sub)


def find_python(params: dict[str, Any]) -> str:
    """The SAM 3.1 environment's interpreter, or a RuntimeError saying how
    to set it up."""
    exe = str(params.get("sam31_python") or os.environ.get(ENV_PYTHON, "") or "")
    if exe and Path(exe).is_file():
        return exe
    if not exe and env_python(default_dir()).is_file():
        return str(env_python(default_dir()))
    where = f"'{exe}' does not exist" if exe else "it is not installed"
    raise RuntimeError(
        f"SAM 3.1 detector: {where}. Run `liveaov setup-sam31` (it builds the Python 3.12 "
        f"environment, see docs/install.md 'SAM 3.1'), or point the `sam31_python` param / "
        f"the {ENV_PYTHON} environment variable at an existing one."
    )


def available(params: dict[str, Any] | None = None) -> tuple[bool, str]:
    """(usable, the interpreter or the reason it is not)."""
    try:
        return True, find_python(params or {})
    except RuntimeError as e:
        return False, str(e)


def run_sam31(
    frames: np.ndarray,
    concepts: list[str],
    seed_local: int,
    python: str,
    timeout_s: float = 3600.0,
    checkpoint: str = "",
) -> list[Sam31Object]:
    """Track `concepts` through `frames` ((T, H, W, 3) display-referred
    0..1) with SAM 3.1. Masks come back at the frames' resolution."""
    from PIL import Image

    work = make_work_dir("laov_sam31_")
    try:
        fdir, odir = work / "frames", work / "out"
        fdir.mkdir()
        for i, f in enumerate(frames):
            Image.fromarray(
                (np.clip(f[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB"
            ).save(fdir / f"{i:05d}.jpg", quality=95)
        cmd = [
            python,
            "-I",  # isolated: no LiveActionAOV env vars / user site in the 3.12 env
            str(WORKER),
            "--frames",
            str(fdir),
            "--out",
            str(odir),
            "--concepts",
            json.dumps([str(c) for c in concepts]),
            "--seed",
            str(int(seed_local)),
        ]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
        env["PYTHONUTF8"] = "1"
        _log.info(
            "SAM 3.1: %d frames, concepts %s, seed frame %d", len(frames), concepts, seed_local
        )
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, env=env, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SAM 3.1 took over {timeout_s:.0f} s") from e
        for line in (proc.stdout or "").splitlines():
            _log.info("SAM 3.1: %s", line)
        if proc.returncode != 0:
            raise RuntimeError(
                f"SAM 3.1 worker failed (exit {proc.returncode}): {(proc.stderr or '')[-1500:]}"
            )
        return read_results(odir)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def read_results(out_dir: Path) -> list[Sam31Object]:
    res = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    out = []
    for o in res["objects"]:
        with np.load(out_dir / o["file"]) as z:
            out.append(Sam31Object(str(o["concept"]), int(o["obj_id"]), z["mask"].astype(np.uint8)))
    return out


def keep_object(stack: np.ndarray, min_peak_coverage: float, min_frames: int) -> bool:
    """Drop SAM 3.1's false hits: an object must cover `min_peak_coverage`
    of the frame on its biggest frame (shelf items, posters: under 0.2 %)
    and be present on at least `min_frames` frames."""
    cov = stack.reshape(stack.shape[0], -1).mean(1)
    return bool(cov.max() >= min_peak_coverage and (cov > 0).sum() >= min_frames)


# --- small objects (the matte pass's `small_objects` on) ---------------------


@dataclass
class Sam31Track:
    """One SAM 3.1 object of the small-objects run, on the pass's frames:
    `masks[k]` is frame k as a BoxMask of (h, w)."""

    label: str
    obj_id: int
    src: int  # 0 = the whole frame, 1.. = tile number
    seed: int  # local seed frame of its session
    masks: list[Any]
    session: int = 0  # 0 = the primary session (the plate on the first seed)

    def present(self) -> list[int]:
        return [k for k, m in enumerate(self.masks) if m.crop.size]


def tile_windows(h: int, w: int, grid: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """(y0, x0, y1, x1) of `grid` x `grid` overlapping tiles: SAM 3's tiled
    detection layout, with the last row / column reaching the plate edge."""
    if grid <= 1:
        return []
    th, tw = h / (grid - (grid - 1) * overlap), w / (grid - (grid - 1) * overlap)
    out = []
    for iy in range(grid):
        for ix in range(grid):
            y0, x0 = int(iy * th * (1 - overlap)), int(ix * tw * (1 - overlap))
            y1 = h if iy == grid - 1 else min(h, int(y0 + th))
            x1 = w if ix == grid - 1 else min(w, int(x0 + tw))
            out.append((y0, x0, y1, x1))
    return out


def _fit(img: np.ndarray, long_side: int) -> np.ndarray:
    """`img` shrunk to `long_side` px on its long side (as is if smaller or 0)."""
    h, w = int(img.shape[0]), int(img.shape[1])
    if long_side <= 0 or max(h, w) <= long_side:
        return img
    import cv2

    s = long_side / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return cv2.resize(
        np.ascontiguousarray(img[..., :3], np.float32), size, interpolation=cv2.INTER_AREA
    )


def _save_jpg(img: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray((np.clip(img[..., :3], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB").save(
        path, quality=95
    )


def run_sam31_small(
    frames: np.ndarray,
    concepts: list[str],
    seeds: list[int],
    python: str,
    *,
    max_objects: int = 0,
    post_batch: int = 0,
    tiles: list[tuple[int, int, int, int]] | None = None,
    plate_frame: Any = None,
    plate_hw: tuple[int, int] | None = None,
    work_size: int = 0,
    timeout_s: float = 3600.0,
    checkpoint: str = "",
) -> list[Sam31Track]:
    """`run_sam31` for small subjects: every seed frame in `seeds` is its own
    session, `max_objects` lifts the model's object cap, and each window of
    `tiles` ((y0, x0, y1, x1) on the plate, `plate_hw`) is tracked as its
    own zoomed-in clip. Tile frames come from `plate_frame(k)` (the original
    plate in proxy mode) else from `frames`. `work_size` > 0: every clip
    SAM 3.1 sees (the frames, each tile) is shrunk to that long side first;
    its memory grows with objects x frames x clip size. Masks come back as
    box crops on the frames' (h, w), never as whole-clip stacks. `seeds[0]`
    is the primary seed: its session on the frames must work, any other
    session SAM 3.1 fails on is logged and skipped."""
    from live_action_aov.passes.matte.boxmask import BoxMask

    n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    tiles = list(tiles or [])
    ph, pw = plate_hw or (h, w)
    work = make_work_dir("laov_sam31_")
    try:
        fdir, odir = work / "frames", work / "out"
        fdir.mkdir()
        tdirs = [work / f"tile{i:02d}" for i in range(len(tiles))]
        for d in tdirs:
            d.mkdir()
        for i, f in enumerate(frames):
            _save_jpg(_fit(f, work_size), fdir / f"{i:05d}.jpg")
            if tiles:
                src = np.asarray(plate_frame(i), np.float32) if plate_frame is not None else f
                for d, (y0, x0, y1, x1) in zip(tdirs, tiles, strict=True):
                    _save_jpg(_fit(src[y0:y1, x0:x1], work_size), d / f"{i:05d}.jpg")
        cmd = [
            python,
            "-I",
            str(WORKER),
            "--frames",
            str(fdir),
            "--out",
            str(odir),
            "--concepts",
            json.dumps([str(c) for c in concepts]),
            "--seed",
            str(int(seeds[0]) if seeds else n // 2),
            "--seeds",
            json.dumps([int(s) for s in seeds]),
            "--crops",
        ]
        if max_objects > 0:
            cmd += ["--max-objects", str(int(max_objects))]
        if post_batch > 0:
            cmd += ["--post-batch", str(int(post_batch))]
        if tdirs:
            cmd += ["--tiles", json.dumps([str(d) for d in tdirs])]
        if checkpoint:
            cmd += ["--checkpoint", checkpoint]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
        env["PYTHONUTF8"] = "1"
        _log.info(
            "SAM 3.1 (small objects): %d frames, concepts %s, seed frames %s, %d tile(s), "
            "up to %s objects per session",
            n,
            concepts,
            seeds,
            len(tiles),
            max_objects or "the model's default",
        )
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, env=env, check=False
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"SAM 3.1 took over {timeout_s:.0f} s") from e
        for line in (proc.stdout or "").splitlines():
            _log.info("SAM 3.1: %s", line)
        if proc.returncode != 0:
            raise RuntimeError(
                f"SAM 3.1 worker failed (exit {proc.returncode}): {(proc.stderr or '')[-1500:]}"
            )
        res = json.loads((odir / "result.json").read_text(encoding="utf-8"))
        _log.info(
            "SAM 3.1: worker %s s, peak VRAM %s GB (torch allocated).",
            res.get("seconds"),
            res.get("peak_vram_gb"),
        )
        for ss in res.get("sessions", []):
            if "skipped" in ss:
                _log.warning(
                    "SAM 3.1: '%s' session (src %s, seed frame %s) skipped (SAM 3.1 error: %s); "
                    "the other sessions are used.",
                    ss.get("concept"),
                    ss.get("src"),
                    ss.get("seed"),
                    str(ss["skipped"])[-300:],
                )
        out: list[Sam31Track] = []
        for o in res["objects"]:
            src = int(o.get("src", 0))
            win = tiles[src - 1] if src > 0 else None
            with np.load(odir / o["file"]) as z:
                boxes, data, hw = z["boxes"], z["data"], (int(z["hw"][0]), int(z["hw"][1]))
            masks, pos = [], 0
            for k in range(n):
                y0, x0, y1, x1 = (int(v) for v in boxes[k])
                if y1 <= y0 or x1 <= x0:
                    masks.append(BoxMask((h, w)))
                    continue
                crop = data[pos : pos + (y1 - y0) * (x1 - x0)].reshape(y1 - y0, x1 - x0)
                pos += crop.size
                if win is None:  # whole frame: SAM 3.1 answers at the JPEG size = (h, w)
                    m = BoxMask.from_crop(hw, y0, x0, crop)
                    masks.append(m if hw == (h, w) else m.resized((h, w)))
                else:  # a tile of the plate: back to its window, placed, plate -> frames
                    wh = (win[2] - win[0], win[3] - win[1])
                    t = BoxMask.from_crop(hw, y0, x0, crop)
                    t = t if hw == wh else t.resized(wh)
                    m = BoxMask.from_crop((ph, pw), win[0] + t.y0, win[1] + t.x0, t.crop)
                    masks.append(m if (ph, pw) == (h, w) else m.resized((h, w)))
            out.append(
                Sam31Track(
                    str(o["concept"]),
                    int(o["obj_id"]),
                    src,
                    int(o["seed"]),
                    masks,
                    int(o.get("session", 0)),
                )
            )
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def keep_small(
    t: Sam31Track,
    min_height_px: float,
    min_peak_coverage: float,
    min_frames: int,
    seen_area: float = 0.0,
) -> bool:
    """The small-objects floor: present on `min_frames` frames, at least
    `min_height_px` tall (box height) and `min_peak_coverage` of the clip
    SAM 3.1 saw on its biggest frame. `seen_area`: that clip's area in
    frame px (a tile's window; 0 = the whole frame). A relative floor, not
    0.5 %: that one drops anyone shorter than ~1/5 of the frame."""
    pres = t.present()
    if len(pres) < min_frames:
        return False
    hw = t.masks[0].shape
    area = seen_area if seen_area > 0 else float(hw[0] * hw[1])
    peak = max(int(t.masks[k].sum()) for k in pres) / area
    tall = max(t.masks[k].height for k in pres)
    return bool(peak >= min_peak_coverage and tall >= min_height_px)


def _dup_of(
    a: Sam31Track, b: Sam31Track, contain: float, frames_frac: float, size_ratio: float
) -> bool:
    """`a` and `b` are the same object tracked twice: on at least
    `frames_frac` of the frames both are present, the smaller lies at least
    `contain` inside the other, and over those shared frames the smaller is
    on median at least `size_ratio` of the bigger (a person inside a "group"
    mask of four is contained but four times smaller: not a duplicate)."""
    from live_action_aov.passes.matte.boxmask import boxes_touch, inter_area

    shared = [k for k in range(len(a.masks)) if a.masks[k].crop.size and b.masks[k].crop.size]
    if not shared:
        return False
    hits, ratios = 0, []
    for k in shared:
        ma, mb = a.masks[k], b.masks[k]
        sa, sb = int(ma.sum()), int(mb.sum())
        lo, hi = min(sa, sb), max(sa, sb)
        ratios.append(lo / hi if hi else 0.0)
        if lo and boxes_touch(ma, mb) and inter_area(ma, mb) / lo >= contain:
            hits += 1
    return hits >= frames_frac * len(shared) and float(np.median(ratios)) >= size_ratio


def merge_duplicates(
    tracks: list[Sam31Track],
    contain: float = 0.6,
    frames_frac: float = 0.6,
    size_ratio: float = 0.5,
) -> tuple[list[Sam31Track], list[tuple[int, int]]]:
    """One track per object across seed sessions and tiles (same label only).

    Longest track first; each later track is compared with the kept ones
    on their ORIGINAL masks, never on a merged union, so merges cannot chain
    through a tight group. A duplicate only lends the kept track the frames
    where that one is empty (someone lost and found again by another seed).
    Returns (kept tracks, [(dropped index, kept index)])."""
    order = sorted(
        range(len(tracks)),
        key=lambda i: (-len(tracks[i].present()), -sum(int(m.sum()) for m in tracks[i].masks), i),
    )
    keep: list[int] = []
    merged: list[tuple[int, int]] = []
    lent: dict[int, list[Any]] = {}
    for i in order:
        t = tracks[i]
        hit = next(
            (
                k
                for k in keep
                if tracks[k].label == t.label
                and _dup_of(t, tracks[k], contain, frames_frac, size_ratio)
            ),
            None,
        )
        if hit is None:
            keep.append(i)
            continue
        merged.append((i, hit))
        cur = lent.setdefault(hit, list(tracks[hit].masks))
        for k, m in enumerate(t.masks):
            if m.crop.size and not cur[k].crop.size:
                cur[k] = m
    out = []
    for i in sorted(keep):
        t = tracks[i]
        out.append(Sam31Track(t.label, t.obj_id, t.src, t.seed, lent.get(i, t.masks)))
    return out, merged


def merge_sessions(
    tracks: list[Sam31Track], cfg: Any
) -> tuple[list[Sam31Track], Any, list[tuple[int, int, int]]]:
    """The tracks of several sessions (seed frames, tiles) merged into one
    track per object by `sam31_merge` (link across sessions, stitch
    fragments, duplicates of comparable size, glitch frames dropped), on
    box crops. Returns (merged tracks in output order, the merge report,
    per merged track (frames its members saw, glitch frames among them,
    frames it holds))."""
    from live_action_aov.passes.matte import sam31_merge
    from live_action_aov.passes.matte.boxmask import BoxMask

    raw: list[Any] = []
    keep: list[int] = []
    for i, t in enumerate(tracks):
        r = _raw_track(t, cfg)
        if r is not None:
            raw.append(r)
            keep.append(i)
    if not raw:
        return [], sam31_merge.MergeReport(raw_tracks=len(tracks)), []
    clusters, rep = sam31_merge.merge_tracks(raw, cfg)
    out: list[Sam31Track] = []
    frames: list[tuple[int, int, int]] = []
    for members in clusters:
        src = [tracks[keep[m]] for m in members]
        first = src[0]
        n = len(first.masks)
        masks: list[Any] = []
        for f in range(n):
            own = [m for m in members if f in raw[m].small]
            if not own:
                masks.append(BoxMask(first.masks[0].shape))
                continue
            best = sam31_merge.best_member(raw, own, f, cfg)
            parts = [tracks[keep[m]].masks[f] for m in own if raw[m].session == raw[best].session]
            masks.append(_union(parts))
        seen = set().union(*(set(t.present()) for t in src))
        bad = set().union(*(set(raw[m].glitches) for m in members))
        held = sum(1 for m in masks if m.crop.size)
        frames.append((len(seen), len(bad), held))
        out.append(
            Sam31Track(
                first.label,
                first.obj_id,
                min(t.src for t in src),
                first.seed,
                masks,
                first.session,
            )
        )
    return out, rep, frames


def _raw_track(t: Sam31Track, cfg: Any) -> Any:
    """`sam31_merge.make_track` for a track kept as box crops: glitch frames
    out, the comparison copy at `cfg.work_width` (a mask too thin for it
    keeps the pixels it touches)."""
    from live_action_aov.passes.matte import sam31_merge

    h, w = t.masks[0].shape
    cov = np.array([float(m.sum()) for m in t.masks]) / float(h * w)
    bad = sam31_merge.glitch_frames(cov, cfg.glitch_factor, cfg.glitch_min_jump, cfg.glitch_window)
    s = min(1.0, cfg.work_width / float(w)) if cfg.work_width > 0 else 1.0
    shape = (max(1, round(h * s)), max(1, round(w * s)))
    small = {}
    for k in t.present():
        if k in bad:
            continue
        m = t.masks[k] if s >= 1.0 else t.masks[k].resized(shape)
        if m.crop.size and m.crop.any():
            y0, x0, _, _ = m.box
            small[k] = sam31_merge.Crop(y0, x0, m.crop > 0)
    if not small:
        return None
    return sam31_merge.RawTrack(t.label, t.session, t.seed, t.obj_id, None, small, bad)


def _union(masks: list[Any]) -> Any:
    """One BoxMask holding every mask of `masks` (same plate)."""
    from live_action_aov.passes.matte.boxmask import BoxMask, union_box

    if len(masks) == 1:
        return masks[0]
    shape = masks[0].shape
    box = union_box(masks, 0, shape)
    if box is None:
        return BoxMask(shape)
    y0, x0, y1, x1 = box
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for m in masks:
        if m.crop.size:
            my0, mx0, my1, mx1 = m.box
            np.maximum(
                out[my0 - y0 : my1 - y0, mx0 - x0 : mx1 - x0],
                m.crop.astype(np.uint8),
                out=out[my0 - y0 : my1 - y0, mx0 - x0 : mx1 - x0],
            )
    return BoxMask.from_crop(shape, y0, x0, out)


__all__ = [
    "Sam31Object",
    "Sam31Track",
    "find_python",
    "keep_object",
    "keep_small",
    "merge_duplicates",
    "merge_sessions",
    "read_results",
    "run_sam31",
    "run_sam31_small",
    "tile_windows",
]
