# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Camera track check on a made-up clip. For CI and for a first day on a
new machine: no plate and no model download needed.

    uv sync --extra dev --extra camera
    uv run python scripts/camera_check.py --out camera_check_out

It draws a short clip (a few textured cards in front of a wall, the camera
sliding sideways), loads PyTorch, then runs `liveaov run-shot` with the
camera pass in this same process, like a real run that mixes the camera
with model passes.

The check passes when the run finishes and a camera file is written. When
the camera solver is kept out of this process (`colmap_isolate`, the
default on a Mac) it also has to stay out: loading pycolmap here is then an
error.

    --isolate auto   the pass's own default for this machine (default)
    --isolate on     keep pycolmap out of this process
    --isolate off    pycolmap in this process
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.machinery
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

WIDTH, HEIGHT, FOCAL_PX = 960, 540, 820.0
FIRST_FRAME, FRAME_COUNT = 1001, 16


# ---------------------------------------------------------------------------
# The clip
# ---------------------------------------------------------------------------


def _texture(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A grey image with detail at every size, values 0.1 .. 0.9."""
    import cv2

    img = np.zeros((h, w), np.float32)
    for cells, gain in ((6, 1.0), (24, 0.7), (96, 0.5), (384, 0.3)):
        gw, gh = max(2, cells), max(2, round(cells * h / w))
        coarse = rng.random((gh, gw), dtype=np.float32)
        img += gain * cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(60):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(max(3, w // 80), max(6, w // 14)))
        tone = float(rng.uniform(0.0, 2.5))
        if rng.random() < 0.5:
            cv2.circle(img, (x, y), r, tone, -1, lineType=cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x - r, y - r // 2), (x + r, y + r // 2), tone, -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    lo, hi = float(img.min()), float(img.max())
    return 0.1 + 0.8 * (img - lo) / max(hi - lo, 1e-6)


def _scene(rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray, tuple[float, ...]]]:
    """Flat cards as (four corners in metres, texture, RGB tint), far to
    near. Axes: x right, y down, z away from the camera."""
    floor_y = 1.6
    cards = [
        (  # back wall
            np.array([[-14, -7, 14], [14, -7, 14], [14, floor_y, 14], [-14, floor_y, 14]], float),
            _texture(rng, 2048, 640),
            (0.95, 1.0, 1.05),
        ),
        (  # floor
            np.array(
                [[-9, floor_y, 14], [9, floor_y, 14], [9, floor_y, 3], [-9, floor_y, 3]], float
            ),
            _texture(rng, 1536, 960),
            (1.05, 1.0, 0.92),
        ),
    ]
    stands = [(-3.4, 10.5), (2.6, 9.5), (-0.6, 8.0), (3.6, 6.8), (-2.6, 6.0), (0.9, 5.0)]
    for x, z in stands:  # upright cards standing on the floor
        half_w, tall = float(rng.uniform(0.5, 0.9)), float(rng.uniform(1.6, 2.6))
        top = floor_y - tall
        corners = np.array(
            [
                [x - half_w, top, z],
                [x + half_w, top, z],
                [x + half_w, floor_y, z],
                [x - half_w, floor_y, z],
            ],
            float,
        )
        tint = tuple(float(v) for v in rng.uniform(0.85, 1.1, 3))
        cards.append((corners, _texture(rng, 384, 640), tint))
    return cards


def _camera(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotation (world to camera) and position at `t` in 0 .. 1: a slide to
    the right, a little forward, always looking at the middle of the set."""
    pos = np.array([-0.7 + 1.4 * t, -0.1 + 0.08 * np.sin(np.pi * t), 0.5 * t])
    forward = np.array([0.0, 0.2, 8.0]) - pos
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward]), pos


def make_clip(folder: Path) -> list[Path]:
    """Write the clip as an EXR sequence; returns its files."""
    import cv2

    from live_action_aov.io.oiio_io import write_exr

    folder.mkdir(parents=True, exist_ok=True)
    cards = _scene(np.random.default_rng(7))
    out: list[Path] = []
    for i in range(FRAME_COUNT):
        R, pos = _camera(i / (FRAME_COUNT - 1))
        frame = np.full((HEIGHT, WIDTH, 3), 0.5, np.float32)
        for corners, tex, tint in cards:
            cam = (corners - pos) @ R.T
            px = np.stack(
                [
                    FOCAL_PX * cam[:, 0] / cam[:, 2] + WIDTH / 2,
                    FOCAL_PX * cam[:, 1] / cam[:, 2] + HEIGHT / 2,
                ],
                axis=1,
            ).astype(np.float32)
            th, tw = tex.shape
            src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
            H = cv2.getPerspectiveTransform(src, px)
            grey = cv2.warpPerspective(tex, H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
            cover = cv2.warpPerspective(
                np.ones_like(tex), H, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR
            )[..., None]
            frame = frame * (1.0 - cover) + grey[..., None] * np.asarray(tint, np.float32) * cover
        path = folder / f"camera_check.{FIRST_FRAME + i:04d}.exr"
        write_exr(
            path,
            np.clip(frame, 0.0, 1.0).astype(np.float32),
            channel_names=["R", "G", "B"],
            attrs={"compression": "zip"},
        )
        out.append(path)
    return out


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------


class _RefuseLoader(importlib.abc.Loader):
    def create_module(self, spec):  # type: ignore[no-untyped-def]
        return None

    def exec_module(self, module):  # type: ignore[no-untyped-def]
        raise AssertionError("camera_check: pycolmap was loaded in the main process")


class _RefusePycolmap(importlib.abc.MetaPathFinder):
    """While this is installed, importing pycolmap in this process fails.
    Child processes are not affected."""

    def find_spec(self, name, path=None, target=None):  # type: ignore[no-untyped-def]
        if name == "pycolmap":
            return importlib.machinery.ModuleSpec(name, _RefuseLoader())
        return None


def _torch_works(torch) -> bool:  # type: ignore[no-untyped-def]
    a = torch.rand(256, 256)
    return bool(torch.isfinite((a @ a).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=Path("camera_check_out"))
    ap.add_argument("--isolate", choices=("auto", "on", "off"), default="auto")
    args = ap.parse_args()
    clip_dir, run_dir = args.out / "clip", args.out / "run"

    files = make_clip(clip_dir)
    print(f"[camera_check] clip: {len(files)} frames {WIDTH}x{HEIGHT} in {clip_dir}", flush=True)

    import torch

    print(
        f"[camera_check] {platform.system()} {platform.machine()} | python "
        f"{platform.python_version()} | torch {torch.__version__} loaded first, works: "
        f"{_torch_works(torch)}",
        flush=True,
    )

    from live_action_aov.passes.camera import pycolmap_track
    from live_action_aov.passes.camera.pycolmap_track import PyColmapCameraPass

    # `run-shot` has no switch for pass parameters: set them on the pass.
    if args.isolate != "auto":
        PyColmapCameraPass.DEFAULT_PARAMS["colmap_isolate"] = args.isolate == "on"
    # No model download in this check: the fallback that needs one is off.
    PyColmapCameraPass.DEFAULT_PARAMS["pnp_fallback"] = False
    isolated = pycolmap_track.pycolmap_isolated(dict(PyColmapCameraPass.DEFAULT_PARAMS))
    if isolated:
        sys.meta_path.insert(0, _RefusePycolmap())
    print(
        f"[camera_check] camera solver kept out of this process: {isolated} "
        f"(--isolate {args.isolate})",
        flush=True,
    )

    from live_action_aov.cli.app import main as liveaov

    argv, t0, code = sys.argv, time.monotonic(), 0
    sys.argv = [
        "liveaov",
        "run-shot",
        str(clip_dir),
        "-p",
        "camera_pycolmap",
        "--out",
        str(run_dir),
    ]
    try:
        liveaov()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else int(bool(e.code))
    finally:
        sys.argv = argv
    took = time.monotonic() - t0

    loaded_here = "pycolmap" in sys.modules
    cameras = sorted(run_dir.rglob("*camera*.json"))
    cameras = [p for p in cameras if p.parent.name == "camera"]
    problems: list[str] = []
    if code != 0:
        problems.append(f"the run ended with exit code {code}")
    if not cameras:
        problems.append("no camera file was written")
    if isolated and loaded_here:
        problems.append("pycolmap was loaded in the main process")
    if not _torch_works(torch):
        problems.append("PyTorch stopped working after the camera")

    print(f"[camera_check] run-shot exit code: {code} ({took:.0f} s)")
    print(f"[camera_check] pycolmap loaded in the main process: {loaded_here}")
    for p in cameras:
        solve = json.loads(p.read_text(encoding="utf-8"))
        q = solve.get("quality", {})
        print(
            f"[camera_check] camera file: {p} ({p.stat().st_size} bytes) | frames with a "
            f"camera: {q.get('registered_frame_count')}/{q.get('total_frame_count')} | "
            f"solver: {solve.get('backend', {}).get('name')}"
        )
    if problems:
        print("[camera_check] FAILED: " + "; ".join(problems))
        return 1
    print("[camera_check] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
