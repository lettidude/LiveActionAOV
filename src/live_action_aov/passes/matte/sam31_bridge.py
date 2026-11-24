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

The device: SAM 3.1 runs on an NVIDIA card only. When `LAOV_DEVICE` names a
device, the worker is told (`worker_device`): `cuda:1` runs it on that card,
`cpu` / `mps` are refused before anything is started.
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


def worker_device(
    device: str, env: dict[str, str] | None = None
) -> tuple[list[str], dict[str, str]]:
    """Worker arguments and environment changes that hold SAM 3.1 to
    `device` (`cuda`, `cuda:1`, ...). "" = nothing: the worker's default.

    SAM 3.1 puts its model on the first card it can see, so a card number
    becomes `CUDA_VISIBLE_DEVICES` for the worker (counted inside a list
    the user already set). Another kind of device raises: SAM 3.1 has no
    CPU or Apple GPU path."""
    if not device:
        return [], {}
    kind, _, idx = device.partition(":")
    if kind != "cuda":
        raise RuntimeError(
            f"SAM 3.1 runs on an NVIDIA card only, and this run is on {device} "
            "(LAOV_DEVICE). Use the SAM 3 detector, or unset LAOV_DEVICE."
        )
    if not idx:
        return ["--device", "cuda"], {}
    listed = (env or {}).get("CUDA_VISIBLE_DEVICES", "")
    visible = [v.strip() for v in listed.split(",") if v.strip()]
    card = visible[int(idx)] if idx.isdigit() and int(idx) < len(visible) else idx
    return ["--device", "cuda"], {"CUDA_VISIBLE_DEVICES": card}


def run_sam31(
    frames: np.ndarray,
    concepts: list[str],
    seed_local: int,
    python: str,
    timeout_s: float = 3600.0,
    checkpoint: str = "",
    device: str = "",
) -> list[Sam31Object]:
    """Track `concepts` through `frames` ((T, H, W, 3) display-referred
    0..1) with SAM 3.1. Masks come back at the frames' resolution.
    `device`: the device the run was asked to use ("" = the default)."""
    from PIL import Image

    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    env["PYTHONUTF8"] = "1"
    device_args, device_env = worker_device(device, env)  # refuses before any work
    env.update(device_env)
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
        cmd += device_args
        _log.info(
            "SAM 3.1: %d frames, concepts %s, seed frame %d%s",
            len(frames),
            concepts,
            seed_local,
            f", device {device}" if device else "",
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


__all__ = [
    "Sam31Object",
    "find_python",
    "keep_object",
    "read_results",
    "run_sam31",
    "worker_device",
]
