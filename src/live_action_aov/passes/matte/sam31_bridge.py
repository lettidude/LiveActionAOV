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
environment variable. Setup: docs/install.md, "SAM 3.1 (optional)".
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


def find_python(params: dict[str, Any]) -> str:
    """The SAM 3.1 environment's interpreter, or a RuntimeError saying how
    to set it up."""
    exe = str(params.get("sam31_python") or os.environ.get(ENV_PYTHON, "") or "")
    if exe and Path(exe).is_file():
        return exe
    where = f"'{exe}' does not exist" if exe else "no interpreter is set"
    raise RuntimeError(
        f"SAM 3.1 detector: {where}. Install the SAM 3.1 environment (Python 3.12 + "
        f"facebookresearch/sam3, see docs/install.md 'SAM 3.1') and point the "
        f"`sam31_python` param or the {ENV_PYTHON} environment variable at its python."
    )


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


__all__ = ["Sam31Object", "find_python", "keep_object", "read_results", "run_sam31"]
