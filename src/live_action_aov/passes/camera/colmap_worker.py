# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Child process for the camera pass: runs COLMAP and writes the result.

`python -m live_action_aov.passes.camera.colmap_worker <job.json>`. The job
names the work folder, the images and masks, the COLMAP params and where to
write `{"model_dir": ...}` (or `{"error": ...}` when nothing reconstructs).
A native crash leaves no result file: the parent turns that into a
RuntimeError for the camera pass alone.

Two extras for a parent that must not load pycolmap itself
(`colmap_isolate`, see `colmap_plain`): a job with `"snapshot"` also saves
the reconstruction as plain data, and a job with `"mode": "pairs"` reads
the verified matches out of COLMAP's database instead of solving.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any


def _solve(job: dict[str, Any]) -> dict[str, Any]:
    from live_action_aov.passes.camera.pycolmap_track import run_colmap

    mask_dir = Path(job["mask_dir"]) if job.get("mask_dir") else None
    _rec, model_dir = run_colmap(
        Path(job["workdir"]),
        Path(job["image_dir"]),
        mask_dir,
        job["image_names"],
        job["params"],
    )
    if job.get("snapshot"):
        from live_action_aov.passes.camera.colmap_plain import write_snapshot

        # Read back from disk, like a parent that loads the model itself.
        pycolmap: Any = importlib.import_module("pycolmap")
        write_snapshot(pycolmap.Reconstruction(str(model_dir)), Path(job["snapshot"]))
    return {"model_dir": str(model_dir)}


def _pairs(job: dict[str, Any]) -> dict[str, Any]:
    from live_action_aov.passes.camera.colmap_plain import save_pairs
    from live_action_aov.passes.camera.pycolmap_track import read_verified_pairs

    frame_list = [(int(f), str(name)) for f, name in job["frame_list"]]
    pairs = read_verified_pairs(Path(job["db"]), frame_list, int(job["min_inliers"]))
    save_pairs(Path(job["pairs"]), pairs)
    return {"pairs": str(job["pairs"])}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        out = _pairs(job) if job.get("mode") == "pairs" else _solve(job)
    except RuntimeError as e:
        out = {"error": str(e)}
    Path(job["result"]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
