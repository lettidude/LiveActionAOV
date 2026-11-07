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
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    from live_action_aov.passes.camera.pycolmap_track import run_colmap

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    mask_dir = Path(job["mask_dir"]) if job.get("mask_dir") else None
    try:
        _rec, model_dir = run_colmap(
            Path(job["workdir"]),
            Path(job["image_dir"]),
            mask_dir,
            job["image_names"],
            job["params"],
        )
        out = {"model_dir": str(model_dir)}
    except RuntimeError as e:
        out = {"error": str(e)}
    Path(job["result"]).write_text(json.dumps(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
