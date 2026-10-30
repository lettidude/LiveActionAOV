# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Where a run's outputs land and what they are called.

The user always picks an output ROOT; every run of a shot gets its own
version folder, so a re-run never overwrites the last one::

    <root>/<shot>/v001/
        <shot>_v001.comp.nk                       <- open this in Nuke
        passes/<shot>_v001.utility.####.exr      (per pass: passes/<group>/...)
        passes/<shot>_v001.utility.####.crypto.exr
        camera/<shot>_v001.camera.nk  .camera.json  .points.obj/.ply  _colmap/
        splat/<shot>_v001.splat.ply  .splat.json
        logs/submit_<timestamp>.log

A run dir that is not a `vNNN` folder (library callers, tests) gets the
same subfolders, named after the bare shot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PASSES_DIR = "passes"
CAMERA_DIR = "camera"
SPLAT_DIR = "splat"
LOGS_DIR = "logs"

_VERSION_RE = re.compile(r"^v(\d{3,})$")


def next_version_dir(shot_root: Path) -> Path:
    """`<root>/<shot>` -> the first free `v###` folder inside it."""
    taken = [
        int(m.group(1))
        for p in (shot_root.iterdir() if shot_root.is_dir() else [])
        if p.is_dir() and (m := _VERSION_RE.match(p.name))
    ]
    return shot_root / f"v{max(taken, default=0) + 1:03d}"


def version_dirs(shot_root: Path) -> list[Path]:
    """A shot's `v###` folders, newest first."""
    if not shot_root.is_dir():
        return []
    found = [
        (int(m.group(1)), p)
        for p in shot_root.iterdir()
        if p.is_dir() and (m := _VERSION_RE.match(p.name))
    ]
    return [p for _, p in sorted(found, reverse=True)]


def latest_version_dir(shot_root: Path) -> Path | None:
    """Newest `v###` folder of a shot, or None before its first run."""
    found = version_dirs(shot_root)
    return found[0] if found else None


def run_dir(shot: Any) -> Path:
    return Path(shot.output_dir or shot.folder)


def run_stem(shot: Any) -> str:
    """`sh010` in a plain dir, `sh010_v002` inside a `v002` folder."""
    d = run_dir(shot)
    return f"{shot.name}_{d.name}" if _VERSION_RE.match(d.name) else str(shot.name)


def frame_padding(sequence_pattern: str) -> int:
    """Frame-number width of the plate pattern (`####` / `%04d`), default 4."""
    m = re.search(r"#+", sequence_pattern)
    if m:
        return len(m.group(0))
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    if m and m.group(1):
        return int(m.group(1))
    return 4


def sidecar_template(shot: Any) -> str:
    """Run-dir-relative EXR template: `passes/<stem>.utility.{frame:04d}.exr`."""
    pad = frame_padding(shot.sequence_pattern)
    return f"{PASSES_DIR}/{run_stem(shot)}.utility.{{frame:0{pad}d}}.exr"


def sub_dir(shot: Any, name: str) -> Path:
    """A run subfolder (`camera`, `splat`, `logs` ...), created on demand."""
    d = run_dir(shot) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "CAMERA_DIR",
    "LOGS_DIR",
    "PASSES_DIR",
    "SPLAT_DIR",
    "frame_padding",
    "latest_version_dir",
    "next_version_dir",
    "run_dir",
    "run_stem",
    "sidecar_template",
    "sub_dir",
    "version_dirs",
]
