# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Where a run's outputs land and what they are called.

Every run of a shot gets its own version folder, so a re-run never
overwrites the last one. By default the shot folder is a SIBLING of the
plate folder (`plate_sibling_dir`: production trees keep the utility next
to its plate, the plate folder itself stays clean); or the user picks one
root for the whole session::

    <plate parent>/<plate folder>_utility/v001/     (default)
    <root>/<shot>/v001/                             (chosen root)

        <shot>_v001.comp.nk                       <- open this in Nuke
        passes/<shot>_v001.utility.####.exr      (per pass: passes/<group>/...)
        passes/<shot>_v001.crypto.####.exr       (only with a lossy delivery)
        camera/<shot>_v001.camera.nk  .camera.json  .points.obj/.ply  _colmap/
        splat/<shot>_v001.splat.ply  .splat.json
        logs/submit_<timestamp>.log

File names come from ONE template (`Shot.naming`), so a studio can match
its own convention::

    {shot}_{version}.{pass}.{frame}.{ext}        (default, above)
    {shot}_{pass}_{version}.{frame}.{ext}        SH010_utility_v001.1001.exr

Tokens: `{shot}` the shot name, `{plate}` the plate folder name, `{version}`
the run's `v###`, `{pass}` what the file holds (`utility`, `crypto`, a pass
group in the per-pass layout, `camera`, `points`, `splat`, `comp`),
`{frame}` the frame number padded like the plate, `{ext}`. Files without a
frame (the comp script, the camera, the splat) drop the `{frame}` token
and the separator before it; a run dir that is not a `vNNN` folder (library
callers, tests) drops `{version}` the same way.
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

#: The historic layout: `<shot>_v001.utility.1001.exr`.
DEFAULT_NAMING = "{shot}_{version}.{pass}.{frame}.{ext}"

#: (label, template) pairs the GUI offers; any template with the tokens
#: below is accepted too.
NAMING_PRESETS: tuple[tuple[str, str], ...] = (
    ("Default  (shot_v001.utility.1001.exr)", DEFAULT_NAMING),
    ("Pass before version  (shot_utility_v001.1001.exr)", "{shot}_{pass}_{version}.{frame}.{ext}"),
    ("Version before pass  (shot_v001_utility.1001.exr)", "{shot}_{version}_{pass}.{frame}.{ext}"),
    ("Plate folder name  (plate_utility_v001.1001.exr)", "{plate}_{pass}_{version}.{frame}.{ext}"),
)

NAMING_TOKENS: tuple[str, ...] = ("shot", "plate", "version", "pass", "frame", "ext")
_REQUIRED_TOKENS: tuple[str, ...] = ("pass", "frame", "ext")
_FORBIDDEN_CHARS = '\\/:*?"<>|'


def validate_naming(template: str) -> str:
    """The template stripped, or `DEFAULT_NAMING` for an empty one.
    Raises ValueError with a one-line reason the GUI can show."""
    t = (template or "").strip()
    if not t:
        return DEFAULT_NAMING
    bad = [ch for ch in _FORBIDDEN_CHARS if ch in t]
    if bad:
        raise ValueError(f"naming template: {''.join(bad)!r} not allowed in a file name")
    try:
        tokens = [f.group(1) for f in re.finditer(r"\{([^{}]*)\}", t)]
        if t.count("{") != len(tokens) or t.count("}") != len(tokens):
            raise ValueError
    except ValueError:
        raise ValueError("naming template: unbalanced braces") from None
    unknown = sorted({tok for tok in tokens if tok not in NAMING_TOKENS})
    if unknown:
        raise ValueError(
            f"naming template: unknown token(s) {', '.join('{' + u + '}' for u in unknown)}; "
            f"use {', '.join('{' + k + '}' for k in NAMING_TOKENS)}"
        )
    missing = [tok for tok in _REQUIRED_TOKENS if tok not in tokens]
    if missing:
        raise ValueError(
            f"naming template: needs {', '.join('{' + m + '}' for m in missing)} "
            "(every output has a kind and an extension; frame files need a frame number)"
        )
    return t


def plate_sibling_dir(plate_folder: Path, name: str = "") -> Path:
    """`.../plates/PF0052_v001` -> `.../plates/PF0052_v001_utility` (or
    `name` next to the plate folder when given)."""
    plate_folder = Path(plate_folder)
    return plate_folder.parent / (name.strip() or f"{plate_folder.name}_utility")


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


def version_label(shot: Any) -> str:
    """`v002` when the run dir is a version folder, else ``""``."""
    d = run_dir(shot)
    return d.name if _VERSION_RE.match(d.name) else ""


def run_stem(shot: Any) -> str:
    """`sh010` in a plain dir, `sh010_v002` inside a `v002` folder."""
    v = version_label(shot)
    return f"{shot.name}_{v}" if v else str(shot.name)


def frame_padding(sequence_pattern: str) -> int:
    """Frame-number width of the plate pattern (`####` / `%04d`), default 4."""
    m = re.search(r"#+", sequence_pattern)
    if m:
        return len(m.group(0))
    m = re.search(r"%0?(\d*)d", sequence_pattern)
    if m and m.group(1):
        return int(m.group(1))
    return 4


def _drop_token(template: str, token: str) -> str:
    """Remove `{token}` and the one separator glued to it (`_` `.` `-`),
    the one before it when there is one, so `a_{v}.{p}` -> `a.{p}`."""
    out = re.sub(rf"[._-]\{{{token}\}}", "", template, count=1)
    if out == template:
        out = re.sub(rf"\{{{token}\}}[._-]?", "", template, count=1)
    return out


def file_name(shot: Any, kind: str, ext: str, frame: int | str | None = None) -> str:
    """One output file name from the shot's naming template.

    `kind` fills `{pass}` (`utility`, `crypto`, `camera`, `splat`, `comp`,
    a pass group...). `frame` is an int (padded like the plate), a literal
    placeholder string such as ``"{frame:04d}"`` (kept verbatim, for
    sequence templates), or None for a file without a frame number.
    """
    template = validate_naming(getattr(shot, "naming", "") or "")
    version = version_label(shot)
    if not version:
        template = _drop_token(template, "version")
    if frame is None:
        template = _drop_token(template, "frame")
        frame_str = ""
    elif isinstance(frame, str):
        frame_str = frame
    else:
        frame_str = f"{int(frame):0{frame_padding(shot.sequence_pattern)}d}"
    fields = {
        "shot": str(shot.name),
        "plate": Path(shot.folder).name,
        "version": version,
        "pass": kind,
        "frame": frame_str,
        "ext": ext.lstrip("."),
    }
    return template.format(**fields)


def frame_template(shot: Any, kind: str, sub: str = "") -> str:
    """Run-dir-relative EXR sequence template with a `{frame:0Nd}` slot:
    `passes/[sub/]<name>`. `kind` = `utility` (multi-layer), `crypto`
    (the lossless Cryptomatte sibling) or a pass group (per-pass layout)."""
    pad = frame_padding(shot.sequence_pattern)
    name = file_name(shot, kind, "exr", frame=f"{{frame:0{pad}d}}")
    return f"{PASSES_DIR}/{sub}/{name}" if sub else f"{PASSES_DIR}/{name}"


def sidecar_template(shot: Any) -> str:
    """Run-dir-relative EXR template: `passes/<stem>.utility.{frame:04d}.exr`."""
    return frame_template(shot, "utility")


def sub_dir(shot: Any, name: str) -> Path:
    """A run subfolder (`camera`, `splat`, `logs` ...), created on demand."""
    d = run_dir(shot) / name
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = [
    "CAMERA_DIR",
    "DEFAULT_NAMING",
    "LOGS_DIR",
    "NAMING_PRESETS",
    "NAMING_TOKENS",
    "PASSES_DIR",
    "SPLAT_DIR",
    "file_name",
    "frame_padding",
    "frame_template",
    "latest_version_dir",
    "next_version_dir",
    "plate_sibling_dir",
    "run_dir",
    "run_stem",
    "sidecar_template",
    "sub_dir",
    "validate_naming",
    "version_dirs",
    "version_label",
]
