"""Session save/load — the whole prep state survives a crash or a restart.

A session file is plain JSON capturing every `ShotState` the GUI holds:
identity, colour decisions (detected + override + exposure), enabled
models, SAM 3 concepts, click-to-mask instances, output routing, proxy,
queue flags. Loading reconstructs the states verbatim — no re-probing,
so a 50-shot prep reopens exactly as it was left, instantly.

Two consumers:
- **Save/Open session** (File menu): explicit `.laov.json` files the user
  manages.
- **Autosave** (MainWindow): debounced write to a fixed per-user path on
  every registry change; on startup the user is offered a restore. This
  is the crash insurance — the artist never loses a prep again.

Runtime-only fields (status, last_error, last_sidecar_dir) are NOT
persisted: a restored session always starts "new"/idle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from live_action_aov.gui.shot_state import ClickInstance, ShotState
from live_action_aov.io.colorspace_detect import DetectedColorspace

SESSION_VERSION = 1
SESSION_SUFFIX = ".laov.json"


def autosave_path() -> Path:
    """Fixed per-user autosave location (created on first save)."""
    d = Path(user_data_dir("LiveActionAOV", appauthor=False))
    return d / "autosave" / f"session{SESSION_SUFFIX}"


# --- ShotState <-> dict ----------------------------------------------------


def shot_to_dict(s: ShotState) -> dict[str, Any]:
    return {
        "name": s.name,
        "folder": str(s.folder),
        "sequence_pattern": s.sequence_pattern,
        "frame_range": [int(s.frame_range[0]), int(s.frame_range[1])],
        "resolution": [int(s.resolution[0]), int(s.resolution[1])],
        "pixel_aspect": float(s.pixel_aspect),
        "detected": (
            {
                "detected": s.detected.detected,
                "reason": s.detected.reason,
                "confident": bool(s.detected.confident),
            }
            if s.detected is not None
            else None
        ),
        "override": s.override,
        "current_frame": int(s.current_frame),
        "view_mode": s.view_mode,
        "exposure_ev": float(s.exposure_ev),
        "auto_ev": None if s.auto_ev is None else float(s.auto_ev),
        "auto_ev_source": s.auto_ev_source,
        "sampled_luma": None if s.sampled_luma is None else float(s.sampled_luma),
        "enabled_models": list(s.enabled_models),
        "sam3_concepts": s.sam3_concepts,
        "refine_all_masks": bool(s.refine_all_masks),
        "refiner_model": s.refiner_model,
        "preview_refiner": s.preview_refiner,
        "click_instances": [
            {
                "name": ci.name,
                "seed_frame": int(ci.seed_frame),
                "points": [[float(x), float(y), int(lbl)] for (x, y, lbl) in ci.points],
                "box": None if ci.box is None else [float(v) for v in ci.box],
            }
            for ci in s.click_instances
        ],
        "output_mode": s.output_mode,
        "output_external_root": (
            None if s.output_external_root is None else str(s.output_external_root)
        ),
        "output_subfolder_name": s.output_subfolder_name,
        "output_external_name": s.output_external_name,
        "proxy_long_edge": s.proxy_long_edge,
        "delivery_compression": s.delivery_compression,
        "delivery_dtype": s.delivery_dtype,
        "queued": bool(s.queued),
    }


def shot_from_dict(d: dict[str, Any]) -> ShotState:
    det = d.get("detected")
    detected = (
        DetectedColorspace(
            detected=str(det["detected"]),
            reason=str(det.get("reason", "restored from session")),
            confident=bool(det.get("confident", False)),
        )
        if det
        else None
    )
    clicks = [
        ClickInstance(
            name=str(ci.get("name", "")),
            seed_frame=int(ci.get("seed_frame", 0)),
            points=[(float(p[0]), float(p[1]), int(p[2])) for p in (ci.get("points") or [])],
            box=(tuple(float(v) for v in ci["box"]) if ci.get("box") else None),  # type: ignore[arg-type]
        )
        for ci in (d.get("click_instances") or [])
    ]
    ext_root = d.get("output_external_root")
    fr = d.get("frame_range") or [0, 0]
    res = d.get("resolution") or [0, 0]
    return ShotState(
        name=str(d["name"]),
        folder=Path(str(d["folder"])),
        sequence_pattern=str(d["sequence_pattern"]),
        frame_range=(int(fr[0]), int(fr[1])),
        resolution=(int(res[0]), int(res[1])),
        pixel_aspect=float(d.get("pixel_aspect", 1.0)),
        detected=detected,
        override=d.get("override"),
        current_frame=int(d.get("current_frame", fr[0])),
        view_mode=d.get("view_mode", "transformed"),
        exposure_ev=float(d.get("exposure_ev", 0.0)),
        auto_ev=(None if d.get("auto_ev") is None else float(d["auto_ev"])),
        auto_ev_source=str(d.get("auto_ev_source", "")),
        sampled_luma=(None if d.get("sampled_luma") is None else float(d["sampled_luma"])),
        enabled_models=[str(m) for m in (d.get("enabled_models") or [])],
        sam3_concepts=str(d.get("sam3_concepts", "")),
        refine_all_masks=bool(d.get("refine_all_masks", False)),
        refiner_model=str(d.get("refiner_model", "")),
        preview_refiner=str(d.get("preview_refiner", "auto")),
        click_instances=clicks,
        output_mode=str(d.get("output_mode", "inplace")),
        output_external_root=(None if ext_root is None else Path(str(ext_root))),
        output_subfolder_name=str(d.get("output_subfolder_name", "utility")),
        output_external_name=str(d.get("output_external_name", "")),
        proxy_long_edge=d.get("proxy_long_edge"),
        delivery_compression=str(d.get("delivery_compression", "zip")),
        delivery_dtype=str(d.get("delivery_dtype", "float32")),
        queued=bool(d.get("queued", True)),
    )


# --- File I/O ---------------------------------------------------------------


def save_session(path: Path, shots: list[ShotState]) -> None:
    """Write the session atomically (tmp + replace) so a crash mid-write
    never corrupts the previous good file — this IS the crash insurance."""
    payload = {
        "app": "LiveActionAOV",
        "kind": "session",
        "version": SESSION_VERSION,
        "shots": [shot_to_dict(s) for s in shots],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_session(path: Path) -> tuple[list[ShotState], list[str]]:
    """Read a session file. Returns (shots, warnings).

    Shots whose plate folder no longer exists are skipped with a warning
    rather than crashing the load — drives unmount, projects move."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "session":
        raise ValueError(f"Not a LiveActionAOV session file: {path}")
    version = int(data.get("version", 0))
    if version > SESSION_VERSION:
        raise ValueError(
            f"Session file is version {version}; this build reads up to "
            f"{SESSION_VERSION}. Update LiveActionAOV."
        )
    shots: list[ShotState] = []
    warnings: list[str] = []
    for entry in data.get("shots") or []:
        try:
            shot = shot_from_dict(entry)
        except Exception as e:
            warnings.append(f"Skipped malformed shot entry: {e}")
            continue
        if not shot.folder.is_dir():
            warnings.append(f"Skipped '{shot.name}' — plate folder missing: {shot.folder}")
            continue
        shots.append(shot)
    return shots, warnings


__all__ = [
    "SESSION_SUFFIX",
    "SESSION_VERSION",
    "autosave_path",
    "load_session",
    "save_session",
    "shot_from_dict",
    "shot_to_dict",
]
