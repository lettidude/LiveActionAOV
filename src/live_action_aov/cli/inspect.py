# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Sidecar EXR inspector — rendering logic for `liveaov inspect`.

Separated from the Typer command so the pure render path is unit-testable
without `CliRunner`. `build_report` consumes the `(pixels, attrs)` tuple
that `live_action_aov.io.oiio_io.read_exr` already returns, so there is
no new I/O code; `format_text` / `format_json` dress the resulting dict
for terminal output or scripting respectively.

Design notes:

- **Channel bucketing** follows the comper's mental model: everything in
  `CANONICAL_CHANNEL_ORDER` is "canonical", `matte.{r,g,b,a}` is its own
  sub-bucket (heroes), `mask.*` is dynamic concepts, anything else is
  "other". `matte` is split off from "canonical" so the Heroes summary
  below reads naturally next to it.
- **Value ranges use `[min, max]`** rounded to 3 decimals. A `⚠` suffix
  flags matte/mask channels that land outside `[0, 1]` beyond a small
  tolerance — the integration test asserts the same envelope, so any
  sidecar that trips the warning here would also trip CI.
- **Hero summary** is a one-line-per-slot read from the
  `liveaov/matte/hero_{r,g,b,a}/{label,track_id,score}` attrs.
  Empty slots are labeled `(empty)`; a sidecar with no heroes at all
  (no `matte/detector` attr) skips the block entirely.
- **JSON output** pins a stable top-level key set —
  `{file, resolution, channels, metadata, heroes}` — so batch runners
  and future CI tooling have a contract to bind against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.channels import (
    CANONICAL_CHANNEL_ORDER,
    MASK_PREFIX,
    MATTE_CHANNELS,
)
from live_action_aov.io.oiio_io import read_exr

# Tolerance for [0, 1] envelope check on matte/mask channels. Matches the
# ±1e-6 slop used in the real-weights integration test.
_VALUE_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ChannelStat:
    name: str
    bucket: str  # canonical | matte | mask | other
    min: float
    max: float
    mean: float
    out_of_range: bool = False  # True when matte/mask strays from [0,1]


@dataclass
class HeroSlot:
    slot: str  # r | g | b | a
    label: str | None
    track_id: int | None
    score: float | None

    @property
    def is_empty(self) -> bool:
        return self.label is None and self.track_id is None and self.score is None


@dataclass
class SidecarReport:
    file: str
    width: int
    height: int
    channels: list[ChannelStat]
    metadata: dict[str, Any]  # flat `liveaov/*` subset
    heroes: list[HeroSlot] = field(default_factory=list)
    has_matte_metadata: bool = False  # True iff any liveaov/matte/*


# ---------------------------------------------------------------------------
# Build the report from a sidecar file
# ---------------------------------------------------------------------------


def build_report(path: Path | str) -> SidecarReport:
    """Read the sidecar at `path` and return a structured report."""
    pixels, attrs = read_exr(path)
    names = list(attrs.get("channelnames") or [])
    width = int(attrs.get("width", 0))
    height = int(attrs.get("height", 0))

    channels = [_channel_stat(name, pixels, names) for name in names]
    metadata = _filter_liveaov_attrs(attrs)
    heroes = _build_hero_summary(metadata)
    has_matte_metadata = any(k.startswith("liveaov/matte/") for k in metadata)

    return SidecarReport(
        file=str(path),
        width=width,
        height=height,
        channels=channels,
        metadata=metadata,
        heroes=heroes,
        has_matte_metadata=has_matte_metadata,
    )


def _channel_stat(name: str, pixels: np.ndarray, names: list[str]) -> ChannelStat:
    idx = names.index(name)
    band = pixels[..., idx]
    mn = float(band.min()) if band.size else 0.0
    mx = float(band.max()) if band.size else 0.0
    mean = float(band.mean()) if band.size else 0.0
    bucket = _bucket_for(name)
    out_of_range = False
    if bucket in {"matte", "mask"}:
        if mn < -_VALUE_TOLERANCE or mx > 1.0 + _VALUE_TOLERANCE:
            out_of_range = True
    return ChannelStat(
        name=name,
        bucket=bucket,
        min=mn,
        max=mx,
        mean=mean,
        out_of_range=out_of_range,
    )


def _bucket_for(name: str) -> str:
    if name in MATTE_CHANNELS:
        return "matte"
    if name.startswith(MASK_PREFIX) and len(name) > len(MASK_PREFIX):
        return "mask"
    if name in CANONICAL_CHANNEL_ORDER:
        return "canonical"
    return "other"


def _filter_liveaov_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Only project-owned attrs are interesting; strip OIIO's scalar bookkeeping
    (width / height / nchannels / channelnames / pixelAspectRatio) and anything
    not under our namespace."""
    return {
        k: v for k, v in sorted(attrs.items()) if isinstance(k, str) and k.startswith("liveaov/")
    }


def _build_hero_summary(metadata: dict[str, Any]) -> list[HeroSlot]:
    """Translate `liveaov/matte/hero_{slot}/{field}` attrs into HeroSlots.

    Any slot with at least one field present is emitted; slots with no fields
    at all are emitted as empty placeholders so the output table always
    shows r/g/b/a. If no hero attrs exist at all, returns `[]` so callers
    can skip the Heroes block entirely.
    """
    prefix = "liveaov/matte/hero_"
    any_field = any(k.startswith(prefix) for k in metadata)
    if not any_field:
        return []

    def _get(slot: str, field_: str) -> Any:
        return metadata.get(f"{prefix}{slot}/{field_}")

    slots: list[HeroSlot] = []
    for slot in ("r", "g", "b", "a"):
        label = _get(slot, "label")
        tid = _get(slot, "track_id")
        score = _get(slot, "score")
        slots.append(
            HeroSlot(
                slot=slot,
                label=str(label) if label is not None else None,
                track_id=_try_int(tid),
                score=_try_float(score),
            )
        )
    return slots


def _try_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _try_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Render: plain text
# ---------------------------------------------------------------------------


def format_text(report: SidecarReport) -> str:
    """Human-readable report for terminals."""
    lines: list[str] = []
    lines.append(f"File: {report.file}")
    lines.append(f"Resolution: {report.width} x {report.height}")
    lines.append(f"Channels ({len(report.channels)}):")

    # Column width based on the longest channel name — keeps stats aligned
    # even when there's a long concept like `mask.construction_vehicle`.
    name_width = max((len(c.name) for c in report.channels), default=10)
    name_width = max(name_width, 12)

    for bucket_label, bucket_key in (
        ("canonical", "canonical"),
        ("matte", "matte"),
        ("mask", "mask"),
        ("other", "other"),
    ):
        group = [c for c in report.channels if c.bucket == bucket_key]
        if not group:
            continue
        lines.append(f"  {bucket_label}:")
        for c in group:
            warn = "  ⚠ out of [0,1]" if c.out_of_range else ""
            lines.append(
                f"    {c.name.ljust(name_width)}  "
                f"[{c.min:.3f}, {c.max:.3f}]  mean={c.mean:.3f}{warn}"
            )

    # Metadata block (only liveaov/*, already filtered).
    lines.append("")
    if report.metadata:
        lines.append("liveaov metadata:")
        meta_key_width = max(len(k) for k in report.metadata)
        for k, v in report.metadata.items():
            short = k.removeprefix("liveaov/")
            lines.append(f"  {short.ljust(meta_key_width - len('liveaov/'))}  = {_fmt_value(v)}")
    else:
        lines.append("liveaov metadata: (none)")

    # Hero summary — only when matte metadata is present.
    if report.has_matte_metadata and report.heroes:
        filled = sum(1 for h in report.heroes if not h.is_empty)
        lines.append("")
        lines.append(f"Heroes ({filled} of {len(report.heroes)} slots filled):")
        for h in report.heroes:
            if h.is_empty:
                lines.append(f"  matte.{h.slot} = (empty)")
            else:
                score_str = f"{h.score:.2f}" if h.score is not None else "?"
                tid_str = str(h.track_id) if h.track_id is not None else "?"
                lines.append(f"  matte.{h.slot} = {h.label} (track {tid_str}, score {score_str})")
    elif report.has_matte_metadata:
        # Matte pipeline ran but produced no heroes — worth flagging,
        # it means the detector found nothing or the refiner dropped them.
        lines.append("")
        lines.append("Heroes: (none — detector found nothing or refiner dropped all slots)")

    return "\n".join(lines)


def _fmt_value(v: Any) -> str:
    """Render a metadata value for display. Strings get quoted so the
    boundary between `matte/commercial = "true"` (string) and a numeric
    attribute is visually obvious. Floats get clipped to 4 decimals —
    OIIO round-trips `0.91` as `0.8700000047683716` through float32, which
    is technically accurate but garbage to read in a terminal."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") or "0"
    return str(v)


# ---------------------------------------------------------------------------
# Render: JSON
# ---------------------------------------------------------------------------


def format_json(report: SidecarReport) -> dict[str, Any]:
    """Stable key set: {file, resolution, channels, metadata, heroes}.

    Batch runners bind against this — any change here is a downstream
    contract break. Keep additions additive (extra keys fine; rename /
    drop is not)."""
    return {
        "file": report.file,
        "resolution": {"width": report.width, "height": report.height},
        "channels": [
            {
                "name": c.name,
                "bucket": c.bucket,
                "min": c.min,
                "max": c.max,
                "mean": c.mean,
                "out_of_range": c.out_of_range,
            }
            for c in report.channels
        ],
        "metadata": {k: _json_safe(v) for k, v in report.metadata.items()},
        "heroes": [
            {
                "slot": h.slot,
                "label": h.label,
                "track_id": h.track_id,
                "score": h.score,
                "empty": h.is_empty,
            }
            for h in report.heroes
        ],
    }


def _json_safe(v: Any) -> Any:
    """Coerce non-JSON-native values (numpy scalars, anything with __str__)
    to primitives. Keeps `json.dumps` from choking on OIIO return types."""
    if isinstance(v, (str, bool, int, float)) or v is None:
        return v
    # numpy scalars expose .item() for native-type coercion.
    if hasattr(v, "item"):
        try:
            return v.item()
        except (TypeError, ValueError):
            pass
    return str(v)


def format_json_str(report: SidecarReport) -> str:
    """Convenience: pretty-printed JSON string."""
    return json.dumps(format_json(report), indent=2)


__all__ = [
    "ChannelStat",
    "HeroSlot",
    "SidecarReport",
    "build_report",
    "format_json",
    "format_json_str",
    "format_text",
]
