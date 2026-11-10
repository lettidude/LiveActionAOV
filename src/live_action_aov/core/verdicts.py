# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Verdict thresholds for the run report: the one place they live.

A pass reports *metrics* (numbers with a name, e.g. ``splat.holdout_psnr_db``);
this module turns a metric into a traffic light - GOOD / WEAK / FAILED - and
holds the *policy* that decides what lands in the report's ``needs_human``
list. The defaults are generic and few. Other values come in without a code
change, as an override (format: ``{"limits": {<metric>: {"good": x,
"weak": y}}, "policy": {<key>: value}}``; a limit set to null is switched off).

Sources, later wins: the defaults below, the JSON file named by the
``LAOV_REPORT_THRESHOLDS`` environment variable, ``Job.report_thresholds``.

A pass that already judges itself (the camera's held-out GOOD / WEAK / FAILED)
keeps its own verdict; the limits here judge the metrics no pass judged.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

ENV_VAR = "LAOV_REPORT_THRESHOLDS"

GOOD, WEAK, FAILED = "GOOD", "WEAK", "FAILED"
_RANK = {"": 0, GOOD: 1, WEAK: 2, FAILED: 3}


@dataclass(frozen=True)
class Limit:
    """GOOD up to `good`, WEAK up to `weak`, FAILED past it (reversed when
    `higher_is_better`)."""

    good: float
    weak: float
    higher_is_better: bool = False
    unit: str = ""

    def judge(self, value: float) -> str:
        if self.higher_is_better:
            return GOOD if value >= self.good else WEAK if value >= self.weak else FAILED
        return GOOD if value <= self.good else WEAK if value <= self.weak else FAILED


#: Generic limits. Metric names are `<area>.<what>_<unit>`.
DEFAULT_LIMITS: dict[str, Limit] = {
    # Only judges a camera solve that brings no verdict of its own.
    "camera.reprojection_error_px": Limit(1.0, 2.0, unit="px"),
    "splat.holdout_psnr_db": Limit(25.0, 20.0, higher_is_better=True, unit="dB"),
    "splat.holdout_psnr_min_db": Limit(20.0, 15.0, higher_is_better=True, unit="dB"),
    "splat.masked_fraction": Limit(0.25, 0.4),
}

#: What goes to `needs_human`, and the matte area checks.
DEFAULT_POLICY: dict[str, Any] = {
    "failed_needs_human": True,  # a pass that failed / was skipped
    "verdict_failed_needs_human": True,
    "weak_needs_human": True,
    "fallback_needs_human": True,  # a pass ran on its fallback engine
    "noncommercial_needs_human": True,  # a non-commercial model ran
    # An object's area under this fraction of its own median, for at least
    # `matte_collapse_min_frames` frames in a row: "lost or collapsed".
    "matte_collapse_ratio": 0.2,
    "matte_collapse_min_frames": 2,
    # Frame-to-frame area change above this factor: a warning.
    "matte_jump_ratio": 3.0,
    # Ignore objects smaller than this fraction of the frame for the checks.
    "matte_min_area": 0.0005,
}


@dataclass(frozen=True)
class Thresholds:
    limits: dict[str, Limit] = field(default_factory=lambda: dict(DEFAULT_LIMITS))
    policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_POLICY))
    sources: tuple[str, ...] = ("defaults",)

    def with_overrides(self, data: dict[str, Any] | None, source: str) -> Thresholds:
        """A copy with `data` (`{"limits": ..., "policy": ...}`) applied."""
        if not data:
            return self
        limits = dict(self.limits)
        for name, spec in (data.get("limits") or {}).items():
            base = limits.get(name)
            if isinstance(spec, dict):
                if base is None:
                    limits[name] = Limit(
                        float(spec["good"]),
                        float(spec["weak"]),
                        bool(spec.get("higher_is_better", False)),
                        str(spec.get("unit", "")),
                    )
                else:
                    kw: dict[str, Any] = {}
                    for k, v in spec.items():
                        if k in ("good", "weak"):
                            kw[k] = float(v)
                        elif k == "higher_is_better":
                            kw[k] = bool(v)
                        elif k == "unit":
                            kw[k] = str(v)
                    limits[name] = replace(base, **kw)
            elif spec is None:
                limits.pop(name, None)  # switch a check off
        policy = {**self.policy, **(data.get("policy") or {})}
        return Thresholds(limits, policy, (*self.sources, source))

    def judge(self, metric: str, value: float) -> str:
        lim = self.limits.get(metric)
        return lim.judge(float(value)) if lim is not None else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "limits": {
                k: {
                    "good": v.good,
                    "weak": v.weak,
                    "higher_is_better": v.higher_is_better,
                    **({"unit": v.unit} if v.unit else {}),
                }
                for k, v in self.limits.items()
            },
            "policy": dict(self.policy),
        }


def load_thresholds(job_overrides: dict[str, Any] | None = None) -> Thresholds:
    """Defaults, then the env-var file, then the job's own overrides. A bad
    override file is logged and ignored: it never stops a run."""
    t = Thresholds()
    path = os.environ.get(ENV_VAR, "").strip()
    if path:
        try:
            t = t.with_overrides(json.loads(Path(path).read_text(encoding="utf-8")), path)
        except (OSError, ValueError, KeyError, TypeError) as e:
            _log.warning("run report: could not read %s=%s (%s) - defaults kept", ENV_VAR, path, e)
    if job_overrides:
        try:
            t = t.with_overrides(job_overrides, "job")
        except (ValueError, KeyError, TypeError) as e:
            _log.warning("run report: bad Job.report_thresholds (%s) - ignored", e)
    return t


def worst(verdicts: list[str]) -> str:
    """The worst of these verdicts ("" when none)."""
    return max(verdicts, key=lambda v: _RANK.get(v, 0), default="") if verdicts else ""


def spans(frames: list[int]) -> list[tuple[int, int]]:
    """[3, 4, 5, 9] -> [(3, 5), (9, 9)]."""
    out: list[tuple[int, int]] = []
    for f in sorted(set(frames)):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def collapse_spans(
    area: list[float | None], first: int, ratio: float, min_frames: int
) -> list[tuple[int, int]]:
    """Frame spans where an object's area falls under `ratio` x its median
    (a missing frame counts as 0), at least `min_frames` long. `area[i]` is
    frame `first + i`."""
    present = [a for a in area if a is not None and a > 0]
    if len(present) < 3:
        return []
    present.sort()
    median = present[len(present) // 2]
    low = [first + i for i, a in enumerate(area) if (a or 0.0) < ratio * median]
    return [s for s in spans(low) if s[1] - s[0] + 1 >= max(1, int(min_frames))]


def jump_frames(area: list[float | None], first: int, factor: float) -> list[int]:
    """Frames whose area differs from the previous frame's by more than
    `factor` (either way). Missing frames break the chain, not flagged."""
    out: list[int] = []
    for i in range(1, len(area)):
        a, b = area[i - 1], area[i]
        if not a or not b:
            continue
        if max(a, b) / min(a, b) > factor:
            out.append(first + i)
    return out


__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_POLICY",
    "ENV_VAR",
    "FAILED",
    "GOOD",
    "WEAK",
    "Limit",
    "Thresholds",
    "collapse_spans",
    "jump_frames",
    "load_thresholds",
    "spans",
    "worst",
]
