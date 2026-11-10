# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""The run report: one machine-readable file per run that says what happened.

``logs/<shot>_v###.run_report.json``, written by the executor at the end of
every run (done, failed or cancelled). Per pass: ran / skipped / fell back
(to what, why) / failed (why), timings, peak VRAM and RAM, the engine that
actually ran, the objects found, metrics and a verdict; for the run: a
``needs_human`` list (one reason and, when known, the frames concerned) and a
one-line summary the GUI and the CLI show.

Passes add their own fields through `UtilityPass.report_fields()` (a dict,
see `PassContribution`); the executor has no pass-specific code. Verdict
thresholds and the needs-a-human policy live in `core.verdicts`.

The schema is versioned (`SCHEMA_VERSION`): a reader checks the major
number; fields are only ever added within a major version.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from live_action_aov.core.verdicts import (
    Thresholds,
    collapse_spans,
    jump_frames,
    load_thresholds,
    worst,
)
from live_action_aov.core.verdicts import spans as frame_spans

_log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
REPORT_KIND = "liveaov.run_report"
REPORT_SUFFIX = ".run_report.json"

PassStatus = Literal["ran", "fell_back", "skipped", "failed", "not_run"]
RunStatus = Literal["done", "failed", "cancelled", "running"]

NOTES = [
    "seconds: wall clock of the pass, model load included.",
    "peak_vram_gb: what torch reserved in THIS process during the pass; a model run in a "
    "child process (SAM 3.1, the COLMAP worker) or in ComfyUI (LTX) is not counted.",
    "peak_ram_gb: resident memory of this process and its child processes, sampled "
    "every 0.5 s (a spike shorter than that can be missed).",
    "frames: plate frame numbers, ends included.",
]


# ----------------------------------------------------------------------
# Schema
# ----------------------------------------------------------------------


class Fallback(BaseModel):
    """The pass did not run the engine it was asked for."""

    wanted: str
    used: str
    reason: str = ""
    #: Only these objects (empty = all of the pass's work).
    objects: list[str] = Field(default_factory=list)
    #: True when it is only some objects, by design (a prop LTX does not see).
    partial: bool = False
    #: A designed route judged by its own verdict (SfM -> nodal -> PnP
    #: camera): reported, but no human needed for the route itself.
    expected: bool = False


class NeedsHuman(BaseModel):
    source: str  # a pass name, or "run"
    kind: str  # machine tag: pass_failed, verdict, fallback, matte_collapse, ...
    reason: str
    frames: tuple[int, int] | None = None


class Check(BaseModel):
    """One metric judged against `core.verdicts` limits."""

    metric: str
    value: float
    verdict: str
    good: float
    weak: float
    higher_is_better: bool = False
    frames: tuple[int, int] | None = None


class ObjectReport(BaseModel):
    """One object a pass found (passes may add their own fields)."""

    model_config = ConfigDict(extra="allow")

    id: int | str
    name: str = ""
    frames: tuple[int, int] | None = None  # first / last frame tracked
    frames_present: int = 0
    #: Fraction of the frame covered, one value per frame from `frames[0]`
    #: to `frames[1]`, None where the object is absent.
    area: list[float | None] = Field(default_factory=list)


class PassReport(BaseModel):
    name: str
    model: str = ""
    version: str = ""
    license: str = ""
    commercial: bool = True
    status: PassStatus = "not_run"
    reason: str = ""  # why skipped / failed
    engine: str = ""  # what actually ran ("sam3.1", "ltx", "vitmatte", "pycolmap/depth_pnp" ...)
    engine_reason: str = ""
    fallbacks: list[Fallback] = Field(default_factory=list)
    frames: tuple[int, int] | None = None
    runs: int = 0  # >1 when the clip ran in windows
    seconds: float = 0.0
    peak_vram_gb: float | None = None
    peak_ram_gb: float | None = None
    channels: list[str] = Field(default_factory=list)
    objects: list[ObjectReport] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    checks: list[Check] = Field(default_factory=list)
    verdict: str = ""  # GOOD / WEAK / FAILED / "" (not judged)
    verdict_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ShotInfo(BaseModel):
    name: str
    run: str = ""  # <shot>_v###
    plate: str = ""
    pattern: str = ""
    frame_range: tuple[int, int] = (0, 0)
    resolution: tuple[int, int] = (0, 0)
    proxy_long_edge: int | None = None
    crop: tuple[int, int, int, int] | None = None
    colorspace: str = ""
    output_dir: str = ""


class Environment(BaseModel):
    tool_version: str = ""
    python: str = ""
    platform: str = ""
    torch: str = ""
    cuda: str = ""
    gpu: str = ""
    params_hash: str = ""


class RunReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    kind: str = REPORT_KIND
    job_id: str = ""
    status: RunStatus = "running"
    error: str = ""
    started: str = ""
    finished: str = ""
    seconds: float = 0.0
    peak_ram_gb: float | None = None
    peak_vram_gb: float | None = None
    shot: ShotInfo
    environment: Environment = Field(default_factory=Environment)
    passes: list[PassReport] = Field(default_factory=list)
    #: Sections a pass asked for (`PassContribution.section`): "camera", "splat" ...
    summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    needs_human: list[NeedsHuman] = Field(default_factory=list)
    sidecars: dict[str, str] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    notes: list[str] = Field(default_factory=lambda: list(NOTES))


class PassContribution(BaseModel):
    """What `UtilityPass.report_fields()` may return (all optional).

    `metrics` values are numbers, or ``{"value": x, "frames": [a, b]}`` to
    say which frames a number is about (the worst held-out frame ...).
    `section` + `summary` put a block at the top of the report
    (``summaries[section]``) - the camera's verdict, the splat's PSNR.
    """

    model_config = ConfigDict(extra="forbid")

    engine: str = ""
    engine_reason: str = ""
    fallbacks: list[Fallback] = Field(default_factory=list)
    objects: list[ObjectReport] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    verdict: str = ""
    verdict_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    needs_human: list[NeedsHuman] = Field(default_factory=list)
    section: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# ----------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------


class PeakRamSampler:
    """Peak resident RAM of this process + its children, sampled on a
    background thread. `window_start()` / `window_peak()` give the peak of
    one pass; `peak_gb` the whole run's. No psutil: everything is None."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self._run_peak = 0
        self._win_peak = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._proc: Any = psutil.Process()
        except Exception:
            self._proc = None

    def _rss(self) -> int:
        if self._proc is None:
            return 0
        try:
            total = int(self._proc.memory_info().rss)
        except Exception:
            return 0
        try:
            kids = self._proc.children(recursive=True)
        except Exception:
            kids = []
        for c in kids:
            try:
                total += int(c.memory_info().rss)
            except Exception:
                pass
        return total

    def sample(self) -> None:
        v = self._rss()
        with self._lock:
            self._run_peak = max(self._run_peak, v)
            self._win_peak = max(self._win_peak, v)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def start(self) -> PeakRamSampler:
        if self._proc is not None and self._thread is None:
            self.sample()
            self._thread = threading.Thread(target=self._loop, name="laov-ram", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def window_start(self) -> None:
        with self._lock:
            self._win_peak = 0
        self.sample()

    def window_peak(self) -> float | None:
        self.sample()
        with self._lock:
            return _gb(self._win_peak) if self._proc is not None else None

    @property
    def peak_gb(self) -> float | None:
        with self._lock:
            return _gb(self._run_peak) if self._proc is not None else None


def _gb(n_bytes: int) -> float:
    return round(n_bytes / 2**30, 2)


# ----------------------------------------------------------------------
# Builder (used by the executor)
# ----------------------------------------------------------------------


class RunReportBuilder:
    """Collects a run's report while it runs. Every method swallows its own
    errors (logged): the report must never fail a run."""

    def __init__(self, job: Any, thresholds: Thresholds | None = None) -> None:
        self.job = job
        self.t0 = time.time()
        self.started = _now()
        self.thresholds = thresholds or load_thresholds(getattr(job, "report_thresholds", None))
        self.passes: dict[str, PassReport] = {}
        self._areas: dict[str, dict[str, dict[int, float]]] = {}  # pass -> obj id -> frame -> area
        self.summaries: dict[str, dict[str, Any]] = {}
        self.extra_human: list[NeedsHuman] = []
        self.ram = PeakRamSampler().start()
        self._vram_peak: float | None = None
        self._pass_t0: dict[str, float] = {}

    # --- passes ---------------------------------------------------------

    def declare(self, name: str, cls: Any) -> None:
        try:
            lic = cls.declared_license()
            self.passes[name] = PassReport(
                name=name,
                model=str(getattr(cls, "name", name)),
                version=str(getattr(cls, "version", "")),
                license=str(lic.spdx),
                commercial=bool(lic.commercial_use),
            )
        except Exception:
            _log.debug("run report: declare %s failed", name, exc_info=True)
            self.passes[name] = PassReport(name=name)

    def pass_started(self, name: str) -> None:
        self._pass_t0[name] = time.time()
        self.ram.window_start()

    def _end(
        self,
        name: str,
        frame_range: tuple[int, int] | None,
        vram_gb: float | None,
        channels: set[str] | None,
    ) -> PassReport:
        pr = self.passes.setdefault(name, PassReport(name=name))
        t0 = self._pass_t0.pop(name, None)
        if t0 is not None:
            pr.seconds = round(pr.seconds + time.time() - t0, 2)
            pr.runs += 1
        ram = self.ram.window_peak()
        if ram is not None:
            pr.peak_ram_gb = max(pr.peak_ram_gb or 0.0, ram)
        if vram_gb is not None:
            v = round(float(vram_gb), 2)
            pr.peak_vram_gb = max(pr.peak_vram_gb or 0.0, v)
            self._vram_peak = max(self._vram_peak or 0.0, v)
        if frame_range is not None:
            a, b = int(frame_range[0]), int(frame_range[1])
            pr.frames = (min(a, pr.frames[0]), max(b, pr.frames[1])) if pr.frames else (a, b)
        if channels:
            pr.channels = sorted(set(pr.channels) | set(channels))
        return pr

    def pass_done(
        self,
        name: str,
        *,
        frame_range: tuple[int, int] | None = None,
        vram_gb: float | None = None,
        channels: set[str] | None = None,
        instance: Any = None,
    ) -> None:
        try:
            pr = self._end(name, frame_range, vram_gb, channels)
            if pr.status not in ("failed", "skipped"):
                pr.status = "ran"
            self._contribute(pr, instance)
        except Exception:
            _log.warning("run report: could not record %s", name, exc_info=True)

    def pass_failed(
        self,
        name: str,
        reason: str,
        *,
        frame_range: tuple[int, int] | None = None,
        vram_gb: float | None = None,
        instance: Any = None,
    ) -> None:
        try:
            pr = self._end(name, frame_range, vram_gb, None)
            pr.status, pr.reason = "failed", _short(reason)
            self._contribute(pr, instance)
        except Exception:
            _log.warning("run report: could not record %s", name, exc_info=True)

    def pass_skipped(self, name: str, reason: str) -> None:
        pr = self.passes.setdefault(name, PassReport(name=name))
        pr.status, pr.reason = "skipped", _short(reason)

    def _contribute(self, pr: PassReport, instance: Any) -> None:
        hook = getattr(instance, "report_fields", None)
        if instance is None or not callable(hook):
            return
        try:
            raw = hook() or {}
            c = PassContribution.model_validate(raw)
        except Exception as e:
            _log.warning("run report: %s.report_fields() unusable: %s", pr.name, e)
            pr.warnings.append(f"report_fields() unusable: {_short(str(e))}")
            return
        if c.engine:
            pr.engine = c.engine
        if c.engine_reason:
            pr.engine_reason = c.engine_reason
        seen = {(f.wanted, f.used, f.reason) for f in pr.fallbacks}
        for f in c.fallbacks:
            if (f.wanted, f.used, f.reason) not in seen:
                pr.fallbacks.append(f)
        areas = self._areas.setdefault(pr.name, {})
        by_id = {str(o.id): o for o in pr.objects}
        for o in c.objects:
            key = str(o.id)
            track = areas.setdefault(key, {})
            if o.area and o.frames:
                for i, a in enumerate(o.area):
                    if a is not None:
                        track[o.frames[0] + i] = float(a)
            have = by_id.get(key)
            if have is None:
                by_id[key] = o.model_copy(update={"area": []})
                pr.objects.append(by_id[key])
            elif o.frames:
                fr = have.frames
                have.frames = (min(fr[0], o.frames[0]), max(fr[1], o.frames[1])) if fr else o.frames
        pr.metrics.update(c.metrics)
        if c.verdict:
            pr.verdict = c.verdict.upper()
        if c.verdict_reason:
            pr.verdict_reason = c.verdict_reason
        for w in c.warnings:
            if w not in pr.warnings:
                pr.warnings.append(w)
        for n in c.needs_human:
            self.extra_human.append(n.model_copy(update={"source": n.source or pr.name}))
        if c.section:
            self.summaries.setdefault(c.section, {}).update({"pass": pr.name, **c.summary})
        pr.extra.update(c.extra)

    # --- the end --------------------------------------------------------

    def finish(self, shot: Any, status: str, error: str = "") -> RunReport:
        self.ram.stop()
        th = self.thresholds
        pol = th.policy
        human: list[NeedsHuman] = []
        for pr in self.passes.values():
            self._finish_objects(pr, pol, human)
            self._judge(pr)
            full = [f for f in pr.fallbacks if not f.partial]
            if pr.status == "ran" and full:
                pr.status = "fell_back"
            if pr.status in ("failed", "skipped") and pol.get("failed_needs_human", True):
                human.append(
                    NeedsHuman(
                        source=pr.name,
                        kind=f"pass_{pr.status}",
                        reason=f"{pr.name} {pr.status}: {pr.reason}",
                    )
                )
            if (pr.verdict == "FAILED" and pol.get("verdict_failed_needs_human", True)) or (
                pr.verdict == "WEAK" and pol.get("weak_needs_human", True)
            ):
                frames = next(
                    (c.frames for c in pr.checks if c.verdict == pr.verdict and c.frames), None
                )
                human.append(
                    NeedsHuman(
                        source=pr.name,
                        kind="verdict",
                        reason=f"{pr.name} {pr.verdict}"
                        + (f": {pr.verdict_reason}" if pr.verdict_reason else ""),
                        frames=frames,
                    )
                )
            if full and pol.get("fallback_needs_human", True):
                for f in (f for f in full if not f.expected):
                    human.append(
                        NeedsHuman(
                            source=pr.name,
                            kind="fallback",
                            reason=f"{pr.name} used {f.used} instead of {f.wanted}"
                            + (f": {f.reason}" if f.reason else ""),
                        )
                    )
            if (
                not pr.commercial
                and pr.status in ("ran", "fell_back")
                and pol.get("noncommercial_needs_human", True)
            ):
                human.append(
                    NeedsHuman(
                        source=pr.name,
                        kind="non_commercial",
                        reason=f"{pr.name} is non-commercial ({pr.license}): not for paid work",
                    )
                )
        human.extend(self.extra_human)
        if status == "failed":
            human.insert(
                0, NeedsHuman(source="run", kind="run_failed", reason=_short(error) or "failed")
            )
        rep = RunReport(
            job_id=str(getattr(self.job, "job_id", "")),
            status=cast(
                RunStatus, status if status in ("done", "failed", "cancelled") else "running"
            ),
            error=_short(error),
            started=self.started,
            finished=_now(),
            seconds=round(time.time() - self.t0, 1),
            peak_ram_gb=self.ram.peak_gb,
            peak_vram_gb=self._vram_peak,
            shot=_shot_info(shot),
            environment=_environment(self.job),
            passes=list(self.passes.values()),
            summaries=self.summaries,
            needs_human=human,
            sidecars={k: str(v) for k, v in (getattr(shot, "sidecars", {}) or {}).items()},
            thresholds=th.as_dict(),
        )
        rep.summary = one_line(rep)
        return rep

    def _finish_objects(self, pr: PassReport, pol: dict[str, Any], human: list[NeedsHuman]) -> None:
        areas = self._areas.get(pr.name, {})
        ratio = float(pol.get("matte_collapse_ratio", 0.2))
        min_frames = int(pol.get("matte_collapse_min_frames", 2))
        jump = float(pol.get("matte_jump_ratio", 3.0))
        min_area = float(pol.get("matte_min_area", 0.0))
        for o in pr.objects:
            track = areas.get(str(o.id), {})
            if not track:
                continue
            a, b = min(track), max(track)
            o.frames = (a, b)
            o.frames_present = len(track)
            o.area = [round(track[f], 6) if f in track else None for f in range(a, b + 1)]
            present = sorted(v for v in track.values() if v > 0)
            if not present or present[len(present) // 2] < min_area:
                continue
            label = f"'{o.name}' (id {o.id})" if o.name else f"id {o.id}"
            for s in collapse_spans(o.area, a, ratio, min_frames):
                human.append(
                    NeedsHuman(
                        source=pr.name,
                        kind="matte_collapse",
                        reason=f"{pr.name}: {label} lost or collapsed "
                        f"(area < {ratio:.0%} of its median)",
                        frames=s,
                    )
                )
            jumps = jump_frames(o.area, a, jump)
            if jumps:
                where = ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in frame_spans(jumps))
                pr.warnings.append(f"{label}: area jumps more than {jump:g}x on frames {where}")

    def _judge(self, pr: PassReport) -> None:
        pr.checks = []
        for metric, raw in pr.metrics.items():
            lim = self.thresholds.limits.get(metric)
            if lim is None:
                continue
            value, frames = raw, None
            if isinstance(raw, dict):
                value, fr = raw.get("value"), raw.get("frames")
                frames = (int(fr[0]), int(fr[1])) if fr else None
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            pr.checks.append(
                Check(
                    metric=metric,
                    value=float(value),
                    verdict=lim.judge(float(value)),
                    good=lim.good,
                    weak=lim.weak,
                    higher_is_better=lim.higher_is_better,
                    frames=frames,
                )
            )
        if pr.verdict or not pr.checks:
            return  # the pass judged itself, or nothing to judge
        pr.verdict = worst([c.verdict for c in pr.checks])
        bad = [c for c in pr.checks if c.verdict == pr.verdict and c.verdict != "GOOD"]
        pr.verdict_reason = "; ".join(
            f"{c.metric} {c.value:g} ({'want' if c.verdict == 'WEAK' else 'needs'} "
            f"{'>=' if c.higher_is_better else '<='} "
            f"{c.good if c.verdict == 'WEAK' else c.weak:g})"
            for c in bad
        )


# ----------------------------------------------------------------------
# Reading / writing / summary
# ----------------------------------------------------------------------


def report_path(shot: Any) -> Path:
    """`<run dir>/logs/<shot>_v###.run_report.json`."""
    from live_action_aov.core.output_layout import LOGS_DIR, run_dir, run_stem

    return run_dir(shot) / LOGS_DIR / f"{run_stem(shot)}{REPORT_SUFFIX}"


def write_report(report: RunReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_report(path: Path) -> RunReport:
    """Read a report; refuses another major schema version."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    major = str(data.get("schema_version", "0")).split(".")[0]
    if major != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"run report schema {data.get('schema_version')} (this reads {SCHEMA_VERSION})"
        )
    return RunReport.model_validate(data)


def one_line(rep: RunReport) -> str:
    """`sh010_v003 done: 5 ran, 1 fell back (ltx_alpha_refiner -> vitmatte);
    camera_pycolmap GOOD; 2 need a human; 3m12s, VRAM 18.2 GB, RAM 31 GB`."""
    counts: dict[str, list[PassReport]] = {}
    for p in rep.passes:
        counts.setdefault(p.status, []).append(p)
    bits: list[str] = []
    if counts.get("ran"):
        bits.append(f"{len(counts['ran'])} ran")
    if counts.get("fell_back"):
        fb = ", ".join(
            f"{p.name} -> {next((f.used for f in p.fallbacks if not f.partial), '?')}"
            for p in counts["fell_back"]
        )
        bits.append(f"{len(counts['fell_back'])} fell back ({fb})")
    for st in ("failed", "skipped", "not_run"):
        if counts.get(st):
            names = ", ".join(p.name for p in counts[st])
            bits.append(f"{len(counts[st])} {st.replace('_', ' ')} ({names})")
    head = f"{rep.shot.run or rep.shot.name} {rep.status}: " + (", ".join(bits) or "no passes")
    parts = [head]
    verdicts = [f"{p.name} {p.verdict}" for p in rep.passes if p.verdict]
    if verdicts:
        parts.append(", ".join(verdicts))
    n = len(rep.needs_human)
    parts.append(f"{n} need{'s' if n == 1 else ''} a human" if n else "nothing for a human")
    res = [_dur(rep.seconds)]
    if rep.peak_vram_gb is not None:
        res.append(f"VRAM {rep.peak_vram_gb:.1f} GB")
    if rep.peak_ram_gb is not None:
        res.append(f"RAM {rep.peak_ram_gb:.0f} GB")
    parts.append(", ".join(res))
    return "; ".join(parts)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _short(text: str, n: int = 600) -> str:
    text = str(text or "").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def _shot_info(shot: Any) -> ShotInfo:
    from live_action_aov.core.output_layout import run_dir, run_stem

    crop = getattr(shot, "crop", None)
    return ShotInfo(
        name=str(shot.name),
        run=run_stem(shot),
        plate=str(getattr(shot, "folder", "")),
        pattern=str(getattr(shot, "sequence_pattern", "")),
        frame_range=tuple(shot.frame_range),
        resolution=tuple(getattr(shot, "resolution", (0, 0))),
        proxy_long_edge=getattr(shot, "proxy_long_edge", None),
        crop=tuple(crop) if crop is not None else None,
        colorspace=str(getattr(shot, "colorspace", "")),
        output_dir=str(run_dir(shot)),
    )


def _environment(job: Any) -> Environment:
    env = Environment(
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
    )
    try:
        from live_action_aov import __version__

        env.tool_version = str(__version__)
    except Exception:
        pass
    torch = sys.modules.get("torch")  # never import torch just for the report
    if torch is not None:
        try:
            env.torch = str(torch.__version__)
            env.cuda = str(torch.version.cuda or "")
            if torch.cuda.is_available():
                env.gpu = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    try:
        shot = job.shot
        blob = {
            "passes": [p.model_dump(mode="json") for p in job.passes],
            "post": [p.model_dump(mode="json") for p in job.post],
            "overrides": shot.pass_overrides,
            "frame_range": list(shot.frame_range),
            "proxy": shot.proxy_long_edge,
            "crop": shot.crop,
            "colorspace": shot.colorspace,
        }
        env.params_hash = hashlib.sha256(
            json.dumps(blob, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        pass
    return env


__all__ = [
    "REPORT_KIND",
    "SCHEMA_VERSION",
    "Check",
    "Fallback",
    "NeedsHuman",
    "ObjectReport",
    "PassContribution",
    "PassReport",
    "PeakRamSampler",
    "RunReport",
    "RunReportBuilder",
    "load_report",
    "one_line",
    "report_path",
    "write_report",
]
