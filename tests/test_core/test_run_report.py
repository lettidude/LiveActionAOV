"""Run report schema, verdicts and needs-a-human rules (no GPU, no plate)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import License
from live_action_aov.core.run_report import (
    SCHEMA_VERSION,
    RunReportBuilder,
    load_report,
    report_path,
    write_report,
)
from live_action_aov.core.verdicts import (
    ENV_VAR,
    Limit,
    Thresholds,
    collapse_spans,
    jump_frames,
    load_thresholds,
    spans,
    worst,
)


class _Cls:
    name = "fake"
    version = "1.0"

    def __init__(self, commercial: bool = True) -> None:
        self._lic = License(spdx="MIT" if commercial else "CC-BY-NC-4.0", commercial_use=commercial)

    def declared_license(self) -> License:
        return self._lic


class _Inst:
    def __init__(self, fields: dict[str, Any]) -> None:
        self._f = fields

    def report_fields(self) -> dict[str, Any]:
        return self._f


def _job(tmp_path: Path, **kw: Any) -> Job:
    out = tmp_path / "sh010" / "v003"
    shot = Shot(
        name="sh010",
        folder=tmp_path,
        output_dir=out,
        sequence_pattern="p.####.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
    )
    return Job(shot=shot, passes=[PassConfig(name="a")], **kw)


# --- verdicts --------------------------------------------------------------


def test_limit_both_directions() -> None:
    low = Limit(1.0, 2.0)
    assert [low.judge(v) for v in (0.5, 1.5, 3.0)] == ["GOOD", "WEAK", "FAILED"]
    high = Limit(25.0, 20.0, higher_is_better=True)
    assert [high.judge(v) for v in (30, 22, 10)] == ["GOOD", "WEAK", "FAILED"]


def test_worst_and_spans() -> None:
    assert worst(["GOOD", "FAILED", "WEAK"]) == "FAILED"
    assert worst([]) == ""
    assert spans([5, 3, 4, 9]) == [(3, 5), (9, 9)]


def test_collapse_and_jump() -> None:
    area: list[float | None] = [0.1, 0.1, 0.1, 0.01, None, 0.1, 0.1, 0.005, 0.1]
    # frames 1003-1004 low (a None counts as 0), 1007 alone is too short
    assert collapse_spans(area, 1000, 0.2, 2) == [(1003, 1004)]
    assert collapse_spans(area, 1000, 0.2, 1) == [(1003, 1004), (1007, 1007)]
    assert jump_frames(area, 1000, 3.0) == [1003, 1007, 1008]


def test_overrides_env_then_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "t.json"
    f.write_text(
        json.dumps(
            {
                "limits": {
                    "splat.holdout_psnr_db": {"good": 30},
                    "new.metric": {"good": 1, "weak": 2},
                },
                "policy": {"weak_needs_human": False},
            }
        )
    )
    monkeypatch.setenv(ENV_VAR, str(f))
    t = load_thresholds({"limits": {"camera.reprojection_error_px": None}})
    assert t.limits["splat.holdout_psnr_db"].good == 30.0
    assert t.limits["splat.holdout_psnr_db"].weak == 20.0  # untouched half kept
    assert t.limits["splat.holdout_psnr_db"].higher_is_better
    assert t.limits["new.metric"].weak == 2.0
    assert "camera.reprojection_error_px" not in t.limits  # None switches it off
    assert t.policy["weak_needs_human"] is False
    assert t.sources == ("defaults", str(f), "job")


def test_bad_override_file_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing.json"))
    assert load_thresholds().sources == ("defaults",)


# --- builder ---------------------------------------------------------------


def _run(tmp_path: Path, plan: list[tuple[str, Any, str, dict[str, Any]]], **job_kw: Any):
    job = _job(tmp_path, **job_kw)
    rr = RunReportBuilder(job)
    for name, cls, what, fields in plan:
        rr.declare(name, cls)
        if what == "skip":
            rr.pass_skipped(name, "needs camera_solve")
            continue
        rr.pass_started(name)
        if what == "fail":
            rr.pass_failed(name, "boom", frame_range=(1001, 1010), instance=_Inst(fields))
        else:
            rr.pass_done(
                name, frame_range=(1001, 1010), vram_gb=3.21, channels={"Z"}, instance=_Inst(fields)
            )
    return job, rr.finish(job.shot, "done")


def test_statuses_fallbacks_and_flags(tmp_path: Path) -> None:
    job, rep = _run(
        tmp_path,
        [
            ("depth", _Cls(), "ok", {}),
            (
                "ltx",
                _Cls(),
                "ok",
                {
                    "engine": "vitmatte",
                    "fallbacks": [
                        {"wanted": "ltx", "used": "vitmatte", "reason": "nothing running"}
                    ],
                },
            ),
            ("cam", _Cls(), "fail", {}),
            ("splat", _Cls(), "skip", {}),
            ("nc", _Cls(commercial=False), "ok", {}),
        ],
    )
    st = {p.name: p.status for p in rep.passes}
    assert st == {
        "depth": "ran",
        "ltx": "fell_back",
        "cam": "failed",
        "splat": "skipped",
        "nc": "ran",
    }
    kinds = sorted(n.kind for n in rep.needs_human)
    assert kinds == ["fallback", "non_commercial", "pass_failed", "pass_skipped"]
    ltx = next(p for p in rep.passes if p.name == "ltx")
    assert ltx.engine == "vitmatte" and ltx.peak_vram_gb == 3.21 and ltx.channels == ["Z"]
    assert rep.schema_version == SCHEMA_VERSION
    assert rep.shot.run == "sh010_v003"
    assert "1 fell back (ltx -> vitmatte)" in rep.summary
    assert "4 need a human" in rep.summary


def test_partial_and_expected_fallbacks_need_no_human(tmp_path: Path) -> None:
    _, rep = _run(
        tmp_path,
        [
            (
                "ltx",
                _Cls(),
                "ok",
                {
                    "engine": "ltx",
                    "fallbacks": [
                        {"wanted": "ltx", "used": "vitmatte", "objects": ["car"], "partial": True}
                    ],
                },
            ),
            (
                "cam",
                _Cls(),
                "ok",
                {
                    "engine": "pycolmap/depth_pnp",
                    "verdict": "GOOD",
                    "fallbacks": [
                        {
                            "wanted": "pycolmap/incremental",
                            "used": "pycolmap/depth_pnp",
                            "expected": True,
                        }
                    ],
                },
            ),
        ],
    )
    st = {p.name: p.status for p in rep.passes}
    assert st == {"ltx": "ran", "cam": "fell_back"}
    assert rep.needs_human == []


def test_metric_verdicts_and_pass_verdict_wins(tmp_path: Path) -> None:
    _, rep = _run(
        tmp_path,
        [
            (
                "splat",
                _Cls(),
                "ok",
                {
                    "metrics": {
                        "splat.holdout_psnr_db": 22.0,
                        "splat.holdout_psnr_min_db": {"value": 12.0, "frames": [1005, 1005]},
                        "splat.gaussians": 1000,
                    },
                    "section": "splat",
                    "summary": {"psnr": 22.0},
                },
            ),
            # the solver says GOOD; its in-sample error alone would say FAILED
            (
                "cam",
                _Cls(),
                "ok",
                {"verdict": "GOOD", "metrics": {"camera.reprojection_error_px": 5.0}},
            ),
        ],
    )
    splat = next(p for p in rep.passes if p.name == "splat")
    assert splat.verdict == "FAILED"
    assert "splat.holdout_psnr_min_db 12" in splat.verdict_reason
    assert {c.metric for c in splat.checks} == {
        "splat.holdout_psnr_db",
        "splat.holdout_psnr_min_db",
    }
    cam = next(p for p in rep.passes if p.name == "cam")
    assert cam.verdict == "GOOD" and cam.checks[0].verdict == "FAILED"
    flag = next(n for n in rep.needs_human if n.kind == "verdict")
    assert flag.source == "splat" and flag.frames == (1005, 1005)
    assert rep.summaries["splat"] == {"pass": "splat", "psnr": 22.0, "verdict": "FAILED"}


def test_job_thresholds_change_the_verdict(tmp_path: Path) -> None:
    fields = {"metrics": {"splat.holdout_psnr_db": 22.0}}
    _, rep = _run(
        tmp_path,
        [("splat", _Cls(), "ok", fields)],
        report_thresholds={"limits": {"splat.holdout_psnr_db": {"good": 21}}},
    )
    assert rep.passes[0].verdict == "GOOD"
    assert rep.thresholds["sources"] == ["defaults", "job"]


def test_objects_area_collapse(tmp_path: Path) -> None:
    area = [0.05] * 4 + [0.001, 0.0, 0.0] + [0.05] * 3
    _, rep = _run(
        tmp_path,
        [
            (
                "sam",
                _Cls(),
                "ok",
                {
                    "engine": "sam3.1",
                    "objects": [{"id": 3, "name": "person", "frames": [1001, 1010], "area": area}],
                },
            )
        ],
    )
    obj = rep.passes[0].objects[0]
    assert obj.frames == (1001, 1010) and obj.frames_present == 10
    flag = next(n for n in rep.needs_human if n.kind == "matte_collapse")
    assert flag.frames == (1005, 1007)
    assert any("area jumps" in w for w in rep.passes[0].warnings)


def test_windows_merge(tmp_path: Path) -> None:
    job = _job(tmp_path)
    rr = RunReportBuilder(job)
    rr.declare("flow", _Cls())
    for a, b in ((1001, 1005), (1006, 1010)):
        rr.pass_started("flow")
        rr.pass_done(
            "flow",
            frame_range=(a, b),
            vram_gb=float(b - 1000),
            instance=_Inst({"objects": [{"id": 1, "frames": [a, b], "area": [0.1] * (b - a + 1)}]}),
        )
    rep = rr.finish(job.shot, "done")
    p = rep.passes[0]
    assert p.runs == 2 and p.frames == (1001, 1010) and p.peak_vram_gb == 10.0
    assert p.objects[0].frames == (1001, 1010) and len(p.objects[0].area) == 10


def test_bad_contribution_is_a_warning_not_a_crash(tmp_path: Path) -> None:
    _, rep = _run(tmp_path, [("x", _Cls(), "ok", {"no_such_field": 1})])
    assert rep.passes[0].status == "ran"
    assert any("report_fields() unusable" in w for w in rep.passes[0].warnings)


def test_failed_run_and_roundtrip(tmp_path: Path) -> None:
    job = _job(tmp_path)
    rr = RunReportBuilder(job)
    rr.declare("a", _Cls())
    rep = rr.finish(job.shot, "failed", "RuntimeError: out of memory")
    assert rep.passes[0].status == "not_run"
    assert rep.needs_human[0].kind == "run_failed"
    path = write_report(rep, report_path(job.shot))
    assert path == tmp_path / "sh010" / "v003" / "logs" / "sh010_v003.run_report.json"
    back = load_report(path)
    assert back.summary == rep.summary and back.status == "failed"
    data = json.loads(path.read_text())
    data["schema_version"] = "2.0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="schema"):
        load_report(path)


def test_thresholds_dict_is_serialisable() -> None:
    json.dumps(Thresholds().as_dict())
