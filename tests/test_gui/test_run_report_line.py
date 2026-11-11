"""After a run the shot list shows the run report: a needs-a-human badge on
the row, the one-line summary and the reasons in the tooltip."""

from __future__ import annotations

from pathlib import Path

from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.run_report import RunReportBuilder, report_path, write_report
from live_action_aov.gui.shot_list import _format_item_label, _format_item_tooltip
from live_action_aov.gui.shot_state import ShotState
from live_action_aov.gui.submit_worker import read_run_report


def _state() -> ShotState:
    return ShotState(
        name="sh010",
        folder=Path("A:/plates/sh010"),
        sequence_pattern="sh010.####.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
    )


def test_badge_and_tooltip() -> None:
    s = _state()
    s.status = "done"
    assert "⚑" not in _format_item_label(s)
    s.last_report_summary = "sh010_v001 done: 2 ran; 1 needs a human; 0m10s"
    s.last_needs_human = ["cam FAILED: held-out <b>4 px</b>"]
    assert _format_item_label(s).endswith("✓ done   ⚑ 1")
    s.status = "cancelled"
    assert _format_item_label(s).endswith("⚑ 1")
    tip = _format_item_tooltip(s)
    assert "Last run: sh010_v001 done" in tip
    assert "&lt;b&gt;4 px&lt;/b&gt;" in tip  # escaped, not rendered


def test_read_run_report(tmp_path: Path) -> None:
    shot = Shot(
        name="sh010",
        folder=tmp_path,
        output_dir=tmp_path / "v001",
        sequence_pattern="p.####.exr",
        frame_range=(1, 2),
        resolution=(64, 64),
    )
    assert read_run_report(shot) == ("", ())
    job = Job(shot=shot, passes=[PassConfig(name="a")])
    rr = RunReportBuilder(job)
    rep = rr.finish(shot, "failed", "boom")
    shot.sidecars["run_report"] = write_report(rep, report_path(shot))
    summary, human = read_run_report(shot)
    assert summary.startswith("sh010_v001 failed")
    assert human == ("boom",)
