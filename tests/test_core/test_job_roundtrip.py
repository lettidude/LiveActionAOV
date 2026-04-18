"""Job ↔ YAML round-trip, PassConfig / Shot / Task validation."""

from __future__ import annotations

from pathlib import Path

from live_action_aov.core.job import Job, PassConfig, Shot


def _make_shot() -> Shot:
    return Shot(
        name="sh010",
        folder=Path("/tmp/sh010"),
        sequence_pattern="sh010.####.exr",
        frame_range=(1001, 1010),
        resolution=(1920, 1080),
        pixel_aspect=1.0,
        passes_enabled=["flow", "depth"],
    )


def test_job_yaml_roundtrip() -> None:
    job = Job(
        shot=_make_shot(),
        passes=[PassConfig(name="flow"), PassConfig(name="depth", params={"smooth": True})],
    )
    yaml_src = job.to_yaml()
    job2 = Job.from_yaml(yaml_src)
    assert job2.shot.name == job.shot.name
    assert job2.shot.frame_range == job.shot.frame_range
    assert job2.shot.resolution == job.shot.resolution
    assert [p.name for p in job2.passes] == ["flow", "depth"]
    assert job2.passes[1].params == {"smooth": True}


def test_job_to_tasks_chunks_by_frame_range() -> None:
    job = Job(
        shot=_make_shot(),
        passes=[PassConfig(name="flow")],
        chunk_size=4,
    )
    tasks = job.to_tasks()
    # frames 1001..1010 with chunk_size 4 = three tasks of sizes 4, 4, 2
    assert len(tasks) == 3
    assert tasks[0].frame_range == (1001, 1004)
    assert tasks[1].frame_range == (1005, 1008)
    assert tasks[2].frame_range == (1009, 1010)
    assert all(t.pass_name == "flow" for t in tasks)
