"""LocalExecutor honours CancelToken at safe checkpoints.

Two end-to-end tests using `NoOpPass` (no HF / torch download):

1. A token cancelled BEFORE submit() raises immediately at the first
   checkpoint and stamps `shot.status == "cancelled"`. Sidecar files
   may or may not exist depending on which checkpoint fired first —
   we don't assert on disk state, only on the lifecycle.

2. A token cancelled DURING submit() (via the progress callback as a
   convenient hook into the running executor) likewise raises at the
   next checkpoint and stamps cancelled status.

We deliberately don't test mid-pass cancellation — torch inference
calls hold the GIL in C++ land and the executor's documented
guarantee is "between passes / before each sidecar write".
"""

from __future__ import annotations

from pathlib import Path

import pytest

import live_action_aov
from live_action_aov.core.cancel import CancelledError, CancelToken
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.io.oiio_io import HAS_OIIO

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed"),
]


def _build_job(folder: Path) -> Job:
    shot = Shot(
        name="cancel_test",
        folder=folder,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, 5),
        resolution=(640, 360),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=["noop"],
    )
    return Job(shot=shot, passes=[PassConfig(name="noop")])


def test_cancel_before_submit_raises_immediately(test_plate_1080p: Path) -> None:
    """A token already flipped at submit() entry trips the first
    checkpoint (between passes) and the executor never reaches the
    sidecar-write phase."""
    job = _build_job(test_plate_1080p)
    token = CancelToken()
    token.cancel("pre-flipped")

    with pytest.raises(CancelledError):
        live_action_aov.run(job, cancel=token)

    assert job.shot.status == "cancelled"


def test_cancel_during_submit_raises_at_next_checkpoint(test_plate_1080p: Path) -> None:
    """Flipping the token from inside the progress callback simulates
    a Cancel-button click landing while the executor is mid-flight.
    The next checkpoint observes the flag and raises."""
    job = _build_job(test_plate_1080p)
    token = CancelToken()

    # Use the progress callback as a thread-safe-enough hook into the
    # running executor's progression. As soon as the executor reports
    # ANY progress past resolution (>= 0.05), flip the token.
    triggered = {"value": False}

    def _progress(fraction: float, label: str) -> None:
        if not triggered["value"] and fraction >= 0.05:
            token.cancel("triggered from progress")
            triggered["value"] = True

    from live_action_aov.executors.local import LocalExecutor

    with pytest.raises(CancelledError):
        LocalExecutor().submit(job, progress_callback=_progress, cancel=token)

    assert job.shot.status == "cancelled"
    assert triggered["value"] is True


def test_uncancelled_run_completes_normally(test_plate_1080p: Path) -> None:
    """Sanity check — passing `cancel=None` (or just leaving the
    default) keeps the historic non-cancellable behaviour. Catches
    the obvious regression where adding the cancel parameter
    accidentally short-circuits all runs."""
    job = _build_job(test_plate_1080p)
    live_action_aov.run(job)  # no cancel kwarg
    assert job.shot.status == "done"
