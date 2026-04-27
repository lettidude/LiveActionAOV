# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Job / Shot / Task — the data model that moves through the pipeline.

Shot is the human-prepared unit; Job is the execution unit (a Shot plus a
pass list plus farm-shaped fields); Task is the farm chunk (unused in v1 but
already shaped for v2 Deadline — design §14, decision 1).

YAML round-trip is first-class: both CLI and GUI read/write YAML as the
source of truth (design §12).
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from live_action_aov.core.pass_base import DisplayTransformParams

ShotStatus = Literal[
    "new",
    "analyzed",
    "reviewed",
    "queued",
    "running",
    "done",
    "failed",
    "cancelled",
]


class PassConfig(BaseModel):
    """One pass invocation in a job: plugin name + user-supplied params."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class PostConfig(BaseModel):
    """One post-processor invocation in a job (e.g. temporal smoother).

    Post-processors are NOT passes — they mutate per-frame channel outputs
    after all passes have run. Resolved via the `live_action_aov.post`
    entry-point group (design §9).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Shot(BaseModel):
    """Human-prepared ingest unit: a folder of EXRs plus transform + passes.

    Pixel aspect is carried through the pipeline untouched; we work in pixel
    space and never desqueeze (design §8.2).
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    folder: Path
    sequence_pattern: str
    frame_range: tuple[int, int]
    resolution: tuple[int, int]
    pixel_aspect: float = 1.0
    colorspace: str = "auto"

    transform: DisplayTransformParams = Field(default_factory=DisplayTransformParams)
    # Toggle the executor-side display transform wire-up. Off by default
    # so synthetic-array unit tests keep their pass-through behaviour; the
    # CLI flips it on (--display-transform) for real scene-referred plates.
    apply_display_transform: bool = False
    passes_enabled: list[str] = Field(default_factory=list)
    pass_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

    # Dict not single path — Shot.sidecars holds multiple output types from
    # v2a onward (utility/camera/scene) (design §14, decision 9).
    sidecars: dict[str, Path] = Field(default_factory=dict)

    # Optional override for where sidecar EXRs are written. `None` =
    # next to the plate (the historic default). GUI Phase 5 surfaces
    # this so users can route outputs to a subfolder, a reviews drive,
    # or a shot-named folder under a shared render root.
    output_dir: Path | None = None

    # Proxy resolution for fast iteration. `None` = plate-native.
    # When set, every plate frame is resized down to this long-edge
    # before passes see it; sidecars land at the proxy resolution.
    # Big disk-I/O + plate-size-scaling-pass savings on 4K / 6K
    # plates. Pixel aspect is preserved. GUI Output tab flips it.
    proxy_long_edge: int | None = None

    status: ShotStatus = "new"
    notes: str = ""

    # External pipeline IDs — None in v1, populated by v2+ integrations.
    prism_task_id: str | None = None
    shotgrid_version_id: int | None = None
    openpype_version_id: str | None = None
    external_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("folder", mode="before")
    @classmethod
    def _coerce_folder(cls, v: Any) -> Path:
        return Path(v) if not isinstance(v, Path) else v

    @field_validator("output_dir", mode="before")
    @classmethod
    def _coerce_output_dir(cls, v: Any) -> Path | None:
        if v is None or v == "":
            return None
        return Path(v) if not isinstance(v, Path) else v

    @field_validator("frame_range", "resolution", mode="before")
    @classmethod
    def _coerce_tuple(cls, v: Any) -> tuple[int, int]:
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_serializer("folder")
    def _serialize_folder(self, v: Path) -> str:
        return str(v)

    @field_serializer("sidecars")
    def _serialize_sidecars(self, v: dict[str, Path]) -> dict[str, str]:
        return {k: str(p) for k, p in v.items()}


class Task(BaseModel):
    """A farm-chunk unit of work. LocalExecutor iterates these sequentially
    in v1; DeadlineExecutor will dispatch them in v2."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_id: str
    pass_name: str
    frame_range: tuple[int, int]
    dependencies: list[str] = Field(default_factory=list)

    @field_validator("frame_range", mode="before")
    @classmethod
    def _coerce_frame_range(cls, v: Any) -> tuple[int, int]:
        if isinstance(v, list):
            return tuple(v)
        return v


class Job(BaseModel):
    """Executable unit: one Shot + a list of passes to run on it.

    Farm-shaped fields (priority, pool, chunk_size, etc.) are ignored by
    LocalExecutor in v1 — their presence means DeadlineExecutor can be
    dropped in v2 without refactoring core (design §14, decision 1).
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    shot: Shot
    passes: list[PassConfig]
    post: list[PostConfig] = Field(default_factory=list)

    # Farm-shaped fields — LocalExecutor ignores in v1.
    priority: int = 50
    pool: str = "gpu"
    chunk_size: int = 10
    dependencies: list[Job] = Field(default_factory=list)
    gpu_affinity: str | None = None
    max_retries: int = 2
    timeout_minutes: int = 120

    def to_tasks(self) -> list[Task]:
        """Chunk a job into tasks of at most `chunk_size` frames per pass.

        LocalExecutor iterates these sequentially. v2 DeadlineExecutor
        dispatches them in parallel respecting `dependencies`.
        """
        start, end = self.shot.frame_range
        tasks: list[Task] = []
        for pc in self.passes:
            cursor = start
            while cursor <= end:
                chunk_end = min(cursor + self.chunk_size - 1, end)
                tasks.append(
                    Task(
                        job_id=self.job_id,
                        pass_name=pc.name,
                        frame_range=(cursor, chunk_end),
                    )
                )
                cursor = chunk_end + 1
        return tasks

    # --- YAML round-trip ---

    def to_yaml(self) -> str:
        """Serialize to YAML for hand-off between CLI/GUI/farm."""
        data = self.model_dump(mode="json", exclude={"dependencies"})
        return yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, src: str | Path) -> Job:
        """Load a Job from a YAML file or string."""
        if isinstance(src, Path):
            text = src.read_text(encoding="utf-8")
        elif isinstance(src, str) and "\n" not in src and Path(src).exists():
            text = Path(src).read_text(encoding="utf-8")
        else:
            text = src
        data = yaml.safe_load(text)
        return cls.model_validate(data)


__all__ = ["Job", "PassConfig", "PostConfig", "Shot", "ShotStatus", "Task"]
