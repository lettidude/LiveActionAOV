"""Camera-track JSON sidecar schema (spec §20.5).

This is the permanent record for every camera-track pass. `.nk` / `.abc` /
`.fbx` exports all derive from it, so once this shape is pinned it's cheap
to add new exporters later.

Axis convention:
- World is Y-up right-handed (matches Maya / Houdini / Nuke / Blender
  import interpretation).
- Camera local frame is OpenGL (+X right, +Y up, +Z back from look).
  `rotation_matrix` is `world_from_camera` in that convention.
- `rotation_euler_zxy_deg` is the Euler decomposition for Nuke's default
  `rot_order ZXY` (degrees). The matrix is the source of truth; if a
  downstream tool prefers a different rot_order the importer can
  re-decompose from the matrix without a re-solve.

Units: arbitrary in v2a — pycolmap's incremental SfM doesn't recover
metric scale. `conventions.units == "unitless"` flags this. Scale
recovery (from Depth Pro or user-hint) is a v2b task (spec §20.4).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ShotRef(BaseModel):
    """Enough of the Shot to identify what was solved, without dragging
    the full Shot schema into the sidecar."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    plate_source: str = ""
    frame_range: tuple[int, int] = (0, 0)
    resolution: tuple[int, int] = (0, 0)
    pixel_aspect: float = 1.0


class BackendInfo(BaseModel):
    """Which solver produced the result + knobs it was given."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    matcher: str = "sequential"
    sequential_overlap: int = 10
    camera_model: str = "PINHOLE"
    sift_max_features: int = 4096
    max_image_size: int = 1920


class Conventions(BaseModel):
    """Coordinate / unit declarations so importers can't silently mis-read."""

    model_config = ConfigDict(extra="forbid")

    axis_system: str = "y_up_rh"
    camera_convention: str = "opengl"
    rot_order: str = "ZXY"
    angle_unit: str = "degrees"
    units: str = "unitless"
    unit_note: str = (
        "pycolmap incremental SfM produces arbitrary-scale reconstructions. "
        "Metric recovery (depth prior / user hint) lands in v2b."
    )


class Intrinsics(BaseModel):
    """Shared intrinsics for the clip (v2a assumes single camera = SINGLE mode)."""

    model_config = ConfigDict(extra="forbid")

    mode: str = "shared"
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    width_px: int
    height_px: int
    haperture_mm_hint: float = 24.892
    vaperture_mm_hint: float = 18.669
    focal_mm_hint: float | None = None
    distortion_model: str = "pinhole"
    distortion_params: list[float] = Field(default_factory=list)
    source: str = "colmap_solver"


class Extrinsic(BaseModel):
    """One per registered frame. Frames that failed to register are omitted;
    `CameraSolve.unregistered_frames` lists them so importers can interpolate
    or leave keyframes blank."""

    model_config = ConfigDict(extra="forbid")

    frame: int
    translation: tuple[float, float, float]
    rotation_euler_zxy_deg: tuple[float, float, float]
    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


class ScenePoint(BaseModel):
    """One 3D feature point. Sparse set — SfM typically resolves a few
    thousand per clip. Consumers use these as survey points in viewports."""

    model_config = ConfigDict(extra="forbid")

    xyz: tuple[float, float, float]
    color_rgb: tuple[int, int, int] = (255, 255, 255)
    reprojection_error_px: float = 0.0
    track_length: int = 0


class Quality(BaseModel):
    """Metrics that let a compositor decide whether to trust the solve
    without rendering. `solve_confidence` aggregates the per-frame errors
    into a 0..1 score for traffic-light UI."""

    model_config = ConfigDict(extra="forbid")

    per_frame_reprojection_error_px: dict[int, float] = Field(default_factory=dict)
    aggregate_reprojection_error_px: float = 0.0
    tracked_points_count: dict[int, int] = Field(default_factory=dict)
    registered_frame_count: int = 0
    total_frame_count: int = 0
    solve_confidence: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class CameraSolve(BaseModel):
    """Top-level container written verbatim to `<shot>.camera.json`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    shot: ShotRef = Field(default_factory=ShotRef)
    backend: BackendInfo
    conventions: Conventions = Field(default_factory=Conventions)
    intrinsics: Intrinsics
    extrinsics: list[Extrinsic] = Field(default_factory=list)
    unregistered_frames: list[int] = Field(default_factory=list)
    points: list[ScenePoint] = Field(default_factory=list)
    quality: Quality = Field(default_factory=Quality)


__all__ = [
    "BackendInfo",
    "CameraSolve",
    "Conventions",
    "Extrinsic",
    "Intrinsics",
    "Quality",
    "ScenePoint",
    "ShotRef",
]
