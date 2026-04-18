"""Pass plugin contract.

Every built-in and third-party pass subclasses `UtilityPass` and is discovered
via the `live_action_aov.passes` entry point group (design §24).

The types here are the comper-visible contract (channel names, license flags,
temporal mode) and the DAG-visible contract (provides/requires artifacts,
channel declarations). Resizing, smoothing, and metadata recording all run
off these declarations without pass-specific special cases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class License(BaseModel):
    """Per-plugin license declaration.

    The CLI license gate reads `commercial_use`; non-commercial passes require
    `--allow-noncommercial` to run (design §17).

    `commercial_tool_resale` is used to distinguish models like CorridorKey
    where outputs are commercially OK but the tool/inference itself cannot be
    resold.
    """

    spdx: str
    commercial_use: bool
    commercial_tool_resale: bool = True
    notes: str = ""


class PassType(str, Enum):
    """Pass family. The v2+ values are declared now so the architecture flows
    through them from day one (design §14, decision 5)."""

    GEOMETRIC = "geometric"
    MOTION = "motion"
    SEMANTIC = "semantic"
    RADIOMETRIC = "radiometric"
    CAMERA = "camera"
    SCENE_3D = "scene_3d"


class TemporalMode(str, Enum):
    """How a pass consumes time.

    - `PER_FRAME`: one frame in, one frame out. Temporal smoother applies.
    - `VIDEO_CLIP`: whole clip at once (e.g. DepthCrafter window).
    - `SLIDING_WINDOW`: overlapping chunks.
    - `PAIR`: two adjacent frames (e.g. RAFT).
    """

    PER_FRAME = "per_frame"
    VIDEO_CLIP = "video_clip"
    SLIDING_WINDOW = "sliding_window"
    PAIR = "pair"


class ChannelSpec(BaseModel):
    """One output channel in the sidecar EXR.

    The ExrSidecarWriter walks these to assemble the output. Dynamic channels
    (e.g. `mask.<concept>` from the matte pass) may be returned by
    `postprocess()` without pre-declaration as long as they follow the naming
    conventions in `io/channels.py` (design §5.1).
    """

    name: str
    dtype: Literal["float16", "float32"] = "float32"
    description: str = ""


class SidecarSpec(BaseModel):
    """A non-EXR sidecar a pass emits (JSON, Alembic, Nuke script, etc.).

    v1 only uses EXR and JSON; v2a adds `abc` and `nk`; v2b adds `fbx`; v3
    adds `ply` for Gaussian Splats. The shape is declared now so the writer
    registry flows through from day one (design §20.10).
    """

    name: str
    format: Literal["exr", "json", "abc", "fbx", "ply", "nk"] = "exr"


class UtilityPass(ABC):
    """Abstract base class every pass implements.

    Subclasses are plain Python classes (not Pydantic) because they carry
    heavy runtime state (loaded models, tensors). The class-level attributes
    are the declarative contract; the three lifecycle methods are the
    inference pipeline.

    Construction should be cheap: lazy-load weights in `infer` or via a
    separate `setup()` if needed, not in `__init__`. The scheduler may
    instantiate many passes during planning without running any of them.
    """

    # --- Identity ---
    name: str
    version: str
    license: License
    pass_type: PassType

    # --- Resource planning ---
    # (w, h) -> estimated VRAM in GB for a single inference call
    vram_estimate_gb_fn: Callable[[int, int], float] = staticmethod(lambda w, h: 0.0)
    model_native_resolution: tuple[int, int] | None = None
    supports_tiling: bool = False

    # --- Temporal behavior ---
    temporal_mode: TemporalMode = TemporalMode.PER_FRAME
    temporal_window: int | None = None

    # --- Colorspace expectation ---
    # What colorspace the pass wants to receive from the display transform.
    # Most AI models trained on display-referred data want "srgb_display".
    input_colorspace: str = "srgb_display"

    # --- Dependency graph (DAG) ---
    produces_channels: list[ChannelSpec] = []
    produces_sidecars: list[SidecarSpec] = []
    provides_artifacts: list[str] = []
    requires_artifacts: list[str] = []

    # --- Smoother wiring ---
    # Which of this pass's channels the temporal smoother should touch when
    # `smooth: auto` is enabled. Channels omitted from this list are never
    # smoothed even if the pass is PER_FRAME. Default empty = "nothing to
    # smooth" (used by the flow pass itself and by stubs).
    smoothable_channels: list[str] = []

    # --- Pass parameters (populated from YAML via `configure`) ---
    params: dict[str, Any]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    # --- Lifecycle ---
    @abstractmethod
    def preprocess(self, frames: np.ndarray) -> Any:
        """Convert plate pixels → model input tensor.

        Input: (N, H, W, C) float32 numpy array in this pass's
        `input_colorspace`. Output: whatever the model wants (usually
        torch.Tensor on device).
        """

    @abstractmethod
    def infer(self, tensor: Any) -> Any:
        """Run the model. Output shape is model-defined."""

    @abstractmethod
    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        """Convert model output → `{channel_name: array}` matching
        `produces_channels`.

        Each array should be (H, W) float32 at plate resolution. The ExrSidecarWriter
        packs them into the sidecar using the channel name as the EXR layer path.
        """

    # --- Shot-level iteration (overridable) ---
    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        """Run the pass across a shot's frame range.

        Default implementation: one preprocess → infer → postprocess call per
        frame (for PER_FRAME passes). PAIR / VIDEO_CLIP passes override this
        to handle batched or paired inference and still return a per-frame
        `{frame_idx: {channel_name: array}}` dict.

        Passes may also populate internal state that `emit_artifacts()` later
        exposes to post-processors and downstream passes.
        """
        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(frame_range[0], frame_range[1] + 1):
            frame, _ = reader.read_frame(f)
            model_in = self.preprocess(frame[None, ...])
            model_out = self.infer(model_in)
            out[f] = self.postprocess(model_out)
        return out

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        """Return any non-channel artifacts produced by this pass.

        Format: `{artifact_name: {frame_idx: value}}`. Called once per shot,
        after `run_shot` completes. Default: empty. Flow publishes
        `forward_flow` / `backward_flow` / `occlusion_mask` here.

        `value` is usually a numpy array, but the framework doesn't enforce
        the type — passes that publish richer structures (e.g. SAM 3's
        per-track mask stacks, a ranked instance list) can stash whatever
        shape they like, and downstream consumers read it by convention.
        """
        return {}

    # --- Cross-pass artifact consumption ---
    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        """Receive the full artifact dict produced by upstream passes.

        Called by the executor *once per shot*, before `run_shot`, iff this
        pass declares any `requires_artifacts`. Default is a no-op — passes
        that genuinely need upstream data (e.g. the RVM refiner reading
        SAM 3's `sam3_hard_masks`) override this to stash references on
        `self` and read them during `run_shot`.

        The dict is shared, not copied — do not mutate it.
        """

    # --- Convenience ---
    @classmethod
    def declared_license(cls) -> License:
        """Return the license of the *class* (bypasses instance state).

        Used by the CLI license gate before instantiation so we never
        construct a pass just to check if it's allowed to run.
        """
        lic = getattr(cls, "license", None)
        if not isinstance(lic, License):
            raise TypeError(
                f"{cls.__name__} must declare a class-level `license: License` attribute"
            )
        return lic


class PassProtocolError(RuntimeError):
    """Raised when a pass violates the plugin contract (e.g. missing channel
    in `postprocess()` output, wrong dtype, etc.)."""


class DisplayTransformParams(BaseModel):
    """Per-shot display transform configuration (design §7).

    Duplicated here rather than in `io/display_transform.py` so `Shot` can
    reference it without a circular import. The algorithm implementation
    lives in io/.
    """

    model_config = ConfigDict(extra="forbid")

    input_colorspace: str = "auto"
    auto_exposure_enabled: bool = True
    exposure_anchor: Literal["median", "p75", "mean_log"] = "median"
    exposure_target: float = 0.18
    sample_frames: int = 10
    tonemap: str = "agx"
    output_eotf: Literal["srgb", "rec709", "linear"] = "srgb"
    manual_exposure_ev: float | None = None
    clamp: bool = True
    # Computed during analysis; None until `analyze_clip` has run.
    computed_exposure_ev: float | None = Field(default=None)


__all__ = [
    "ChannelSpec",
    "DisplayTransformParams",
    "License",
    "PassProtocolError",
    "PassType",
    "SidecarSpec",
    "TemporalMode",
    "UtilityPass",
]
