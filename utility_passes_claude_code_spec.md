# Utility Passes — Claude Code Implementation Specification (v1)

**Status**: ready for implementation
**Companion**: `utility_passes_design_notes.md` — the design record (the "why"). This spec is the "what and how." When in doubt about a design decision, consult the design record.
**Target hardware**: Linux/Windows workstation with NVIDIA GPU (Titan-class, ~24GB VRAM for dev). macOS Apple Silicon supported via MPS but slow.
**Target release**: public GitHub, MIT-licensed core.

---

## 0. How to Use This Document

- This is the **implementation brief**. Build phases 0–6 sequentially.
- Each phase has concrete deliverables and an **exit criterion**. Do not advance until the exit criterion is met.
- Design rationale is in `utility_passes_design_notes.md` (1480 lines). This spec references sections as `design §N.M`. When a decision seems arbitrary, it almost always has rationale in the design doc.
- Scaffolding files (pyproject.toml, install scripts, CI) are included **inline** in Section 5. Create them verbatim.
- **Non-goals for v1 are explicit** (§1.3). Do not drift into scope creep — v2+ features are architecturally accommodated but not implemented.

---

## 1. Project Overview & Scope

### 1.1 What this tool does

A VFX plate-to-AOV preprocessor. Reads EXR image sequences, runs AI-driven passes (depth, normals, optical flow, matte), and writes sidecar EXRs with Nuke/CG AOV channel conventions. Original plate is never modified. GUI for preparation, CLI for execution, library for integration.

### 1.2 v1 scope (locked)

- **Passes**: depth, normals, matte, flow
- **Post-processor**: flow-guided temporal smoothing (applied to per-frame passes)
- **Execution**: local single-GPU
- **Interface**: CLI + YAML + PySide6 prep GUI
- **Output**: EXR sidecar matching VFX conventions
- **Licensing**: per-pass commercial flags, `--allow-noncommercial` gate
- **Packaging**: uv-based self-contained `.venv`, install scripts for Linux/Windows/macOS
- **Plugins**: entry-points-based discovery from day one; core has zero hardcoded pass imports

### 1.3 Explicit non-goals for v1

Do not implement these, but do declare their architectural hooks where specified:

- Farm execution (Deadline) — `DeadlineExecutor` stub only
- Nuke panel integration — defer to v2/v3
- Prism / ShotGrid / OpenPype adapters — empty stub modules only
- Cryptomatte-style instance IDs — v2
- Camera tracking (VGGT/MegaSAM) — `CameraPass` stub + `pass_type="camera"` enum value only
- 3DGS/4DGS reconstruction — `pass_type="scene_3d"` enum value only
- Intrinsic decomposition — `pass_type="radiometric"` enum value only
- Green screen keying (CorridorKey) — v2c
- Non-EXR input formats (DPX, MOV, R3D) — `ImageSequenceReader` ABC declared, only `OIIOExrReader` implemented
- Tiling for full-res inference — YAML schema supports `strategy: tile`, implementation defers
- FBX export — v2a/v2b
- Web UI / REST API — v3+

### 1.4 Core principles

Every implementation decision flows from these (design §2):

1. **Sidecar never merged** — original plate untouched
2. **Scene-referred in, display-referred for models, scene-ready out**
3. **License is first-class** — declared per-plugin, gated by CLI flag
4. **Prep-human / execute-hands-off** — GUI for review, execution is batch
5. **CLI/library is the source of truth** — GUI and future Nuke panel are thin consumers
6. **Artistic-grade, not metrology-grade** — outputs are for comp, not measurement
7. **Plugins all the way down** — built-in passes declared identically to third-party

---

## 2. Technology Stack

- **Python 3.11** (provisioned by uv)
- **uv** for environment and package management (not conda, not pip directly)
- **PyTorch 2.3+** with CUDA 12.1 wheels (pinned in `pyproject.toml`)
- **OpenImageIO** (`oiio-python` wheel) — EXR I/O
- **OpenColorIO** (`opencolorio` wheel) — colorspace transforms
- **NumPy** — tensor math not on GPU
- **PySide6** — GUI (LGPL, Nuke-compatible)
- **Typer** — CLI
- **Pydantic v2** — config validation, Job/Shot models
- **ruff** — lint + format (configured via pyproject.toml)
- **pytest** — testing
- **huggingface_hub** — model checkpoint download/caching

Specific model dependencies declared per-phase (below).

---

## 3. Repository Structure

```
utility-passes/
├── pyproject.toml                      # Load-bearing: deps, entry points, tool config
├── uv.lock                             # Generated, committed
├── install.sh                          # Linux/macOS installer (see §5)
├── install.bat                         # Windows installer (see §5)
├── uninstall.sh                        # Removes .venv, offers model cache cleanup
├── uninstall.bat
├── README.md                           # User-facing intro (see §5)
├── LICENSE                             # MIT for core
├── .gitignore
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Lint + tests on push/PR
│       └── release.yml                 # (stub for v1, flesh out for v1.0 tag)
├── docs/
│   ├── developing-plugins.md           # How to write a plugin (Phase 6)
│   ├── architecture.md                 # Points at design notes
│   └── user-guide.md                   # (Phase 6)
├── src/
│   └── utility_passes/
│       ├── __init__.py                 # Exports: Job, Shot, run, __version__
│       ├── core/
│       │   ├── __init__.py
│       │   ├── pass_base.py            # UtilityPass ABC, License, ChannelSpec, PassType
│       │   ├── job.py                  # Shot, Job, Task, PassConfig
│       │   ├── dag.py                  # DAG scheduler with artifact dependencies
│       │   ├── registry.py             # PluginRegistry — entry point discovery
│       │   └── vram.py                 # VRAM estimation + planning
│       ├── io/
│       │   ├── __init__.py
│       │   ├── oiio_io.py              # EXR read/write via OIIO
│       │   ├── ocio_color.py           # OCIO colorspace transforms
│       │   ├── resize.py               # Pass-type-aware resizing
│       │   ├── display_transform.py    # Auto-exposure + tonemap + EOTF
│       │   ├── metadata.py             # CameraMetadata extraction (full spec, v2a uses it)
│       │   ├── channels.py             # Channel naming contract
│       │   ├── readers/
│       │   │   ├── __init__.py
│       │   │   ├── base.py             # ImageSequenceReader ABC
│       │   │   └── oiio_exr.py         # OIIOExrReader
│       │   └── writers/
│       │       ├── __init__.py
│       │       ├── base.py             # SidecarWriter ABC
│       │       ├── exr.py              # ExrSidecarWriter
│       │       └── json.py             # JsonSidecarWriter (minimal v1, expanded v2)
│       ├── shared/
│       │   └── optical_flow/           # Shared flow cache for temporal smoother
│       │       ├── __init__.py
│       │       └── cache.py
│       ├── post/
│       │   ├── __init__.py
│       │   └── temporal_smooth.py      # Flow-guided warp-and-blend
│       ├── passes/
│       │   ├── __init__.py
│       │   ├── flow/
│       │   │   ├── __init__.py
│       │   │   └── raft.py             # RAFTPass (bidirectional)
│       │   ├── depth/
│       │   │   ├── __init__.py
│       │   │   ├── depth_anything_v2.py
│       │   │   ├── depthcrafter.py
│       │   │   └── depth_pro.py
│       │   ├── normals/
│       │   │   ├── __init__.py
│       │   │   ├── dsine.py
│       │   │   └── normalcrafter.py
│       │   ├── matte/
│       │   │   ├── __init__.py
│       │   │   ├── sam3.py             # Detector + tracker
│       │   │   ├── matanyone2.py       # Soft alpha refiner (non-commercial)
│       │   │   ├── rvm.py              # Soft alpha refiner (commercial-safe)
│       │   │   └── rank.py             # Hero ranking logic
│       │   └── camera/                 # STUB ONLY for v1
│       │       ├── __init__.py
│       │       └── stub.py             # CameraPass stub with pass_type="camera"
│       ├── executors/
│       │   ├── __init__.py
│       │   ├── base.py                 # Executor ABC
│       │   ├── local.py                # LocalExecutor (v1 implementation)
│       │   └── deadline.py             # DeadlineExecutor stub (v2)
│       ├── integrations/
│       │   ├── __init__.py
│       │   ├── base.py                 # PipelineAdapter ABC
│       │   ├── standalone.py           # Default, no-integration
│       │   ├── prism.py                # Stub (v2)
│       │   ├── shotgrid.py             # Stub (v2)
│       │   └── openpype.py             # Stub (v2)
│       ├── models/
│       │   ├── __init__.py
│       │   └── registry.py             # ModelRegistry — lazy load, VRAM tracking
│       ├── cli/
│       │   ├── __init__.py
│       │   └── app.py                  # Typer app
│       └── gui/
│           ├── __init__.py
│           ├── app.py                  # QApplication entry
│           ├── main_window.py
│           ├── viewport.py             # Image viewer with transform modes
│           ├── shot_list.py
│           ├── inspector.py            # Right panel
│           ├── session.py              # Autosave / load
│           └── widgets/
│               ├── __init__.py
│               ├── exposure_slider.py
│               ├── pixel_inspector.py
│               └── histogram.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py                     # Shared fixtures (test plates, OCIO config)
│   ├── fixtures/
│   │   ├── test_plate_1080p/           # Small test EXR sequence
│   │   └── ocio_config/
│   ├── test_io/
│   ├── test_passes/
│   ├── test_cli/
│   └── test_gui/
└── plugin-template/                    # Cookiecutter-ready example
    ├── cookiecutter.json
    └── {{cookiecutter.project_slug}}/
        ├── pyproject.toml              # Pre-filled entry points scaffolding
        ├── src/
        │   └── {{cookiecutter.module_name}}/
        │       ├── __init__.py
        │       └── pass.py             # UtilityPass stub with TODOs
        ├── tests/
        └── README.md
```

---

## 4. Data Contracts (Locked — Implement Exactly)

### 4.1 EXR output channels

Pass outputs written to sidecar EXR using these exact channel names. Refer to design §5.1 for rationale.

```
# Depth
Z                           # primary depth, "best available" convention
Z_raw                       # raw model output pre-normalization
depth.confidence            # optional [0,1]

# Normals (camera-space, [-1,1], unit-length, +X right / +Y up / +Z toward camera)
N.x, N.y, N.z
normals.confidence          # optional [0,1]

# Forward motion (frame t → t+1, in pixels at plate resolution)
motion.x, motion.y

# Backward motion (frame t → t-1, in pixels at plate resolution)
back.x, back.y
flow.confidence             # F-B consistency [0,1]

# Hero mattes (top-4 soft alpha)
matte.r, matte.g, matte.b, matte.a      # all float [0,1]

# Semantic masks (one channel per detected concept, hard-mask quality)
mask.<concept>              # e.g. mask.person, mask.vehicle, mask.sky
```

### 4.2 Metadata schema (written to sidecar EXR header)

```
liveaov/version                   "1.0.0"
liveaov/created                   ISO 8601 timestamp
liveaov/plate_source              absolute path to source plate

liveaov/input/colorspace          "acescg" | "linear_rec709" | ...
liveaov/input/exposure_offset     float EV stops
liveaov/input/exposure_anchor     "median_p50" | "p75" | "mean_log"
liveaov/input/tonemap             "agx" | "filmic" | "reinhard" | "none"
liveaov/input/eotf                "srgb" | "rec709" | "linear"

liveaov/<pass_name>/model         model identifier
liveaov/<pass_name>/version       model version or hash
liveaov/<pass_name>/license       "apache-2.0" | "cc-by-nc-4.0" | ...
liveaov/<pass_name>/commercial    "true" | "false"
liveaov/<pass_name>/params        JSON-serialized pass params

# Depth-specific
liveaov/depth/space               "metric" | "relative_clip" | "relative_frame"
liveaov/depth/units               "meters" | "normalized"

# Normals-specific
liveaov/normals/space             "camera"
liveaov/normals/convention        "+X_right,+Y_up,+Z_toward_camera"

# Flow-specific
liveaov/flow/direction            "forward" | "backward" | "bidirectional"
liveaov/flow/unit                 "pixels_at_plate_res"

# Matte-specific
liveaov/matte/detector            "sam3"
liveaov/matte/refiner             "matanyone2" | "rvm" | "none"
liveaov/matte/concepts            JSON array of detected concepts
liveaov/matte/hero_r/label        string
liveaov/matte/hero_r/track_id     int
liveaov/matte/hero_r/score        float
# same for hero_g, hero_b, hero_a

# Smoother
liveaov/smooth/applied_to         JSON array of pass names
liveaov/smooth/algorithm          "flow_guided_ema_v1"
liveaov/smooth/fb_threshold       float
```

### 4.3 Core dataclasses (use Pydantic v2)

```python
# src/utility_passes/core/pass_base.py

from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field

class License(BaseModel):
    spdx: str                                    # "MIT", "Apache-2.0", "CC-BY-NC-4.0-variant", etc.
    commercial_use: bool                         # Can outputs be used commercially?
    commercial_tool_resale: bool = True          # Can we resell the tool/inference as a service?
    notes: str = ""                              # Free-form clarifications

class PassType(str, Enum):
    GEOMETRIC = "geometric"       # depth, normals
    MOTION = "motion"             # flow
    SEMANTIC = "semantic"         # matte, greenscreen
    RADIOMETRIC = "radiometric"   # intrinsic decomp (v2+)
    CAMERA = "camera"             # camera track (v2+)
    SCENE_3D = "scene_3d"         # 3DGS/4DGS (v3+)

class TemporalMode(str, Enum):
    PER_FRAME = "per_frame"
    VIDEO_CLIP = "video_clip"
    SLIDING_WINDOW = "sliding_window"
    PAIR = "pair"

class ChannelSpec(BaseModel):
    name: str
    dtype: Literal["float16", "float32"] = "float32"
    description: str = ""

class SidecarSpec(BaseModel):
    name: str                        # "utility", "camera", "scene"
    format: Literal["exr", "json", "abc", "fbx", "ply", "nk"] = "exr"
```

```python
# src/utility_passes/core/job.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

class Shot(BaseModel):
    name: str
    folder: Path
    sequence_pattern: str                        # "sh020_plt_v003.####.exr"
    frame_range: tuple[int, int]
    resolution: tuple[int, int]
    pixel_aspect: float = 1.0
    colorspace: str = "auto"

    transform: "DisplayTransformParams" = Field(default_factory=lambda: DisplayTransformParams())
    passes_enabled: list[str] = Field(default_factory=list)
    pass_overrides: dict = Field(default_factory=dict)

    sidecars: dict[str, Path] = Field(default_factory=dict)

    status: Literal["new", "analyzed", "reviewed", "queued",
                    "running", "done", "failed"] = "new"
    notes: str = ""

    # External IDs — None in v1, populated by integrations in v2+
    prism_task_id: str | None = None
    shotgrid_version_id: int | None = None
    openpype_version_id: str | None = None
    external_metadata: dict = Field(default_factory=dict)

class PassConfig(BaseModel):
    name: str                                    # plugin name e.g. "depth_anything_v2"
    params: dict = Field(default_factory=dict)

class Job(BaseModel):
    shot: Shot
    passes: list[PassConfig]

    # Farm-shaped fields — LocalExecutor ignores in v1
    priority: int = 50
    pool: str = "gpu"
    chunk_size: int = 10
    dependencies: list["Job"] = Field(default_factory=list)
    gpu_affinity: str | None = None
    max_retries: int = 2
    timeout_minutes: int = 120

    def to_tasks(self) -> list["Task"]:
        """Chunk job into tasks. LocalExecutor iterates them sequentially."""
        ...

class Task(BaseModel):
    job_id: str
    pass_name: str
    frame_range: tuple[int, int]
    dependencies: list[str] = Field(default_factory=list)
```

### 4.4 Pass plugin contract (the ABC)

```python
# src/utility_passes/core/pass_base.py (continued)

from abc import ABC, abstractmethod
import numpy as np
import torch

class UtilityPass(ABC):
    """All passes (built-in and third-party) subclass this."""

    # --- Identity ---
    name: str                                    # plugin name, unique across registry
    version: str                                 # semver
    license: License
    pass_type: PassType

    # --- Resource planning ---
    vram_estimate_gb_fn: callable                # (w, h) -> float
    model_native_resolution: tuple[int, int] | None = None
    supports_tiling: bool = False

    # --- Temporal behavior ---
    temporal_mode: TemporalMode
    temporal_window: int | None = None

    # --- Colorspace expectation ---
    input_colorspace: str = "srgb_linear"        # what the pass wants to receive

    # --- Dependency graph ---
    produces_channels: list[ChannelSpec]
    produces_sidecars: list[SidecarSpec] = []
    provides_artifacts: list[str] = []           # in-memory artifacts other passes can consume
    requires_artifacts: list[str] = []

    # --- Lifecycle ---
    @abstractmethod
    def preprocess(self, frames: np.ndarray) -> torch.Tensor: ...

    @abstractmethod
    def infer(self, tensor: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def postprocess(self, tensor: torch.Tensor) -> dict[str, np.ndarray]:
        """Return {channel_name: array} matching produces_channels."""
        ...
```

---

## 5. Scaffolding Files (Create Exactly)

### 5.1 `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "utility-passes"
version = "0.1.0"
description = "AI-driven VFX plate preprocessor: depth, normals, motion, mattes as sidecar EXRs"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.11"
authors = [
    { name = "TODO", email = "TODO@TODO" }
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: End Users/Desktop",
    "License :: OSI Approved :: MIT License",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Multimedia :: Video :: Non-Linear Editor",
]

dependencies = [
    # Core
    "pydantic>=2.5",
    "typer>=0.12",
    "pyyaml>=6.0",
    "numpy>=1.26,<2.0",
    # I/O
    "oiio-python>=2.5",
    "opencolorio>=2.3",
    # Torch stack (CUDA 12.1 wheels)
    "torch>=2.3,<3.0",
    "torchvision>=0.18",
    # Model hub
    "huggingface-hub>=0.23",
    # GUI
    "PySide6>=6.6",
    # Misc
    "tqdm>=4.66",
    "rich>=13.7",                               # pretty CLI output
    "platformdirs>=4.2",                        # user cache dirs
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
]

# Model-specific extras (user installs what they need)
depthcrafter = ["diffusers>=0.30"]
normalcrafter = ["diffusers>=0.30"]
sam3 = []                                       # installed via its own package
matanyone2 = []                                 # installed via its own package

[project.urls]
Homepage = "https://github.com/TODO/utility-passes"
Documentation = "https://github.com/TODO/utility-passes/blob/main/docs/user-guide.md"
Repository = "https://github.com/TODO/utility-passes"
Issues = "https://github.com/TODO/utility-passes/issues"

[project.scripts]
utility-passes = "utility_passes.cli.app:main"
utility-passes-gui = "utility_passes.gui.app:main"

# -----------------------------------------------------------------------------
# PLUGIN ENTRY POINTS
# -----------------------------------------------------------------------------
# All built-in passes declared here. Third-party plugins declare the same
# groups in their own pyproject.toml. Core has zero hardcoded imports.

[project.entry-points."utility_passes.passes"]
# Flow
raft                 = "utility_passes.passes.flow.raft:RAFTPass"
# Depth
depth_anything_v2    = "utility_passes.passes.depth.depth_anything_v2:DepthAnythingV2Pass"
depthcrafter         = "utility_passes.passes.depth.depthcrafter:DepthCrafterPass"
depth_pro            = "utility_passes.passes.depth.depth_pro:DepthProPass"
# Normals
dsine                = "utility_passes.passes.normals.dsine:DSINEPass"
normalcrafter        = "utility_passes.passes.normals.normalcrafter:NormalCrafterPass"
# Matte
sam3_matte           = "utility_passes.passes.matte.sam3:SAM3MattePass"
matanyone2_refiner   = "utility_passes.passes.matte.matanyone2:MatAnyone2Refiner"
rvm_refiner          = "utility_passes.passes.matte.rvm:RVMRefiner"
# Camera (v1 stub)
camera_stub          = "utility_passes.passes.camera.stub:CameraPassStub"

[project.entry-points."utility_passes.post"]
temporal_smooth      = "utility_passes.post.temporal_smooth:TemporalSmoother"

[project.entry-points."utility_passes.executors"]
local                = "utility_passes.executors.local:LocalExecutor"
deadline             = "utility_passes.executors.deadline:DeadlineExecutorStub"

[project.entry-points."utility_passes.io.readers"]
oiio_exr             = "utility_passes.io.readers.oiio_exr:OIIOExrReader"

[project.entry-points."utility_passes.io.writers"]
exr                  = "utility_passes.io.writers.exr:ExrSidecarWriter"
json                 = "utility_passes.io.writers.json:JsonSidecarWriter"

[project.entry-points."utility_passes.integrations"]
standalone           = "utility_passes.integrations.standalone:StandaloneAdapter"
prism                = "utility_passes.integrations.prism:PrismAdapterStub"
shotgrid             = "utility_passes.integrations.shotgrid:ShotGridAdapterStub"
openpype             = "utility_passes.integrations.openpype:OpenPypeAdapterStub"

# -----------------------------------------------------------------------------
# TOOL CONFIGURATIONS
# -----------------------------------------------------------------------------

[tool.hatch.build.targets.wheel]
packages = ["src/utility_passes"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "F", "W",      # pycodestyle + pyflakes
    "I",                # isort
    "N",                # pep8-naming
    "UP",               # pyupgrade
    "B",                # flake8-bugbear
    "C4",               # flake8-comprehensions
    "SIM",              # flake8-simplify
    "RUF",              # ruff-specific
]
ignore = [
    "E501",             # line too long (covered by formatter)
    "B008",             # function calls in argument defaults (FastAPI/Typer pattern)
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = [
    "-ra",
    "--strict-markers",
    "--cov=utility_passes",
    "--cov-report=term-missing",
]
markers = [
    "gpu: tests that require a GPU (deselect with '-m \"not gpu\"')",
    "slow: tests that take >10s",
    "integration: full-pipeline tests",
]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unreachable = true
ignore_missing_imports = true                  # for oiio, opencolorio, etc.

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
]
```

### 5.2 `install.sh` (Linux/macOS)

```bash
#!/usr/bin/env bash
# Utility Passes — installation script
# Creates project-local .venv, installs dependencies, verifies.

set -u  # unset vars are errors; don't set -e — we want to continue and log failures

LOG_FILE="install.log"
: > "$LOG_FILE"

log() {
    local msg="$1"
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$timestamp] $msg" | tee -a "$LOG_FILE"
}

fail() {
    local msg="$1"
    local remediation="${2:-}"
    log "✗ FAILED: $msg"
    if [ -n "$remediation" ]; then
        log "                     → $remediation"
    fi
    log "Install incomplete. See $LOG_FILE for full details."
    exit 1
}

ok() { log "✓ $1"; }

log "Utility Passes installer started"

# 1. Check / install uv
if ! command -v uv >/dev/null 2>&1; then
    log "uv not found, installing..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh >>"$LOG_FILE" 2>&1; then
        fail "Failed to install uv" "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env" 2>/dev/null || source "$HOME/.local/bin/env" 2>/dev/null || true
fi
UV_VERSION="$(uv --version 2>/dev/null || echo 'unknown')"
ok "uv: $UV_VERSION"

# 2. Check NVIDIA driver (Linux only)
if [[ "$(uname)" == "Linux" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1)"
        CUDA_VER="$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo '')"
        ok "NVIDIA driver $DRIVER_VER (CUDA $CUDA_VER)"
    else
        log "⚠ WARNING: nvidia-smi not found. GPU passes will not run."
        log "            If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/Download/index.aspx"
    fi
elif [[ "$(uname)" == "Darwin" ]]; then
    log "ℹ macOS detected — will use MPS backend (Apple Silicon) or CPU (Intel). GPU passes may be slow."
fi

# 3. Provision Python 3.11
if ! uv python install 3.11 >>"$LOG_FILE" 2>&1; then
    fail "Failed to provision Python 3.11 via uv" "Check network connectivity and disk space"
fi
ok "Python 3.11 provisioned"

# 4. Sync dependencies
log "Installing dependencies (this may take several minutes)..."
if ! uv sync --extra dev >>"$LOG_FILE" 2>&1; then
    fail "uv sync failed" "See $LOG_FILE for details. Common causes: network issues, incompatible torch wheel for platform"
fi
ok "Dependencies installed"

# 5. Smoke test
if ! uv run utility-passes --version >>"$LOG_FILE" 2>&1; then
    fail "Smoke test failed — utility-passes --version did not succeed" "See $LOG_FILE"
fi
VERSION_OUTPUT="$(uv run utility-passes --version 2>&1)"
ok "Smoke test: $VERSION_OUTPUT"

log ""
log "✓ Installation complete."
log ""
log "Next steps:"
log "  uv run utility-passes --help          # see available commands"
log "  uv run utility-passes-gui             # launch the preparation GUI"
log "  uv run utility-passes models pull --all   # pre-download model checkpoints (optional)"
log ""
log "Activate the environment manually with:"
log "  source .venv/bin/activate   (Linux/macOS)"
log "  .venv\\Scripts\\activate     (Windows)"
```

### 5.3 `install.bat` (Windows)

```batch
@echo off
setlocal enabledelayedexpansion

set LOG_FILE=install.log
echo. > %LOG_FILE%

call :log "Utility Passes installer started"

REM 1. Check / install uv
where uv >nul 2>&1
if errorlevel 1 (
    call :log "uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >> %LOG_FILE% 2>&1
    if errorlevel 1 (
        call :fail "Failed to install uv" "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit /b 1
    )
)
for /f "tokens=*" %%V in ('uv --version 2^>nul') do set UV_VERSION=%%V
call :log "[OK] uv: %UV_VERSION%"

REM 2. Check NVIDIA driver
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :log "[WARN] nvidia-smi not found. GPU passes will not run."
    call :log "        If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/Download/index.aspx"
) else (
    for /f "tokens=*" %%D in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2^>nul') do set DRIVER_VER=%%D
    call :log "[OK] NVIDIA driver: !DRIVER_VER!"
)

REM 3. Provision Python 3.11
uv python install 3.11 >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "Failed to provision Python 3.11" "Check network connectivity and disk space"
    exit /b 1
)
call :log "[OK] Python 3.11 provisioned"

REM 4. Sync dependencies
call :log "Installing dependencies (this may take several minutes)..."
uv sync --extra dev >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "uv sync failed" "See %LOG_FILE% for details. Common causes: network issues, incompatible torch wheel"
    exit /b 1
)
call :log "[OK] Dependencies installed"

REM 5. Smoke test
uv run utility-passes --version >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "Smoke test failed" "See %LOG_FILE%"
    exit /b 1
)
for /f "tokens=*" %%O in ('uv run utility-passes --version 2^>nul') do set VERSION_OUTPUT=%%O
call :log "[OK] Smoke test: %VERSION_OUTPUT%"

call :log ""
call :log "[OK] Installation complete."
call :log ""
call :log "Next steps:"
call :log "  uv run utility-passes --help"
call :log "  uv run utility-passes-gui"

goto :eof

:log
set TS=%date% %time%
echo [%TS%] %~1
echo [%TS%] %~1 >> %LOG_FILE%
goto :eof

:fail
call :log "[FAIL] %~1"
if not "%~2"=="" call :log "        -> %~2"
call :log "Install incomplete. See %LOG_FILE% for full details."
goto :eof
```

### 5.4 `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --extra dev
      - run: uv run ruff check .
      - run: uv run ruff format --check .

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python-version }}
      - run: uv sync --extra dev
      - name: Run tests (skip GPU)
        run: uv run pytest -m "not gpu" -v

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"
      - run: uv sync --extra dev
      - run: uv run mypy src/utility_passes
```

### 5.5 `README.md` (stub)

```markdown
# Utility Passes

**AI-driven VFX plate preprocessor.** Reads EXR image sequences, runs depth/normals/motion/matte passes, writes sidecar EXRs with Nuke/CG AOV channel conventions.

> **Status**: alpha, pre-v1 release

## Quick start

```bash
git clone https://github.com/TODO/utility-passes
cd utility-passes
./install.sh        # or install.bat on Windows
```

Then:
```bash
uv run utility-passes-gui           # preparation GUI
uv run utility-passes --help        # CLI reference
```

## What it does

Given a plate like `/shots/sh020/plate/v003/sh020_plt.####.exr`, the tool produces:

- `/shots/sh020/plate/v003/sh020_plt.utility.####.exr` — sidecar with:
  - `Z` depth channel
  - `N.x/N.y/N.z` camera-space normals
  - `motion.x/motion.y` forward motion vectors (pixels)
  - `back.x/back.y` backward motion vectors
  - `matte.r/g/b/a` top-4 soft hero mattes
  - `mask.<concept>` semantic hard masks

Original plate is never modified. See [design notes](docs/architecture.md) for architectural details.

## Documentation

- [User guide](docs/user-guide.md)
- [Developing plugins](docs/developing-plugins.md)
- [Architecture](docs/architecture.md)

## License

Core: MIT. Individual model plugins have their own licenses — see the license matrix in [architecture notes](docs/architecture.md#license-matrix).
```

### 5.6 `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
.venv/
env/
venv/
dist/
build/
*.egg-info/
*.egg

# Testing
.pytest_cache/
.coverage
.coverage.*
htmlcov/
.tox/

# Type checking
.mypy_cache/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Our tool
install.log
.utility_passes/
sessions/

# Model cache (too big for repo)
*.pt
*.safetensors
checkpoints/
models/*.bin
```

---

## 6. Build Plan — Six Phases

Implement in order. Each phase has concrete deliverables and a verifiable exit criterion. **Do not skip ahead.**

---

### Phase 0 — Foundation (target: week 1)

**Goal**: end-to-end IO + pass plugin infrastructure without any AI inference. A "no-op pass" should produce a valid sidecar EXR that opens cleanly in Nuke.

**Modules to implement:**

- `core/pass_base.py` — `UtilityPass` ABC, `License`, `PassType`, `TemporalMode`, `ChannelSpec`, `SidecarSpec` (all the data types from §4.3-4.4)
- `core/job.py` — `Shot`, `Job`, `Task`, `PassConfig` Pydantic models, YAML serialization
- `core/registry.py` — `PluginRegistry` with entry-point discovery. Loads all `utility_passes.*` entry-point groups.
- `core/dag.py` — DAG scheduler with `requires_artifacts` / `provides_artifacts` resolution. Topological sort, cycle detection.
- `core/vram.py` — simple VRAM estimation interface; actual estimation stubbed
- `io/oiio_io.py` — EXR read (multi-channel, multi-part, metadata preservation) and write via OpenImageIO
- `io/ocio_color.py` — OCIO colorspace transforms; auto-sniff from EXR header where possible
- `io/channels.py` — channel naming constants matching §4.1
- `io/display_transform.py` — full implementation (see §7 for algorithm)
- `io/resize.py` — pass-type-aware resize (see §8)
- `io/metadata.py` — `CameraMetadata` dataclass + EXR header parser (v2a will use it; implement fully in v1)
- `io/readers/base.py` + `io/readers/oiio_exr.py` — abstract + EXR implementation
- `io/writers/base.py` + `io/writers/exr.py` + `io/writers/json.py`
- `executors/base.py` + `executors/local.py` — minimal local executor that calls `to_tasks()` and iterates
- `executors/deadline.py` — stub that raises `NotImplementedError` with clear message
- `integrations/base.py` + `standalone.py` + stubs for `prism.py`, `shotgrid.py`, `openpype.py`
- `models/registry.py` — `ModelRegistry` with lazy loading + reference counting (stub the actual loading for Phase 0)

**Test fixture**: create a small synthetic EXR sequence (`tests/fixtures/test_plate_1080p/`) — 10 frames at 1920×1080, simple gradient, ACEScg colorspace, to use for all phase tests.

**Implement a throwaway `NoOpPass` in tests** that just writes a zero-valued channel — proves the full pipeline works without needing a real model.

**Exit criterion** (verify before moving to Phase 1):

```bash
# Given: tests/fixtures/test_plate_1080p/ contains a 10-frame EXR sequence
# When: run the NoOpPass
uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ --passes noop
# Then:
# - tests/fixtures/test_plate_1080p/test_plate.utility.####.exr exists for each frame
# - Output EXR has correct resolution, pixel aspect, metadata preserved
# - Output opens cleanly in Nuke (manual verification) — channels listed correctly
# - oiio --info on the output shows the expected liveaov/* metadata entries
```

Also: CI passes (lint + tests), `utility-passes --version` works, entry-point discovery returns `NoOpPass` via `PluginRegistry.list_passes()`.

---

### Phase 1 — Flow Pass (target: week 2)

**Goal**: first real AI pass. Bidirectional RAFT with F-B consistency. This is the keystone intermediate everything else reuses.

**Modules:**

- `passes/flow/raft.py` — `RAFTPass` implementing `UtilityPass`
  - Backend: use the official `torchvision.models.optical_flow.raft_large` (BSD-3, ships with torchvision — no separate model download)
  - Temporal mode: `PAIR`
  - Produces channels: `motion.x`, `motion.y`, `back.x`, `back.y`, `flow.confidence`
  - Provides artifacts: `forward_flow`, `backward_flow`, `occlusion_mask`, `parallax_estimate`
  - Unit: pixels at plate resolution (scale flow vectors by `plate_w / inference_w` on upscale)
- `shared/optical_flow/cache.py` — `FlowCache` that stores flow tensors keyed by `(shot_id, frame, direction)`, spills to disk if too big
- `post/temporal_smooth.py` — `TemporalSmoother` post-processor
  - Algorithm: flow-guided EMA warp-and-blend with F-B occlusion rejection (design §9.1)
  - Works on any per-frame pass output that declares compatibility
  - Not registered as a pass itself — registered as `utility_passes.post.temporal_smooth`

**Key implementation notes:**

- RAFT output is in pixels at its inference resolution. After upscale to plate resolution, **scale vectors by the upscale ratio**. This is the #1 thing to get right for Nuke VectorBlur to work.
- Forward-backward consistency: for pixel `p`, compute `p_warped = p + forward_flow[p]`, then `p_back = p_warped + backward_flow[p_warped]`. If `|| p - p_back || > threshold` (default 1.0px), mark `p` as occluded in `occlusion_mask`.
- `parallax_estimate` = median magnitude of forward flow across the clip, normalized by image width. Used by v2a camera pass for backend routing.

**YAML config shape:**

```yaml
passes:
  flow:
    params:
      backend: raft_large                       # raft_large | raft_small | sea_raft (future)
      precision: fp32                           # fp16 | fp32
      fb_threshold_px: 1.0
      inference_resolution: 960                 # scale down for speed; upscale flow back
```

**Exit criterion**:

```bash
# Given: tests/fixtures/test_plate_1080p/ with visible motion between frames
# When: run the flow pass
uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ --passes flow
# Then:
# - Output sidecar has motion.x, motion.y, back.x, back.y, flow.confidence channels
# - Channel values are in pixels (verify magnitude matches visible motion)
# - Open the output in Nuke:
#     - Drop a ShuffleCopy to move motion.x/motion.y into rgba.red/rgba.green
#     - Plug into a VectorBlur node
#     - Result should show plausible motion blur matching the motion in the plate
# - occlusion_mask artifact is accessible in memory for Phase 2's temporal smoother
# - parallax_estimate is recorded in job metadata
```

---

### Phase 2 — Depth + Normals (target: week 3)

**Goal**: commercial-safe passes first (DA V2 + DSINE), then add the temporal-native options (DepthCrafter, NormalCrafter), then Depth Pro for metric. Wire the temporal smoother into the non-temporal backends.

#### 2.1 Depth pass (multiple backends)

**Module: `passes/depth/depth_anything_v2.py`** (commercial-safe fallback, build FIRST)

- Source: [depth-anything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- Ship with the **Small and Base variants only** in v1 (Apache 2.0). Gate Large/Giant behind `--allow-noncommercial` (CC-BY-NC-4.0).
- Temporal mode: `PER_FRAME`
- Produces: `Z` (relative-normalized-per-clip, not per-frame — important), `Z_raw`
- Model native resolution: multiples of 14 (ViT patch size); resize to 518 on short edge typically

**Module: `passes/depth/depth_pro.py`** (metric, commercial-safe)

- Source: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- Apple ML Research License (read terms carefully, confirm commercial use before production)
- Temporal mode: `PER_FRAME`
- Produces: `Z` (metric, meters), `Z_raw`, optional `depth.confidence`
- Metadata: `depth/space = "metric"`, `depth/units = "meters"`

**Module: `passes/depth/depthcrafter.py`** (primary temporal, license-ambiguous)

- Source: [Tencent/DepthCrafter](https://github.com/Tencent/DepthCrafter)
- **License gating critical**: declare `License(spdx="apache-2.0+svd-nc", commercial_use=False, notes="Weights are Apache 2.0 but built on Stable Video Diffusion which has non-commercial terms. Verify with authors for commercial use.")`
- Temporal mode: `VIDEO_CLIP` with sliding window (default window=110, overlap=25)
- Produces: `Z` (relative-clip-normalized), `Z_raw`

**Policy for `Z` channel** (non-negotiable):

- If backend is metric (Depth Pro) → `Z` is metric distance
- If backend is relative (DA V2, DepthCrafter) → `Z` is normalized **per-clip, not per-frame** (prevents flicker in the smoothed output)
- Always write `Z_raw` = raw model output
- Record `depth/space` metadata accurately

#### 2.2 Normals pass

**Module: `passes/normals/dsine.py`** (commercial-safe fallback, build FIRST)

- Source: [baegwangbin/DSINE](https://github.com/baegwangbin/DSINE)
- License: MIT on weights
- Temporal mode: `PER_FRAME`
- Produces: `N.x`, `N.y`, `N.z` (camera-space, [-1,1], unit-length)
- Intrinsics handling: auto-read from `CameraMetadata` if present; fall back to "approximate" (50mm equivalent); **scale intrinsics by inference/plate ratio on resize** (design §8.4 — this is THE critical bug to not reproduce)

**Module: `passes/normals/normalcrafter.py`** (primary temporal)

- Source: [Binyr/NormalCrafter](https://github.com/Binyr/NormalCrafter)
- Same license ambiguity as DepthCrafter (Apache weights on SVD base)
- Temporal mode: `VIDEO_CLIP`

**Convention lock** (non-negotiable, same as design §10.3):

- Output camera-space: +X right, +Y up, +Z toward camera
- [-1,1] float range, **unit-length per pixel** (renormalize `N / ||N||` after resize)
- If a model outputs [0,1] display-ranged normals (for visualization), `postprocess()` converts to [-1,1]

#### 2.3 Temporal smoother wiring

The per-frame backends (DA V2, DSINE, Depth Pro) benefit from the smoother. Video-native backends (DepthCrafter, NormalCrafter) don't need it.

```yaml
passes:
  depth:
    name: depth_anything_v2        # or depth_pro, depthcrafter
    params:
      smooth: auto                 # auto enables for per-frame backends
      # smooth: true | false       # explicit override
```

#### 2.4 Central model registry used

`ModelRegistry.get("depth_anything_v2_base")` lazy-loads checkpoint from HuggingFace on first use, caches to `~/.cache/utility-passes/models/`, reference-counts for unload. Each pass calls this rather than managing its own loading.

**Exit criterion**:

```bash
# Given: tests/fixtures/test_plate_1080p/ with motion and varied depth
# When: run depth + normals with smoother
uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ \
    --passes flow,depth,normals \
    --depth-backend depth_anything_v2 \
    --normals-backend dsine
# Then:
# - Output sidecar has Z, N.x, N.y, N.z, plus flow channels
# - In Nuke:
#     - Drop a Relight node (or Position2Normal), feed N.x/N.y/N.z in → relight works
#     - Z channel usable for a PositionFromDepth node
# - Normals are unit-length (verify: sqrt(N.x^2 + N.y^2 + N.z^2) ≈ 1.0 per pixel)
# - License metadata correctly records "apache-2.0" and commercial=true

# Then verify non-commercial gate:
uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ \
    --passes depth --depth-backend depthcrafter
# Expected: exits with license error, hints at --allow-noncommercial flag

uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ \
    --passes depth --depth-backend depthcrafter --allow-noncommercial
# Expected: runs successfully, metadata records commercial=false
```

---

### Phase 3 — Matte Pass (target: week 4)

**Goal**: SAM 3 concept detection + video tracking, with MatAnyone 2 (non-commercial) or RVM (commercial-safe) as soft-alpha refiner. RGBA hero packing + semantic masks.

#### 3.1 Architecture

```
Plate → SAM 3 (concept detect + track) → hard masks + track IDs
             │
             ├─ mask.<concept>             ← semantic union masks (all concepts)
             └─ top-N instances → refiner → matte.r/g/b/a (soft alpha)
```

#### 3.2 Modules

**`passes/matte/sam3.py`** — detector + tracker

- Source: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- License: `License(spdx="SAM-License-1.0", commercial_use=True, notes="Meta custom license. Prohibits military/ITAR.")`
- Pass type: `SEMANTIC`
- Temporal mode: `VIDEO_CLIP` (uses its own memory bank)
- Produces: `mask.<concept>` channels (dynamic — one per detected concept above threshold)
- Provides artifact: `sam3_hard_masks` — dict of `{instance_id: per_frame_mask_stack}` for the refiner to consume. **Critical for v2c CorridorKey integration** (design §21.8).

**`passes/matte/matanyone2.py`** — soft alpha refiner (non-commercial)

- Source: [pq-yang/MatAnyone2](https://github.com/pq-yang/MatAnyone2)
- License: `License(spdx="NTU-S-Lab-1.0", commercial_use=False)`
- Takes: original plate + first-frame hard mask from SAM 3
- Produces: soft alpha per instance

**`passes/matte/rvm.py`** — soft alpha refiner (commercial-safe fallback)

- Source: [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting)
- License: `License(spdx="MIT", commercial_use=True)`
- Older (2021), human-focused, but serviceable

**`passes/matte/rank.py`** — hero ranking

```python
def rank_instances(instances: list[Instance],
                   flow: FlowData,
                   weights: RankWeights) -> list[Instance]:
    """Score each instance, return top-N for RGBA packing."""
    # score = w1*area + w2*centrality + w3*motion_energy
    #       + w4*duration + w5*user_priority
```

Motion energy comes from the flow pass — another reuse of the keystone intermediate. Requires `flow` in the job's pass list.

#### 3.3 YAML config

```yaml
passes:
  matte:
    params:
      detector: sam3
      refiner: rvm              # rvm | matanyone2 | none
      auto_detect:
        concepts: [person, vehicle, tree, building, sky, water, animal]
        confidence_threshold: 0.4
        min_area_fraction: 0.005
        sample_frame: middle
      ranking:
        weights:
          area: 0.4
          centrality: 0.2
          motion: 0.2
          duration: 0.2
          user_priority: 0.0
        max_heroes: 4
      overrides:
        # user can force specific hero assignment
        heroes: []              # list of {concept, track_id, slot}
```

#### 3.4 Channel naming dynamism

`produces_channels` for the matte pass is partly dynamic — `mask.<concept>` depends on what's detected. `ExrSidecarWriter` must handle this: writer accepts the channel dict returned by `postprocess()` and writes whatever channels are there. No pre-declaration required for dynamic channel names (they still follow the `mask.*` or `matte.*` naming convention).

#### 3.5 Metadata

Extensive — record ranking decisions so the Nuke-side (future gizmo) can label correctly:

```
liveaov/matte/detector             "sam3"
liveaov/matte/refiner              "rvm"
liveaov/matte/concepts             JSON array
liveaov/matte/hero_r/label         "person"
liveaov/matte/hero_r/track_id      17
liveaov/matte/hero_r/score         0.87
# ...for each of r, g, b, a
```

**Exit criterion**:

```bash
# Given: test plate with a person walking (synthesize one if needed)
uv run utility-passes run-shot tests/fixtures/test_plate_1080p/ \
    --passes flow,matte
# Then:
# - Output has mask.person channel (hard-ish), matte.r channel (soft alpha)
# - Metadata records hero_r as person with track_id and score
# - In Nuke: ShuffleCopy matte.r into alpha → composite person over new BG → hair/edges look good
# - Verify license gate: --allow-noncommercial flag needed for MatAnyone 2 refiner
```

---

### Phase 4 — CLI (target: week 5)

**Goal**: full pipeline invokable from terminal, no GUI dependency. Typer-based.

#### 4.1 Commands

```bash
# Discovery
utility-passes discover <folder>
# Walks folder, finds EXR sequences, prints them with detected metadata.

# Analyze (auto-exposure etc., no execution)
utility-passes analyze <folder> [--output shot.yaml]
# Creates a Shot object with sensible defaults, writes to YAML for review.

# Run a prepared job
utility-passes run <job.yaml>
# Full execution path.

# Quick ad-hoc (no YAML)
utility-passes run-shot <folder> \
    --passes flow,depth,normals,matte \
    --depth-backend depth_anything_v2 \
    --normals-backend dsine \
    [--allow-noncommercial]

# Preflight
utility-passes preflight <job.yaml>
# Check: models available, VRAM estimated fits, licenses acceptable, outputs writable

# Plugins
utility-passes plugins list
utility-passes plugins list --type depth

# Models
utility-passes models pull --all                # pre-download checkpoints
utility-passes models pull depth_anything_v2
utility-passes models cache-info                # show cache size / location
utility-passes models clear-cache

# Version
utility-passes --version
```

#### 4.2 Output

- Use `rich` for progress bars, tables, colored output
- Errors use `rich.console.Console` with traceback formatting
- Machine-readable mode: `--json` flag on commands makes output JSON for pipeline integration

#### 4.3 Exit criterion

```bash
# All of the above commands work and produce correct output.
# In particular:
utility-passes discover tests/fixtures/
# ...returns a list of detected sequences with metadata

utility-passes preflight some-job.yaml
# ...validates without running; reports any missing models, VRAM issues, license needs
```

---

### Phase 5 — GUI (target: weeks 6-7)

**Goal**: PySide6 prep GUI with three-panel layout, viewport with transform modes, session persistence.

#### 5.1 Main window layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Utility Passes — Shot Prep                                              │
├──────────────┬────────────────────────────────────┬──────────────────────┤
│  SHOT LIST   │  VIEWPORT                          │  INSPECTOR           │
│              │                                    │  - Shot info         │
│  ▸ shot_01   │  [Frame display, scrub bar below]  │  - Colorspace select │
│  ▾ shot_02   │                                    │  - View mode radio   │
│    ✓ ready   │  [Original / Transformed /         │  - Exposure slider   │
│  ▸ shot_03   │   Compare / (future) Keyed]        │  - Tonemap dropdown  │
│              │                                    │  - Pass toggles      │
│  [+ Add]     │                                    │  - Status + notes    │
├──────────────┴────────────────────────────────────┴──────────────────────┤
│ [Preflight] [Submit local] [Submit Deadline (v2)] [Export YAML]          │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 5.2 Modules

**`gui/app.py`** — QApplication entry, `main()` function registered as `utility-passes-gui` console script

**`gui/main_window.py`** — `MainWindow(QMainWindow)`. Holds the three panels via QSplitter. Menu bar with File → Save/Load Session, Edit → Preferences, Help → About.

**`gui/viewport.py`** — `Viewport(QWidget)` wrapping a QLabel with QImage or QGraphicsView:

- Three view modes (radio group): Original / Transformed / Compare
- Compare mode: split-wipe slider OR side-by-side toggle
- Scrub bar: current frame, range slider, playback controls
- Pixel inspector: on hover, tooltip shows RGB in both scene-linear and display-referred
- **v1 performance strategy**: proxy resolution (1920 long-edge). Load full EXR with OIIO, downsample to proxy, transform in NumPy on CPU. 30-60fps scrubbing easily. v2 adds GPU shader path.
- Frame cache: LRU of 20 frames; background prefetch ±5 around current position

**`gui/shot_list.py`** — `ShotListPanel(QWidget)`:

- QTreeView backed by a custom QAbstractItemModel
- Status icons per row (new/analyzed/reviewed/ready/...)
- Drag-and-drop folder → add shots
- "+ Add folder" / "+ Add list (recursive)" buttons
- Context menu: rename, remove, duplicate, open folder in file manager

**`gui/inspector.py`** — `InspectorPanel(QWidget)`:

- Form-style layout, QFormLayout
- Colorspace QComboBox (populated from OCIO config)
- Exposure slider (-5 to +5 EV, 0.1 step) + numeric spinbox
- Tonemap QComboBox
- Pass QCheckBox grid (populated from `PluginRegistry.list_passes()`)
- Notes QTextEdit
- "Auto-analyze" button (computes exposure from sample frames)
- "Reset to defaults" button

**`gui/session.py`** — session persistence:

- Autosave on every change (debounced 500ms) to `~/.utility_passes/sessions/autosave.yaml`
- Manual File → Save / Open
- "Open last session" on launch
- Session format: YAML, serializes list of `Shot` objects

**`gui/widgets/`** — small reusable widgets:

- `exposure_slider.py` — slider + spinbox bound together
- `pixel_inspector.py` — overlay widget for hover RGB display
- `histogram.py` — small histogram over current frame, toggleable

#### 5.3 GUI ↔ core integration

GUI calls `utility_passes` library directly (same process). For execution:

1. User clicks "Submit local" → GUI builds `Job` from `Shot` + configuration → calls `LocalExecutor.submit(job)` in a worker thread
2. Worker emits Qt signals for progress, per-frame updates, errors
3. Main window updates shot list status from signals
4. On completion, sidecar path is filled into `shot.sidecars["utility"]`

#### 5.4 Auto-detect green screen (Phase 5 polish, optional)

- In the inspector, if the first frame has >40% green-hue coverage, show a "Possible green screen — enable keying pass (v2c)?" hint (disabled button, tooltip)
- Documents the feature as v2c

#### 5.5 Exit criterion

- Launch GUI, drag in the test fixture folder
- Shot appears in list with status "new"
- Click shot → viewport loads, shows Original view
- Click "Auto-analyze" → exposure value fills in, status becomes "analyzed"
- Toggle to Transformed view → see tonemapped result
- Scrub through frames → smooth playback
- Adjust exposure slider → transformed view updates in real-time
- Toggle off "matte" pass → pass will not run on submit
- Click "Submit local" → execution runs, progress updates in shot list
- On completion: shot status = "done", sidecar EXR exists
- Close GUI, relaunch → "Open last session" restores prep state

---

### Phase 6 — Polish & Ship (target: week 8)

**Goal**: the details that make a tool shippable rather than demoable.

#### 6.1 Implement

- **License gate in CLI and GUI**: warn/refuse non-commercial passes without `--allow-noncommercial` (CLI) or explicit dialog acceptance (GUI)
- **VRAM estimation**: each pass implements `vram_estimate_gb(w, h)`. Scheduler pre-flight checks available VRAM via `torch.cuda.mem_get_info()`. If insufficient, auto-downscale with a warning dialog/log, or refuse with actionable message.
- **Progress reporting during execution**: per-frame progress bars for long passes; time-remaining estimates
- **Pixel inspector in viewport** — hover shows RGB in scene-linear and display-referred
- **Histogram overlay** — toggleable in viewport
- **Error recovery**: crashes at frame N should not restart from frame 0. Frame-level success markers in a `.utility_progress.json` sidecar during execution, cleaned up on success.

#### 6.2 Documentation

- `docs/user-guide.md` — how to install, how to prep shots, how to run
- `docs/developing-plugins.md` — walkthrough of writing a `UtilityPass` plugin. Reference `plugin-template/`.
- `docs/architecture.md` — points to design notes, summarizes for new contributors
- Example `job.yaml` files in `examples/`

#### 6.3 Plugin template

Ship `plugin-template/` as a cookiecutter-ready example. Minimal but working `UtilityPass` subclass that produces a single test channel. Researcher can clone, fill in their model, publish.

#### 6.4 Release prep

- Bump version to `0.1.0`
- `CHANGELOG.md`
- GitHub Release with install script link
- First model pull (DA V2 Base) bundled as a "getting started" example

#### 6.5 Exit criterion

- Everything above works
- Fresh clone + `./install.sh` + run GUI → works on a blank machine
- CI green
- Tag `v0.1.0`, publish release

---

## 7. Display Transform — Full Spec

`io/display_transform.py` (called out separately because it's load-bearing for all AI passes).

### 7.1 Algorithm

Per clip, applied uniformly to all frames:

1. **Linearize via OCIO**: input colorspace → linear working space (scene-referred)
2. **Auto-exposure**: sample `sample_frames` evenly-spaced frames. Compute per-frame luminance (Rec.709 or ACEScg coefficients). Take the `anchor` percentile (default p50 / median). Solve for exposure offset `E` such that `pow(2, E) * Y_percentile = target` (default target=0.18). **`E` is clip-wide; not per-frame.**
3. Apply exposure: `frame *= pow(2, E)`
4. Tone map (default AgX): compress HDR into ~[0,1] with smooth highlight roll-off
5. sRGB EOTF (gamma 2.2)
6. Clamp [0,1]

### 7.2 Tonemappers

Implement as plugins (`utility_passes.tonemappers` entry point group):

- `agx` (default) — use a clean reference implementation; multiple open-source ports exist under permissive licenses
- `filmic` — Hable curve
- `reinhard` — simple `x / (1 + x)`
- `none` — passthrough (for debugging)

### 7.3 API

```python
class DisplayTransformParams(BaseModel):
    input_colorspace: str = "auto"
    auto_exposure_enabled: bool = True
    exposure_anchor: Literal["median", "p75", "mean_log"] = "median"
    exposure_target: float = 0.18
    sample_frames: int = 10
    tonemap: str = "agx"
    output_eotf: Literal["srgb", "rec709", "linear"] = "srgb"
    manual_exposure_ev: float | None = None
    clamp: bool = True

class DisplayTransform:
    def analyze_clip(self, reader: ImageSequenceReader, params: DisplayTransformParams) -> dict:
        """Compute clip-wide exposure. Return analysis result (cacheable, GUI displays)."""

    def apply(self, frames: np.ndarray, params: DisplayTransformParams, analysis: dict) -> np.ndarray:
        """Apply transform. Fast to call repeatedly (for GUI slider scrubbing)."""
```

Critical property: `apply()` must be cheap (milliseconds) so the GUI can update on slider drag.

---

## 8. Resize Module — Full Spec

`io/resize.py`. Pass-type-aware, not a one-size-fits-all function.

### 8.1 API

```python
class ResizeMode(str, Enum):
    FIT_LONG_EDGE = "fit_long_edge"
    FIT_SHORT_EDGE = "fit_short_edge"
    FRACTION = "fraction"
    EXACT = "exact"

class ResizeParams(BaseModel):
    mode: ResizeMode = ResizeMode.FIT_LONG_EDGE
    target: int | tuple[int, int] = 1920
    max_vram_gb: float = 16.0
    upscale_back: bool = True
    strategy: Literal["downscale", "tile", "auto"] = "downscale"  # tile is v2

def downscale(frame: np.ndarray, params: ResizeParams, model_constraints: dict) -> np.ndarray:
    """Resize for inference input, respecting model constraints (multiples of N, etc.)"""

def upscale(result: np.ndarray, target_size: tuple[int, int], channel_type: str) -> np.ndarray:
    """Upscale inference output to plate resolution."""
    # channel_type ∈ {"continuous", "discrete", "normal_vector", "flow_vector"}
    # continuous → bilinear (depth, soft alpha)
    # discrete → nearest (hard masks, semantic IDs)
    # normal_vector → bilinear, then renormalize N/||N||
    # flow_vector → bilinear, then scale vectors by upscale ratio
```

### 8.2 Intrinsics scaling (for DSINE etc.)

```python
def scale_intrinsics(intrinsics: dict, from_res: tuple[int, int], to_res: tuple[int, int]) -> dict:
    """Scale fx, fy, cx, cy when resizing between resolutions.
    Failing to do this causes 'results vary with resolution' bugs."""
    scale_x = to_res[0] / from_res[0]
    scale_y = to_res[1] / from_res[1]
    return {
        "fx": intrinsics["fx"] * scale_x,
        "fy": intrinsics["fy"] * scale_y,
        "cx": intrinsics["cx"] * scale_x,
        "cy": intrinsics["cy"] * scale_y,
    }
```

### 8.3 Anamorphic

Work in pixel space. Preserve `pixelAspectRatio` from EXR header. Never desqueeze for inference — models see pixels and don't care about PAR.

---

## 9. Testing Strategy

### 9.1 Unit tests

- Every `io/` module — round-trip EXR read/write with all channel types, metadata preservation
- `display_transform` — known input → known output for each tonemap
- `resize` — verify vector scaling, normal renormalization, nearest vs bilinear
- `core/registry` — entry point discovery, plugin loading
- `core/dag` — dependency resolution, cycle detection
- Dataclass serialization (Job ↔ YAML round-trip)

### 9.2 Integration tests

- **NoOpPass end-to-end** (Phase 0) — verifies full pipeline
- **Flow pass end-to-end** — RAFT on synthetic motion, check flow magnitudes
- **Depth + smoother** — per-frame backend with smoother, verify temporal consistency improvement
- **Matte pass** — SAM 3 + RVM on a test plate with a person, verify channels exist
- **CLI end-to-end** — all commands produce correct output
- **GUI smoke tests** — launch, load shot, submit, verify output (headless with QT_QPA_PLATFORM=offscreen)

### 9.3 GPU tests

Mark tests requiring GPU with `@pytest.mark.gpu`. CI runs `pytest -m "not gpu"`. Developers run full suite locally on Titan.

### 9.4 Fixtures

`tests/fixtures/test_plate_1080p/` — small real-looking test plate (10 frames, 1920×1080, ACEScg). Generate with a procedural Python script so the repo stays small; the script creates plates with controlled motion, depth variation, a subject to matte, etc.

---

## 10. Documentation Expectations

### 10.1 Inline

- Every public module has a docstring stating its responsibility
- Every public class/function has a docstring with args, returns, raises
- Type hints throughout (enforced by mypy)

### 10.2 Docs folder

- `docs/user-guide.md` — end-user walkthrough
- `docs/developing-plugins.md` — how to write a new pass (Phase 6)
- `docs/architecture.md` — 1-page overview + link to design notes for depth
- `docs/cli-reference.md` — generated or hand-written command reference

### 10.3 Examples

- `examples/simple-job.yaml` — minimal job config
- `examples/full-job.yaml` — all passes, all options, annotated
- `examples/commercial-safe.yaml` — only commercial-clean models

---

## 11. References & Handoff Notes

### 11.1 Design document

`utility_passes_design_notes.md` — 1480 lines. The "why" behind every decision here. Consult when:

- A design decision in this spec seems arbitrary (rationale is almost certainly there)
- A future addon comes up (Tier 1/2/3 triage is explicit)
- A license question arises (full matrix is there)
- Planning beyond v1 (v2a/b/c/d/e, v3a/b roadmap with week estimates)

### 11.2 Open questions to resolve during implementation

- Confirm NormalCrafter and DepthCrafter actual license status (email authors)
- Verify Depth Pro's Apple ML Research License terms for commercial pipeline use
- Stress-test DA V2 Base on real VFX plates (night exteriors, low-light) for quality baseline
- Evaluate whether `agx` or a simpler tonemap gives better AI input — A/B on a few plates

### 11.3 Known implementation traps

From the design discussions — specific things to get right:

1. **Flow vector scaling on upscale** — if plate is 3840 wide and RAFT infers at 960, multiply flow vectors by 4.0 on upscale. Otherwise Nuke VectorBlur produces either microscopic or catastrophic smearing.
2. **Normals renormalization after resize** — bilinear upscale of [-1,1] unit vectors produces non-unit vectors. `N / ||N||` per pixel is one line of code; forgetting it is the #1 bug in normal-pass implementations.
3. **DSINE intrinsics scaling** — the "results vary with resolution" complaint in the existing DSINE-for-Nuke comes from NOT scaling intrinsics. Scale `fx, fy, cx, cy` by `inference_res / plate_res`.
4. **Clip-wide exposure, not per-frame** — per-frame auto-exposure creates flicker in model input which propagates into flicker in output. Single `E` for the whole clip.
5. **Per-clip depth normalization, not per-frame** — for relative depth backends, normalize once across the whole clip. Per-frame normalization causes severe temporal instability.
6. **Nearest-neighbor upscale for hard masks** — bilinear invents non-existent class IDs between two adjacent class pixels.
7. **Pixel aspect preserved through pipeline** — anamorphic plates stay in pixel space; PAR is metadata only.
8. **EXR header channel ordering** — Nuke reads channels in a specific order expected convention. Verify by opening test output in Nuke after each phase.

### 11.4 Success criteria for v1.0 release

- All 6 phases complete, exit criteria met
- Fresh-clone install on Linux, Windows, macOS (Apple Silicon) all work
- CI green
- Documentation complete
- At least one example plate + its utility sidecar committed as a reference for users
- `utility-passes-plugin-template` companion repo published
- MIT license core, per-plugin licenses correctly declared

---

*Spec version: 1.0*
*Target: v1 release*
*Estimated duration: 8 weeks for a focused developer / Claude Code agent*
*Companion document: `utility_passes_design_notes.md` v4.0*
