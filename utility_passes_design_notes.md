# Utility Passes Tool — Architecture & Design Notes

*A scene-understanding preprocessor for VFX plates. Reads image sequences (primarily EXR), runs AI-driven per-pass inference, writes sidecar EXRs with Nuke/CG AOV channel conventions. Future-extensible toward 4D Gaussian Splatting reconstruction.*

**Status**: design-locked, pre-implementation. Target repo: public GitHub release, Titan workstation dev target, eventual farm execution.

---

## 1. Project Context

### 1.1 The problem

VFX plates need CG-style auxiliary passes (depth, normals, motion vectors, mattes) for compositing work, but they come from real cameras without any of those passes — unlike CG renders which ship them for free as AOVs.

Modern AI models can estimate all of these from the plate, but:
- Most AI models are trained on display-referred 8-bit imagery; plates are scene-referred float EXR, often very dark
- Each model has different input assumptions (resolution, colorspace, range)
- No consistent output conventions (coordinate spaces, value ranges, channel naming) matching VFX pipeline expectations
- Temporal consistency is the hard problem — per-frame estimators flicker
- Commercial licensing is a minefield (many SOTA models are non-commercial)

### 1.2 The tool

A production-grade ingest tool that:
- Takes EXR plate folders as input
- Produces Nuke-compatible sidecar EXRs with correctly-named, correctly-scaled AOV channels
- Handles the scene-referred → model-input conversion (display transform / normalizer)
- Manages temporal consistency via shared optical flow + flow-guided smoothing
- Exposes commercial vs non-commercial model paths as first-class config
- Separates human-in-loop **preparation** from hands-off **execution**

### 1.3 Users

VFX TDs, compositors, pipeline engineers. Shipped via pip + GitHub. Usable from CLI, Python library, GUI, eventually Nuke panel.

---

## 2. Core Philosophy

Load-bearing design principles that every decision flows from:

1. **Sidecar, never merged.** Original plate is never modified. All outputs go to separate sidecar files.
2. **Scene-referred in, display-referred for models, scene-ready out.** Correct colorspace handling at every boundary, metadata-documented.
3. **License is first-class.** Every pass declares its license and commercial-use flag. Non-commercial backends are gated behind an explicit flag.
4. **Prep-human, execute-hands-off.** Shot preparation (including visual review of normalization) requires a human; execution is batched and farm-friendly.
5. **CLI/library is the source of truth.** GUI and Nuke panel are thin consumers of the same core API.
6. **Artistic-grade, not metrology-grade.** We output perceptually-plausible AOVs for compositing, not measured geometry. This framing allows aggressive resizing and permits imperfect absolute scales.
7. **Future-extensible.** v1 architecture is shaped by foreseeable v2+ addons (farm, Nuke panel, GS reconstruction, pipeline integrations).

---

## 3. v1 Scope (Locked)

```
v1 deliverable:   CLI + YAML + PySide6 prep GUI
v1 passes:        depth, normals, matte, flow
v1 post-process:  flow-guided temporal smoothing (for per-frame passes)
v1 execution:     local (single GPU, Titan-class hardware)
v1 stack:         Python 3.11+, PyTorch, OIIO, OCIO, PySide6
v1 output:        EXR sidecar with Nuke/CG AOV channel conventions
v1 license gate:  per-pass license flags, --allow-noncommercial gate
```

### 3.1 Explicit non-goals for v1

- Farm execution (Deadline) — stub only, v2
- Nuke panel integration — stub only, v2/v3
- Prism/ShotGrid/OpenPype hooks — stubs only, v2
- Cryptomatte-style instance IDs — v2
- Camera tracking (VGGT-based) — v2
- 3DGS/4DGS reconstruction — v3+
- Intrinsic decomposition (albedo/shading) — v2+
- Non-EXR input formats (DPX, MOV, R3D) — v2
- Web UI / REST API — v3+

---

## 4. Architecture — Layered

```
utility_passes/
├── core/                # Pass ABC, Job/Shot models, DAG scheduler
│   ├── pass_base.py
│   ├── job.py
│   ├── dag.py
│   └── vram.py          # VRAM estimation + planning
├── io/                  # I/O, colorspace, resize, normalization
│   ├── oiio_io.py
│   ├── ocio_color.py
│   ├── resize.py        # pass-type-aware
│   ├── display_transform.py   # exposure + tonemap + EOTF
│   └── channels.py      # naming contract
├── shared/              # Shared intermediates
│   └── optical_flow/    # RAFT/SEA-RAFT wrappers with caching
├── post/                # Post-processors
│   └── temporal_smooth.py
├── passes/              # One folder per pass
│   ├── depth/
│   ├── normals/
│   ├── matte/
│   └── flow/
├── models/              # Central model registry
│   └── registry.py
├── executors/           # Execution backends
│   ├── local.py         # v1
│   ├── deadline.py      # v2 stub
│   └── tractor.py       # v2+ stub
├── integrations/        # Pipeline adapters (stubs in v1)
│   ├── base.py
│   ├── standalone.py
│   ├── prism.py         # stub
│   ├── shotgrid.py      # stub
│   └── openpype.py      # stub
├── cli/                 # Typer app
└── gui/                 # PySide6 prep GUI
    ├── app.py
    ├── main_window.py
    ├── viewport.py
    ├── shot_list.py
    ├── inspector.py
    └── session.py
```

**Dependency direction**: `core` is foundational. `io`, `shared`, `models` depend on `core`. `passes` depends on everything below it. `post`, `executors`, `integrations` depend on `core` + `passes`. `cli`, `gui` are pure consumers — never imported by core.

---

## 5. Data Contracts

### 5.1 EXR channel conventions (the comper-visible contract)

Everything below is **camera-space, float EXR, VFX convention** (+X right, +Y up, +Z toward camera). Scene-referred where applicable.

```
# Depth
Z                           # primary depth, "best available" convention
Z_raw                       # raw model output
depth.confidence            # optional, [0,1]

# Normals
N.x, N.y, N.z               # camera-space, [-1,1], unit-length per pixel
normals.confidence          # optional, [0,1]

# Motion (forward — frame t → t+1)
motion.x, motion.y          # pixels at plate resolution

# Motion (backward — frame t → t-1)
back.x, back.y              # pixels at plate resolution
flow.confidence             # forward-backward consistency, [0,1]

# Hero mattes (top-N ranked instances, soft alpha)
matte.r   float [0,1]       # hero instance 1
matte.g   float [0,1]       # hero instance 2
matte.b   float [0,1]       # hero instance 3
matte.a   float [0,1]       # hero instance 4

# Semantic masks (all detected concepts, hard-mask quality)
mask.person    float [0,1]  # union of all person instances
mask.vehicle   float [0,1]
mask.sky       float [0,1]
mask.<concept> float [0,1]  # one channel per detected concept
```

### 5.2 Metadata schema

Every sidecar EXR records the full pipeline state for reproducibility:

```
# Input
utilityPass/input/colorspace         "acescg" | "linear_rec709" | "arri_logc4" | ...
utilityPass/input/exposure_offset    +2.3
utilityPass/input/exposure_anchor    "median_p50" | "p75" | "mean_log"
utilityPass/input/tonemap            "agx" | "filmic" | "reinhard" | "none"
utilityPass/input/eotf               "srgb" | "rec709" | "linear"

# Per-pass provenance
utilityPass/depth/model              "depthcrafter" | "depth_pro" | "depth_anything_v2"
utilityPass/depth/version            model version hash
utilityPass/depth/space              "metric" | "relative_clip" | "relative_frame"
utilityPass/depth/units              "meters" | "normalized"
utilityPass/depth/license            "apache-2.0" | "sam-license" | ...
utilityPass/depth/commercial         true | false

utilityPass/normals/model            "normalcrafter" | "dsine"
utilityPass/normals/space            "camera"
utilityPass/normals/convention       "+X_right,+Y_up,+Z_toward_camera"

utilityPass/flow/model               "raft" | "sea_raft" | "unimatch"
utilityPass/flow/direction           "forward" | "backward" | "bidirectional"
utilityPass/flow/unit                "pixels_at_plate_res"

utilityPass/matte/detector           "sam3" | "sam2"
utilityPass/matte/refiner            "matanyone2" | "rvm" | "none"
utilityPass/matte/concepts           ["person", "vehicle", "tree", ...]
utilityPass/matte/hero_r/label       "person"
utilityPass/matte/hero_r/track_id    17
utilityPass/matte/hero_r/score       0.87

# Temporal smoothing
utilityPass/smooth/applied_to        ["depth", "normals"]
utilityPass/smooth/algorithm         "flow_guided_ema_v1"
utilityPass/smooth/fb_threshold      1.0
```

### 5.3 Shot data model

```python
@dataclass
class Shot:
    # Identity
    name: str                             # "sh020_plt" — user-editable
    folder: Path
    sequence_pattern: str                 # "sh020_plt_v003.####.exr"
    frame_range: tuple[int, int]
    resolution: tuple[int, int]
    pixel_aspect: float                   # 1.0 spherical, 2.0 anamorphic
    colorspace: str                       # sniffed or user-set

    # Transform
    transform: DisplayTransformParams

    # Passes
    passes_enabled: list[str]             # ["depth", "normals", "flow", "matte"]
    pass_overrides: dict                  # per-pass param overrides

    # Outputs (v2+ will hold multiple sidecar types)
    sidecars: dict[str, Path]             # {"utility": ..., "camera": ..., "scene": ...}

    # Status
    status: Literal["new", "analyzed", "reviewed", "queued",
                    "running", "done", "failed"]
    notes: str = ""

    # Future — external pipeline integration (v2+)
    prism_task_id: str | None = None
    shotgrid_version_id: int | None = None
    openpype_version_id: str | None = None
    external_metadata: dict = field(default_factory=dict)

    def to_job_yaml(self) -> dict: ...
    @classmethod
    def from_folder(cls, path: Path) -> "Shot": ...
```

### 5.4 Job / Task model (farm-ready from day one)

```python
@dataclass
class Job:
    shot: Shot
    passes: list[PassConfig]

    # Farm-shaped fields — LocalExecutor ignores in v1, DeadlineExecutor uses in v2
    priority: int = 50
    pool: str = "gpu"
    chunk_size: int = 10
    dependencies: list["Job"] = None
    gpu_affinity: str | None = None
    max_retries: int = 2
    timeout_minutes: int = 120

    def to_tasks(self) -> list[Task]:
        """Chunks the job into farm-sized units. LocalExecutor runs them sequentially."""
```

---

## 6. Pass Plugin Contract

All AI passes implement `UtilityPass`:

```python
class UtilityPass(ABC):
    name: str
    version: str
    license: License                    # with commercial flag
    pass_type: Literal[
        "geometric",     # depth, normals
        "motion",        # flow
        "semantic",      # matte
        "radiometric",   # future (intrinsic decomp)
        "camera",        # future (camera track / VGGT)
        "scene_3d",      # future (3DGS / 4DGS)
    ]

    # Resource planning
    vram_estimate_gb: Callable[[int, int], float]
    model_native_resolution: tuple[int, int] | None
    supports_tiling: bool

    # Temporal behavior
    temporal_mode: Literal["per_frame", "video_clip", "sliding_window", "pair"]
    temporal_window: int | None

    # DAG — produces for EXR output, provides/requires for in-memory artifacts
    produces_channels: list[ChannelSpec]
    provides_artifacts: list[str]
    requires_artifacts: list[str]

    # I/O colorspace expectations
    input_colorspace: str               # usually "srgb_display"

    # The inference pipeline
    def preprocess(self, frames: np.ndarray) -> torch.Tensor: ...
    def infer(self, tensor: torch.Tensor) -> torch.Tensor: ...
    def postprocess(self, tensor: torch.Tensor) -> ChannelData: ...
```

**Shared artifact pattern**: flow is computed once, reused by temporal smoother for depth/normals/matte. DAG scheduler handles this via `provides` / `requires`.

---

## 7. Display Transform (Normalizer)

The critical preprocessing step: scene-referred EXR → model-friendly input.

### 7.1 Algorithm (per clip, applied uniformly to all frames)

1. **Linearize to working space** (OCIO). Handles ACEScg, ARRI LogC4, RED Log3G10, Sony S-Log3, Rec.709, etc.
2. **Auto-exposure**: sample N frames, compute per-frame luminance percentile (default median = p50), solve for exposure offset `E` such that `pow(2,E) * Y_percentile = target` (default target = 0.18 scene-linear gray). **Single clip-wide E** — no per-frame drift.
3. **Apply exposure**: `frame = frame * pow(2, E)`
4. **Tone map**: AgX (default), Filmic, or Reinhard. Compresses HDR into ~[0,1] with smooth highlight roll-off.
5. **sRGB EOTF** (gamma 2.2): brings midtones into the distribution AI models were trained on.
6. **Clamp [0,1]**.

After step 5: image looks like a well-exposed phone photo. This is what AI models want.

### 7.2 YAML config

```yaml
display_transform:
  input_colorspace: auto               # auto-sniff from EXR, fall back to user-set
  auto_exposure:
    enabled: true
    anchor: median                     # median | p75 | mean_log
    target: 0.18
    sample_frames: 10
  tonemap: agx                         # agx | filmic | reinhard | none
  output_eotf: srgb                    # srgb | rec709 | linear
  manual_exposure_ev: null             # override: e.g. -1.0 for -1 stop
  clamp: true
```

### 7.3 Per-pass override (v2+)

Some passes may benefit from different exposure (e.g., matte on dark exterior wants +0.5 EV). Not in v1; YAML schema supports it later.

### 7.4 Reverse direction

Not needed in v1. Depth, normals, flow, matte are invariant to input exposure. Only radiometric outputs (albedo, shading — Phase 2+) need transform inversion.

---

## 8. Resizer

Plates vary enormously: 1920×1080 to 6K anamorphic. Each model has its own resolution constraints (DINOv2 patch size, 1536×1536 pad, etc.). VRAM is the hard limit on 16GB consumer GPUs.

### 8.1 Core behavior

1. Preprocess plate to model-friendly resolution
2. Run inference at that resolution
3. Upscale result back to plate native resolution for sidecar EXR

### 8.2 Anamorphic handling

Work in **pixel space, not squeezed/desqueezed**. Models have no concept of pixel aspect ratio; they see pixels. Preserve `pixelAspectRatio` from EXR header, pass through untouched. Nuke desqueezes at viewer.

### 8.3 Per-pass-type interpolation (non-negotiable)

- **Depth upscale**: bilinear (bicubic creates halos at discontinuities)
- **Normals upscale**: bilinear, **then renormalize** `N / ||N||` per pixel
- **Soft masks upscale**: bilinear
- **Hard masks / semantic IDs**: **nearest** (bilinear invents non-existent class IDs)
- **Flow upscale**: bilinear, **then scale vectors** by plate_w / inference_w

### 8.4 DSINE-specific: intrinsics scaling trap

DSINE uses per-pixel ray direction as an inductive bias. If plate is 3840 wide and we infer at 1024, **intrinsics (fx, fy, cx, cy) must be scaled** by 1024/3840, or DSINE computes wrong rays and produces subtly wrong normals. This is why the existing DSINE-for-Nuke has "result varies with resolution" as a documented problem.

### 8.5 YAML

```yaml
resize:
  mode: fit_long_edge | fit_short_edge | fraction | exact
  target: 1920
  max_vram_gb: 16
  upscale_back: true
  strategy: downscale | tile | auto    # v1: downscale only
  # tile_size, tile_overlap — v2
```

Presets: `hd` (1920 long-edge), `half`, `quarter`, `native`.

---

## 9. Temporal Smoother

Flow-guided warp-and-blend post-processor for per-frame passes.

### 9.1 v1 algorithm

For each frame t:
1. Get `forward_flow[t-1]` and `backward_flow[t]` from flow cache
2. Warp `Z[t-1]` into frame t using forward flow → `Z_warped`
3. Compute F-B consistency: if `||p - (p + fwd + bwd)|| > threshold`, mark p as occluded
4. For non-occluded pixels: `Z_smooth[t] = alpha * Z_warped + (1-alpha) * Z[t]`
5. For occluded pixels: `Z_smooth[t] = Z[t]` (no warp from previous)

Defaults: `alpha = 0.4`, `threshold = 1.0 px`. Closes ~70% of flicker gap in ~50 lines of code.

### 9.2 Applicable to

Any per-frame pass: depth (DA V2, Depth Pro), normals (DSINE), matte (when using per-frame detector).

Video-native models (DepthCrafter, NormalCrafter) don't need it and should skip the smoother by default.

### 9.3 v2+ extensions

- Bidirectional smoothing (blend from both neighbors)
- Sliding-window optimization
- Per-pixel alpha based on flow confidence

---

## 10. Per-Pass Specifications

### 10.1 Flow (build first — it's the keystone)

Shared intermediate: computed once per plate, consumed by everything else.

**Backends:**
- **RAFT** (primary) — BSD-3, ~4GB VRAM at 1080p, battle-tested. Bidirectional required.
- **SEA-RAFT** (optional) — faster RAFT variant, permissive.
- **UniMatch/GMFlow** (skip v1) — better quality on large motion but fiddly.

**Output**: `motion.x/y`, `back.x/y`, `flow.confidence` (F-B consistency).

**Exit criterion**: Nuke VectorBlur reading `motion.x/y` produces clean motion blur on test plate.

### 10.2 Depth

**Backends** (triage from 8-candidate list):
- **DepthCrafter** (primary) — video-native, temporally coherent. **License ambiguous** (Apache weights on SVD base; SVD is non-commercial). Treat as non-commercial pending author confirmation.
- **Depth Pro** (commercial-clean) — metric, sharp edges, Apple ML Research License. Used as metric anchor for Phase 2 scale alignment.
- **Depth Anything V2** (fallback) — per-frame, relative. Small/Base = Apache; **Large/Giant = CC-BY-NC 4.0**.

**Skipped**: Marigold (superseded), DepthFM, Geowizard (per-frame diffusion, slow), Metric3D, Sapiens, Lotus-G (revisit for normals joint-output in v2).

**Output**:
- `Z` = best-available (metric if model provides, else relative-normalized-per-clip — not per-frame)
- `Z_raw` = raw model output
- `depth.confidence` (optional)

**Scale policy**: metadata declares space. v2 can add `align_to_metric` utility that combines DepthCrafter (temporal) + Depth Pro (metric anchor) → aligned metric depth.

### 10.3 Normals

**Backends:**
- **NormalCrafter** (primary) — ICCV 2025, video-native, SVD-based. Same license ambiguity as DepthCrafter. HF model card says Apache 2.0, but built on SVD (non-commercial). Treat as non-commercial pending confirmation.
- **DSINE** (commercial-safe fallback) — CVPR 2024 Oral, MIT weights on HF. Per-frame ViT. Already has a Nuke integration (vinavfx/DSINE-for-Nuke) — useful reference. Pair with temporal smoother.

**Skipped**: StableNormal (middle ground, SD license issues), Lotus (v2 candidate for joint depth+normals).

**Output**: `N.x/N.y/N.z` camera-space, [-1,1], unit-length per pixel (renormalize after resize).

**Convention lock**: `+X right, +Y up, +Z toward camera` (matching Arnold/Redshift/V-Ray). Model output must be normalized to this convention in `postprocess()`. Metadata records the convention.

**DSINE-specific**: takes camera intrinsics as input. Auto-read from EXR metadata when present (ARRI Alexa/Cooke /i), fall back to "approximate" (50mm equivalent), warn loudly if neither. Scale intrinsics per the resize ratio.

### 10.4 Matte

**Three problems conflated into one**: detection, segmentation, soft matting.

**SAM 3 (Nov 2025, SAM 3.1 Mar 2026) changes the field**: Promptable Concept Segmentation — text prompt like `"person"` returns all instances, masked, video-tracked with stable IDs. One model does detection + segmentation + tracking. 30ms/frame on H200, fits in 16GB. **SAM License** (custom Meta license, permits commercial use with restrictions on military/ITAR/etc.).

**Soft alpha separately**: SAM 3 outputs hard masks. For hero subjects we need soft alpha (hair, motion blur, semi-transparency).

**Pipeline:**
```
Plate → SAM 3 (detect + track all concepts) → hard masks + IDs
              │
              ├─ mask.<concept> channels (all semantic masks, hard)
              └─ top-N hero instances → MatAnyone2 or RVM → soft alpha
                                          │                    │
                                          │                    matte.r/g/b/a
                                          (non-commercial)    (commercial-safe)
```

**Backends:**
- **SAM 3 / SAM 3.1** (primary detector) — SAM License, commercial OK with restrictions.
- **MatAnyone 2** (primary refiner, CVPR 2026 Highlight) — NTU S-Lab License 1.0 (**non-commercial**).
- **RVM** (commercial-safe refiner) — MIT. Older (2021), human-focused, serviceable.
- **SAM 2** alone (alternative primary) — Apache 2.0, if avoiding SAM License entirely.

**Auto-detect strategy (v1)**: curated concept list run against a sample frame, keep above-threshold + above-min-area detections.

```yaml
auto_detect:
  concepts: [person, vehicle, tree, building, sky, water, animal]
  confidence_threshold: 0.4
  min_area_fraction: 0.005
  sample_frame: middle
```

**Hero ranking** (for RGBA packing — max 4):
```
score = w1 * area_fraction
      + w2 * centrality
      + w3 * motion_energy    (from flow pass — another reason flow is keystone)
      + w4 * duration
      + w5 * user_priority
```

Top 4 → `matte.r/g/b/a`. Metadata records which slot = which label/track_id/score.

**Phase 2**: VLM-based concept generation (Florence-2 → noun phrases → SAM 3), Cryptomatte-style 32-bit ID hashing.

---

## 11. GUI Specification

### 11.1 Three panels

```
[SHOT LIST]   [VIEWPORT]   [INSPECTOR]
```

**Shot list (left)**: shots with status icons (new/analyzed/reviewed/ready/queued/running/done/failed). "+ Add folder" / "+ Add list" (batch recursive).

**Viewport (center)**: image view with frame scrubber. Three modes:
1. **Original** — raw EXR with neutral display transform (sRGB EOTF + clamp)
2. **Transformed** — what actually goes into the model (full pipeline: linearize → exposure → tonemap → EOTF → clamp)
3. **Compare** — side-by-side or split-wipe

Pixel inspector (hover shows RGB in both spaces). Optional histogram overlay.

**Inspector (right)**: exposure EV slider, tonemap dropdown, colorspace override, pass toggles, status indicator, per-shot notes.

### 11.2 Performance strategy

v1: proxy-resolution display (1920 long-edge for viewport), full-res at submit. Transform math in NumPy, ~30 lines.
v2: GPU shader path for 4K+ scrubbing.

Frame cache: preload first/middle/last; background-prefetch ±5 frames around scrubbed position.

### 11.3 Session persistence

Autosave to `~/.utility_passes/sessions/<timestamp>.yaml` on every change. Manual save/load. "Last session" opens on launch. TD can hand off partially-prepped session to colleague.

### 11.4 Stack

- **PySide6** (LGPL, Nuke-compatible)
- **QImage painting** for viewport in v1 (QtOpenGL for v2 GPU path)
- **OIIO** for image reading (shared with core)
- **qdarkstyle** (dark theme — VFX standard)

---

## 12. CLI Specification

```bash
# Analyze a folder without submitting — fill in defaults, user reviews
utility-passes analyze <folder>

# Execute a prepared job (YAML written by GUI or hand-crafted)
utility-passes run <job.yaml>

# Quick ad-hoc run (no prep phase)
utility-passes run-shot <folder> --passes depth,normals --exposure +2.0

# Preflight (check models are available, VRAM fits, etc.)
utility-passes preflight <job.yaml>

# Discovery
utility-passes discover <folder>     # list detected sequences
```

YAML is the source of truth. CLI reads/writes YAML. GUI reads/writes YAML. Anything the GUI can do, the CLI can do.

---

## 13. Build Order (6 Phases)

### Phase 0 — Foundation (week 1)
`io/`, `core/pass_base.py`, `core/job.py`.
**Exit**: load EXR → no-op pass → sidecar EXR → open in Nuke, channels correct.

### Phase 1 — Flow (week 2)
`passes/flow/raft.py` (bidirectional), `post/temporal_smooth.py` (even before consumers exist).
**Exit**: Nuke VectorBlur works on `motion.x/y`.

### Phase 2 — Depth + Normals (week 3)
Commercial-safe fallbacks first (DA V2, DSINE) + smoother integration. Then temporal-native (DepthCrafter, NormalCrafter) + Depth Pro metric.
**Exit**: all backends produce valid channels; Nuke Relight works on output.

### Phase 3 — Matte (week 4)
SAM 3 detect+track → MatAnyone 2 / RVM refine → RGBA hero packing + semantic masks.
**Exit**: auto-detect on test plate produces correct `matte.r/g/b/a` + `mask.<concept>`.

### Phase 4 — CLI (week 5)
Typer wrapper for full pipeline.
**Exit**: entire pipeline runnable from terminal without GUI.

### Phase 5 — GUI (weeks 6-7)
Three panels, viewport with transform modes, session persistence.
**Exit**: drag folder → review → submit → sidecars appear.

### Phase 6 — Polish (week 8)
Pixel inspector, histogram, license gate, VRAM estimation, progress reporting, README, example YAMLs.
Ship to GitHub.

---

## 14. Tier 1 Architectural Decisions for Future-Proofing

Decisions made now in v1 design that enable v2+ without refactor:

1. **Farm-ready Job model**: priority/pool/chunks/affinity/retries as fields, even though LocalExecutor ignores them. `to_tasks()` returns chunks; LocalExecutor iterates.
2. **Rich pass plugin contract**: enum values for `radiometric`, `camera`, `scene_3d` declared now; implementations come later.
3. **Codec abstraction in IO**: `ImageSequenceReader.for_path()` registry. DPX/MOV/R3D added as siblings of `OIIOExrReader` without touching core.
4. **Integration adapters as empty modules**: `integrations/prism.py`, `integrations/shotgrid.py`, `integrations/openpype.py` exist as stubs. Architecture flows through them.
5. **Core is a real library**: importable as `from utility_passes import Job, run`. CLI and GUI are consumers. Makes Nuke panel trivial later.
6. **Central model registry**: lazy load, reference count, VRAM tracking. v2 adds multi-GPU affinity without touching passes.
7. **Self-describing output specs**: passes declare `ChannelSpec` objects; writer walks the spec. New pass types with weird outputs (point clouds, PLY, JSON) fit without changing the writer.
8. **External ID fields on Shot**: `prism_task_id`, `shotgrid_version_id` etc. are `None` in v1. Zero cost, saves migration later.
9. **Sidecar as dict not single path**: `Shot.sidecars: dict[str, Path]` from day one. `{"utility": ...}` in v1, `{"utility": ..., "camera": ..., "scene": ...}` in v3.

---

## 15. Future Addons — Tiered Catalog

### Tier 1 (influences v1 architecture — already handled above)
- Deadline / farm execution
- Prism / ShotGrid / OpenPype integrations
- Nuke panel integration
- Camera tracking (VGGT) as ambient capability
- Additional pass types (intrinsic decomposition, HDRI estimation, shadow/reflection mattes)
- Non-EXR input formats (DPX, MOV, R3D)
- Multi-GPU distribution

### Tier 2 (reversible, add in v2+ without rewrite)
- VLM-based concept generation for matte (Florence-2)
- Cryptomatte-style instance IDs (32-bit hash)
- Proxy generation (Nuke proxy, ProRes preview)
- Metric alignment utility (DepthCrafter + Depth Pro anchor)
- QC / review features (flicker detection, comparison renders, confidence heatmaps)
- Web UI / REST API
- Batch command-line operations

### Tier 3 (revisit when prioritized)
- Cloud inference backends (fal, Replicate)
- Model quantization (fp8/int8)
- New tonemappers, new display transforms
- Model fine-tuning workflows

---

## 16. Future Addon: Gaussian Splatting — Research Notes (April 2026)

### 16.1 Why this is architecturally relevant

**Our utility passes tool is the preprocessing pipeline for video-to-4DGS reconstruction.** Current SOTA monocular-video-to-4DGS methods (Shape of Motion, MoSca, Prior-Enhanced GS, Deblur4DGS) require as input: dense video depth + dynamic masks + optical flow + camera poses. We are building the first four. Camera poses add trivially via VGGT.

This means: v2/v3 expansion toward 4DGS reconstruction is a **natural extension**, not a pivot. The pass system already handles this shape if we declare `scene_3d` in the pass type enum now.

### 16.2 Current state of the art

**Image → 3DGS (static)** — solved, feed-forward, <1s
- AnySplat (Feb 2026), Flash3D, SparseSplat
- Single-view quality bounded by occlusion ambiguity

**Multi-view → 3DGS (static)** — solved, <1s
- **VGGT** (CVPR 2025 Best Paper, facebookresearch/vggt): feed-forward transformer predicts extrinsics + intrinsics + depth + point maps + 3D tracks from 1 to 100s of views. Outperforms optimization-based methods. Commercial license on checkpoint. Current keystone tool.
- AnySplat, DepthSplat, Fast3R

**Monocular video → 4DGS (dynamic)** — partially solved, **optimization-based**
- Shape of Motion, MoSca, Deblur4DGS (AAAI 2026), Prior-Enhanced GS (Dec 2025)
- Training: minutes to hours per clip
- Inputs: dense video depth (DepthCrafter/MegaSAM), dynamic masks (SAM2/SAM3), optical flow/2D tracks (RAFT/CoTracker), camera poses (VGGT/MegaSAM)

**Feed-forward monocular video → 4DGS** — research frontier, not production-ready. 12-24 months out for VFX-usable quality.

**Standardization** — Khronos `KHR_gaussian_splatting` extension for glTF 2.0 reaches ratification Q2 2026. glTF becomes universal interchange format.

### 16.3 Nuke V2V vision — the transformative use case

Once a plate has a 4DGS sidecar, the Nuke workflow becomes:

1. Read plate + 4DGS sidecar
2. Render novel view from 4DGS with slightly-modified camera (true parallax, not 2.5D)
3. Feed novel view + plate through V2V stylization model (conditioned on 4DGS-rendered target look)
4. Composite back

Step 2 is the breakthrough. V2V currently breaks temporal/spatial consistency because models can't see 3D. With a 4DGS sidecar, the AI has a consistent 3D world. This turns V2V from research demo into shot-ready workflow.

### 16.4 Integration path

**v2a** (first real addon after v1): add `CameraTrackPass` — see Section 20 for full spec. Produces per-frame extrinsics + intrinsics + scene points + distortion (distortion v2b). Outputs multiple sidecars: JSON (complete record), Nuke `.nk` script, Alembic `.abc`, FBX (v2b). Completes the 4DGS input requirements and delivers the "automatic matchmove draft" workflow.

**v3**: add `SceneReconstructionPass`. For static plates: feed-forward via AnySplat/VGGT-backbone, seconds. For dynamic plates: optimization-based via Shape of Motion / MoSca / Prior-Enhanced GS, background job. Output: `utility.scene.ply` or `utility.scene.spz` or `utility.scene.gltf` sidecar.

**v3+**: Nuke panel with 3DGS viewer + novel-view renderer.

---

## 17. License Considerations (summary)

Every pass declares license + commercial flag as first-class metadata. CLI refuses non-commercial by default; requires `--allow-noncommercial`. GUI shows warning dialogs.

| Model | License | Commercial? |
|---|---|---|
| RAFT | BSD-3 | ✅ |
| DSINE | MIT (weights) | ✅ |
| Depth Pro | Apple ML Research | ✅ (read terms) |
| Depth Anything V2 Small/Base | Apache 2.0 | ✅ |
| Depth Anything V2 Large/Giant | CC-BY-NC 4.0 | ❌ |
| DepthCrafter | Apache weights / SVD base non-commercial | ⚠️ ambiguous — treat as ❌ pending confirmation |
| NormalCrafter | Apache weights / SVD base non-commercial | ⚠️ ambiguous — treat as ❌ pending confirmation |
| SAM 2 | Apache 2.0 | ✅ |
| SAM 3 / 3.1 | SAM License (custom) | ✅ with restrictions (no military/ITAR) |
| MatAnyone / MatAnyone 2 | NTU S-Lab License 1.0 | ❌ (business contact available) |
| RVM | MIT | ✅ |
| VGGT | Custom (checkpoint permits commercial) | ✅ (verify) |
| MegaSAM | Apache 2.0 + CC-BY | ✅ |
| MapAnything | TBD — verify | ⚠️ |
| Alembic (SDK) | BSD 3-Clause | ✅ |
| Autodesk FBX SDK | Commercial/EULA | ⚠️ avoid — use ASCII FBX or Alembic |
| CorridorKey | CC BY-NC-SA 4.0 variant | ✅ for processed outputs; ❌ for tool resale |
| GVM (alpha hint generator) | BSD-2-Clause | ✅ |
| VideoMaMa (alpha hint alternative) | CC BY-NC 4.0 (+ Stability weights) | ⚠️ non-commercial |

---

## 18. Conversation Log — Chronological Decisions

For traceability, the design decisions made and why:

1. **Sidecar never merged** — non-destructive ingest, farm-friendly, easy regen of single pass.
2. **Scene-referred EXR throughout, display-referred only for model input** — VFX-correct, auditable via metadata.
3. **Prep-human / execute-hands-off split** — quality control must be human; execution is batch.
4. **CLI + library is source of truth, GUI is a consumer** — enables Nuke panel trivially later.
5. **Pass plugin contract with `pass_type` enum declared for future types** — `scene_3d`/`camera`/`radiometric` namespaced now, implemented later.
6. **Central model registry** — cleanly handles multi-GPU/affinity/caching in v2+.
7. **Flow-guided temporal smoother as reusable post-pass** — closes per-frame flicker cheaply; flow becomes keystone intermediate.
8. **Resize + renormalize + intrinsics-scaling per pass type** — prevents the "resolution-dependent results" bug in DSINE-class models.
9. **Display transform with clip-wide auto-exposure + tonemap + sRGB EOTF** — single exposure avoids flicker in model input.
10. **GUI viewport with original / transformed / compare modes** — TD must see what the model sees to trust the output.
11. **License is first-class; commercial-safe fallback path exists for every pass** — DA V2 + DSINE + RVM + RAFT = complete commercial pipeline.
12. **SAM 3 obsoletes custom orchestrator for matte detection** — one model does detection + segmentation + video tracking with stable IDs.
13. **Top-N ranking for matte RGBA packing uses flow for motion energy** — reusing the keystone intermediate again.
14. **VGGT as ambient future capability** — declaring it early means v2 camera pass and v3 scene reconstruction slot in cleanly.
15. **Sidecar generalized to dict** — `{"utility": ..., "camera": ..., "scene": ...}` supports v3 GS outputs natively.
16. **Camera pass elevated to v2a priority** — matchmove draft in every ingest is the highest-impact single addition after v1; changes starting point of every shot.
17. **MegaSAM as primary camera backend, not VGGT** — VGGT is fast but has hard 60-80 frame limit, pinhole-only, and brittle on low-parallax/dynamic scenes (common in VFX). MegaSAM is slower but video-native, handles low-parallax, dynamic, unknown FOV, Apache 2.0 license.
18. **Dual-backend with auto-routing via flow-pass parallax estimate** — another reuse of the flow intermediate. Short static scenes go to VGGT for speed; video/dynamic/low-parallax go to MegaSAM for robustness.
19. **Metadata reconciliation with solver output** — cinema camera metadata (ARRI, Cooke /i, RED) constrains solver output. Our value-add over pure model inference is this reconciliation logic.
20. **Pinhole assumption in v2a, distortion in v2b** — ships useful sooner; MM artists can refine distortion in their own tool if needed.
21. **FBX deferred to v2b; v2a ships JSON + Nuke .nk + Alembic** — Autodesk Python FBX SDK is dead-on-arrival for a shippable tool; ASCII FBX is writable ourselves but not critical-path. Nuke .nk covers 80% of use case; Alembic covers Maya/Houdini/Blender.
22. **Quality metrics ship with every solve** — compositors won't trust automatic MM without confidence data. Traffic-light per-shot indicator in GUI.
23. **Green screen keying (CorridorKey) as v2c loose-coupled satellite pass** — self-contained, consumes original EXR directly, doesn't need display transform. Integrates via subprocess wrapper rather than library import, keeping license separation clean.
24. **SAM 3 matte pass provides the CorridorKey alpha hint** — better than GVM standalone. The integration is the value-add; we're not just packaging CorridorKey, we're improving its input via our existing matte pass.
25. **Matte pass exposes hard mask as addressable artifact** — small v1 detail that enables v2c consumption via `provides_artifacts: ["sam3_hard_mask"]` declaration.

---

## 19. Open Questions for Handoff to Claude Code

- [ ] Confirm NormalCrafter and DepthCrafter license status with authors (email business contact)
- [ ] Format of handoff doc: markdown spec only / spec + Claude Code prompt / prompt only
- [ ] Granularity: full v1 spec / Phase 0 first / Phase 0+1
- [ ] Repo scaffolding included (pyproject.toml, ruff, pytest, CI) — yes/partial/no
- [ ] Verify MapAnything license before considering for metric scale
- [ ] Stress-test MegaSAM on real VFX plates before locking as v2a primary (low-parallax long-lens, anamorphic, night exterior)

---

## 20. v2a Addon: Camera Pass (Matchmove Draft)

**The feature that changes the tool from "useful" to "essential."** Every ingested plate ships with a ready-to-use 3D camera in Nuke/Maya/Houdini. Compositors start on day one with a working 3D scene instead of waiting for matchmove.

### 20.1 What "basic matchmove" contains

- **Camera intrinsics**: focal length (possibly animated for zooms), principal point, sensor size, pixel aspect
- **Camera extrinsics per frame**: position (tx, ty, tz) + rotation (quaternion or Euler)
- **World orientation**: aligned to ground plane (Y-up convention)
- **World scale**: metric where possible (via metadata/depth priors), otherwise documented-unscaled
- **Sparse scene points**: "survey points" for compositor reference
- **Quality metrics**: per-frame reprojection error, track confidence, solve quality score
- **Lens distortion**: **deferred to v2b** — v2a assumes pinhole, documents limitation

### 20.2 Backend triage — MegaSAM primary, VGGT secondary

After research, the video-native behavior of MegaSAM outperforms VGGT for VFX plates.

**VGGT (CVPR 2025 Best Paper) — known limits for VFX use:**
- Memory scales quadratically → OOM at ~60–80 frames per batch on single GPU
- Trained on unordered images, not video → doesn't exploit temporal structure
- Brittle on difficult scenes (low-parallax, dynamic) → BA fails with "not enough inliers"
- Pinhole assumption
- **Good at:** fast feed-forward (0.2s), static scenes with good parallax, short sequences, establishing shots

**MegaSAM (Google, CVPR 2025, Apache 2.0 + CC-BY) — addresses VGGT's gaps:**
- Built on DROID-SLAM differentiable BA + learned motion segmentation + mono-depth priors
- **Explicitly designed for monocular dynamic videos with unconstrained camera paths, including low-parallax cases**
- Uses DepthAnything + UniDepth as mono-depth priors for robust initialization
- Focal length optimization via `--opt_focal`
- **Commercial-clean license**
- **Slow:** optimization-based, minutes per clip (not feed-forward). But handles exactly the shots VFX cares about.

**Architecture:** both backends, routed by shot characteristics.

```yaml
camera_pass:
  backend: auto | megasam | vggt | vggt_long
  # auto routing: use parallax from existing flow pass to decide
  #   low parallax OR dynamic scene OR long sequence → megasam
  #   static multi-view OR short seq OR speed priority → vggt
  auto_route:
    parallax_threshold: 0.02      # median flow magnitude / image width
    length_threshold: 80          # frames; above this, vggt needs chunking
    dynamic_threshold: 0.3        # fraction of dynamic mask coverage
```

**Long sequences**: if VGGT selected and clip >60 frames, auto-enable VGGT-Long's chunk+loop+align strategy (or use MegaSAM instead).

**Tertiary (future eval)**: MapAnything was recently integrated into VGGT-Long's pipeline (Dec 2025) and provides **metric scale feed-forward** predictions. Potential option for scale recovery when available with commercial license. Verify before using.

### 20.3 Metadata pipeline — the secret weapon

This is where our tool outperforms pure model output. Cinema cameras embed extensive per-frame metadata; using it constrains the solve.

```python
@dataclass
class CameraMetadata:
    # From EXR header
    resolution: tuple[int, int]
    pixel_aspect: float
    colorspace: str

    # Camera body (ARRI / RED / Sony)
    camera_make: str | None = None            # "ARRI", "RED", "Sony"
    camera_model: str | None = None           # "Alexa 35", "V-Raptor"
    sensor_size_mm: tuple[float, float] | None = None
    iso: int | None = None
    shutter_angle: float | None = None

    # Lens data — per-frame if available (Cooke /i, ARRI LDS)
    focal_length_mm: float | list[float] | None = None
    focus_distance_m: float | list[float] | None = None
    t_stop: float | list[float] | None = None
    lens_model: str | None = None
    lens_serial: str | None = None

    # Reel / clip identity
    timecode_start: str | None = None
    reel_id: str | None = None
    clip_name: str | None = None

    # Sidecar metadata neighbors (MOV/MXF/R3D/CDL)
    sidecar_paths: list[Path] = field(default_factory=list)
```

**Metadata sources (read order):**
1. EXR header custom attributes (Nuke-written EXRs, ARRI ProRes-to-EXR converters)
2. Sidecar files next to the sequence (`.ale`, `.xml`, `.rmd`, `.cdl`)
3. User-provided hints file (`shot.hints.json` dropped in the sequence folder)
4. Per-clip overrides in the GUI inspector (manual entry)

**Reconciliation policy for focal length**:

```python
def reconcile_focal(metadata_fx, solver_fx, trust_weights):
    # Rule 1: if metadata absent → trust solver
    # Rule 2: if metadata and solver agree within 5% → average, weighted to metadata
    # Rule 3: if metadata and solver disagree by 5-15% → warn, prefer metadata
    # Rule 4: if metadata and solver disagree by >15% → flag as low-confidence,
    #         show BOTH in UI, require user choice
    # Rule 5: if metadata says zoom (focal animated) and solver says static →
    #         refit solver with variable focal constraint
```

This logic is what makes the output usable on professional plates where a 32mm Cooke prime on a Super35 sensor should give `fx ≈ 2050px` on a 3840-wide frame, and any significant deviation from that is almost certainly a solver error.

### 20.4 Scale and ground plane — human-assisted

VGGT and MegaSAM both produce solves in arbitrary coordinate systems at arbitrary scale. For the FBX/Alembic to be useful, we need to align.

**Ground plane detection (automatic with override):**
- Option A: RANSAC-fit plane to depth points below principal axis → rotate to Y=0
- Option B: leverage SAM 3 `mask.ground`/`mask.floor`/`mask.water` detections
- Option C: user clicks ground in GUI viewport on keyframe → solve aligns to that plane

Default: automatic with B + A fallback. If confidence is low, GUI prompts user for manual override.

**Scale recovery (automatic with override):**
- Option A: if metric depth available (Depth Pro in depth pass), compute scene scale from median depth comparison
- Option B: detect known-size concepts (SAM 3 `person` → average human 1.7m) → scale scene
- Option C: user specifies "this doorway is 2.1m" → manual scale
- Option D: MapAnything (future, if commercial license clears) → metric feed-forward

Default: Option A if metric depth ran, Option B as fallback, "unscaled with metadata flag" otherwise.

**GUI widget**: new inspector panel for camera pass — two sliders (pitch for ground alignment fine-tune, scale multiplier), plus "Set ground from this frame" button and "Scale so this object = N meters" tool.

### 20.5 Export formats — tiered priority

Based on research into FBX tooling pain and DCC ecosystem:

**v2a priority (implement in this order):**

1. **JSON** (our format, complete record) — trivial, first to implement
   - `shot.utility.camera.json` with full per-frame data + metadata + confidence
   - The permanent record; all other exports derive from this
   - Future-proof schema (versioned)

2. **Nuke `.nk` script snippet** — 80% of use case, trivial to write
   - Text file with `Camera3` node + animated focal length + STMap placeholder
   - ~20 lines of templated Python string generation, no dependencies
   - Dropped in the shot folder, compers just "Import Script" in Nuke

3. **Alembic (`.abc`)** — universal DCC
   - Official `alembic` Python bindings (BSD 3-Clause, maintained, pip-installable)
   - Native read in Maya, Houdini, Blender, Katana
   - Nuke reads but less smoothly than FBX for cameras (acceptable — Nuke users take .nk path)
   - Estimated ~1 day of implementation

**v2b (deferred by one release):**

4. **FBX (ASCII)** — we write it ourselves
   - FBX ASCII format is text, well-documented, camera subset is small
   - ~200–300 lines of templated string output
   - **Why not the Autodesk Python FBX SDK**: binary-only, pinned Python versions (2.7/3.7 historically), platform-specific, cannot pip install cleanly — dead-on-arrival for a shippable tool
   - **Why not Blender bpy**: ~150MB dependency for just FBX writing — not worth it
   - Reference: Blender's own `export_fbx_bin.py` for `CAMERA_FOCAL` animation curve structure (Apache-compatible reading)

**v3:**

5. **USD** (`.usda` or `.usdc`) — modern pipelines moving here; USD Python bindings are clean

**Axis/unit conventions — lock these**:
- Y-up right-handed (matches Maya, Houdini, Nuke, Blender's import interpretation)
- Units: meters (with metadata declaring the unit)
- FBX axis metadata set explicitly on every export
- Camera aperture (sensor size) exported so Nuke correctly interprets FOV

### 20.6 Distortion — deferred to v2b

v2a: **pinhole assumption, documented**. Output carries metadata `utilityPass/camera/distortion_model: "pinhole"`. For shots that need distortion handled, MM artists use our solve as starting point in their tool of choice.

v2b options to evaluate:
- **Classical self-calibration**: find straight edges in the scene (Hough + edge detection), fit radial distortion coefficients. Works OK for architectural plates; bad on organic scenes.
- **Learned lens distortion estimation**: research; newer models exist (need investigation closer to v2b).
- **3DE-compatible output**: if we can estimate `k1, k2, k3` or the custom 3DE model, we can write an `.nk` STMap node or direct 3DE project files for hero-quality work.

Architectural hook in v2a: the camera pass declares `produces_sidecars` including a `distortion` type, implemented as empty/no-op in v2a. v2b fills in the estimator.

### 20.7 Quality metrics — compositor-trust enablers

Compositors won't trust an automatic solve without visible confidence data. Every export includes:

```json
{
  "quality": {
    "per_frame_reprojection_error_px": [0.34, 0.41, 0.38, ...],
    "solve_confidence": 0.82,              // 0-1, per-frame and aggregate
    "tracked_points_count": [142, 138, ...],
    "focal_stability_stdev": 0.012,        // in normalized units
    "backend_used": "megasam",
    "metadata_agreement": {
      "focal_length_vs_metadata": 0.98,    // 1.0 = perfect agreement
      "reconciliation_applied": "averaged_weighted_metadata"
    },
    "warnings": [
      "Low parallax in frames 1045-1072; solve may be underconstrained",
      "Rolling shutter detected; consider using rolling-shutter-aware tool for hero work"
    ]
  }
}
```

GUI exposes this as a traffic-light indicator per shot (green / yellow / red). Shots flagged red → MM artist takes over. Shots flagged green → comp uses directly.

### 20.8 Known failure modes to document

To set expectations for users and prevent silent bad solves:

- **Long-lens locked-off shots**: low parallax, MegaSAM handles better than VGGT but still unreliable. Document in the GUI warning.
- **Water / reflective surfaces**: tracking fails on moving water. Detected via high residual error → flag.
- **Rolling shutter**: CMOS sensors with fast pans produce per-scanline distortion that neither backend models. Flag when camera metadata declares CMOS + high pan rate.
- **Dollies with zero parallax** (push-in on a flat subject): scale ambiguous, solve may collapse. Flag.
- **Motion control / repeatable moves**: if metadata indicates MoCo, note in output (MM artists will want to know).
- **Crash zooms**: focal length changes faster than solver can track. Flag based on metadata focal curve.

These become entries in the `warnings` field and trigger GUI flags during prep.

### 20.9 Revised roadmap with v2a positioned

```
v1   (8 weeks)  — Foundation: depth, normals, matte, flow + GUI + CLI (locked)
v2a  (4 weeks)  — Camera pass: MegaSAM + VGGT + metadata + JSON/.nk/Alembic export
v2b  (3 weeks)  — FBX ASCII export + distortion estimation
v2c  (2 weeks)  — Green screen keying pass (CorridorKey integration — see Section 21)
v2d  (4 weeks)  — Farm execution (Deadline)
v2e  (2 weeks)  — Pipeline integrations (Prism, ShotGrid)
v3a  (6 weeks)  — Nuke panel integration (loads .nk, monitors jobs, inline preview)
v3b  (8 weeks)  — Scene reconstruction: 3DGS (static) + 4DGS (dynamic) sidecar export
```

v2a comes first because it's the highest artistic-impact single addition after v1 — it changes the starting point of every shot.

### 20.10 What to add to v1 now to prepare

Tier 1 architectural preparations (cheap now, expensive later):

1. **`CameraMetadata` extraction in `io/metadata.py`** — implement fully in v1. Used by depth Z-scale hints, normals intrinsics, matte DOF heuristics, and critically by v2a camera pass.
2. **`SidecarWriter` registry in `io/writers/`** — abstract interface. v1 only instantiates `ExrSidecarWriter`. v2a adds `JsonSidecarWriter`, `NukeScriptWriter`, `AbcSidecarWriter`. v2b adds `FbxAsciiWriter`. v3 adds `PlySidecarWriter` for GS.
3. **`Shot.sidecars: dict[str, Path]`** — already planned, confirm this is a dict not a single path.
4. **`CameraPass` stub** — empty class with declared `pass_type = "camera"` and `produces_sidecars = [...]`. Forces the architecture to flow through non-EXR outputs from day one.
5. **Flow pass exposes parallax estimate** — add `provides_artifacts: ["parallax_estimate"]` so v2a auto-routing can consume it without recomputing.

None of these require implementing the camera pass logic in v1 — just declaring the shapes.

---

## 21. v2c Addon: Green Screen Keying Pass (CorridorKey)

**A loose-coupled satellite pass for green/blue screen plates.** Runs beside the main pipeline, consumes the original EXR directly, produces premultiplied RGBA with hair/motion-blur/translucency preserved. Not needed for most shots; essential for productions with significant VFX-against-screen work.

### 21.1 Why this fits the architecture so well

Three properties that make it an ideal Tier-2 addon:

1. **Self-contained** — CorridorKey handles its own colorspace (sRGB or linear sRGB/Rec.709), doesn't need our display transform, doesn't need flow, doesn't need temporal smoothing. It's a standalone AI inference over individual frames.
2. **Modest VRAM** — 6–8GB on consumer GPUs after community optimizations (was 24GB originally). Fits alongside our other passes without forcing a bigger machine.
3. **Clean input contract** — takes original plate + coarse alpha hint → outputs premult RGBA. No dependency graph entanglement with other passes.

### 21.2 The value-add — why not just point users at CorridorKey directly

CorridorKey needs an **alpha hint** (coarse mask) as input. Standalone options are GVM (auto, BSD-2) or VideoMaMa (CC-BY-NC, needs hand-drawn seed). Quality of the hint directly drives quality of the final key.

**We already produce higher-quality hints than GVM as part of v1's matte pass.** SAM 3 with concept prompt `"person"` (or whatever the subject is) gives cleaner, more temporally-consistent per-frame masks than GVM's auto-matters. The integration becomes:

```
Green screen plate ─┬─► SAM 3 matte pass (hard mask per frame) ──► alpha hint
                    │                                                  │
                    └─► Original plate ───────────────────────────────► CorridorKey
                                                                       │
                                                                       Premultiplied RGBA EXR
                                                                       (hair, motion blur, translucency preserved)
```

Result: better key than CorridorKey standalone, because we supply it SAM 3 quality hints. **The integration is the value-add, not just the packaging.**

### 21.3 Pass architecture

```python
class GreenScreenKeyPass(UtilityPass):
    name = "greenscreen"
    version = "corridorkey-1.x"
    license = License("CC-BY-NC-SA-4.0-variant",
                      commercial_outputs_ok=True,
                      commercial_tool_resale=False)
    pass_type = "semantic"   # family match with matte

    input_colorspace = "srgb_linear"  # NOT the display-transform output
    temporal_mode = "per_frame"       # CorridorKey processes frames independently
    vram_estimate_gb = lambda w, h: 6.0 if min(w, h) <= 1080 else 8.0
    model_native_resolution = (2048, 2048)
    supports_tiling = True            # CorridorKey has built-in tiling

    requires_artifacts = ["sam3_hard_mask"]  # from matte pass
    provides_artifacts = []
    produces_channels = [
        ChannelSpec("R",      dtype="float32"),   # unmultiplied foreground
        ChannelSpec("G",      dtype="float32"),
        ChannelSpec("B",      dtype="float32"),
        ChannelSpec("A",      dtype="float32"),   # linear alpha
        # Optional premultiplied pair for convenience
        ChannelSpec("key.r",  dtype="float32"),   # premult R = R*A
        ChannelSpec("key.g",  dtype="float32"),
        ChannelSpec("key.b",  dtype="float32"),
        ChannelSpec("key.a",  dtype="float32"),
    ]
```

**Output sidecar**: `<plate>.key.exr` — separate from `<plate>.utility.exr` because the keying output is conceptually different (it's a *replacement* for the plate's foreground, not an auxiliary pass). Nuke compers read it directly into their key branch.

### 21.4 YAML config

```yaml
passes:
  greenscreen:
    enabled: false                    # off by default; opt-in per shot
    backend: corridorkey
    params:
      # CorridorKey-specific
      resolution: 2048                # internal inference size; auto-scales to plate
      tile: auto                      # auto-tile at 4K+
      precision: fp16                 # fp16 or fp32
    alpha_hint:
      source: sam3_matte              # use our SAM 3 output (preferred)
      # alternatives:
      # source: gvm                   # CorridorKey's bundled auto-matte
      # source: videomama             # needs videomama_seed: <path>
      # source: manual                # needs hint_path: <path>
      concept: person                 # what to hint for if using sam3_matte
      erosion_px: 4                   # CorridorKey works best with slightly eroded hints
    output:
      unmultiplied: true              # write R/G/B straight color
      premultiplied: true             # also write key.r/g/b/a
      sidecar_name: key               # → <plate>.key.exr
```

### 21.5 GUI integration

Satellite status in the shot list:
- Shot list shows a small `[GS]` badge on shots where greenscreen pass is enabled
- Inspector has a "Green Screen" toggle — when enabled:
  - Shows screen color picker (green / blue / custom)
  - Shows alpha hint source dropdown (sam3 / gvm / manual)
  - Shows concept field for sam3 hint (`"person"` default)
  - Adds a 4th viewport mode: **Keyed** — shows the composited result over a user-chosen background

Auto-detect stretch goal: run a quick green-dominance check on the first frame; if >40% of pixels fall in the green-screen hue range, suggest enabling the pass. v2c ships with manual toggle; auto-detect is v2c+1 polish.

### 21.6 Implementation notes

**Integration strategy**: wrap the CorridorKey CLI via subprocess rather than importing as a library. Reasons:
- CorridorKey uses `uv` for dependency management — mixing its env with ours is painful
- Its CLI is stable; refactors don't break our integration
- License separation is cleaner (we call it as a tool, don't embed it)
- Updates are independent (user can upgrade CorridorKey without rebuilding our tool)

Concretely: our pass writes the alpha hint as a temporary EXR, invokes CorridorKey with the original plate + hint, reads back the premult RGBA EXR into our sidecar system.

**Colorspace path**: CorridorKey wants sRGB/Rec.709 linear. Our OCIO layer converts ACEScg/LogC4/etc. → linear Rec.709 as the input to this pass, **skipping the display transform** (no tone mapping — CorridorKey isn't an AI model trained on display-referred data, it's trained on linear synthetic data).

**License gating**: because CorridorKey allows commercial use of outputs but restricts tool resale, the pass declares `commercial_outputs_ok=True, commercial_tool_resale=False`. Our license gate accepts this for studio use without the `--allow-noncommercial` flag. If we ever build a paid SaaS around the utility passes tool and CorridorKey inference is part of that offering, we'd need written permission from Niko/Corridor.

### 21.7 Why this is v2c, not v1

- Applies to a subset of shots (GS plates only) → narrower impact than camera pass
- Requires v1 matte pass (SAM 3) to provide good alpha hints → must come after v1
- Not on the critical path for 4DGS reconstruction (v3) → no dependency pressure
- 2 weeks of integration work rather than v1 scope risk

**Skipping it until v2c is the right call.** The reasoning is exactly what you identified: it runs itself, takes the original EXR, and can be added without touching the rest of the pipeline.

### 21.8 v1 preparation — essentially none

Unlike the camera pass (which needed metadata extraction + sidecar registry + stubs in v1), the greenscreen pass needs almost no v1 prep:

- `SidecarWriter` registry already generalized (from camera pass prep) — CorridorKey output is just another EXR sidecar
- SAM 3 matte pass already produces hard masks in v1 — the alpha hint is free
- Pass plugin contract already supports per-pass `input_colorspace` declarations — CorridorKey's "linear sRGB, no tonemap" is natively expressible
- License flag system already handles commercial-OK / tool-resale-restricted — v2c just adds the entry

The only v1 thing worth verifying: **ensure the matte pass's hard-mask output is available as a standalone artifact** (not only packed into the sidecar EXR). If a v2c pass wants to consume it, it needs to be addressable. The `provides_artifacts: ["sam3_hard_mask"]` declaration on the matte pass handles this — make sure it's there.

---

## 22. Packaging & Environment — Self-Contained by Design

The tool ships self-contained. No system Python contamination, no conda conflicts, no "works on my machine." VFX has a specific pain here: every DCC ships its own Python, and cross-contamination destroys production pipelines. We solve it with modern Python tooling.

### 22.1 Stack

- **`uv`** — project-local environment + package manager. Written in Rust, 10-100x faster than pip, handles Python installation itself. Single binary (~20MB). Already the standard for modern Python tooling in 2026.
- **`uv.lock`** — committed to the repo; reproducible builds across machines.
- **`pyproject.toml`** — single source of truth for all dependencies, metadata, entry points.
- **Project-local `.venv`** — lives in the project folder; deleting it resets everything cleanly.
- **Model checkpoints** — downloaded lazily on first pass invocation via `huggingface_hub`, cached to user-level cache (not per-project — models are big, shouldn't re-download).

### 22.2 Python and CUDA handling

- Python 3.11 provisioned by `uv` — user never needs to know what system Python they have.
- CUDA: PyTorch wheels bundle CUDA runtime. Users need only an NVIDIA driver supporting our chosen CUDA version (12.1 or 12.4 — pinned in `pyproject.toml`). We do NOT require users to install CUDA Toolkit separately.
- macOS: PyTorch MPS backend for Apple Silicon; CPU fallback for Intel. Most passes are GPU-heavy and will be slow on Apple Silicon (realistically v1 targets Linux/Windows + NVIDIA).

### 22.3 Non-Python dependencies

| Dependency | Source | v1 required? |
|---|---|---|
| OpenImageIO Python bindings | `oiio-python` wheel | ✅ |
| OpenColorIO Python bindings | `opencolorio` wheel | ✅ |
| FFmpeg | system PATH | v2 (non-EXR inputs) |
| CUDA driver | NVIDIA | ✅ (GPU passes) |
| Alembic Python bindings | `alembic` wheel | v2a (camera pass) |
| CorridorKey | separate `uv pip install` | v2c |

Each checked at startup; missing dependencies produce **clear error messages with remediation hints**, not stack traces.

### 22.4 What the user installs

```bash
# One line to install uv (cross-platform, official installer)
curl -LsSf https://astral.sh/uv/install.sh | sh       # Linux/macOS
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# Then our tool
git clone https://github.com/<org>/utility-passes
cd utility-passes
./install.sh    # or install.bat on Windows
```

That's it. `install.sh` handles everything else. See Section 23.

---

## 23. Installer & Bootstrapper

Two-tier pattern, proven by CorridorKey and other modern AI tools.

### 23.1 Tier A: universal install scripts (v1)

```
install.sh   (Linux/macOS)
install.bat  (Windows)
```

~100 lines each. What they do, with logging to `install.log`:

1. Check for `uv`; install if missing (one-liner from astral.sh)
2. Check NVIDIA driver sufficiency (or warn if CPU-only)
3. Provision Python 3.11 via `uv python install 3.11`
4. Run `uv sync` to install all deps from `uv.lock`
5. Verify install with smoke test (`uv run utility-passes --version`)
6. Offer to pre-download model weights (`utility-passes models pull --all`)

**Logging format** (copy exactly, this is non-negotiable UX):

```
[2026-04-16 14:23:01] ✓ uv 0.4.18 found
[2026-04-16 14:23:02] ✓ NVIDIA driver 560.35 (CUDA 12.6) — OK
[2026-04-16 14:23:05] ✓ Python 3.11.10 provisioned in .venv
[2026-04-16 14:23:42] ✓ torch 2.3.0+cu121 installed
[2026-04-16 14:24:11] ✗ FAILED: oiio-python wheel not available for platform
                     → See: https://docs.openimageio.org/en/latest/installation.html
                     → Or run: conda install -c conda-forge openimageio
[2026-04-16 14:24:12] Install incomplete. See install.log for full details.
```

Every failure ships with remediation hints. "✗ FAILED + what to do next" is what separates usable studio tools from rage-quit hellware.

### 23.2 Tier B: one-click installer (v2+)

Windows primarily (most friction there for non-technical artists):

- `.msi` or signed `.exe` bundling uv + Python + wheels + models
- Similar to EZ-CorridorKey's Windows installer approach (~1-2GB download, zero setup)
- Linux: AppImage or Flatpak (less critical — Linux users run install.sh fine)
- macOS: `.pkg` for Apple Silicon Macs
- **Slot in v2+** when non-technical users show up. v1 audience = TDs/compers who handle a terminal.

### 23.3 Update mechanism

```bash
utility-passes update            # checks git, pulls if newer, runs uv sync
utility-passes update --check    # dry-run: just report if update available
utility-passes update --version 1.2.3   # pin to specific version
```

For source installs: `git pull` + `uv sync`.
For binary installers: download delta, restart.

Model updates handled separately (`utility-passes models update`).

### 23.4 Uninstall

`uninstall.sh` / `uninstall.bat`: removes `.venv`, optionally removes model cache (asks first — multi-GB download to re-fetch), leaves user data / sessions / sidecars untouched. The directory delete reclaims everything; no registry entries or system-wide hooks on Tier A.

### 23.5 Studio deployment considerations

v2+ concerns, but worth noting now so v1 doesn't block them:

- **Shared install**: multiple users on same machine. Pattern: install the `.venv` to a shared location (`/opt/utility-passes` or `C:\ProgramData\utility-passes`); user sessions + preferences live per-user. uv supports shared envs naturally.
- **Air-gapped studios**: no internet access. Pattern: build offline wheelhouse + model bundle; install script has `--offline` mode that reads from local cache. Achievable in v2, requires no v1 changes.
- **Farm deployment**: identical install on every farm node. uv.lock guarantees this.

---

## 24. Plugin System — Plugins All the Way Down

Everything is a plugin, from day one. There is no distinction between "built-in" passes and "third-party" passes — the ones we ship are declared identically to ones anyone else writes. This is the architectural unlock for community growth, academic adoption, and eventual commercial plugin marketplaces.

### 24.1 Discovery mechanism — Python entry points

Standard Python ecosystem pattern. Our `pyproject.toml` declares every pass as an entry point:

```toml
[project.entry-points."utility_passes.passes"]
raft                = "utility_passes.passes.flow:RAFTPass"
depth_anything_v2   = "utility_passes.passes.depth:DepthAnythingV2Pass"
depthcrafter        = "utility_passes.passes.depth:DepthCrafterPass"
depth_pro           = "utility_passes.passes.depth:DepthProPass"
dsine               = "utility_passes.passes.normals:DSINEPass"
normalcrafter       = "utility_passes.passes.normals:NormalCrafterPass"
sam3_matte          = "utility_passes.passes.matte:SAM3MattePass"
matanyone2          = "utility_passes.passes.matte:MatAnyone2Refiner"
rvm                 = "utility_passes.passes.matte:RVMRefiner"

[project.entry-points."utility_passes.executors"]
local               = "utility_passes.executors.local:LocalExecutor"

[project.entry-points."utility_passes.io.readers"]
oiio_exr            = "utility_passes.io.readers:OIIOExrReader"

[project.entry-points."utility_passes.io.writers"]
oiio_exr            = "utility_passes.io.writers:ExrSidecarWriter"
json                = "utility_passes.io.writers:JsonSidecarWriter"

[project.entry-points."utility_passes.integrations"]
standalone          = "utility_passes.integrations.standalone:StandaloneAdapter"
```

A third-party plugin package declares matching entry points in its own `pyproject.toml`:

```toml
[project.entry-points."utility_passes.passes"]
my_new_depth_model  = "my_package.depth:MyNewDepthModel"
```

User installs with `uv pip install utility-passes-plugin-mynewmodel` → our tool auto-discovers it on next launch. No code changes to our tool, no recompilation, no config file edits. It appears in the CLI `--passes` option, the GUI pass-toggle list, everything.

### 24.2 The plugin contract IS the `UtilityPass` ABC

Already defined in Section 6. Any class implementing it + declared via entry points gets:

- Auto-registration in CLI and GUI
- License gating applied from the class's `license` attribute
- VRAM planning, resize handling, temporal smoothing integration — all inherited
- Metadata recording in sidecar EXR (`utilityPass/<name>/...`) — free
- Access to shared artifacts (flow, depth, masks) via `requires_artifacts`

### 24.3 Plugin registry (core implementation)

```python
class PluginRegistry:
    """Central discovery of all plugins via entry points."""

    def __init__(self):
        self._passes: dict[str, type[UtilityPass]] = {}
        self._executors: dict[str, type[Executor]] = {}
        self._readers: dict[str, type[ImageSequenceReader]] = {}
        self._writers: dict[str, type[SidecarWriter]] = {}
        self._integrations: dict[str, type[PipelineAdapter]] = {}

    def load_all(self):
        """Walk all entry points, register everything. Called at startup."""
        for ep in importlib.metadata.entry_points(group="utility_passes.passes"):
            cls = ep.load()
            self._passes[ep.name] = cls

        # ... same pattern for executors, readers, writers, integrations

    def list_passes(self) -> list[str]: ...
    def get_pass(self, name: str) -> type[UtilityPass]: ...
    def list_by_type(self, pass_type: str) -> list[str]: ...
    # e.g. list_by_type("depth") → ["depth_anything_v2", "depthcrafter", "depth_pro"]
```

Core has **zero hardcoded pass imports**. Discovers everything at runtime. Our own built-in passes have no special status.

### 24.4 Namespacing categories

```
utility_passes.passes            # UtilityPass plugins
utility_passes.post              # Post-processors (smoothers, etc.)
utility_passes.io.readers        # ImageSequenceReader plugins
utility_passes.io.writers        # SidecarWriter plugins
utility_passes.executors         # Execution backends
utility_passes.integrations      # Pipeline adapters (Prism, ShotGrid, ...)
utility_passes.tonemappers       # Display transform plugins (AgX, Filmic, custom)
utility_passes.gui.panels        # (v2+) GUI panel extensions
```

Everything extensible, all via entry points.

### 24.5 Plugin template repository

We ship a companion repo: `utility-passes-plugin-template` (or a `cookiecutter` template):

```
utility-passes-plugin-template/
├── pyproject.toml           # pre-filled with entry-point scaffolding
├── src/
│   └── my_plugin/
│       ├── __init__.py
│       └── pass.py          # stub UtilityPass with TODOs
├── tests/
│   └── test_pass.py         # minimal smoke tests using our test fixtures
└── README.md                # walkthrough
```

A researcher adds their new depth model:
```
cookiecutter https://github.com/<org>/utility-passes-plugin-template
# fills in their inference code in pass.py
uv pip install .            # → now available in utility-passes
```

~200 lines of plugin wrapper code = their model in every user's toolchain. This is the mechanism that grows the tool beyond what we personally build.

### 24.6 Plugin lifecycle commands

v1 ships with install/remove/list via uv; richer lifecycle in v2+:

```bash
# v1 (trivial — just wraps uv)
utility-passes plugins list                         # show installed plugins
uv pip install utility-passes-plugin-X              # standard uv install
uv pip uninstall utility-passes-plugin-X            # standard uv uninstall

# v2+
utility-passes plugins list --available             # query a central registry
utility-passes plugins install <name>               # our wrapper over uv
utility-passes plugins remove <name>
utility-passes plugins update [<name>]
utility-passes plugins disable <name>               # keep installed but skip loading
utility-passes plugins info <name>                  # show metadata, license, URL
```

The central plugin registry (v2+) is a simple JSON file hosted on GitHub Pages or similar — no server infrastructure needed. Community publishes to it via PRs.

### 24.7 Safety and quality for third-party plugins

Plugins run in the same process as core — they can do anything the core can. Reasonable safeguards:

- **License metadata is mandatory** — every plugin must declare commercial status. Our license gate applies automatically.
- **Plugin metadata includes source URL** — users can inspect before installing.
- **Signed releases (v2+)** — publish signed wheels, users can opt into verified-only mode.
- **Sandbox mode (v3+)** — run untrusted plugins in subprocess isolation. Complexity vs. value tradeoff; probably not worth building unless we hit a security need.

For v1: trust model is same as pip/npm/any package manager. Users install what they trust, at their own risk. Our README makes this clear.

### 24.8 Business model accommodation

This architecture cleanly supports the "free core + paid plugins" model you mentioned.

- **Core library**: MIT or Apache 2.0. Free forever. Builds trust, enables community, meets VFX open-source expectations.
- **Paid plugins**: anyone (including us) can publish commercial plugins as standard Python packages. License key checked in plugin's `__init__`, plugin refuses to load without valid key.
- **Enterprise "pro" distribution**: bundle of core + commercial plugins + enterprise-hardened integrations (Prism/ShotGrid/OpenPype with SSO, audit logs, priority support). Ship as `utility-passes-pro` — same core underneath, just with paid plugins pre-enabled.
- **Plugin marketplace (v4+, if the tool takes off)**: central registry with paid + free plugins. Stripe integration for licensing. Revenue share with community authors.

None of this requires v1 changes. The entry-points plugin system is the load-bearing architecture; monetization layers on top without core refactor.

### 24.9 What to do in v1

Concrete v1 implementation requirements so the plugin system works from launch:

1. **All built-in passes declared via entry points** in our own `pyproject.toml` — no exceptions, no shortcuts.
2. **`PluginRegistry.load_all()` called at startup** — before CLI arg parsing, before GUI launch.
3. **CLI `--passes` argument lists dynamically** from registry, not hardcoded.
4. **GUI pass toggles populated dynamically** from registry.
5. **License gate inspects plugin metadata, not hardcoded per-model lists.**
6. **Ship `utility-passes-plugin-template` repo alongside v1 release** — minimal but working example.
7. **Document the plugin contract in `docs/developing-plugins.md`** — walkthrough of writing a new pass.

Items 1-5 change nothing about what we're building — just how it's wired. Items 6-7 are documentation + template, maybe 1-2 days of work.

---

## 25. Status: Design Phase Complete — Ready for Handoff to Claude Code

This design document is now **ready for Claude Code spec generation**. All major architectural decisions are recorded, v1 scope is locked with v2+ roadmap clear, Tier 1 future-proofing decisions are documented, and the seams for every known future addon are declared.

Open questions remaining (see Section 19):
- [ ] Confirm NormalCrafter and DepthCrafter license status with authors
- [ ] Decide handoff format for Claude Code (full spec / phased / prompt-only)
- [ ] Decide repo scaffolding inclusion (pyproject.toml, ruff, pytest, CI)
- [ ] Stress-test MegaSAM on real VFX plates before locking as v2a primary
- [ ] Verify MapAnything license before considering for metric scale

**Next step**: answer the handoff format / granularity / scaffolding questions, then generate the Claude Code spec from this document.

---

*Document version: 4.0-packaging-installer-plugins-added*
*Last updated: 2026-04-16*
*This document is the design record. The formal Claude Code spec (when written) will reference it.*
