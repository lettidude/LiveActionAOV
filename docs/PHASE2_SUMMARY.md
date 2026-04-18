# Phase 2 — Depth + Normals (round 1: commercial-safe backends)

**Status:** ✅ Complete — 57/57 tests pass in ~9 s. Awaiting confirmation to
proceed to Phase 2 round 2 (DepthCrafter/NormalCrafter diffusers wiring) or
Phase 3 (Matte pass).

**Exit criterion (spec §13.1 Phase 2):**
> `uv run liveaov run-shot <folder> --passes flow,depth,normals --depth-backend depth_anything_v2 --normals-backend dsine` writes a sidecar with `Z`, `N.x/y/z` + flow channels, normals are unit-length per pixel, licenses recorded. The non-commercial gate (`depthcrafter` without `--allow-noncommercial`) refuses to run with exit 2.

Both clauses met. The normal-triplet renormalization runs after the temporal
smoother so `|N| = 1` survives flow-guided blending.

---

## 1. Scope cut for round 1

| Backend | License | Status in round 1 |
|---|---|---|
| `depth_anything_v2` (Small/Base) | Apache-2.0 | **Full implementation** |
| `dsine` | MIT | **Full implementation** |
| `depthcrafter` | Apache-2.0 + SVD-NC | **License-gated stub** (gate works, inference in round 2) |
| `normalcrafter` | Apache-2.0 + SVD-NC | **License-gated stub** |
| `depth_pro` | Apple ML Research | **Deferred** — licensing review + code port |

Round 2 will wire real inference into DepthCrafter + NormalCrafter via
`diffusers` (already a registered optional extra), then bring Depth Pro in
with its own license gate.

---

## 2. Modules added

### `src/live_action_aov/passes/depth/depth_anything_v2.py`
`DepthAnythingV2Pass` — commercial-safe fallback depth backend.

| Field | Value |
|---|---|
| `name` | `depth_anything_v2` |
| `pass_type` | `GEOMETRIC` |
| `temporal_mode` | `PER_FRAME` |
| `license` | `Apache-2.0`, commercial OK |
| Variants | `small`, `base` (refuses `large`/`giant`; those are CC-BY-NC-4.0) |
| Produces | `Z` (per-clip normalized, larger = nearer), `Z_raw` (raw model output) |
| Artifacts | `depth_norm_min`, `depth_norm_max` (for sidecar metadata) |
| Default params | `variant=small`, `inference_short_edge=518`, `smooth=auto` |

**Trap 5 (per-clip depth normalization)** — `run_shot` collects raw depth for
every frame, computes one `(d_min, d_max)` pair across the whole clip, then
applies `Z = 1 - (d - d_min) / (d_max - d_min)` per frame. `Z_raw` stays
un-normalized so downstream tools can apply their own policy.

Backend: HuggingFace `transformers.AutoModelForDepthEstimation` +
`AutoImageProcessor` loading `depth-anything/Depth-Anything-V2-Small-hf` (or
`-Base-hf`). Weights stream to `~/.cache/huggingface` on first use.

### `src/live_action_aov/passes/normals/dsine.py`
`DSINEPass` — commercial-safe fallback normals backend.

| Field | Value |
|---|---|
| `name` | `dsine` |
| `pass_type` | `GEOMETRIC` |
| `temporal_mode` | `PER_FRAME` |
| `license` | `MIT`, commercial OK |
| Produces | `N.x`, `N.y`, `N.z` — camera-space, [-1,1], **unit-length per pixel** |
| Default params | `inference_short_edge=480`, `smooth=auto`, axes `opencv → opengl` |

**Trap 2 (renormalize after resize)** — bilinear upscale breaks unit length;
`postprocess` divides by `max(||N||, 1e-6)` then clamps to [-1, 1].

**Trap 3 (intrinsics scaling)** — `_scaled_intrinsics` scales `fx, fy, cx, cy`
by the inference/plate ratio. When shot intrinsics aren't provided, it falls
back to a 50mm-equivalent heuristic: `f = 0.8 * max(W, H)` at plate res, then
scaled to inference res.

**Axis convention** — DSINE ships OpenCV (+X right, +Y down, +Z forward into
scene); spec §10.3 requires OpenGL (+X right, +Y up, +Z toward camera). The
conversion flips Y and Z sign. A dedicated `_convert_axes` helper handles
the switch and is fully involutory (round-trip is identity).

Backend: `torch.hub.load('baegwangbin/DSINE', 'DSINE', trust_repo=True)` on
first use. Tests bypass this with a fake model so CI never touches the hub.

### `src/live_action_aov/passes/depth/depthcrafter.py` + `passes/normals/normalcrafter.py`
License-gated stubs (same shape as `passes/camera/stub.py`): declare
`license.commercial_use = False` so the CLI's license gate blocks them
without `--allow-noncommercial`. All three lifecycle methods raise
`NotImplementedError` with a pointer to round 2.

### `src/live_action_aov/core/pass_base.py` — one-line extension
New class attribute `smoothable_channels: list[str] = []`. The executor's
auto-smoother reads it to know which of a pass's outputs are worth
smoothing. Non-smoothable channels (like `Z_raw`) are deliberately omitted.

### `src/live_action_aov/post/temporal_smooth.py` — triplet renormalize
When `applied_to` contains `N.x`, `N.y`, `N.z` together, the smoother
renormalizes the triplet post-blend so `|N| = 1` holds per pixel. Nuke's
Relight math depends on this, and bilinear-blended unit normals
catastrophically break it (magnitudes of ~0.7 from two orthogonal vectors
blended 50/50).

### `src/live_action_aov/executors/local.py` — auto-wired smoother + metadata
- After all passes run, the executor scans `ordered` for `PER_FRAME`
  passes whose `smooth` param resolves to truthy-or-"auto" via
  `_smooth_wanted`. Each such pass gets its own `PostConfig(name="temporal_smooth", params={"applied_to": <smoothable_channels>, "_auto_for": <pass_name>})` appended to the post-processor queue.
- The auto-wire only fires when `forward_flow` was actually emitted — if
  no flow pass ran, the smoother would have nothing to consume.
- Metadata key disambiguation: each applied smoother writes to
  `liveActionAOV/smooth/temporal_smooth::<pass_name>/*` when auto-wired, so
  a job with both depth and normals smoothing doesn't collide in the EXR
  header.
- New metadata: `liveActionAOV/depth/normalization/min`, `.../max`,
  `liveActionAOV/depth/space = "relative"`, `.../unit = "normalized_per_clip"`.

### `src/live_action_aov/cli/app.py` — semantic backend rewrite
Two new options:
- `--depth-backend` (default `depth_anything_v2`)
- `--normals-backend` (default `dsine`)

`_resolve_semantic_passes` rewrites `depth` / `normals` in `--passes` to the
chosen backend, deduplicating in order. Concrete backend names pass through
unchanged so power users who write `--passes flow,depthcrafter,normalcrafter`
aren't affected.

### `pyproject.toml`
- New dependency: `transformers>=4.45` (DA V2's HF pipeline).
- Entry-point registrations:
  ```toml
  depth_anything_v2  = "live_action_aov.passes.depth.depth_anything_v2:DepthAnythingV2Pass"
  depthcrafter       = "live_action_aov.passes.depth.depthcrafter:DepthCrafterPass"
  dsine              = "live_action_aov.passes.normals.dsine:DSINEPass"
  normalcrafter      = "live_action_aov.passes.normals.normalcrafter:NormalCrafterPass"
  ```

---

## 3. Tests added (23 new, all passing)

| File | Count | Covers |
|---|---|---|
| `tests/test_passes/test_depth_anything_v2.py` | 6 | Fake-model contract: Z+Z_raw shape/dtype, per-clip norm spans [0,1] across frames (not per-frame), raw ordering preserved, `emit_artifacts` exposes min/max, license flag, refuses `large`/`giant` |
| `tests/test_passes/test_dsine.py` | 6 | Fake-model contract: unit-length after upscale, axis flip opencv→opengl, `_convert_axes` involutory, `_scaled_intrinsics` matches resize ratio + falls back to 50mm heuristic, license flag |
| `tests/test_cli/test_cli_phase2.py` | 8 | Semantic rewrite (`depth` → backend, dedup, passthrough), plugins list visibility, non-commercial gate blocks `depthcrafter` + `normalcrafter` with exit 2, commercial backends clear gate |
| `tests/test_executors/test_auto_smoother.py` | 2 | Auto-wire fires when fake flow + PER_FRAME depth both present, suppressed when no flow pass |
| `tests/test_post/test_temporal_smooth.py` (+1) | 1 | Triplet renormalize keeps \|N\|=1 after blending (1,0,0) ⊕ (0,1,0) at α=0.5 |

Total suite: **57 passed in ~9 s on CPU.** No GPU or HuggingFace downloads
needed for unit tests — real-model integration is reserved for dedicated
slow/gpu-marked tests in round 2.

---

## 4. By-hand exit-criterion verification

```bash
# Happy path:
uv run liveaov run-shot A:/tmp/plate2 --passes flow,depth,normals \
    --depth-backend depth_anything_v2 --normals-backend dsine
# → sidecar EXRs with Z, Z_raw, N.x, N.y, N.z + 9 flow channels
# → `liveActionAOV/depth/normalization/min|max`, `depth/space=relative`
# → auto-wired `smooth/temporal_smooth::depth_anything_v2/applied_to=Z`
# → auto-wired `smooth/temporal_smooth::dsine/applied_to=N.x,N.y,N.z`

# License gate:
uv run liveaov run-shot A:/tmp/plate2 --passes depth \
    --depth-backend depthcrafter
# → exit 2, message points at --allow-noncommercial

uv run liveaov run-shot A:/tmp/plate2 --passes depth \
    --depth-backend depthcrafter --allow-noncommercial
# → exit 1 with "inference lands in Phase 2 round 2" (stub raises)
```

Plugins list:
```
$ uv run liveaov plugins list
 name              | type      | license           | commercial
-------------------+-----------+-------------------+-----------
 depth_anything_v2 | geometric | Apache-2.0        | yes
 depthcrafter      | geometric | Apache-2.0+SVD-NC | no
 dsine             | geometric | MIT               | yes
 flow              | motion    | BSD-3-Clause      | yes
 normalcrafter     | geometric | Apache-2.0+SVD-NC | no
```

---

## 5. Non-goals respected

- ❌ No Depth Pro pass — Apple ML Research License needs a separate review
  pass before we wire it up.
- ❌ No DepthCrafter/NormalCrafter real inference — gated stubs land the
  license surface; real work in round 2 with the `live-action-aov[depthcrafter]`
  optional extra.
- ❌ No GUI-side backend picker — YAML `params.variant` / `params.smooth`
  are the source of truth; the CLI flags are syntactic sugar.
- ❌ No VRAM preflight gate — still Phase 4's concern.
- ❌ No bidirectional smoother — still one-sided EMA (v1 design).

---

## 6. Known limitations

- **DA V2 depth flip convention**: we emit `Z = 1 - (d - min)/(max - min)`
  so larger `Z` = nearer (matches Nuke PositionFromDepth). Shots where the
  raw model output interprets larger = nearer (rare) would get inverted.
  Users can inspect `Z_raw` to verify; a future `invert_depth: false` param
  can short-circuit the flip if needed.
- **DSINE intrinsics from reader**: we fall back to the 50mm heuristic when
  shot intrinsics aren't provided. Reader-side `CameraMetadata` propagation
  is a Phase 2b follow-up once the shot analyzer lands.
- **Transformers 5.x**: we installed `transformers>=4.45`; the resolver
  picked v5.5.4. No breaking API changes observed, but if DA V2 weights
  load under 5.x changes their pipeline surface we may need to pin.

---

## Phase 2 round 2 — shipped

Round 2 wires the VIDEO_CLIP crafters into real diffusers inference and
adds Apple's metric-depth backend. All three additions keep the fast-CI
property of round 1: tests bypass model downloads via `_load_model` /
`_infer_window` subclass overrides.

### New modules

- `shared/video_clip/sliding_window.py` — shared window-planning +
  overlap-stitching math, pure-numpy and side-effect free so DepthCrafter
  and NormalCrafter can be reasoned about without a diffusers stack.
  Three exports:
  - `plan_window_starts(n_frames, window, overlap)` — guaranteed coverage;
    last window backtracks so its right edge hits `n_frames` exactly (no
    drifting mismatch when the clip length isn't a multiple of stride).
  - `trapezoid_weight(window, overlap)` — flat=1 interior, linear ramp
    across the overlap, ramps sum to 1 for a clean crossfade.
  - `stitch_windowed_predictions(preds, starts, n_frames, overlap, *,
    endpoint_unramped=True)` — weighted-average stitch; the first window's
    left ramp and the last window's right ramp are replaced with 1.0 so
    the absolute clip endpoints are untouched.

- `passes/depth/depthcrafter.py` — replaces the license-gated stub with a
  real `DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id,
  trust_remote_code=True)` pipeline. VIDEO_CLIP mode, `window=110`,
  `overlap=25`, `inference_short_edge=640`, `precision="fp16"`.
  - `_infer_window` is split out from `infer` so tests inject a fake
    without reimplementing preprocess/postprocess plumbing.
  - `run_shot` reads all frames → `plan_window_starts` → per-window
    inference → `stitch_windowed_predictions` → per-clip min/max
    normalization (trap 5) → flip so larger Z = nearer.
  - Emits `depth_norm_min` / `depth_norm_max` artifacts (same contract as
    DA V2) so the executor stamps `depth/normalization/*` metadata
    uniformly.
  - `smoothable_channels = []` — VIDEO_CLIP output is already coherent,
    so the auto-smoother must not attach.

- `passes/normals/normalcrafter.py` — same pattern as DepthCrafter, but
  the output is a 3-channel (N.x, N.y, N.z) unit-normal field.
  - VIDEO_CLIP, `window=14`, `overlap=2`, `inference_short_edge=576`.
  - Postprocess upscales then renormalizes to unit length (trap 2,
    per-window pass).
  - `run_shot` renormalizes the whole clip *again* after stitching (the
    overlap blend breaks unit length), then converts OpenCV → OpenGL axes
    once (flip Y and Z) with the same helper DSINE uses.
  - `smoothable_channels = []`.

- `passes/depth/depthpro.py` — Apple Depth Pro via
  `transformers.AutoModelForDepthEstimation`. **Metric** depth in meters +
  `depth.confidence`. Apple ML Research License (non-commercial); gated
  behind `--allow-noncommercial`.
  - PER_FRAME, `inference_short_edge=1536`, `precision="fp16"`.
  - `Z` = metric meters; `Z_raw` = identical (schema parity with the
    relative backends where Z_raw is the un-normalized source).
  - `depth.confidence` — pulled from `outputs.confidence` /
    `outputs.depth_confidence` with a fall-back field of 1.0 when the
    backend doesn't emit one.
  - Emits `depth_metric` scalar artifact; executor reads it to stamp
    `depth/space=metric` + `depth/unit=meters`.
  - `smoothable_channels = [CH_Z]` — Z gets the temporal smoother when a
    flow pass is present; Z_raw and confidence are deliberately untouched
    so analysis tools see the un-smoothed signal and raw uncertainty.

### Executor changes

`executors/local.py :: _base_attrs` gained a branch for metric depth:

```python
if artifacts.get("depth_metric"):
    base["liveActionAOV/depth/space"] = "metric"
    base["liveActionAOV/depth/unit"]  = "meters"
elif artifacts.get("depth_norm_min") and artifacts.get("depth_norm_max"):
    base["liveActionAOV/depth/space"] = "relative"
    base["liveActionAOV/depth/unit"]  = "normalized_per_clip"
```

Metric wins if both are present (mixed-backend jobs are undefined, but
metric is the less-lossy signal so we prefer it).

### Registry

`pyproject.toml` adds the `depthpro` entry point and a
`live-action-aov[depthpro]` extra (empty today — DepthPro ships through
the `transformers` core dep; the extra exists for discoverability and
future-proofing if we swap in `apple-ml-depth-pro` as the backend).

### Tests added (round 2)

- `tests/test_shared/test_sliding_window.py` — 13 tests on the
  plan / weight / stitch helpers. Verifies endpoint pinning, monotonic
  crossfade in the overlap region, and shape/length validation.
- `tests/test_passes/test_depthcrafter.py` — 7 tests driving
  `_FakeDepthCrafter(_FakeReader)` across 30-frame clips. Asserts every
  frame is covered, per-clip normalization spans exactly [0, 1], Z_raw
  stays unflipped, and `smoothable_channels = []`.
- `tests/test_passes/test_normalcrafter.py` — 5 tests. Unit-length
  after stitch+renormalize, OpenCV→OpenGL axis flip, pass-through when
  `output_axes="opencv"`, `smoothable_channels = []`.
- `tests/test_passes/test_depthpro.py` — 6 tests. Metric depth (left
  edge ~2m, right edge ~20m), confidence clipped to [0,1], `depth_metric`
  artifact emitted, `smoothable_channels = [CH_Z]` only.
- `tests/test_cli/test_cli_phase2.py` — extended to assert `depthpro`
  appears in `liveaov plugins list` and is blocked by the gate without
  `--allow-noncommercial`.
- `tests/test_executors/test_metric_depth_metadata.py` — 1 end-to-end
  test with a `_FakeMetricDepthPass` registered via fixture. Runs through
  the CLI, reads the sidecar EXR back, asserts
  `liveActionAOV/depth/space=metric` + `liveActionAOV/depth/unit=meters`
  and that the relative-depth wiring did NOT also fire.

**Test totals: 92 passed (57 → 92, +35 for round 2).** Full suite under
3 seconds; no HF/torch.hub downloads; no GPU required.

### Round 2 known limitations

- **DepthCrafter / NormalCrafter weights**: we call `from_pretrained` with
  `custom_pipeline=model_id, trust_remote_code=True`. Upstream hub
  repository layouts can drift — if Tencent/Stable-X renames the custom
  pipeline module, the `_load_model` call breaks until we add a shim.
  The fake-model tests shield CI from this.
- **DepthPro confidence path**: `outputs.confidence` is the current HF
  attribute name; we also try `outputs.depth_confidence` as a fallback.
  If Apple's HF export settles on a different name, add it to the list
  in `DepthProPass.infer`.
- **Metric+relative mixing**: running DepthCrafter and DepthPro in the
  same job produces ambiguous metadata (metric wins per the branch above,
  but the Z channel contents from DepthCrafter are still normalized).
  A future "primary depth backend" flag could make this explicit; for
  now, don't do that.

## Phase 3 preview (spec §13.1)

- SAM 3 concept-based detection + video tracking.
- MatAnyone 2 (non-commercial) or RVM (commercial-safe) soft-alpha refiner.
- RGBA hero packing (`matte.r/g/b/a`) + dynamic semantic masks (`mask.<concept>`).
