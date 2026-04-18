# Phase 1 — Flow Pass (RAFT) + Temporal Smoother

**Status:** ✅ Complete — awaiting confirmation to begin Phase 2.

**Exit criterion (spec §13.1, Phase 1):**
> `uv run liveaov run-shot <folder> --passes flow` writes a sidecar EXR with
> `motion.x`, `motion.y`, `back.x`, `back.y`, `flow.confidence` channels,
> channel values in pixels at plate resolution, valid `liveActionAOV/flow/*`
> metadata including `parallax_estimate`.

Both met. 34 tests pass in ~2 s (including the CPU-friendly RAFT smoke test).

---

## 1. Modules added

### `src/live_action_aov/passes/flow/raft.py`
`RAFTPass` — first real AI pass. Backend = `torchvision.models.optical_flow.raft_large` (BSD-3, weights ship with torchvision so no Hugging Face dependency). Key properties:

| Field | Value |
|---|---|
| `name` | `flow` (CLI: `--passes flow`) |
| `pass_type` | `MOTION` |
| `temporal_mode` | `PAIR` |
| `input_colorspace` | `srgb_display` (Phase 1 uses a cheap linear→gamma 2.2 shim; v2 wires proper `DisplayTransform`) |
| `license` | `BSD-3-Clause`, commercial use OK |
| Produces channels | `motion.x`, `motion.y`, `back.x`, `back.y`, `flow.confidence` |
| Provides artifacts | `forward_flow`, `backward_flow`, `occlusion_mask`, `parallax_estimate` |
| Default params | `backend=raft_large`, `precision=fp32`, `fb_threshold_px=1.0`, `inference_resolution=520`, `num_flow_updates=12` |

**Trap handling:**
- **Trap 1 (flow vector scaling)** — `_upscale_flow_to_plate` bilinear-upscales flow then multiplies x/y components by `plate_w/inf_w` and `plate_h/inf_h`.
- Inference target is clamped to ≥128 px per side (RAFT's correlation pyramid needs feature maps ≥16 on both axes after the internal 8× downsample).

**F-B consistency** — `_fb_consistency` uses `grid_sample` to sample backward flow at `p + fwd[p]`, computes the residual magnitude, and turns it into a soft confidence via `exp(-err² / threshold²)`.

**`run_shot` override** — iterates consecutive pairs `(f, f+1)` once each, assigns forward flow to frame `f` and backward flow to frame `f+1`, populates endpoint frames with zeros where no pair exists. Two RAFT calls per pair (forward + backward direction), no re-work.

**`parallax_estimate`** — per-shot scalar: median |forward flow| across all pairs, normalized by plate width. Used in v2a by depth/normals passes for backend routing.

### `src/live_action_aov/shared/optical_flow/cache.py`
`FlowCache` — in-memory `{(shot_id, frame, direction): (2, H, W) float32}`. Read by the smoother; written by the executor after `RAFTPass.emit_artifacts()` returns. v2 can swap the in-memory dict for an on-disk spill without touching callers.

### `src/live_action_aov/post/temporal_smooth.py`
`TemporalSmoother` — post-processor (NOT a pass; registered under `live_action_aov.post`). Algorithm (design §9.1):

```
For each frame t (in order):
    warped = backward_warp(channel[t-1], bwd[t])
    occlusion = fb_error(fwd[t-1], bwd[t], threshold_px)
    weight = (1 - occlusion) * alpha
    smoothed[t] = weight * warped + (1 - weight) * channel[t]
```

Defaults: `alpha=0.4`, `fb_threshold_px=1.0`. Empty `applied_to` is a no-op.

### `src/live_action_aov/core/pass_base.py` — extensions
Two new concrete methods on `UtilityPass` (non-breaking):
- `run_shot(reader, frame_range) → {frame: {channel: arr}}` — default is per-frame preprocess/infer/postprocess (covers NoOp). PAIR/VIDEO_CLIP passes override.
- `emit_artifacts() → {name: {frame: arr}}` — default empty. RAFT uses this to expose flow tensors to the executor.

### `src/live_action_aov/core/job.py`
New `PostConfig` + `Job.post: list[PostConfig]` for configuring post-processors from YAML.

### `src/live_action_aov/executors/local.py` — rewritten
- Calls `pass.run_shot()` instead of owning the frame iteration loop (passes now control their own temporal semantics).
- Collects `pass.emit_artifacts()` per pass; mirrors `forward_flow`/`backward_flow` into `FlowCache` keyed by `(shot.name, frame, direction)`.
- Resolves and applies each `Job.post` entry after all passes complete.
- Writes `liveActionAOV/flow/parallax_estimate`, `liveActionAOV/flow/direction`, `liveActionAOV/flow/unit`, and `liveActionAOV/smooth/<name>/*` metadata when present.

### `pyproject.toml`
Entry points registered:
```toml
[project.entry-points."live_action_aov.passes"]
flow = "live_action_aov.passes.flow.raft:RAFTPass"

[project.entry-points."live_action_aov.post"]
temporal_smooth = "live_action_aov.post.temporal_smooth:TemporalSmoother"
```

---

## 2. Tests added (9 new, all passing)

| File | Covers |
|---|---|
| `tests/test_passes/test_flow_raft.py` | 2 — `RAFTPass` single-pair smoke (motion magnitude > 1 px where subject moves, confidence ∈ [0,1]) + `run_shot` + artifacts |
| `tests/test_post/test_temporal_smooth.py` | 4 — zero-flow identity EMA, occluded pixels keep raw, first frame untouched, empty `applied_to` is no-op |
| `tests/test_shared/test_flow_cache.py` | 4 — put/get, missing→None, shape validation, per-shot `clear` |

CPU-friendly: tests use `raft_small`, `inference_resolution=128`, `num_flow_updates=6`. Total suite runs in ~2 s on CPU.

## 3. By-hand exit-criterion run

```bash
uv run liveaov run-shot A:/tmp/plate2 --passes flow
```

Output sidecar inspection:
```
channel names: ['back.x', 'back.y', 'flow.confidence', 'motion.x', 'motion.y']
shape: (192, 256, 5)
motion.x range: -67.5 ... 123.0  (pixels at plate res — ✓)
flow.confidence range: 0.00 ... 0.9999
liveActionAOV/flow/parallax_estimate = 0.295
liveActionAOV/flow/direction = bidirectional
liveActionAOV/flow/unit = pixels_at_plate_res
liveActionAOV/flow/license = BSD-3-Clause
```

(OpenEXR sorts stored channels alphabetically in the file; Nuke/DJV group them back into `motion`/`back`/`flow` layers at read time.)

---

## 4. Non-goals respected

- ❌ No sRGB display transform wiring — RAFT uses a cheap gamma 2.2 shim. Proper `DisplayTransform` integration is a Phase 2 concern.
- ❌ No VRAM preflight gate — `core/vram.py` exists from Phase 0 but isn't wired into the executor yet.
- ❌ No YAML config file in the CLI — `run-shot` still only takes `--passes <csv>`; YAML round-trip is tested separately.
- ❌ No v2+ extensions to the smoother (bidirectional blending, per-pixel alpha from confidence).

---

## Phase 2 preview (not started)

Depth + Normals passes:
- `passes/depth/depth_anything_v2.py` — `DepthAnythingV2Pass` (Apache-2.0, commercial OK). PER_FRAME. Per-clip depth normalization policy (trap 5).
- `passes/depth/depthcrafter.py` — `DepthCrafterPass` (NON-commercial; license gate blocks without `--allow-noncommercial`). VIDEO_CLIP.
- `passes/normals/dsine.py` — `DSINEPass` with intrinsics scaling (trap 3) and renormalization after resize (trap 2).
- CLI `--depth-backend` / `--normals-backend` options so users pick the commercial/non-commercial backend explicitly.
- Wire `TemporalSmoother` into the default CLI path for PER_FRAME passes (depth + normals).
