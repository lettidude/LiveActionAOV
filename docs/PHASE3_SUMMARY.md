# Phase 3 — Matte Pass (rounds 1 + 2: SAM 3 + RVM + MatAnyone 2)

**Status:** ✅ Complete (both rounds) — 159 passed, 1 skipped (the `@slow
@gpu` real-weights integration) in ~3 s on CPU. Round 2 adds MatAnyone 2
as a non-commercial refiner behind the same contract as RVM; swap with
`--refiner matanyone2 --allow-noncommercial`. A pre-release GPU smoke
test runs the real SAM 3 + RVM combination end-to-end — **do not skip
this at a release cut** (brainstorm decision #7). A CorridorKey v2c
schema lock pins `sam3_hard_masks` exactly so any future drift breaks
CI before it breaks CorridorKey at runtime.

Round 2 deltas are summarised in §7 at the bottom; rounds 1 details
(§1–§6) remain accurate as written.

**Exit criterion (spec §13.1 Phase 3):**
> `liveaov run-shot <folder> --passes flow,matte` writes a sidecar with
> `mask.<concept>` channels per detected concept plus four `matte.r/g/b/a`
> soft-alpha hero channels. The per-clip RGBA slot lock holds (slot → track
> identity is fixed for the whole shot; frames where the instance is absent
> are zero'd). The SAM 3 + RVM default combination is commercial-safe and
> clears the license gate without `--allow-noncommercial`. SAM 3's raw
> per-instance hard-mask stacks are exposed as an artifact so future
> CorridorKey (v2c) can consume them without re-running the detector.

Both clauses met. The `matte` CLI alias expands to `sam3_matte + rvm_refiner`
in one invocation; `--refiner matanyone2` will flip to the NC backend in
round 2.

---

## 1. Scope cut for round 1

| Pass | License | Status |
|---|---|---|
| `sam3_matte` (detector + tracker) | SAM-License-1.0 (commercial w/ military carve-out) | **Full implementation** |
| `rvm_refiner` (soft-alpha packer) | MIT | **Full implementation** |
| `matanyone2` (higher-quality refiner) | NTU-S-Lab-1.0 (NC) | **Deferred** to round 2 |
| `rank.py` (per-clip slot assignment) | — | **Full implementation** |

Round 2 drops MatAnyone 2 in behind the same `requires_artifacts`
(`sam3_hard_masks`, `sam3_instances`) contract. No detector or ranker
changes needed.

---

## 2. Modules added

### `src/live_action_aov/passes/matte/rank.py`
Pure-Python ranker + slot assignment. No torch, no OIIO — imported by
both the detector pass (to build the canonical slotted list) and the
user-facing tools (to validate `heroes:` overrides).

| Export | Purpose |
|---|---|
| `SLOT_ORDER = ("r", "g", "b", "a")` | Canonical RGBA slot order |
| `RankWeights` (pydantic) | area=0.4, centrality=0.2, motion=0.2, duration=0.2, user_priority=0.0 |
| `Instance` (dataclass) | Pre-reduced scalars per tracked object |
| `HeroOverride` (frozen dataclass) | `(track_id, slot)` force-assignment |
| `HeroSlot` (dataclass) | Ranked + slotted hero (refiner consumes this) |
| `score_instance(inst, weights, n_clip_frames)` | Weighted sum |
| `rank_and_assign(...)` | Full pipeline: score → overrides → fill → sort by slot |

**Deterministic**: ties broken by (higher area, lower track_id).
**Overrides first**: user-forced slots claim before scored fill.
**Returns in slot order**: refiner indexes directly by slot.

### `src/live_action_aov/passes/matte/sam3.py`
`SAM3MattePass` — concept-based detection + video tracking.

| Field | Value |
|---|---|
| `name` | `sam3_matte` |
| `pass_type` | `SEMANTIC` |
| `temporal_mode` | `VIDEO_CLIP` |
| `license` | `SAM-License-1.0`, commercial OK (military/ITAR carve-out in notes) |
| `produces_channels` | `[]` — **dynamic**, one `mask.<concept>` per detected concept |
| `provides_artifacts` | `sam3_hard_masks`, `sam3_instances`, `matte_concepts` |
| Backend | HF `AutoModel.from_pretrained("facebook/sam3", trust_remote_code=True)` |
| Default params | 7 concepts (person/vehicle/tree/building/sky/water/animal), `confidence=0.4`, `min_area=0.005`, `sample_frame="middle"`, `max_heroes=4` |

**Seed-and-track (v1 default)**: one forward pass through the detector on
the middle frame builds all seed masks; the tracker then propagates each
across the whole clip. Power users can enable per-window re-detection via
`redetect_stride=N`.

**Per-clip slot lock (brainstorm decision #3)**: the ranker runs once over
the whole clip. `matte.r` refers to the same track for the entire shot;
if the track leaves screen, the refiner writes zeros for those frames,
but the slot stays reserved.

**Union by concept (brainstorm decision #6)**: `mask.person` is the OR
of all `person` instances per frame. The per-instance raw masks live in
the `sam3_hard_masks` artifact for the refiner (and future CorridorKey
/ Cryptomatte consumers). No `mask.` channel is emitted for a concept
that produces zero pixels anywhere in the clip — no stray `mask.sky`
when there's no sky.

**Motion-energy feature**: if a flow pass ran earlier, the refiner
reads `forward_flow` from `ingest_artifacts` and reduces
`mean(|flow| over mask) / plate_diag` per instance for the ranker.
Without flow, the motion term is zero (weights still sum to a
meaningful score).

### `src/live_action_aov/passes/matte/rvm.py`
`RVMRefinerPass` — Robust Video Matting soft-alpha refiner + RGBA packer.

| Field | Value |
|---|---|
| `name` | `rvm_refiner` |
| `pass_type` | `SEMANTIC` |
| `temporal_mode` | `VIDEO_CLIP` |
| `license` | `MIT`, commercial OK |
| `produces_channels` | `matte.r`, `matte.g`, `matte.b`, `matte.a` |
| `requires_artifacts` | `sam3_hard_masks`, `sam3_instances` (hard DAG dep) |
| `provides_artifacts` | `matte_heroes` (for executor metadata) |
| Backend | `torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3", trust_repo=True)` |

**Internal loop over heroes (brainstorm decision #5)**: the refiner loops
over the ranker's hero list, refines each instance's hard-mask stack into
a soft alpha (T, H, W), and packs into `matte.{slot}` where `slot` is
the ranker-assigned r/g/b/a. **Packing is deterministic and lives inside
the refiner** — no separate packing pass.

**Frames where the instance is absent get zeros**: the refiner's
recurrent memory can bleed alpha into bordering frames, but the per-clip
slot lock says "if the instance isn't present, the slot is 0 at that
frame". The pass zeros out hallucinated alpha post-inference. The
`test_instance_absent_frames_zeroed` test pins this behavior.

**`matte_heroes` artifact**: published for the executor to stamp
`liveActionAOV/matte/hero_{r,g,b,a}/{label,track_id,score}` onto every
sidecar, plus the commercial flag (derived from this pass's license).
QC pipelines can use `liveActionAOV/matte/commercial = "true"|"false"`
programmatically before delivering to clients.

### `src/live_action_aov/passes/matte/__init__.py`
Package docstring explaining the four-module layout (sam3 / rvm /
matanyone2 / rank).

### `src/live_action_aov/core/pass_base.py` — `ingest_artifacts` hook
New lifecycle method on `UtilityPass`:

```python
def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
    """Receive the full artifact dict produced by upstream passes.
    Called by the executor *once per shot*, before `run_shot`, iff
    this pass declares any `requires_artifacts`. Default is a no-op."""
```

Non-breaking: flow/depth/normals don't override it, so nothing changes
for them. The RVM refiner overrides it to stash SAM 3's per-track hard
masks and hero list. `emit_artifacts` return type also relaxed from
`dict[str, dict[int, np.ndarray]]` to `dict[str, dict[int, Any]]` since
the matte pipeline emits richer structures (lists of hero dicts,
per-track mask stacks with labels).

### `src/live_action_aov/executors/local.py` — matte metadata + ingest call
Two additions:

**Pre-`run_shot` ingest call** for passes declaring `requires_artifacts`:
```python
if getattr(cls, "requires_artifacts", None):
    instance.ingest_artifacts(artifacts)
frame_outputs = instance.run_shot(reader, shot.frame_range)
```

**Matte metadata block** in `_base_attrs`:
- `liveActionAOV/matte/concepts = "person,vehicle,..."` (from `matte_concepts`)
- `liveActionAOV/matte/hero_r/label|track_id|score` for every slot
  covered by a hero (from `matte_heroes`)

Per-sidecar:
- `liveActionAOV/matte/detector = "sam3_matte"`
- `liveActionAOV/matte/refiner  = "rvm_refiner"`
- `liveActionAOV/matte/commercial = "true"` (derived from the refiner's
  license — always a string, not a bool, so Nuke reads it as a scalar)

The detector + refiner are discovered by walking the DAG for passes that
provide `sam3_hard_masks` / `matte_heroes` respectively — keeps the
executor agnostic to which backend is wired in.

### `src/live_action_aov/cli/app.py` — `matte` compound alias
`_SEMANTIC_ALIASES["matte"]` is **compound**: it expands to two passes
(`matte_detector + refiner`) rather than renaming one. `_resolve_semantic_passes`
grew keyword-only args `matte_detector` and `refiner`. Two new options
on `run-shot`:

- `--matte-detector` (default `sam3_matte`)
- `--refiner` (default `rvm_refiner`; `matanyone2` lands round 2)

Users who prefer explicit form can still pass `--passes sam3_matte,rvm_refiner`
directly. The DAG re-sorts if the order is wrong.

### `pyproject.toml`

New entry points:
```toml
sam3_matte  = "live_action_aov.passes.matte.sam3:SAM3MattePass"
rvm_refiner = "live_action_aov.passes.matte.rvm:RVMRefinerPass"
```

New optional extras:
- `sam3 = []` — reserved for future sam3-specific deps
- `rvm = []` — RVM loads via `torch.hub`, torch is already core
- `matte = ["live-action-aov[sam3]", "live-action-aov[rvm]"]` — one-shot install

### `src/live_action_aov/io/channels.py` (no changes this round)
The Phase 2 `MASK_PREFIX = "mask."` + `is_mask_channel()` utility is
used as-is. `MATTE_CHANNELS = (matte.r, matte.g, matte.b, matte.a)` was
already in `CANONICAL_CHANNEL_ORDER` — the writer orders them first,
then appends the dynamic `mask.<concept>` channels in insertion order.

---

## 3. Tests added (39 new, all passing)

| File | Count | Covers |
|---|---|---|
| `tests/test_passes/test_matte_rank.py` | 12 | Scoring math, top-N selection, override-first slot claim, tie-breaking (area → track_id), deduction when fewer instances than slots, `max_heroes > slots` rejected, determinism, duration + user_priority contributions |
| `tests/test_passes/test_sam3_matte.py` | 9 | License w/ carve-out note, seed-frame keyword/int resolution, `mask.<concept>` emitted for detected concepts only, area-floor suppression, multi-instance union, `sam3_hard_masks` + `sam3_instances` + `matte_concepts` artifact shape, flow ingest as soft dep, user override routes to slot |
| `tests/test_passes/test_rvm_refiner.py` | 9 | MIT license + gate off, `requires_artifacts` declared, all four matte channels emitted, slot→channel mapping (r/g unused b/a), instance-absent-frame zeroing (the per-clip slot lock), `matte_heroes` artifact for metadata, missing hard-mask recorded not crashed, `smoothable_channels=[]` |
| `tests/test_io/test_dynamic_mask_channels.py` | 2 | Writer ordering places `matte.{r,g,b,a}` in canonical positions then appends `mask.<concept>` in insertion order; full round-trip with 4 dynamic mask channels + 4 matte channels + matte-metadata attrs |
| `tests/test_executors/test_matte_metadata.py` | 1 | End-to-end CLI run with fake detector + fake refiner; sidecar has `matte/detector`, `matte/refiner`, `matte/commercial="true"`, `matte/concepts`, `matte/hero_r/{label,track_id,score}` |
| `tests/test_cli/test_cli_phase3.py` | 6 | `matte` alias expands to detector+refiner, respects `--matte-detector`/`--refiner`, dedup with explicit passes, plugins-list visibility, both backends marked commercial-safe |

Phase 2 `test_cli_phase2.py` updated to use keyword args for the now-keyword-only
`_resolve_semantic_passes`.

**Test totals: 131 passed (92 → 131, +39 for Phase 3 round 1).** Full
suite under 3 seconds on CPU. No HuggingFace / torch.hub downloads; no
GPU required. Fake-model test pattern (subclass-override
`_load_model` / `_detect_seed` / `_track_instance` / `_refine_instance`)
keeps the real-model integration path separate and marked `@slow @gpu`
for the pre-release smoke test.

---

## 4. By-hand exit-criterion verification

```bash
# Happy path — commercial-safe default combination.
uv run liveaov run-shot A:/tmp/plate3 --passes flow,matte
# → per-frame sidecar EXRs with:
#     - mask.person, mask.vehicle, ... (one per detected concept)
#     - matte.r, matte.g, matte.b, matte.a (top-4 heroes, per-clip slot lock)
#     - 9 flow channels
# → `liveActionAOV/matte/detector = "sam3_matte"`
# → `liveActionAOV/matte/refiner = "rvm_refiner"`
# → `liveActionAOV/matte/commercial = "true"`
# → `liveActionAOV/matte/concepts = "person,vehicle,..."`
# → `liveActionAOV/matte/hero_r/label = "person"` etc.

# Explicit detector + refiner (equivalent).
uv run liveaov run-shot A:/tmp/plate3 --passes flow,sam3_matte,rvm_refiner

# Swap refiner for the NC variant (round 2 lands matanyone2).
uv run liveaov run-shot A:/tmp/plate3 --passes matte --refiner matanyone2 --allow-noncommercial
```

### OIIO-level channel verification (oiiotool --info equivalent)

Programmatic round-trip (the `test_dynamic_mask_channels` test) confirms
what `oiiotool --info` would show:

```
File           : matte_sidecar.exr
Resolution     : 48 x 32
# channels     : 8
Channel names  :
  - mask.animal        ← dynamic layer, grouped by OIIO
  - mask.person
  - mask.sky
  - mask.vehicle
  - matte.r            ← canonical RGBA, stays in order
  - matte.g
  - matte.b
  - matte.a
Matte metadata :
  - liveActionAOV/matte/commercial = true
  - liveActionAOV/matte/concepts = person,vehicle,animal,sky
  - liveActionAOV/matte/detector = sam3_matte
  - liveActionAOV/matte/hero_r/label = person
  - liveActionAOV/matte/hero_r/score = 0.91
  - liveActionAOV/matte/hero_r/track_id = 17
  - liveActionAOV/matte/refiner = rvm_refiner
```

Nuke reads this as two layers: `mask` (with 4 sub-channels) and `matte`
(with RGBA). Shuffle/ChannelMerge nodes auto-populate the pickers.

### Plugins list

```
$ liveaov plugins list
 name              | type      | license           | commercial
-------------------+-----------+-------------------+-----------
 depth_anything_v2 | geometric | Apache-2.0        | yes
 depthcrafter      | geometric | Apache-2.0+SVD-NC | no
 depthpro          | geometric | Apple-ML-Research | no
 dsine             | geometric | MIT               | yes
 flow              | motion    | BSD-3-Clause      | yes
 normalcrafter     | geometric | Apache-2.0+SVD-NC | no
 rvm_refiner       | semantic  | MIT               | yes
 sam3_matte        | semantic  | SAM-License-1.0   | yes
```

---

## 5. Non-goals respected

- ❌ No MatAnyone 2 wiring — round 2 lands the NC high-quality refiner.
- ❌ No Cryptomatte output — the `sam3_hard_masks` artifact ships in the
  shape Cryptomatte / CorridorKey (v2c) will consume, but we don't
  write a Cryptomatte EXR in round 1.
- ❌ No visual user selection GUI — `heroes: [{track_id, slot}]`
  overrides go through YAML or CLI params; the Qt comper that lets
  users click on people in the plate is Phase 4b work.
- ❌ No confidence channel on the hero mattes — RVM doesn't emit one,
  and a synthesized "confidence" from `max(soft_alpha_stack)` would be
  misleading.
- ❌ No per-frame re-ranking — brainstorm decision #3 locks slots at
  the clip level; per-frame identity flips are exactly the UX bug we
  avoid by paying this cost.

---

## 6. Known limitations

- **SAM 3 HuggingFace availability**: we call
  `AutoModel.from_pretrained("facebook/sam3", trust_remote_code=True)`.
  If the upstream repo renames or rearranges the custom pipeline
  module, `_load_model` breaks until we add a shim. The fake-model
  tests shield CI.
- **RVM via torch.hub**: first run downloads weights from the
  `PeterL1n/RobustVideoMatting` GitHub repo via `torch.hub.load`. In
  airgapped environments, users must set `TORCH_HOME` and pre-populate
  the cache.
- **Overlap of detected instances**: when the same pixel is inside
  two `person` instances' masks, the union `mask.person` caps at 1.0
  via `np.maximum`. The per-instance hard masks (in `sam3_hard_masks`)
  preserve the distinction — downstream Cryptomatte can recover the
  full per-instance boundary.
- **Concept list vs model capability**: we pass
  `["person", "vehicle", "tree", ...]` as concept prompts. SAM 3's
  vocabulary is wide but not unlimited; concepts the model doesn't
  recognize simply produce zero detections (and hence no `mask.*`
  channel for that concept — the "no stray `mask.sky`" guarantee).
- **Slot reshuffling across shots**: the per-clip slot lock holds
  *within* a shot. Two shots from the same sequence may assign the
  same character to different slots if the area/centrality ordering
  changes. Use the `heroes:` override to pin characters across shots
  when that matters (standard VFX workflow).

---

## Phase 3 preview — round 2

- MatAnyone 2 (NTU-S-Lab-1.0, non-commercial) as `--refiner matanyone2`,
  behind the license gate. Same `requires_artifacts` contract so SAM 3
  doesn't change.
- `@slow @gpu` integration test with real SAM 3 + RVM weights on a
  small synthetic clip. Never skip this one at a release cut
  (brainstorm decision #7).
- CorridorKey v2c plumbing audit — confirm `sam3_hard_masks` arrives
  in the shape CorridorKey expects; document any gaps before CorridorKey
  lands.

---

## 7. Round 2 addendum — MatAnyone 2 + GPU smoke + CorridorKey lock

### 7.1 Modules added

**`src/live_action_aov/passes/matte/matanyone2.py`** — `MatAnyone2RefinerPass`

| Field | Value |
|---|---|
| `name` | `matanyone2` |
| `license` | `NTU-S-Lab-1.0`, **commercial = False** (notes explain NC-only) |
| `pass_type` / `temporal_mode` | `SEMANTIC` / `VIDEO_CLIP` |
| `produces_channels` | `matte.r`, `matte.g`, `matte.b`, `matte.a` (identical to RVM) |
| `requires_artifacts` | `sam3_hard_masks`, `sam3_instances` (identical to RVM) |
| `provides_artifacts` | `matte_heroes` (identical to RVM — executor-level refiner discovery is contract-agnostic) |
| `smoothable_channels` | `[]` (memory-based / recurrent; auto-smoother must not attach) |
| Backend | `torch.hub.load("pq-yang/MatAnyone", "matanyone2", trust_repo=True)` |
| Default params | `inference_short_edge=720`, `warmup_frames=5`, `memory_every=5` |

**Contract parity with RVM** is the central invariant: the executor
walks the DAG for `provides_artifacts = ["matte_heroes"]` to find the
refiner, so swapping RVM ↔ MatAnyone 2 requires zero executor code.
`test_output_bitwise_matches_rvm_for_same_fake_refine` pins this by
running both passes with an identical fake `_refine_instance` and
asserting channel-stack equality frame-by-frame.

The `_SLOT_TO_CHANNEL` map is **deliberately duplicated** in both
`rvm.py` and `matanyone2.py` rather than hoisted. When future backends
diverge on slot ordering (e.g. a refiner that prefers BGRA), we want
local control rather than a shared constant to refactor.

### 7.2 CorridorKey v2c schema lock

CorridorKey (spec §21.8, Phase 6) will consume `sam3_hard_masks`
directly. That consumer doesn't exist yet, so the contract between
SAM 3 and CorridorKey is a paper contract — `tests/test_passes/
test_sam3_hard_masks_schema.py` (14 tests) turns the paper into code.

The lock pins:

| Aspect | Pinned value |
|---|---|
| Outer wrapper | `dict[int, dict]` with exactly one key (shot-level) |
| Inner keys | Plain `int` track_ids (not `np.int64` — JSON/pickle portability) |
| Per-track schema | Exactly `{"label": str, "frames": list[int], "stack": np.ndarray}` — any extra field is a schema break |
| `frames` | Sorted ascending list of plain ints, **absolute** frame indices (not 0-based local) |
| `stack` | `float32`, 3-D, `shape[0] == len(frames)`, values in `[0, 1]` |
| Plate alignment | All track stacks share the same (H, W) — no per-track resampling |
| Track-id uniqueness | Collisions would race CorridorKey's per-track corridor state |
| Empty-shot semantics | Zero detections → `emit_artifacts() == {}` (not a malformed skeleton with empty inner dicts) |

Sibling `sam3_instances` lock (same file): refiner consumer contract for
RVM + MatAnyone 2 — hero dicts must carry `{track_id, slot, label,
score, frames}`, slots are RGBA singletons, one hero per slot, and each
hero's `frames` must be a subset of its track's hard-mask frames.

When CorridorKey lands and needs a new field (e.g. `bbox`), update
these tests **first**, then sam3.py — not the other way round.

### 7.3 CLI surface additions (incremental)

No structural change from round 1; just the `matanyone2` refiner slotted
in behind the existing `--refiner` flag:

```bash
# Commercial-safe default (unchanged).
uv run liveaov run-shot A:/tmp/plate3 --passes flow,matte

# NC refiner — license gate blocks without opt-in.
uv run liveaov run-shot A:/tmp/plate3 --passes matte --refiner matanyone2
# exit 2, error names matanyone2

# Opt in — gate clears, runs on GPU.
uv run liveaov run-shot A:/tmp/plate3 --passes matte --refiner matanyone2 \
    --allow-noncommercial
# `liveActionAOV/matte/commercial = "false"` stamped on every sidecar
```

Added in `pyproject.toml`:
```toml
matanyone2 = "live_action_aov.passes.matte.matanyone2:MatAnyone2RefinerPass"
# …
matanyone2 = []  # torch.hub pull, reserved extra for future pinning
```

### 7.4 Real-weights GPU smoke test

`tests/test_passes/test_matte_integration_slow.py` — marked `@slow @gpu
@integration`, deselected by default. **Do not skip this at a release
cut** (brainstorm decision #7): fake-model unit tests shield the
contract but cannot catch upstream API drift in HF's SAM 3 loader or
torch.hub's RVM loader.

What it does:
1. Generates 10 frames @ 256×256 of a moving white disk on grey.
2. Runs `liveaov run-shot --passes flow,matte` (commercial-safe default).
3. Reads back one sidecar, asserts *structural* invariants only:
   - ≥1 dynamic `mask.<concept>` channel
   - all four `matte.{r,g,b,a}` channels
   - `liveActionAOV/matte/commercial = "true"`
   - detector + refiner identity stamped
   - matte values in `[0, 1]`, at least one non-zero

Runtime: ~30–120 s on a modern GPU; ~1–2 GB of weights downloaded on
first run. Opt-out: `LIVEAOV_SKIP_MATTE_INTEGRATION=1` (CI lanes
without GPUs); release cuts MUST NOT set this.

Not covered: MatAnyone 2 real-weights — running the NC refiner here
would muddy the invariant *"default pipeline is commercial-safe
end-to-end."* A dedicated MatAnyone integration test can be added later
behind its own marker if/when a commercial mirror ships.

### 7.5 Tests added (round 2)

| File | Count | Covers |
|---|---|---|
| `tests/test_passes/test_matanyone2_refiner.py` | 9 | NC license gate, DAG contract parity with RVM (exact set of requires/provides), all four matte channels, `smoothable_channels==[]`, RGBA-every-frame emission, **bitwise-identical output to RVM given same fake refine**, instance-absent-frame zeroing with a hallucinating refiner, matte_heroes artifact for metadata, missing hard-mask recorded not crashed |
| `tests/test_passes/test_sam3_hard_masks_schema.py` | 14 | CorridorKey v2c contract lock — outer wrapper shape, plain-int track_ids, exact per-track key set, label/frames/stack types, sorted ascending absolute frame indices, float32 3-D alignment, [0,1] range, plate-shape consistency, uniqueness, sibling sam3_instances refiner contract (5 tests), empty-shot returns `{}` |
| `tests/test_passes/test_matte_integration_slow.py` | 1 (skipped off-GPU) | End-to-end `flow,matte` CLI run with real SAM 3 + RVM weights; structural invariants only |
| `tests/test_executors/test_matte_metadata.py` (extended) | +1 | `_FakeNCRefiner` (NTU-S-Lab-1.0) subclass → `matte/commercial = "false"` stamped when NC refiner runs with `--allow-noncommercial` |
| `tests/test_cli/test_cli_phase3.py` (extended) | +4 | `matanyone2` listed in plugins with `NTU` license and commercial=no; `matte` alias respects `--refiner matanyone2`; NC gate **blocks** without `--allow-noncommercial` (exit 2, error names matanyone2); NC gate **clears** with opt-in (exit != 2) |

**Round 2 totals: +28 new tests (9 matanyone2 + 14 schema lock + 1
integration + 1 metadata + 4 CLI − 1 skipped off-GPU).**

**Cumulative Phase 3 totals: 131 (round 1) + 28 (round 2) = 159 passed,
1 skipped.** Full suite under 3 s on CPU. No HF / torch.hub downloads
except the opt-in integration test.

### 7.6 Non-goals respected (round 2)

- ❌ No MatAnyone 2 real-weights test — the NC license would force
  `--allow-noncommercial` in the integration test, contradicting the
  "default pipeline is commercial-safe end-to-end" invariant.
- ❌ No auto-smoother on MatAnyone 2 output — the net is already
  recurrent, stacking temporal smoothing produces mush.
- ❌ No per-shot model swap — one refiner per job. If a sequence needs
  mixed commercial/NC refiners across shots, run two jobs.
- ❌ No CorridorKey integration — the schema lock test **pins the
  contract**, but CorridorKey itself is Phase 6.
- ❌ No parity tests for bit-for-bit output between real RVM and real
  MatAnyone 2 — they're different networks, of course they differ. The
  parity test uses fake identical refine hooks to isolate the
  **plumbing** contract.

### 7.7 Known limitations (round 2 additions)

- **MatAnyone 2 via torch.hub**: first run pulls weights from
  `pq-yang/MatAnyone`. No HF Hub mirror exists. Airgapped installs
  need `TORCH_HOME` pre-populated; no HF fallback.
- **Commercial flag is *refiner-derived***: if a user swaps in a
  commercial detector + NC refiner, the sidecar reads `commercial =
  "false"` even though the detector alone would have been fine. This
  is intentional — the delivered pixels carry the refiner's license,
  not the detector's. Documented in the `test_sidecar_commercial_flag_
  is_false_for_noncommercial_refiner` docstring.
- **Schema lock is enforced only through the fake path**: the schema
  tests verify that the *code that produces* `sam3_hard_masks` emits
  the pinned shape. They don't audit that a real `facebook/sam3`
  upstream hasn't silently restructured its output (that's the real-
  weights integration test's job, indirectly — CorridorKey will fail
  loudly if the upstream drifts *and* we shipped without catching it).

---

## Phase 4 preview

Round 2 closes the matte arc. Next up in Phase 4:

- **Qt comper / hero picker GUI** — click on a person in the plate to
  force them into `matte.r`. `heroes: [{track_id, slot}]` overrides
  already plumb through from CLI/YAML; the GUI just fills in the list
  visually.
- **Cryptomatte output** — consume the same `sam3_hard_masks` artifact
  the CorridorKey lock pins, emit per-instance IDs as a Cryptomatte
  EXR alongside the union `mask.*` channels.
- **MatAnyone 3 (pending upstream release)** — when/if a commercial
  mirror ships, slot it in behind the same contract (drop the NC gate,
  flip the extra name). The contract-parity test is the only thing
  that needs running.
