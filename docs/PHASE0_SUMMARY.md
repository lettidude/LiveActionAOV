# Phase 0 — Scaffolding & NoOp Pipeline

**Status:** ✅ Complete — awaiting confirmation to begin Phase 1.

**Exit criterion (from spec §13):**
> `liveaov run-shot <folder> --passes noop` writes a valid sidecar EXR with `liveaov/*` metadata, and `liveaov plugins list` returns `["noop"]`.

Both met. 19 tests pass, 3 skip cleanly when OIIO / installed entry points are unavailable.

---

## 1. Naming substitution applied

| Spec term | This project |
|---|---|
| `utility-passes` / `utility_passes` | `live-action-aov` / `live_action_aov` |
| CLI name | `liveaov` |
| Source tree | `src/live_action_aov/` |
| Entry-point group prefix | `live_action_aov.*` |
| EXR metadata namespace | `liveaov/` |
| Display name | LiveActionAOV |

---

## 2. Scaffolding delivered

- `pyproject.toml` — hatchling build, pinned Py 3.11+, six plugin entry-point groups (`live_action_aov.passes`, `.post`, `.executors`, `.io.readers`, `.io.writers`, `.integrations`), `liveaov` + `liveaov-gui` scripts, ruff/mypy/pytest config.
- `install.sh` / `install.bat` — uv bootstrap + `uv sync --extra dev`, with timestamped log files and remediation hints.
- `uninstall.sh` / `uninstall.bat` — venv removal + optional HF cache cleanup.
- `.github/workflows/ci.yml` — lint + mypy + pytest matrix on 3.11 / 3.12.
- `.github/workflows/release.yml` — `uv build` on tag (stub).
- `README.md`, `LICENSE` (MIT), `.gitignore`, `docs/architecture.md`, `docs/user-guide.md`, `docs/developing-plugins.md`.

---

## 3. Source modules

### core/
- `pass_base.py` — `License` (SPDX + commercial-use flags), `PassType` enum (GEOMETRIC / MOTION / SEMANTIC / RADIOMETRIC / CAMERA / SCENE_3D), `TemporalMode` enum (PER_FRAME / VIDEO_CLIP / SLIDING_WINDOW / PAIR), `ChannelSpec`, `SidecarSpec`, `DisplayTransformParams`, abstract `UtilityPass` with class-level declared contract + `preprocess → infer → postprocess` lifecycle.
- `job.py` — `Shot` (with pipeline-ID fields reserved for v2+), `Job` with `to_tasks()` chunking, `Task`, `PassConfig`. YAML round-trip via `to_yaml` / `from_yaml`.
- `registry.py` — `PluginRegistry` walking all six entry-point groups via `importlib.metadata.entry_points()`, runtime `register_pass()` for test injection, typed queries (`list_passes`, `list_by_type`, `get_pass`, `list_executors`, `list_writers`, …). Module-level `get_registry()` singleton.
- `dag.py` — `PassNode`, Kahn's algorithm topological sort (stable on input order), `DagCycleError`, `MissingArtifactError`.
- `vram.py` — `VramPlan`, `query_available_gb()` via `torch.cuda.mem_get_info` with CPU sentinel, `plan()` with configurable headroom.

### io/
- `channels.py` — canonical channel-name constants (`CH_Z`, `CH_N_X`, `CH_MOTION_X/Y`, `CH_BACK_X/Y`, `CH_MATTE_R/G/B/A`), `CANONICAL_CHANNEL_ORDER`, `is_mask_channel`.
- `oiio_io.py` — `read_exr` / `write_exr` wrappers with `HAS_OIIO` degradation flag, slash-namespaced attribute I/O.
- `ocio_color.py` — `to_linear` / `from_linear` via PyOpenColorIO when available, hardcoded sRGB↔linear fallback, `sniff_colorspace` from EXR chromaticity attributes.
- `resize.py` — `ResizeMode` enum, `ResizeParams`, `downscale` / `upscale` routed by channel type, `scale_intrinsics` (DSINE fix — design §8.4), pure-numpy bilinear/nearest kernels, `_renormalize_vectors` (trap 2), `_scale_flow_vectors` (trap 1).
- `display_transform.py` — `DisplayTransform` with `analyze_clip` (clip-wide EV — trap 4) and cheap `apply`; AgX polynomial sigmoid, Hable filmic, Reinhard; AP1 vs Rec.709 luma coefficients.
- `metadata.py` — `CameraMetadata` with `from_exr_attrs` multi-key fallback.
- `readers/base.py` + `readers/oiio_exr.py` — `ImageSequenceReader` ABC + `####` / `%04d` / literal pattern support.
- `writers/base.py` + `writers/exr.py` + `writers/json.py` — `SidecarWriter` ABC; EXR writer sorts channels against canonical order (trap 8); JSON writer handles numpy types.

### executors/
- `base.py` — `Executor` ABC.
- `local.py` — `LocalExecutor.submit`: resolve passes via registry → topo-sort → iterate frames → write one sidecar per pass-type per frame with `liveaov/*` metadata header.
- `deadline.py` — `DeadlineExecutorStub` raising `NotImplementedError` (non-goal, §1.3).

### integrations/
- `base.py` — `PipelineAdapter` ABC (`attach_ids`, `publish`).
- `standalone.py` — no-op adapter (v1 default).
- `prism.py`, `shotgrid.py`, `openpype.py` — stubs raising `NotImplementedError` (non-goal, §1.3).

### models/
- `registry.py` — `ModelRegistry` with refcount-based `register` / `get` / `release` / `unload`, `cache_dir()` via `platformdirs`.

### passes/
- `camera/stub.py` — `CameraPassStub` with `PassType.CAMERA`, `SidecarSpec`s for `.json` / `.nk` / `.abc` (forces non-EXR sidecar flow-through architecture even though the pass itself is a stub).

### cli/
- `app.py` — Typer app: `--version`, `plugins list [--type]` rich table, `run-shot <folder> --passes <csv> [--allow-noncommercial]` with license gate before any I/O, `_sniff_sequence` folder scan.

### gui/
- `app.py` — Phase 5 stub printing a pointer.

---

## 4. Tests

All 22 tests live under `tests/`. 19 pass, 3 skip cleanly.

| File | Covers |
|---|---|
| `conftest.py` | Session-autouse `NoOpPass` registration + `test_plate_1080p` fixture |
| `support/noop_pass.py` | Test-only pass emitting zero-valued `Z` |
| `fixtures/generate_test_plate.py` | 5-frame ACEScg EXR generator (gradient + moving Gaussian highlight) |
| `test_core/test_dag.py` | 4 — topo order, stable on input order, missing artifact, cycle detection |
| `test_core/test_job_roundtrip.py` | 2 — YAML round-trip + `to_tasks` chunking |
| `test_core/test_registry.py` | 4 — `noop` registered, `list_by_type`, entry-point discovery (2 skip if not `uv sync`'d) |
| `test_io/test_channels.py` | 2 — canonical order + `is_mask_channel` |
| `test_io/test_resize.py` | 4 — one per resize trap |
| `test_io/test_display_transform.py` | 3 — clip-wide EV, manual override, clamp |
| `test_io/test_oiio_roundtrip.py` | Skipif-OIIO; multi-channel + custom attrs + pixel aspect |
| `test_passes/test_noop_end_to_end.py` | Skipif-OIIO; full pipeline integration, `liveaov/*` header present |
| `test_cli/test_cli_phase0.py` | 3 — `--version`, `plugins list`, `run-shot` sidecar count |

---

## 5. All 8 known traps (§11.3) handled

| # | Trap | Where handled |
|---|---|---|
| 1 | Flow vectors must scale with resize | `io/resize.py::_scale_flow_vectors` |
| 2 | Normals must be renormalized after bilinear | `io/resize.py::_renormalize_vectors` |
| 3 | Camera intrinsics must scale with resize | `io/resize.py::scale_intrinsics` |
| 4 | Auto-exposure must be clip-wide, not per-frame | `io/display_transform.py::analyze_clip` |
| 5 | Depth normalization policy clip-wide | `DisplayTransformParams.depth_normalize_policy` |
| 6 | Hard masks use nearest, not bilinear | `resize.py` channel_type routing |
| 7 | Pixel aspect ratio preserved through writes | `io/oiio_io.py::write_exr` |
| 8 | EXR channel ordering canonical, not insertion-order | `io/writers/exr.py::_order_channels` |

---

## 6. Non-goals (§1.3) respected

- ❌ No Deadline submission logic — stub only.
- ❌ No Prism / ShotGrid / OpenPype integration logic — stubs only.
- ❌ No actual camera solve — `CameraPassStub` declares sidecar shape only.
- ❌ No GUI beyond a pointer message.
- ❌ No model downloads — `ModelRegistry` is plumbing only; no weights fetched in Phase 0.
- ❌ No AI inference — `NoOpPass` writes zeros.

---

## Phase 1 preview (not started)

RAFT optical-flow pass:
- `passes/flow/raft.py` — `RAFTPass` via `torchvision.models.optical_flow.raft_large`, bidirectional, `TemporalMode.PAIR`, produces `motion.x/y`, `back.x/y`, `flow.confidence`; declares `forward_flow` / `backward_flow` / `occlusion_mask` / `parallax_estimate` artifacts.
- `shared/optical_flow/cache.py` — `FlowCache`.
- `post/temporal_smooth.py` — flow-guided EMA warp-and-blend with forward/backward occlusion rejection (α=0.4, threshold=1.0 px).
