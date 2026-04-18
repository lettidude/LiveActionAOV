# Architecture

LiveActionAOV is a layered, plugin-first VFX plate preprocessor. This page is
a one-page orientation for contributors. Depth lives in the design notes at
the repo root (`utility_passes_design_notes.md`).

## Layers (high level)

- **`core/`** — pass ABC, job/shot models, DAG scheduler, plugin registry,
  VRAM planner. Foundational, no I/O.
- **`io/`** — EXR read/write (OIIO), OCIO colorspace transforms, pass-type-aware
  resize, display transform (auto-exposure → tonemap → EOTF), camera metadata
  extraction, reader/writer ABCs.
- **`shared/`** — intermediates produced once, reused by multiple passes
  (optical flow is the keystone).
- **`passes/`** — one folder per pass (depth, normals, flow, matte, camera
  stub).
- **`post/`** — flow-guided temporal smoother.
- **`models/`** — central model registry with lazy loading and reference
  counting.
- **`executors/`** — `LocalExecutor` is v1; `DeadlineExecutor` is a stub for
  v2.
- **`integrations/`** — pipeline adapters; `StandaloneAdapter` is v1;
  Prism / ShotGrid / OpenPype are stubs.
- **`cli/`** — Typer command surface.
- **`gui/`** — PySide6 prep GUI; thin consumer of core.

Dependency direction: `core` is foundational; `io`, `shared`, `models` depend
on `core`; `passes` depend on everything below; `cli` and `gui` are pure
consumers.

## Plugin system

Every pass is discovered via Python entry points in the
`live_action_aov.passes` group. Core has zero hardcoded pass imports. Built-in
and third-party passes are registered identically.

See `docs/developing-plugins.md` for how to write a pass.

## License matrix

Per-pass licenses are declared on each plugin's `License` object. The CLI
enforces `--allow-noncommercial` before running any pass with
`commercial_use=False`. The full model-license matrix lives in
`utility_passes_design_notes.md` §17.

## Design document

The full design record (rationale, tradeoffs, future roadmap v2/v3) is at
`../utility_passes_design_notes.md`. Consult it when a decision in the code
seems arbitrary — rationale is almost always there.
