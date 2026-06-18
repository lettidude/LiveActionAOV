# User Guide

*This guide is a stub for Phase 0. It will be expanded in Phase 6.*

## Installation

See the top-level `README.md` for installation via `install.sh` / `install.bat`.

## CLI

```bash
liveaov --help
liveaov --version
liveaov plugins list
liveaov run-shot <folder> --passes noop       # Phase 0 smoke test
liveaov inspect <sidecar.exr>                 # structural QC of a sidecar
```

More commands land in Phase 4 (`discover`, `analyze`, `run`, `preflight`,
`models`, etc.).

### `liveaov inspect` — sidecar QC at the terminal

After `run-shot` writes its sidecars, `inspect` is the fast structural
check before you reach for Nuke. It answers: *did the pipeline produce
the right channels, metadata, and hero slots?*

```text
$ liveaov inspect plate/plate.utility.0010.exr
File: plate/plate.utility.0010.exr
Resolution: 1920 x 1080
Channels (11):
  canonical:
    motion.x              [-12.340, 11.070]  mean=0.034
    motion.y              [ -8.910,  9.220]  mean=-0.012
    flow.confidence       [  0.000,  1.000]  mean=0.873
  matte:
    matte.r               [  0.000,  1.000]  mean=0.142
    matte.g               [  0.000,  0.000]  mean=0.000
    matte.b               [  0.000,  0.000]  mean=0.000
    matte.a               [  0.000,  0.000]  mean=0.000
  mask:
    mask.person           [  0.000,  1.000]  mean=0.148
    mask.vehicle          [  0.000,  0.980]  mean=0.031

liveaov metadata:
  matte/commercial       = "true"
  matte/detector         = "sam3_matte"
  matte/refiner          = "rvm_refiner"
  matte/concepts         = "person,vehicle"
  matte/hero_r/label     = "person"
  matte/hero_r/track_id  = 3
  matte/hero_r/score     = 0.91

Heroes (1 of 4 slots filled):
  matte.r = person (track 3, score 0.91)
  matte.g = (empty)
  matte.b = (empty)
  matte.a = (empty)
```

- Matte/mask channels outside `[0, 1]` (± tolerance) are flagged `⚠`,
  so regressions jump out visually.
- Pass `--json` for a machine-parseable document with the pinned key
  set `{file, resolution, channels, metadata, heroes}`. Useful for
  batch stress runs that compare many sidecars at once.
- Pixel-level QC (edge quality, temporal flicker) still belongs in
  Nuke — `inspect` is the plumbing check that comes before that.

## GUI

```bash
liveaov-gui
```

**Workflow:**

1. **Load plates** — drag plate folders onto the shot list (or File → Add shot). Each shot auto-detects its colourspace and seeds exposure.
2. **Passes tab** — pick a model per category. For matte, choose a backend:
   - **SAM3 + RVM** — hard masks + RVM soft alpha (default, commercial-safe).
   - **SAM3 + BiRefNet (soft edges)** — roto-grade soft alpha (hair, motion blur, fine edges) in `matte.*`. MIT, commercial-safe.
   - The **"SAM 3 detects"** field takes comma-separated concepts (e.g. `person, red car`) driving both the matte and the Cryptomatte.
3. **Masks tab (interactive click-to-mask, optional)** — seed an object directly in the viewport:
   - **Drag a box** around it (often the strongest single prompt), and/or **left-click = include**, **right-click = exclude**.
   - **Preview mask (this frame)** runs SAM 3 on the seed frame so you can refine before the full run.
   - **Fewer inputs work better** — a box or 2–6 points beats dozens; SAM collapses under too many point constraints.
   - At submit, each object propagates across the shot into a `mask.<name>` channel + a named **Cryptomatte ID**.
4. **Output tab** — choose where sidecars land (next-to-plate / subfolder / external root) and an optional proxy resolution for fast iterations.
5. **Submit local** — runs the pipeline; a log panel shows progress.

**Sessions** autosave continuously — after a crash, the next launch offers a one-click restore. Use **File → Save / Open session** (`*.laov.json`) for explicit project files; the whole prep round-trips (shots, colour, models, concepts, click points, output routing).

## Nuke plugin — UtilityRelight

A companion Nuke node ships with this repo at
`src/live_action_aov/plugins/nuke/UtilityRelight/`. It consumes a
sidecar EXR and a beauty plate and gives the comp artist live 3D
light placement on the subject — six layered light contributions
(key, spec, rim, bounce, glow, fog) computed on the GPU via
BlinkScript.

### Install

1. Copy these two files into your `~/.nuke/` directory (directly,
   NOT in subfolders — Nuke loads `~/.nuke/*.py` at startup):
   - `src/live_action_aov/plugins/nuke/UtilityRelight/utility_relight.py`
   - `src/live_action_aov/plugins/nuke/UtilityRelight/UtilityRelightKernel.blink`

2. Add to your `~/.nuke/menu.py` (create the file if it doesn't
   exist):
   ```python
   import utility_relight
   utility_relight.register()
   ```

3. Restart Nuke. The node appears at
   **Nodes → UtilityPasses → UtilityRelight**.

### Quick test

1. Create the node. Connect plate to input `src`, sidecar EXR to
   input `aov`.
2. Output tab → set **View Mode → 3D Scene Preview**.
3. Press `Tab` in the viewer to enter 3D mode.
4. Drag the **LightAxis** gizmo arrows — the point cloud relights
   in real time.
5. Output tab → switch back to **2D Relight** for the final
   render-layer output. Merge (`plus`) it onto your plate.

Tested on Nuke 16.0. Likely works on 14–15. See the plugin's own
[README](../src/live_action_aov/plugins/nuke/UtilityRelight/README.md)
for the full feature reference, conventions, and Nuke-version
compatibility notes.
