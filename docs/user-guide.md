# User Guide

*Short on purpose: the README covers the first run; this page is the reference for the command line, the output folders and Nuke.*

## Installation

See the top-level `README.md` for installation via `install.sh` / `install.bat`.

## CLI

```bash
liveaov --help
liveaov --version
liveaov plugins list
liveaov run-shot <folder> --out <root> --passes noop   # smoke test
liveaov inspect <sidecar.exr>                 # structural QC of a sidecar
```

`liveaov run-shot --help` lists every option (passes, backends, output root, proxy, colourspace).
Part of a shot only: `--first 1010 --last 1060` (frames) and `--crop x,y,w,h` (one box of the
plate, pixels from the top-left). Studio file names: `--naming "{shot}_{pass}_{version}.{frame}.{ext}"`,
see [File naming](#file-naming).

### `liveaov inspect` — sidecar QC at the terminal

After `run-shot` writes its sidecars, `inspect` is the fast structural
check before you reach for Nuke. It answers: *did the pipeline produce
the right channels, metadata, and hero slots?*

```text
$ liveaov inspect out/plate/v001/passes/plate_v001.utility.0010.exr
File: out/plate/v001/passes/plate_v001.utility.0010.exr
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
   - **Default engine: Auto (recommended)** - ViTMatte + BiRefNet guide — SAM 3 finds the object, BiRefNet says where it is sure, ViTMatte resolves every unsure pixel (hair, motion blur, a sloppy SAM edge). Works on people and on objects BiRefNet does not understand (it is ignored there). MIT, commercial-safe.
   - **Green / blue screen** (Chroma key) is the second entry; the other engines (BiRefNet Portrait / Matting / General, RVM) are under **Advanced** in the same menu. Non-commercial models are marked **⚠ NC** (tooltip: NOT FOR PRODUCTION USE).
   - The **"SAM 3 detects"** field takes comma-separated concepts (e.g. `person, red car`) driving both the matte and the Cryptomatte.
   - **Whole + parts**: list a whole and its parts together (`person, hair, jacket, trousers`). `mask.person` stays the full body; `mask.hair`, `mask.jacket`, ... are cut from the person's matte, so they share its outer edge (hair strands included), add up to it exactly and never leave it. Parts are recognised by name (clothing, body and vehicle parts) when they sit inside a bigger object; a person inside a car stays a separate object.
   - Multi-word concepts become channel-safe names: `brick wall` -> `mask.brick_wall`.
   - **Small objects** (Auto / Max / Off) — SAM 3 sees about 1000 px, so on a 4K plate a 40 px person reaches it as 10 px and is missed. **Auto** (default) also detects on 2x2 zoomed tiles of 2K+ plates, **Max** on 3x3 (crowds, far people: finds ~90% of people 12-96 px tall), **Off** = full frame only. The tracker always follows the full frame, so a person walking across tiles keeps one ID. Small objects keep SAM's edge (anti-aliased, no edge refine) and each gets its own Cryptomatte ID; a click on a small subject is segmented in a zoomed window around the click. With a **proxy** set (Output tab), the search for objects still reads the original plate, so a 1K proxy finds the same small people as the full-res run while tracking and refining stay at proxy cost.
3. **Masks tab (interactive click-to-mask, optional)** — seed an object directly in the viewport:
   - **Drag a box** around it (often the strongest single prompt), and/or **left-click = include**, **right-click = exclude**.
   - **Preview mask (this frame)** runs SAM 3 on the seed frame so you can refine before the full run.
   - **Engine** (beside the object list) sets this object's own soft-edge engine; picking one re-previews the object with it straight away, and the submit uses the same engine. Default = the run engine from the Passes tab.
   - **Fewer inputs work better** — a box or 2–6 points beats dozens; SAM collapses under too many point constraints.
   - At submit, each object propagates across the shot into a `mask.<name>` channel + a named **Cryptomatte ID**.
4. **Output tab** — at the top, **this shot only**: the **frames** to process (in / out inside the plate, *Whole plate* resets) and a **crop**: press *Draw crop in viewer*, drag a box on the image. Passes run on that box only (faster, and a small subject fills the models' input); the EXRs keep the full plate as display window with the box as data window, so Nuke places them where they belong. The camera track and the clean plate need the whole frame and refuse to run with a crop.
   Below, for every shot: where the versions go (next to the plate, or one **root**; remembered between sessions), an optional proxy resolution for fast iterations, the EXR delivery, the **layout** (one multi-layer EXR, one Read in Nuke; or one EXR per pass in subfolders `passes/depth/`, `passes/matte/` ...) and the **file naming** template, see [File naming](#file-naming). Every run is a new version, see [Output folders](#output-folders).
5. **Submit local** — runs the pipeline; a log panel shows progress.
6. **Preview output** (button, or right-click a shot) — look at what the last run made without Nuke: pick a version and a layer (each mask, depth, normals, flow, the splat clean plate ...), scrub frames, show a mask over the plate. Camera and splat solves are summarised (frames solved, reprojection error, PSNR). Right-click also opens the output / plate folder and the comp in Nuke.

## Output folders

```text
<root>/<shot>/v001/
    <shot>_v001.comp.nk                       <- open this in Nuke
    passes/<shot>_v001.utility.####.exr      (per-pass layout: passes/<pass>/<shot>_v001.<pass>.####.exr)
    passes/<shot>_v001.crypto.####.exr       (lossy delivery only: Cryptomatte kept lossless)
    camera/<shot>_v001.camera.nk  .camera.json  .points.obj  .points.ply  _colmap/
    splat/<shot>_v001.splat.ply  .splat.json
    logs/submit_<date>_<shot>.log
<root>/<shot>/v002/ ...                      <- the next run; v001 is never touched
```

Old versions are never deleted by the tool: remove them by hand when you no longer need them. The CLI works the same way (`--out <root>` is required).

**Sessions** autosave continuously — after a crash, the next launch offers a one-click restore. Use **File → Save / Open session** (`*.laov.json`) for explicit project files; the whole prep round-trips (shots, colour, models, concepts, click points, output routing).

### File naming

Studios have their own conventions, so every file of a run is named from **one template**
(Output tab → File naming, or `--naming`). Tokens:

| Token | Value |
|---|---|
| `{shot}` | the shot name (the plate folder name unless renamed) |
| `{plate}` | the plate folder name |
| `{version}` | the run's version folder, `v001` |
| `{pass}` | what the file holds: `utility`, `crypto`, a pass group in the per-pass layout (`depth`, `matte` ...), `camera`, `points`, `splat`, `comp` |
| `{frame}` | the frame number, padded like the plate |
| `{ext}` | the extension |

Default `{shot}_{version}.{pass}.{frame}.{ext}` gives the tree above. `{shot}_{pass}_{version}.{frame}.{ext}`
gives `SH010_utility_v001.1001.exr`, `SH010_camera_v001.nk`, `SH010_comp_v001.nk`: files without a frame
drop `{frame}` and the separator before it. `{pass}`, `{frame}` and `{ext}` are required; the folders
(`passes/`, `camera/`, `splat/`, `v###`) do not change.

## Opening a run in Nuke — `<shot>_v###.comp.nk`

Every run writes `<shot>_v###.comp.nk` at the top of its version folder. Open it (or File → Import Script) and everything is already wired:

| Backdrop | What is in it |
|---|---|
| PLATE | the original plate Read |
| all passes / one per pass | the utility Read(s), `raw` (data, no colour conversion), and one labelled Shuffle2 per pass: every mask on its own, an "every mask" matte, depth / normals / position / flow to RGB, mattes and the splat clean plate to RGBA |
| camera | the solved Camera2, the point cloud (ReadGeo2 of `camera/<shot>_v###.points.obj`) in a Scene, and a ScanlineRender over the plate: drop Cards on the points to check the track |
| example | with a splat and masks: a Keymix that replaces every masked object with the splat clean plate (watch `splat.alpha`: areas the camera never saw are empty) |

Paths are relative to the script, so a whole version folder can be copied to another machine. In Nuke our `mask.*` layer is called `mask_extra.*` (Nuke reserves `mask`); the script already uses the Nuke names.

Nuke caps a script at **1023 channels in total** (every layer of every Read). One Read with 600 masks, 600 Shuffles and the every-mask matte loads and checks fine in Nuke 16; keep a run (plus whatever else is in your script) under ~1000 channels.

The Gaussian splat itself (`splat/<shot>_v###.splat.ply`) is for splat-capable tools; in Nuke 16 use the `splat` EXR layer (clean plate, coverage, depth), which is already rendered through the shot camera.

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
