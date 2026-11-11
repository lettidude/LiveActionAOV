# Changelog

All notable changes to LiveActionAOV are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **Steady a pass over time (opt-in).** Any pass can be stabilised with the run's own motion
  vectors: each frame is combined with its neighbours (window of 3, 5 or 7 frames) after warping
  them onto it, with a **median** (removes one-frame pops), a **mean** (softer passes) or a
  centre-weighted **blend**. Neighbours that the motion does not explain are left out, so moving
  subjects do not smear. Normals stay unit length, depth keeps its scale, mattes stay 0-1. CLI
  `run-shot --stabilise PASS[=MODE[:WINDOW]]` (e.g. `--stabilise normals=blend:5`), pass params
  `stabilise` / `stabilise_window`. Needs the Motion pass in the run. Off by default: runs without
  it are unchanged. Not in the app yet.
- **Camera as USD.** Each camera solve also writes `camera/<shot>_v###.camera.usda` with the camera
  and a coloured point cloud, for Houdini, Maya, Blender, Unreal and Nuke's USD 3D system.
- **A point cloud you can read.** With a depth pass in the run, the point cloud in the Nuke script
  is now a dense, coloured cloud of the set instead of a few grey dots, and it shows in a render.
  It is written as `camera/<shot>_v###.cloud.obj` + `.cloud.png` and goes into the USD too.
- **Process part of a shot.** Per shot, a **frame range** and a **crop** box drawn in the viewer
  (*This shot* tab → *Draw in viewer*; CLI `--first/--last`, `--crop x,y,w,h`). The EXRs line up
  over the plate in Nuke. The camera track and the Gaussian splat need the whole frame and say so.
- **File naming template** (`Shot.naming`, Output tab → *File naming*, CLI `--naming`): one template
  for every file of a run with `{shot} {plate} {version} {pass} {frame} {ext}`, with presets for the
  common studio orders. The default naming is unchanged.
- **Long clips use much less memory.** A clip too long for the machine's RAM is processed in
  overlapping windows of frames by itself (`Shot.window_frames`: `auto`, a number or off;
  `window_overlap`; CLI `--window`). Passes that need the whole clip (the SAM tracker, the camera,
  the splat, video models) still run on it once. The frames written are the same as a whole-clip run.
  Plugin authors: new pass attributes `window_safe` / `window_crossfade` and a `probe_clip` hook.
- **`liveaov doctor`**, and the same line in the app's Output tab: what is installed and the settings
  proposed for a plate (proxy size, detector, windows), with the reason for each. **Use suggested**
  applies them. `run-shot --proxy` takes a long edge in pixels or `auto`. New proxy size 2816.
- **`liveaov setup-sam31`** installs the SAM 3.1 environment in a per-user folder that the matte pass
  finds by itself; a rerun skips the steps already done.
- **Masks tab rebuilt.** *Found by name* (the names SAM looks for, with People / People + vehicles /
  Everything presets) and *Your own objects* (clicks and boxes). The selected object has its name,
  its soft-edge engine, a **matte slot** to pin it to `matte.r/g/b/a` (prompt spec key `slot`) and
  *Go to*, which jumps to the frame it was drawn on.
- **A *This shot* tab** for what belongs to one shot (frames, crop, folder name); the Output tab is
  now for the whole session, so a batch needs no visit to every shot.
- **LTX Alpha crop** (per-object engine in the Masks tab, `crop_labels`): inside a whole-frame run,
  chosen objects get their own LTX run on a rectangle around them, for a subject that needs more
  detail.

### Changed
- **Every camera now gets a GOOD / WEAK / FAILED verdict**, the COLMAP one and the rotation-only one
  included, like the fallback camera already had: it shows in the GUI, in the Nuke camera's label and
  in `liveaov qc`, and a camera that only rotates is flagged NODAL. The Gaussian splat skips a NODAL
  camera.
- **Masks tab: the preview is back in front.** *Preview* sits under the object buttons and, with
  *Auto* on (the default), every click or box previews by itself. *with* picks what the preview is
  made with. The names preset in use stays lit. The pass is listed as *Matte (SAM + soft-edge engine)*.
- **ComfyUI not running is said in plain words** instead of a raw socket error.
- **The Gaussian splat output is called *splat background*, no longer *clean plate*,** in the app,
  the docs and the Nuke script (the example node is now `SplatFill_example`, was
  `CleanPlate_example`). The `splat.*` channels and the files are unchanged.
- **LTX Alpha runs on the whole frame by default** (`mode: whole_frame`, was `per_object`). In the app
  the default matte engine is *LTX Alpha whole frame in your ComfyUI, else ViTMatte*.
- **Lossless EXRs are written with ZIP one scanline at a time** (`zips`, was `zip`): same size, much
  faster to write. Saved sessions move over by themselves; `zip` can still be picked by name.
- **The detector is `auto`** (was `sam3`): SAM 3.1 when it is installed and the card can run it,
  else SAM 3, and the log says which. If SAM 3.1 fails on a shot in `auto`, the shot is redone with
  SAM 3 with a warning; asking for `sam3.1` by name still fails loudly.
- With a lossy delivery the lossless Cryptomatte file is now `<shot>_v001.crypto.####.exr`
  (`{pass}` = `crypto`) instead of `<shot>_v001.utility.####.crypto.exr`, so it follows the naming
  template. The comp script points at the new name.

### Fixed
- **Gaussian splat: one frame could come out washed out** while the others were fine. Fixed; the
  report gains `holdout_psnr_min_db` and `polish_steps`, and the log warns about a weak frame. Areas
  the camera never saw behind the masked objects are still a guess: check `splat.A`.
- **Model licences that were declared wrong.** **DSINE** is non-commercial (Imperial College
  London licence), not MIT: it now asks for `--allow-noncommercial` and is out of the default
  download set (MoGe-3 is the normals model for paid work). **Robust Video Matting** is GPL-3.0,
  not MIT: its mattes can be delivered, its code must not ship inside a closed bundle.
  **Depth Anything V2 Base** and **Video Depth Anything Base / Large** are CC-BY-NC-4.0: the passes
  now load the Apache-2.0 Small models only. The README has the full table.
- **An object LTX does not see as foreground no longer gets an empty matte** (a car, a prop): it falls
  back to ViTMatte or SAM's edge, with a warning.
- **Scenery no longer takes a hero slot.** The `matte.rgba` slots go to subjects first (people,
  animals, vehicles, your clicks); sky, buildings, trees and the like get a slot only when there are
  fewer subjects than slots (`scenery_labels`). Scenery keeps its `mask.<label>` and Cryptomatte id.
- **LTX per-object crops no longer grow to the whole frame** on plates with a lot of tracked scenery,
  which gave each object somebody else's matte.

## [0.12.0] — 2026-11-08

### Added
- **Camera from depth when the solve fails** (`pnp_fallback`, on by default). Fast handheld shots
  that gave no camera now get one. Each camera is graded **GOOD / WEAK / FAILED** in the warnings and
  the Nuke camera label; a FAILED camera is labelled "DO NOT USE" (`pnp_keep_failed: false` drops
  it) and the splat refuses it. `pnp_fov_x_deg` sets the field of view by hand.
- **Temp folder setting** (`--work-root`, `LAOV_WORK_ROOT`, GUI File > Temp folder). Frames exported
  from the plate go there instead of the system temp folder. Keep it on a local disk.
- **LTX Alpha whole-frame mode** (`mode: whole_frame`): one LTX run for the whole frame instead of one
  per object, so edges SAM missed stay in the matte and many objects cost one run. Foreground that
  belongs to no tracked object goes to a new `foreground.a` layer.
- **LTX call cache** (`cache_dir`): a rerun of the same shot does not repeat the LTX work.
- **SAM 3.1 detector for the matte pass** (`detector: sam3.1`): keeps similar objects apart better
  than SAM 3. It runs in its own environment (`sam31_python` param or `LAOV_SAM31_PYTHON`; setup in
  docs/install.md).
- **GUI: SAM 3.1 detector, whole-frame LTX engine, camera from depth.** All three are in the Passes
  tab, saved in the session and copied by "Apply passes to all shots".

### Fixed
- **A crash in COLMAP no longer kills the whole run.** The camera solve runs on its own
  (`colmap_subprocess`, on by default; `colmap_timeout_s` optional): a crash is reported as a camera
  failure and the rest of the run carries on. Its progress shows in the log live.
- **Temporary folders with plate frames are always cleaned up**, also after a failed solve. A folder
  kept for debugging (`keep_workdir`, Marigold `keep_temp`) is now a warning with its path.
- **Nuke's Cryptomatte node crashed on our files.** The provenance keys of the Cryptomatte pass are
  now under `liveaov/cryptomatte_pass/...`. Existing files are fixed by dropping the
  `liveaov/cryptomatte/{model,version,license,commercial}` header keys (no pixel change).
- **LTX Alpha found nothing on a ComfyUI with its own folder layout.** The model files are now found
  in other folders, in other precisions and as GGUF, and the graph uses the names the server reports.
- **A failed shot stopped an overnight batch.** Shots ticked while a batch runs are now added to it,
  and failure dialogs no longer block the queue.
- **A network-share blip no longer aborts the job**: sidecar writes retry long enough for the share
  to come back.
- **LTX Alpha no longer leaves mattes in ComfyUI's `output/`.** With a remote ComfyUI the uploaded
  crops stay in its `input/` (ComfyUI cannot delete them): a warning says so.
- **The camera solve no longer runs a long shot out of RAM** (`sift_num_threads: 0` = auto). The
  log states the solve size and thread count.
- **`max_image_size` above the proxy size now applies**: the camera reads the full-res plate when the
  proxy is smaller.

## [0.11.1] — 2026-11-02

### Added
- **LTX Alpha on a smaller or a remote GPU.** GGUF versions of the LTX models are found on the
  ComfyUI server, so a 12-16 GB card only needs a quantised download. A ComfyUI on another machine
  works too (`comfy_url`; `send_frames`: auto / path / upload). Models > ComfyUI for LTX Alpha shows
  which files it found.

### Fixed
- **MoGe-3 normals next to another depth backend**: MoGe-3's smoothing no longer touches the other
  pass's depth.

## [0.11.0] — 2026-11-01

### Added
- **Portable offline build** (`scripts/build_portable.py`) for workstations with no internet (TPN):
  one folder with Python, dependencies and models, installed or updated per artist with
  `install_local.bat`. Runs offline by default. See docs/install.md.
- **Nodal camera for shots with no parallax.** Tripod pans, tilts and locked-off shots now get a
  camera (rotation only, camera at the origin), with the focal from the pan, the lens metadata or a
  guess, with a warning. The splat skips a nodal camera with a clear message.
- **MoGe-3 geometry pass (`moge3`), the new default normals engine** (MIT). One pass writes metric
  `Z`, `N` and `P`, steady over the shot. GUI: "MoGe-3" in both Depth and Normals. CLI:
  `--normals-backend` defaults to `moge3`. New `moge3` extra (in `all`). The camera record notes
  where a nodal focal came from (`intrinsics.source`).

### Fixed
- `liveaov prefetch` skipped MoGe-3; passes can now expose a download-only `prefetch_weights` hook.
- `--version` / About reported v0.9.0 since the 0.10 releases.
- **UtilityRelight 1.12: works with metric depth.** With MoGe-3 the light radius came out huge and the
  2D picker could put the light behind the camera. Fixed.
- **DSINE normals were inverted.** They now follow the spec (OpenGL: +Y up, +Z toward camera) like
  the other normals engines.

## [0.10.1] — 2026-10-31

### Added
- **LTX Alpha engine (per object).** Pick "LTX Alpha - generative, hair" as an object's engine in the
  Masks tab: the matte is made by LTX-2.5 + the alpha-gen IC-LoRA in YOUR ComfyUI. Big wins on motion
  blur, hair strands and splashes. Models > ComfyUI for LTX Alpha sets the URL and tests the server,
  nodes and model files; the Masks tab shows whether it is ready. If ComfyUI is down or a model is
  missing, the object falls back to ViTMatte with a warning. LTX-2 Community License, weights not
  bundled.

## [0.10.0] — 2026-10-31

### Changed
- **License: MIT -> Business Source License 1.1** from 30 October 2026.
  Free for individuals, freelancers, studios under 10 people and
  non-commercial use; larger studios buy a one-off license per major version
  (COMMERCIAL.md). Each version becomes Apache-2.0 after four years. Code
  published before the change stays MIT; vendored third-party code keeps its
  own license.
- **A camera solve that fails no longer kills the run.** The camera (and the splat) are skipped, the
  rest is written, and the log ends with "Run finished WITHOUT camera_pycolmap: <reason>" (also in
  the shot's notes). A solve that covers too few frames counts as a failure
  (`min_registered_fraction`). The splat warns when the masked objects cover a large part of the
  plate.
- **Camera solves no longer fail at random**: the same plate now gives the same camera
  (`mapping_attempts`).
- **Output next to the plate is back, as the default.** Each shot's versions go to
  `<plate folder>_utility/v###/`, so the plate folder stays clean. "One root for every shot" is the
  other option (Output tab). CLI: `--out` is optional again. Old sessions keep their root.
- **The front object keeps its alpha.** Overlaps are held out front to back like a comp: a hand on a
  car stays solid in the person's matte.
- **A part as big as its whole keeps its mask** (only an arm in frame: `mask.hand` is the full hand).
- **One engine menu.** Run engine (Passes tab) and object engine (Masks tab) are the same menu:
  recommended first, Green / blue screen second, the rest under Advanced. Picking an object's engine
  pins it and re-previews it in one step. Short licence badges (✓ / ⚠ NC, full licence in the
  tooltip).
- **Output folders.** Nothing is written inside the plate folder, and every run is a new version:
  `<shot folder>/v001/` with `<shot>_v001.comp.nk` on top, EXRs in `passes/`, `camera/`, `splat/`
  (was `camera_track/`) and `logs/`. Files are named after the shot. A custom shot-folder name stays
  per shot.
- **Crowds use far less RAM.** Plugin authors: the `sam3_hard_masks` `stack` is now a lazy (T, H, W)
  uint8 stack (`np.asarray(stack)` / `stack[k]` still give dense arrays, `stack.box(k)` the crop).
- Small objects in all-masks mode get SAM's anti-aliased mask instead of a soft-edge refine
  (`refine_min_px`); hero slots are always refined.
- **New default matte engine: "Auto - ViTMatte + BiRefNet guide".** Better hair and strands where
  SAM's mask is blobby. The previous behaviour: `trimap="band"`.
- **Parts are cut from the whole's matte** (`parts_from_whole`, default on): `mask.hair`,
  `mask.jacket` ... share the person's outer edge and add up to `mask.person`.
- **Front-to-back order from depth** (`layer_order="depth"`, default) instead of object size, so a
  shelf behind a person no longer cuts into her hair.
- A refined edge fades out at a neighbour instead of ending on a straight line.

### Added
- **Preview output** (GUI button + right-click on a shot): every layer of any version, frame by
  frame, masks over the plate, camera / splat summary, without opening Nuke. Right-click also opens
  the output folder, the plate folder and the comp in Nuke.
- **Small objects / crowds** (`small_objects`, GUI "Small objects: Auto / Max / Off", default Auto):
  small and distant people in big plates are now found, also in proxy mode (`min_object_px`).
- **Crowds on ordinary GPUs**: big groups fit in much less VRAM (`tracker_max_objects`). Every pass
  logs its peak VRAM.
- Crowds: a person standing next to another is a new track, not part of them, and the gap between
  two close people is no longer filled with shards.
- **Whole + parts roto**: concepts like `person, hair, jacket, trousers` give `mask.person` = the full
  body and parts that split it and stay inside it (`part_labels` to add your own). A person inside a
  car stays a separate object. `sam3_hard_masks` entries gain `parent`.
- **BiRefNet decides the outline** (Passes tab checkbox; `outline="refiner"`): BiRefNet draws the
  whole silhouette and SAM only stops leaks, with a fallback when BiRefNet loses the subject.

### Fixed
- A person's hair came out at half opacity when SAM also found an unrelated "hair" elsewhere in frame.
- ACEScg plates were detected as lin_rec709. Chromaticities now come first, and OIIO's default tag is
  reported as a guess.
- Adding a shot reset the Output settings and the colourspace override; a new shot now inherits them
  from the session.
- Hard chips of SAM's shape no longer appear on soft hair edges next to a touching object.
- BiRefNet squashed tall crops; the crop keeps its aspect now (`keep_aspect`, on by default).
- Multi-word concepts (`brick wall`) now write a channel Nuke can read: `mask.brick_wall`.
- ViTMatte crawled on shots with many objects. Fixed.

## [0.9.0] — 2026-10-28

### Added
- **Camera track** ("Camera" category; plugin `camera_pycolmap`, pycolmap, BSD-3-Clause, CPU, no
  weights). Writes a Nuke Camera2 (`<shot>.camera.nk`), a camera record (`.camera.json`), a sparse
  point cloud (`.points.ply`) and the COLMAP model that Gaussian-splat trainers read. With Matte on in
  the same job, the tracked objects are masked out of the solve (`mask_labels` to restrict).
  Checked in Nuke 16 (`scripts/nuke_check_camera.py`). Limits: arbitrary scale, one focal per clip,
  needs parallax; the default lens model is `SIMPLE_PINHOLE`.
- **Gaussian splat** ("Camera" category; plugin `splat_gsplat`, gsplat Apache-2.0): rebuilds the
  static scene from the shot's own camera and re-renders every frame -> EXR layer `splat`
  (`splat.R/G/B` without the masked objects, `splat.A` coverage, `splat.Z` depth) + a `.splat.ply` in
  the same world as the Nuke camera. gsplat needs a wheel built for the tool's torch: `wheels/` +
  `scripts/build_gsplat_wheel.bat`; install/update scripts re-install `wheels/*.whl`. See
  docs/gaussian_splat.md.
- **Nuke comp starter script** `<shot>.comp.nk` on every run: the plate, the utility EXRs, one
  labelled Shuffle2 per pass in backdrops, the camera with its point cloud in a Scene +
  ScanlineRender and, with a splat and masks, a Keymix that swaps the masked objects for the splat.
  Paths are relative, so the folder can be moved. In Nuke, `mask.*` shows up as `mask_extra.*`.
- **Output layout option** (GUI Output tab, `Shot.delivery_layout`): one multi-layer EXR (default) or
  **one EXR per pass in its own subfolder** (`depth/`, `matte/`, `splat/`, `flow/` ...).
- The camera track also writes `<shot>.points.obj`, which Nuke 16's ReadGeo2 loads as a point cloud.
- Plugin authors: **optional artifacts** (`optional_artifacts`) and an `emit_sidecars()` hook for
  non-EXR outputs.
- **MegaFlow motion backend** ("MegaFlow (large displacement, slower)" in the Motion category; plugin
  `flow_megaflow`, Apache-2.0). Same channels as RAFT. RAFT stays the default; pick MegaFlow for fast
  action, retimes and textureless plates.
- **Marigold V2 bridge passes** (Apache-2.0): `marigold_v2_albedo` (Albedo category) and
  `marigold_v2_depth_seethrough` (Depth category, depth through glass and water). The model runs in
  its own environment (~70 GB setup). Experimental.

### Changed
- **Roto (SAM 3): one owner per pixel, and objects that enter later are found.** Objects are tracked
  together, so `mask.a + mask.b` add up instead of overlapping; re-detection (`redetect_stride`)
  picks up objects that enter mid-shot; a click on an object the detector also found becomes one
  track. Old behaviour: `joint_tracking=False, exclusive_masks=False, redetect_stride=None`.
- **Seams between touching objects are layered like a comp** (new post `matte_layering`, automatic
  with Matte): no more dark line between a person and the car behind. Hair over a neighbour keeps its
  soft edge (`neighbor_exclude_erode: auto`).

### Fixed
- The executor no longer fails when a frame has no EXR channels.

## [0.8.0] — 2026-08-13

### Added
- **Screen Pull keyer** ("Key" category, Passes tab): whole-plate chroma key, `key.rgba` = despilled
  plate premultiplied by the alpha. No models, no weights, MIT. Auto green/blue detection, works on
  long clips, `key.rgb` stays unclamped scene-linear.
- **Chroma Key per-object engine**: SAM decides WHO, the key does the edge. Warns when the clicked
  object looks like the screen itself.
- **Per-object engines**: every mask layer can pin its own soft-edge engine (ViTMatte / BiRefNet /
  Chroma Key / RVM) in one run. The list shows each object's engine.
- **One Matte switch** (Passes tab): Matte Off/On + a "Default engine" dropdown. Old sessions migrate
  automatically.
- **Viewport zoom + pan**: wheel zooms around the cursor, middle-drag pans, middle double-click fits.
- **Models menu**: "Download all models (prefetch)" and a checkable "Offline mode" for air-gapped
  (TPN) runs. Once weights are cached, no network requests.
- **Manual pixel-aspect override** (Colour tab) for anamorphic scans with a wrong header.
- **Zero-match concept warning**: a typed concept that finds nothing logs a loud warning naming the
  missing channel.

### Fixed
- **Edge ring on a clicked object** that a default concept also found. Clicks with an empty concepts
  field now submit clicks only.
- **Hero mattes came out black** when an object pinned its own engine.
- **Boiling edges and per-frame warning spam** on some tracks.
- **Halos too wide on thin objects** (a rifle no longer gets a car-sized halo).
- **Stray specks and pinholes** in SAM's masks are cleaned before refining, without filling car
  windows or arm gaps.
- **TPN compliance**: cached model loads no longer touch the network.
- Prefetch now covers ViTMatte + Video-Depth-Anything and warns when a pass's weights are missing
  from the list.

### Changed
- Masks tab: point placement is on with the tab; the preview shows soft edges as soft.
- launch/update scripts pin `--extra all` so a bare `uv run` cannot remove the refiner extras.

## [0.7.1] — 2026-07-14

### Fixed
- **Fast scrubbing no longer crashes the GUI** on Windows.
- **Anamorphic plates now display desqueezed.** Display only; click coordinates are unaffected.

## [0.7.0] — 2026-07-10

### Added
- **ViTMatte trimap-guided refiner** ("SAM3 + ViTMatte" in Passes). MIT code and weights. It only
  works on the edge, so the inside of a mask is never eroded. More hair detail than BiRefNet Portrait.
- **"Preview with" selector** (Masks tab): preview the seed-frame mask with any refiner engine,
  independently of the run choice, to compare per shot.
- **Compare mode** ("SAM3 + ALL refiners"): RVM, BiRefNet and ViTMatte each write their own layer
  (`matte_rvm.*` / `matte_birefnet.*` / `matte_vitmatte.*`, plus `mask_<engine>.<name>` in all-masks
  mode), to pick the best one per shot in Nuke. About 3x the refiner time.

### Fixed
- The "Refine ALL mask channels" toggle did not reach ViTMatte.

### Changed
- Passes tab: the BiRefNet weights dropdown is disabled when not used; the all-masks checkbox says
  what it does. The Masks tab notes which engine the run will use.

## [0.6.0] — 2026-07-07

### Added
- **Compact EXR delivery** (Output tab): per-shot codec + bit depth, with a "Compact - DWAB 16-bit"
  preset for moving files on limited bandwidth. **Cryptomatte splits by itself** into a lossless
  float32 `.crypto.exr` file when the chosen delivery would break its IDs. Defaults unchanged.
- **Soft edges on ALL masks** (Masks tab): every `mask.<name>` channel gets a soft alpha instead of
  SAM's hard edge. Off by default; one refinement per object.
- **Per-shot refiner model choice** (Masks tab), for the preview and the run: BiRefNet Portrait (new
  default, softer hair), BiRefNet Matting, BiRefNet General, RMBG-2.0 (paid licence for commercial
  use). Benchmark script: `scripts/benchmark_refiner_weights.py`.
- **Mask preview is always soft**, so motion-blurred hands and hair read correctly.

### Fixed
- **Refiners can no longer erode the inside of a mask.**
- **Refiner memory**: lower RAM use on long tracks.

### Changed
- GUI polish: refiner controls on the Masks tab, "Preview" tab renamed "Colour", Output tab marked
  session-wide.

## [0.5.2] — 2026-06-23

### Fixed
- **The SAM 3 matte no longer runs the machine out of RAM on long clips with many objects.** RAM use
  is much lower, and a check right after detection stops at once with a clear message (fewer
  objects, a lower proxy or a shorter range) instead of crashing late.
- **Cryptomatte is faster on long clips.**

### Notes
- Plugin authors: the `sam3_hard_masks` artifact's `stack` is now `uint8` (was `float32`); the
  bundled consumers already cast as needed.

## [0.5.1] — 2026-06-18

### Fixed
- **A network-share blip during sidecar writes no longer throws away a shot**: each EXR write retries
  before failing. Real faults (disk full, permission) still show.

### Docs
- New **Model cache** section: what is cached, where (`C:` by default), rough size, and how to move it
  with `HF_HOME` / `TORCH_HOME`.

## [0.5.0] — 2026-06-18

### Added
- **Offline operation + model prefetch.**
  - **`liveaov prefetch`** downloads every pass's models once, on the CPU, including the extra repos
    some passes need. `--passes a,b,c` for a subset, `--all` for every backend.
  - **`liveaov --offline ...`** never contacts the network, so an outage cannot abort a run.
    Workflow: `prefetch` once online, then run with `--offline`.
- Longer Hugging Face download timeouts, so cold downloads and brief blips do not fail.

### Changed
- The per-run log file name carries the shot name.

## [0.4.2] — 2026-06-18

### Fixed
- **`liveaov --help` crashed on a legacy Windows console or when piped.** CLI text is now ASCII.

### Changed
- `launch.bat` sets `PYTHONUTF8=1`.
- `install.bat` / `install.sh` pass `uv sync --python 3.11` explicitly.

## [0.4.1] — 2026-06-18

### Fixed
- **Install crash on machines with system Python 3.13/3.14** (false "CUDA not available", failed
  `liveaov --version`). Fixed with a `.python-version` (3.11) pin and `requires-python` capped to
  `>=3.11,<3.13`.

## [0.4.0] — 2026-06-18

### Added
- **BiRefNet soft-edge matte refiner**: soft alpha (hair, motion blur, fine edges) in the `matte.*`
  channels. MIT. Pick **"SAM3 + BiRefNet (soft edges)"** in the Passes tab. RVM stays the default.
- **Box-drag prompt** for click-to-mask: drag a rectangle around an object, often better than many
  clicks. Box and points can be combined.

### Notes
- New `[birefnet]` extra (`timm` / `einops` / `kornia`, MIT), included in `[all]`.

## [0.3.0] — 2026-06-13

### Added
- **Interactive click-to-mask** (Masks tab). Click an element in the viewport (left = include,
  right = exclude) and SAM 3 follows it through the shot into a `mask.<name>` channel **and a
  Cryptomatte ID** with the name you typed. Optional, and it works with the text concepts. Headless:
  `prompt_instances` param on `sam3_matte`.
- **Seed-frame mask preview.** "Preview mask (this frame)" shows the mask on one frame so you can
  adjust points before the full run.
- **Undo point**: remove the last point without clearing the set.
- **Session save/load + continuous autosave.** After a crash the next launch offers a one-click
  restore. `File → Save/Open session` (`*.laov.json`) keeps the whole prep. Shots whose plate folder
  is gone are skipped with a warning.

### Notes
- **Few clicks work best**: a few points, preview, one corrective click. The GUI warns when there are
  many points.
- Click-to-mask gives selection / ID-grade masks, not roto-grade alpha.

## [0.2.0] — 2026-06-09

### Added
- **Cryptomatte pass.** SAM 3 tracked objects as a standard Cryptomatte file, read by any
  Cryptomatte-capable compositor (Nuke / After Effects / Fusion / Resolve) like a CG pass. The masks
  are hard-edged: good for selections, holdouts and ID grades, not a roto replacement. The `feather`
  param softens edges.
- **SAM 3 concepts field** (Inspector → Passes): a comma-separated list of what to find, e.g.
  `person, red car, dog`, for Matte and Cryptomatte. Empty = the defaults (`person, vehicle, tree,
  building, sky, water, animal`). SAM 3 only finds what you name.

### Changed
- **Pinned `transformers` to `>=5.5.4,<6`**: other versions gave an empty matte.
- **OpenCV is a core dependency** (`opencv-python-headless`): fixes `No module named 'cv2'`.
- **`geffnet` is a core dependency**, so the default depth / normals / matte run out of the box.
- **Install docs state the tiers**: `uv tool install live-action-aov` = core;
  `uv tool install "live-action-aov[all]"` = every pass; `--torch-backend=auto` is needed on Windows.

### Fixed
- **NaN / Inf pixels in a plate no longer spoil the passes**: they are read as 0 with a warning.
- **A colourspace the OCIO config cannot decode now warns** instead of giving dark passes.
- **VRAM**: a multi-pass batch no longer runs out of GPU memory on the second shot.
- **Ultrawide plates** no longer crop in the viewport.
- **Inspector horizontal scrollbar** appears when needed.
- **Clear messages for missing extras**: selecting a pass whose extra is not installed gives the
  `pip install …[<extra>]` to run.

[Unreleased]: https://github.com/lettidude/LiveActionAOV/compare/v0.12.0...HEAD
[0.12.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.12.0
[0.11.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.11.1
[0.11.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.11.0
[0.10.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.10.1
[0.10.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.10.0
[0.9.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.9.0
[0.8.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.8.0
[0.7.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.7.1
[0.7.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.7.0
[0.6.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.6.0
[0.5.2]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.2
[0.5.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.1
[0.5.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.0
[0.4.2]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.2
[0.4.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.1
[0.4.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.0
[0.3.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.3.0
[0.2.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.2.0
