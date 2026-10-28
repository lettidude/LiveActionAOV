# Changelog

All notable changes to LiveActionAOV are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses [semantic versioning](https://semver.org/).

## [Unreleased]

### Changed
- **New default matte engine: "Auto - ViTMatte + BiRefNet guide".** SAM 3
  now says only WHERE an object is, not where its edge is. A trimap is built
  per frame: sure-inside deep in SAM's mask (never deeper than half the
  object's own thickness, so poles and thin shelves keep a core), sure-
  outside far from it, and a wide unsure zone that ViTMatte resolves.
  BiRefNet Portrait narrows or widens that zone where it is confident, but
  only when it agrees with SAM (IoU >= 0.75); on objects it does not
  understand (a rifle, a steel beam) it is ignored. Before, SAM's eroded
  shape stayed solid and the refiner only touched a thin band around it, so
  a blobby SAM mask (hair volume filled in) came out with hard, jagged hair
  and cut-off strands (field report). Old construction: `trimap="band"`.
- **Parts are cut from the whole's matte** (`parts_from_whole`, default on):
  `mask.hair` / `mask.jacket` = `mask.person` x a soft ownership map (each
  pixel to the nearest part, or to the remainder: face, hands). The outer
  edge is the person's matte, strands beyond SAM's hair mask land in
  `mask.hair`, and the parts add up to the whole. Parts are no longer
  refined on their own (each had its own, disagreeing outline).
- **Front-to-back order from depth** (`layer_order="depth"`, default): SAM 3
  orders objects by median Depth Anything V2 Small disparity (Apache-2.0,
  5 frames) instead of "the smaller object is in front". Thin background
  shelves were ranked in front of the person standing before them, so the
  layering post held her hair out and pasted hard chips of SAM's shape onto
  it. Objects at about the same depth (a rifle in the hands) still fall back
  to the area rule; no depth weights offline -> area order with a warning.
- Neighbour exclusion fades a refined edge out over the seam width instead
  of cutting it: a strand over the wall no longer ends on a straight line.

### Added
- **Whole + parts roto**: concepts like `person, hair, jacket, trousers` now
  give `mask.person` = the full body and parts that split it among
  themselves and stay inside it. Before, one-owner-per-pixel left
  `mask.person` as just face and hands. A track is a part when its label is
  a clothing / body / vehicle-part word (or listed in `part_labels`) and its
  mask lies >= 70% inside a bigger object; a person inside a car stays a
  separate object. SAM 3 joint tracking, the refiners' neighbour exclusion,
  seam normalisation, the layering post and Cryptomatte (the whole's ID is
  what its parts leave) all respect it. `sam3_hard_masks` entries gain
  `parent`.
- **BiRefNet decides the outline** (Passes tab checkbox; param
  `outline="refiner"`): BiRefNet draws the whole silhouette and SAM 3 only
  gates leaks (`gate_px`). Default mode only lets BiRefNet soften the edge
  of SAM's shape, so a blobby SAM mask (arm gaps filled in) stayed blobby.
  Safety net: a frame where BiRefNet keeps less than `min_area_ratio`
  (75%) of SAM's area falls back to the SAM-bounded edge (a pole across the
  subject). From a field roto on a small, distant person.

### Fixed
- ACEScg plates were detected as lin_rec709: OIIO stamps
  `oiio:ColorSpace = lin_rec709` on every EXR without colour info, and that
  tag was read before the file's chromaticities. Chromaticities (AP1 ->
  ACEScg, AP0 -> ACES2065-1) and `acesImageContainerFlag` now come first,
  and OIIO's default stamp is reported as a guess, not a confident read.
- Adding a shot reset the Output settings (destination, proxy, delivery) and
  the colourspace override. A new shot now inherits them from the session;
  a plate whose header is confident (ARRI LogC4) keeps its own colourspace.
- The matte layering post filled SAM's hard shape to 1 next to a touching
  object even where both refined mattes had rejected it, pasting hard chips
  onto soft hair edges. It now fills only pixels a matte actually claims.
- BiRefNet stretched every crop to 1024x1024, so a tall person crop was
  squashed ~2.5:1. The crop now grows to a square from plate context and
  is padded only where the frame ends (`keep_aspect`, on by default).
- Multi-word concepts (`brick wall`) wrote a `mask.brick wall` channel that
  Nuke reads as `mask_extra.brick_wall`, so the comp script pointed at a
  missing channel. Mask channels are now channel-safe: `mask.brick_wall`.
- ViTMatte crawled on multi-object shots (one object ~10 min, PF0093-05 with
  10 objects did not finish in an hour): every crop has its own size, so the
  CUDA cache grew to 25 GiB reserved for 5 GiB used and, next to the run's
  other models, spilled the 32 GB card into shared system memory. The cache
  is now released when it runs far above real use (0.4 GiB reserved, same
  0.3 s per object-frame).

## [0.9.0] — 2026-10-28

### Added
- **Camera track** ("Camera" category; plugin `camera_pycolmap`): classical
  SfM with pycolmap 4 (BSD-3-Clause, PyPI wheels, CPU, no weights). Writes
  `camera_track/` next to the sidecars: `<shot>.camera.nk` (Nuke Camera2,
  Y-up, ZXY), `<shot>.camera.json` (record + per-frame reprojection error),
  `<shot>.points.ply` (sparse cloud) and `<shot>_colmap/` (COLMAP model —
  what Gaussian-splat trainers read). Frames are CLAHE-normalised before
  SIFT: on a dark bar interior (ACTIONVFX PF0081) plain SIFT registered
  2/32 frames, CLAHE + lower peak threshold 64/64. **With Matte on in the
  same job, every SAM 3 object is dilated and masked out of the features**
  (matchmove garbage matte for free; `mask_labels` to restrict). Measured on
  PF0081 (64 frames, 2048x1080 dolly): ~2 min, 0.99 px mean / 0.70 px median
  reprojection at plate resolution, **verified inside Nuke 16** by projecting
  the solved points through the imported camera (`scripts/nuke_check_camera.py`).
  Limits: arbitrary scale, one focal per clip, needs parallax (locked-off /
  nodal pans cannot triangulate); `SIMPLE_RADIAL` solves k1 but the Nuke
  camera does not apply it (11 px off at the edges on PF0081 without an
  undistort), so the default model is `SIMPLE_PINHOLE`.
- **Gaussian splat** ("Camera" category; plugin `splat_gsplat`, gsplat
  Apache-2.0): trains a 3DGS of the static scene on the shot's own camera
  solve and re-renders every frame through it -> EXR layer `splat`
  (`splat.R/G/B` scene-linear clean plate, `splat.A` coverage, `splat.Z`
  multi-view depth) + `camera_track/<shot>.splat.ply` in the same world as
  the Nuke camera. SAM 3 objects (Matte on) are excluded from training, so
  the re-render is the plate without them. PF0081 (64 frames 2048x1080):
  61 s training, 33.9 dB PSNR on frames held out of training; the reaching
  arm disappears from the re-render. gsplat needs a wheel built for the
  tool's torch 2.11 / cu128 (no upstream one): `wheels/` + 
  `scripts/build_gsplat_wheel.bat` (all NVIDIA archs RTX 20xx-50xx), and
  install/update scripts now re-install `wheels/*.whl` after `uv sync`.
  See docs/gaussian_splat.md.
- **Nuke comp starter script** `<shot>.comp.nk`, written next to the
  sidecars on every run: the plate, the utility EXR(s), one labelled
  Shuffle2 per pass (each mask on its own, an "every mask" matte, depth /
  normals / flow to RGB, the splat clean plate to RGBA) inside backdrops,
  the camera track with its point cloud in a Scene + ScanlineRender, and,
  with a splat and masks, a Keymix that swaps the masked objects for the
  splat clean plate. Paths are relative to the script, so the output folder
  can be moved. Verified by loading it in Nuke 16 (`-t`): no node errors,
  every Shuffle equals its source channel, Keymix/ScanlineRender inputs
  wired right, 3582 points loaded on PF0081. Nuke quirks handled: our
  `mask.*` shows up as `mask_extra.*` (collision with Nuke's built-in mask
  layer), custom layers are declared with `add_layer` (otherwise Shuffle2
  mappings load black).
- **Output layout option** (GUI Output tab, `Shot.delivery_layout`): one
  multi-layer EXR (default, as before) or **one EXR per pass in its own
  subfolder** (`depth/`, `matte/`, `splat/`, `flow/` ...).
- Camera track also writes `<shot>.points.obj`: Nuke 16's ReadGeo2 reads
  neither PLY nor USD points; an OBJ with `p` elements loads as a point
  cloud.
- DAG **optional artifacts** (`optional_artifacts`): a pass is ordered after
  a provider when one is scheduled and runs without it otherwise; and a
  pass-level `emit_sidecars()` hook for non-EXR outputs.

- **MegaFlow motion backend** ("MegaFlow (large displacement, slower)" in
  the Motion category; plugin `flow_megaflow`). Vendored Apache-2.0 code
  (ETH CVG, ECCV 2026) on a frozen DINOv2 ViT-L/14; same channels and
  artifacts as RAFT. Bench on PF0071 @1024 (RTX 5090): on consecutive
  frames both are fine (RAFT 68 ms/pair, MegaFlow 320 ms, 1 vs 6 GB VRAM);
  at 8x frame stride RAFT hallucinates flow on textureless water
  (forward/backward error 5.2 px) while MegaFlow stays clean (0.67 px).
  RAFT remains the default; pick MegaFlow for fast action, retimes and
  textureless plates. Weights: HF Kristen-Z/MegaFlow (prefetch covers it).
- **Marigold V2 bridge passes** (Apache-2.0 code + weights, Qwen-Image-Edit
  -2509 Apache-2.0 backbone): `marigold_v2_albedo` (Albedo category) and
  `marigold_v2_depth_seethrough` (Depth category — depth that sees through
  glass/water). The 20B model runs in an EXTERNAL environment
  (`A:/AI/marigold-v2`, one-time setup via `setup_win.sh` + asset download,
  ~70 GB); the passes shell out once per shot. Measured on a real plate
  (PF0071, 16 frames @1024): ~40-50 s per shot including model load;
  see-through depth is temporally very stable (0.2-0.4% of range frame to
  frame); albedo drifts by a global per-channel gain that the pass
  neutralises (`stabilize_gain`). Experimental — per-image model.

### Changed
- **Roto (SAM 3) is one-owner-per-pixel and re-detects.** Field problem:
  overlapping mattes and broken edges where objects touch, and objects
  missed when absent from the seed frame. Root cause: every object was
  tracked in its OWN session, so masks overlapped freely, and detection ran
  on one frame only. Now:
  - objects sharing a seed frame are tracked **jointly** (video encoded
    once; each pixel goes to the highest-logit object, user clicks get a
    bias) and every contested pixel has ONE owner across all tracks (click
    > detector, then the smaller object: rifle over person, person in a car
    window). `mask.a + mask.b` now adds up instead of overlapping;
  - **re-detection** every 12 frames (`redetect_stride`) starts tracks for
    objects that enter later; a same-label detection touching an existing
    track is re-attached to it (a lost arm), not a new "person";
  - duplicates (a click on an object the detector also found) collapse to
    one track; the click wins;
  - refiners **normalise seams**: where soft alphas of different objects
    sum above 1, they are scaled to add up to 1.
  Measured (1024 proxy): PF0048 — the rifle, missed before, is found
  (frame 1073 re-detection) and tracked 1047-1120; contested pixels 2.0% ->
  0; alpha-sum>1 pixels after BiRefNet 2.0% -> 0. PF0052 (garage) — the
  gunman entering at 1082 and the white car, missed before, are tracked.
  PF0071 — same objects, joint tracking 50 -> 38 s. Cost when re-detection
  finds more objects: PF0048 full chain 99 -> 108 s, PF0052 73 -> 125 s.
  Old behaviour: `joint_tracking=False, exclusive_masks=False,
  redetect_stride=None`.
- **Roto seams between objects are layered like a comp** (new post
  `matte_layering`, runs last, automatic with Matte). Field problem: a hard
  dark line between two mattes that touch (person in front of a car).
  Cause: each object is refined on its own, so at the contact BOTH alphas
  fall to ~0.5, and SAM leaves a thin unowned line between the masks. Now
  SAM closes contact gaps up to 0.4% of the long edge (`seam_close_px`),
  publishes a front-to-back rank, and after refining the front object keeps
  its own edge while the back object is filled under it near the contact
  and held out: `back = max(back, fill) * (1 - front)`. Thin unowned cracks
  between hair strands count as contact, wide empty areas stay background,
  edges against background are untouched, sums never exceed 1. Refiners now
  let an object keep its soft edge 1% of the long edge into a neighbour
  (`neighbor_exclude_erode: auto`, was 2 px) so hair over a car survives.
  Measured on the replayed BiRefNet output (alpha sum in the contact zone,
  1024 proxy): PF0052 person/car mean 0.88 -> 0.997, pixels below 0.9 46%
  -> 0.6%; PF0048 greenscreen/person 0.77 -> 0.99 (below 0.9: 77% -> 2%).
  Cost ~6-7 s per shot.
- SAM 3.1 evaluated, not adopted yet: its gain is speed with many objects
  (~7x at 128 objects), quality mixed (-2..+2 pts); transformers' SAM 3.1
  video model is still a draft PR (#46128) and Meta's repo needs Python 3.12.

### Fixed
- The executor no longer fails when a frame has no EXR channels (e.g. SAM 3
  found nothing and the only other pass writes non-EXR sidecars).

## [0.8.0] — 2026-08-13

### Added
- **Screen Pull keyer** ("Key" category, Passes tab): whole-plate chroma
  key — `key.rgba` = despilled plate premultiplied by a pulled alpha
  (screen=0, subject=1, soft in between). Pure colour-difference maths:
  no models, no weights, MIT-clean, streams frame-by-frame (memory-flat
  on 1000-frame clips). Auto green/blue detection; keys on the
  SCENE-LINEAR plate (the display transform's desaturation collapses the
  chroma discriminant) with a robust two-stage per-frame calibration
  (works with the screen down to ~2% of frame; a per-shot floor stops
  frames where the screen leaves view from re-targeting onto foliage).
  `key.rgb` stays unclamped scene-linear — comp keeps the dynamic range.
- **Chroma Key per-object engine**: SAM decides WHO, the key does the
  edge. Auto-calibrated against the measured screen, screen-aware edges
  (non-screen band pixels follow the SAM edge instead of the key), and a
  seed-direction guard that warns when the clicked object looks like the
  screen itself.
- **Per-object engines**: every mask layer can pin its own soft-edge
  engine (ViTMatte / BiRefNet / Chroma Key / RVM) — portrait weights on
  the person, trimap on the car, in one run. The engine picker sits
  beside the mask list; "Apply engine" pins the current preview engine
  onto the selected object; the list shows each object's engine.
- **One Matte switch** (Passes tab): the five per-engine matte combos
  collapse into Matte Off/On + a "Default engine" dropdown (engine +
  BiRefNet weight variant in one choice). Old sessions migrate
  automatically.
- **Viewport zoom + pan**: wheel zooms around the cursor (up to 8x),
  middle-drag pans, middle double-click resets to fit. Click-to-mask
  coordinates follow the zoomed view exactly.
- **Models menu** (menu bar): "Download all models (prefetch)" launches
  the full prefetch in a console window; a checkable "Offline mode"
  pins HF_HUB_OFFLINE for TPN-style air-gapped runs. Model loading is
  local-first everywhere: once weights are cached, no network requests.
- **Manual pixel-aspect override** (Colour tab) for anamorphic scans
  whose EXR header carries PixelAspectRatio=1.0.
- **Zero-match concept warning**: every user-typed SAM 3 concept that
  matches nothing logs a loud warning naming the missing channel — a
  typo no longer silently drops a matte from the delivery.

### Fixed
- **Duplicate-track edge ring**: a clicked object duplicated by a default
  concept (click "ARM" + default "person") made the twin tracks hollow
  each other out via inter-object exclusion, delivering an edge ring.
  Exclusion now never fires inside an object's own hard mask — and
  clicks with an empty concepts field now submit clicks ONLY (no default
  concepts), removing the duplicate at the source.
- **Hero mattes came out black** when any object pinned its own engine:
  the dedicated engine passes emitted empty hero channels that
  overwrote the main engine's. They now emit only their own
  `mask.<name>` channels.
- **Boiling edges + per-frame warning spam**: the edge guardrail decided
  engine-vs-feather per frame, flip-flopping across the clip. It now
  decides once per track (>=25% tripped frames -> consistent feather)
  and logs one aggregated warning.
- **Refiner band sizing** on thin/holey objects: the band radius now
  derives from sqrt(mask AREA) instead of the bbox diagonal (a rifle no
  longer gets a car-sized halo), BiRefNet's core anchor is a small
  fixed erode independent of the band, and guardrail feathering is a
  tight 3px instead of band-sized mush.
- **SAM speck cleanup**: stray specks and pinholes in SAM 3's hard masks
  (top culprits for refiner blobs) are removed before refinement, with
  absolute caps so legitimate structures (car windows, arm gaps) are
  never filled.
- **TPN compliance**: warm-cache model loads no longer touch the
  network (HF revalidation skipped; torch.hub validation skipped).
- Prefetch covered ViTMatte + Video-Depth-Anything (external tester
  report); a coverage guard now warns when a pass's weights are not in
  the prefetch list.

### Changed
- Masks tab: point placement auto-arms with the tab (the "Place points
  in viewport" checkbox is gone); preview overlay alpha is gamma-lifted
  so soft edges read as soft instead of eroded.
- launch/update scripts pin `--extra all` so a bare `uv run` can no
  longer prune the refiner extras.

## [0.7.1] — 2026-07-14

### Fixed
- **Fast scrubbing no longer hard-crashes the GUI** (Windows access
  violation). The preview loader started one decode per slider tick on the
  shared global thread pool — concurrent OIIO/OCIO decodes raced in native
  code. Decodes are now serialized on a private single-thread pool and
  superseded queued requests skip their decode entirely.
- **Anamorphic plates now display desqueezed.** Pixel aspect was detected
  from the EXR header but the viewer never applied it. Display-only: the
  pipeline stays in pixel space, and click-to-mask coordinates are
  unaffected. (If a scan's header carries PixelAspectRatio=1.0 there is
  nothing to detect — say so and we'll add a manual override.)

## [0.7.0] — 2026-07-10

### Added
- **ViTMatte trimap-guided refiner** ("SAM3 + ViTMatte" in Passes). MIT code
  AND weights, native HF transformers loader. The trimap is built per frame
  from SAM 3's hard mask (erode → fg, dilate band → unknown); the model only
  estimates alpha inside the band, so known regions are respected **by
  construction** — interior erosion is structurally impossible. Benchmarked
  on a real plate vs BiRefNet-portrait with the in-repo harness: **+20%
  soft-edge detail in the hair band** (25.5% vs 21.3%) and cleaner outside it.
- **"Preview with" selector** (Masks tab): preview the seed-frame mask with
  ANY refiner engine — ViTMatte / BiRefNet Portrait-Matting-General /
  RMBG-2.0 / RVM — independently of the run choice, to eye-compare
  algorithms per shot before committing. Preview-only; "Same as run"
  mirrors the Passes choice.
- **Compare mode** ("SAM3 + ALL refiners"): RVM, BiRefNet and ViTMatte all
  run on the same SAM 3 tracks, each writing its own layer
  (`matte_rvm.*` / `matte_birefnet.*` / `matte_vitmatte.*`, plus
  `mask_<engine>.<name>` in all-masks mode) — pick the best algorithm per
  shot in Nuke. ~3x refiner time.

### Fixed
- `vitmatte_refiner` was missing from the refiner-params injection set, so
  the "Refine ALL mask channels" toggle never reached it.

### Changed
- Passes tab: the BiRefNet weights dropdown is disabled when the selected
  matte combo doesn't use it; the all-masks checkbox is relabelled to say
  what it does. The Masks tab notes which engine the run will use.

## [0.6.0] — 2026-07-07

### Added
- **Compact EXR delivery** (Output tab): per-shot codec + bit depth. The
  "Compact - DWAB 16-bit" preset measured ~3.5x smaller on a real 6-pass 2K
  shot — built for moving sidecars between machines on limited bandwidth.
  **Cryptomatte auto-splits** into a lossless float32 `.crypto.exr` sibling
  whenever the chosen delivery would corrupt its exact hash IDs (measured:
  even a lossless float16 cast breaks the id->name match in Nuke). Defaults
  unchanged (single lossless float32 file).
- **Soft edges on ALL masks** (Masks tab): the refiner runs on every
  detected/clicked object — each `mask.<name>` channel becomes roto-grade
  soft alpha instead of SAM 3's hard edge (union-by-label preserved). Off by
  default; costs one refinement per object.
- **Per-shot refiner model choice** (Masks tab), drives BOTH the seed-frame
  preview and the submit: BiRefNet Portrait (new default — measured visibly
  softer hair than the general weights), BiRefNet Matting, BiRefNet General
  (cleanest training-data provenance), RMBG-2.0 (labelled: paid licence for
  commercial use). Benchmarked reproducibly in
  `scripts/benchmark_refiner_weights.py`.
- **Mask preview is always soft** and gamma-lifted, so semi-transparent
  regions (motion-blurred hands, hair) read correctly instead of looking
  eroded.

### Fixed
- **Refiners can no longer erode the mask interior.** Trimap-style combine:
  the model only shapes the edge band between erode and dilate; the eroded
  hard core stays solid 1.0 even if the model under-segments.
- **Refiner memory: hard-mask stacks stay uint8** end-to-end — the refiner
  was re-expanding every track's whole-clip stack to float32 on ingest,
  undoing the v0.5.2 saving (~1.2 GB vs 0.3 GB per 129-frame 1920 track).

### Changed
- GUI polish: refiner controls consolidated on the Masks tab, "Preview" tab
  renamed "Colour", Output tab marked session-wide, hint text trimmed.

## [0.5.2] — 2026-06-23

### Fixed
- **SAM 3 matte no longer OOM-crashes the box on long / high-detection
  clips.** The pass held *every* tracked instance's whole-clip mask stack in
  host RAM at once, in **float32** — peak scaled `N_instances x frames x H x W
  x 4 bytes`. A real shot (573 frames, 1080p, ~16 detected instances) peaked
  ~70 GB and died on a raw `MemoryError` after ~15 min of GPU work. Fixes:
  - **Hard masks are stored as `uint8` (0/1), not float32** — mattes are
    binary, so float32 was a 4x waste. The same shot now peaks ~17.5 GB.
    Consumers (Cryptomatte / RVM / BiRefNet refiner) cast to float32 one
    instance at a time, where the cost is bounded.
  - **Pre-flight RAM check (fail fast).** The pass estimates peak host RAM
    right after detection — *before* the minutes-long track — and aborts at
    second 0 with an actionable message ("needs ~X GiB for N instances at WxH
    over F frames; Y GiB free — reduce instances, lower proxy, or shorten the
    range") instead of crashing late and throwing the work away.
  - **Graceful `MemoryError`** at artifact emission surfaces the same guidance
    rather than a raw NumPy traceback, if the estimate was optimistic.
- **Cryptomatte: stop re-expanding the whole mask stack per frame.** The
  encoder cast each instance's entire `(T, H, W)` stack to float32 on *every*
  frame just to read one slice — `O(frames)` wasted work. It now indexes the
  single frame first, then casts.

### Notes
- The `sam3_hard_masks` artifact's `stack` is now `uint8` (was `float32`).
  Anything reading it directly should cast as needed; the bundled consumers
  already do.

## [0.5.1] — 2026-06-18

### Fixed
- **A transient NAS / network-share blip during sidecar writes no longer
  throws away a whole multi-pass shot.** Each EXR write now retries a few
  times (linear backoff) on `OSError` before failing — a momentary share
  glitch recovers instead of aborting minutes of GPU work. Genuine faults
  (disk full, permission) still surface after the retries.

### Docs
- New **Model cache** section: what's cached, that it lives on `C:`
  (`%USERPROFILE%\.cache\huggingface` + `\.cache\torch`) by default, rough
  sizes (~30–40 GB with everything), and how to relocate it to another drive
  with `HF_HOME` / `TORCH_HOME`.

## [0.5.0] — 2026-06-18

### Added
- **Offline operation + model prefetch** (reliability: a real batch failed 8
  shots whose weights were *already cached* when Hugging Face had a ~14-min
  outage — the tool re-validates the cache over the network on every run).
  - **`liveaov prefetch`** — a "dummy preload" that loads each pass's model
    once **on CPU** (no GPU, no VRAM contention) purely to trigger every
    download, including the secondary repos some passes pull (NormalCrafter /
    DepthCrafter → `stable-video-diffusion`) and the torch.hub backends.
    `--passes a,b,c` for a subset, `--all` for every backend.
  - **`liveaov --offline ...`** — sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`
    so a run never contacts the network and a blip can't abort it. Workflow:
    `prefetch` once online → run with `--offline` forever.
- Default HF timeouts raised (`HF_HUB_DOWNLOAD_TIMEOUT=120`,
  `HF_HUB_ETAG_TIMEOUT=30`, up from 10 s) at the entry points + `launch.bat`,
  so cold downloads and brief blips don't fail.

### Changed
- Per-run log filename now carries the shot slug (batch triage).

## [0.4.2] — 2026-06-18

### Fixed
- **`liveaov --help` crashed on a legacy (cp1252) Windows console / when
  stdout is piped.** The CLI help and `inspect` output contained non-ASCII
  glyphs (`—`, `→`, `…`, `⚠`, `±`); Rich raised `UnicodeEncodeError` rendering
  them on non-UTF-8 stdout. All CLI user-facing strings are now ASCII, so
  `--help` works everywhere. (`--version` and the GUI were unaffected.)

### Changed
- `launch.bat` now sets `PYTHONUTF8=1` (defence-in-depth for any future
  non-ASCII output).
- `install.bat` / `install.sh` pass `uv sync --python 3.11` explicitly,
  alongside the `.python-version` pin from 0.4.1 — belt-and-suspenders so the
  venv is never built against an unsupported system Python.

[0.9.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.9.0
[0.8.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.8.0
[0.7.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.7.1
[0.7.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.7.0
[0.6.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.6.0
[0.5.2]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.2
[0.5.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.1
[0.5.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.5.0
[0.4.2]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.2

## [0.4.1] — 2026-06-18

### Fixed
- **Install crash on machines with system Python 3.13/3.14.** With no
  `.python-version` pin and an open-ended `requires-python = ">=3.11"`,
  `uv sync` / `uv tool install` built the venv against the system's newer
  Python instead of the 3.11 the installer provisions. NumPy (pinned `<2.0`,
  i.e. 1.26) doesn't support Python 3.13+ and crashed on import with
  `OverflowError: cannot convert longdouble infinity to integer`, which then
  broke the torch import (surfacing as a false "CUDA not available" warning
  and a failed `liveaov --version` smoke test). Fixed by adding a
  `.python-version` (3.11) pin and capping `requires-python` to `>=3.11,<3.13`.
  Lift the cap when the project migrates to NumPy 2.x.

[0.4.1]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.1

## [0.4.0] — 2026-06-18

### Added
- **BiRefNet soft-edge matte refiner** — turns SAM 3's hard masks into
  roto-grade **soft alpha** (hair, motion blur, fine edges) in the `matte.*`
  channels. MIT-licensed and commercial-clean (production-proven: CorridorKey
  uses BiRefNet internally). Pick **"SAM3 + BiRefNet (soft edges)"** in the
  Passes tab. Per-frame, so the `matte.*` channels are declared smoothable —
  pair with the flow-guided temporal smoother on noisy plates. Reuses the RVM
  refiner's pipeline; RVM stays the default. Verified end-to-end on a real
  plate (5-pass batch, hair/edge detail confirmed in Nuke).
- **Box-drag prompt** for click-to-mask — drag a rectangle around an object
  for a strong single SAM 3 box prompt (often better than many clicks, and it
  sidesteps the point-collapse trap). Click/right-click still add include/
  exclude points; box and points coexist.

### Notes
- New `[birefnet]` extra (`timm` / `einops` / `kornia`, MIT), included in
  `[all]`.

[0.4.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.4.0

## [0.3.0] — 2026-06-13

### Added
- **Interactive click-to-mask** (Masks tab). Click an element in the viewport
  — left = include, right = exclude — and at submit SAM 3 propagates it across
  the whole shot into a `mask.<name>` channel **and a Cryptomatte ID** with the
  name you typed. Entirely optional (an empty object list changes nothing) and
  complementary to the text-concepts field: text grabs known categories,
  clicks grab the element text can't name. Fully reproducible headlessly
  (`prompt_instances` param on `sam3_matte`); proxy-safe coordinates.
- **Seed-frame mask preview.** "Preview mask (this frame)" runs SAM 3 on the
  seed frame only and overlays the result — adjust points and re-preview
  before committing to a full-shot run. The model loads on first use, stays
  resident for instant re-previews, and is freed automatically at Submit.
- **Undo point** — remove the last point added without clearing the set.
- **Session save/load + continuous autosave.** Every change autosaves
  (debounced, atomic) to a per-user file; after a crash the next launch offers
  a one-click restore. `File → Save/Open session` (`*.laov.json`) round-trips
  the entire prep — shots, colour decisions, exposure, models, concepts,
  click-to-mask points, output routing, proxy, queue flags. Shots whose plate
  folder vanished are skipped with a warning instead of failing the load.

### Notes
- **Few clicks work best.** SAM's interactive regime is few-click: measured on
  a real plate, 3+2 points segmented the full car (34.8% of frame) while 53
  points collapsed the mask to ~nothing (1.6%). The GUI teaches the
  2–6-clicks → preview → one-corrective-click workflow and warns past 9 points.
- Hard-edge scope unchanged: click-to-mask produces selection/ID-grade masks,
  not roto-grade alpha. A soft-edge refiner stage is the planned next step.

[0.3.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.3.0

## [0.2.0] — 2026-06-09

### Added
- **Cryptomatte pass.** Turns SAM 3 tracked instances into a spec-correct
  Cryptomatte sidecar (MurmurHash3 IDs, ranked coverage channels, manifest in
  the EXR header). Reads in any Cryptomatte-capable compositor (Nuke / After
  Effects / Fusion / Resolve) exactly like a CG render-engine pass — only the
  source is a live-action plate. **Scope:** SAM 3 masks are hard-edged, so this
  is fast per-object *selection* / holdouts / ID-grades, **not** a roto
  replacement. The `feather` param softens edges.
- **SAM 3 concepts field** (Inspector → Passes). Free-text, comma-separated
  list of what SAM 3 should detect, e.g. `person, red car, dog`. Drives both
  the Matte and Cryptomatte passes (they share the detector). Empty = the
  built-in defaults (`person, vehicle, tree, building, sky, water, animal`).
  SAM 3 only finds what you name — an empty/black matte usually means the
  subject wasn't on the list.

### Changed
- **Pinned `transformers` to `>=5.5.4,<6`.** SAM 3 only landed in transformers
  ~5.5 and its checkpoint key layout is version-coupled; an out-of-range build
  silently random-initialises the SAM 3 text encoder, which produces an empty
  matte. The old `>=4.45` pin let installs resolve a broken version.
- **OpenCV promoted to a core dependency** (`opencv-python-headless`) — fixes a
  `No module named 'cv2'` at submit time on clean installs.
- **`geffnet` promoted to a core dependency** — DSINE is the default normals
  backend, so a bare install would otherwise fail on normals with
  `Missing dependencies: geffnet`. The `core` install now runs the full
  default trio (depth / normals / matte) out of the box.
- **Install docs now state the tiers explicitly:** `uv tool install
  live-action-aov` = core (commercial-safe depth/normals/matte);
  `uv tool install "live-action-aov[all]"` = every pass. Documents the
  end-user `uv tool` path (previously only the cloned-repo dev path was
  covered) and `--torch-backend=auto`, which is required on Windows —
  otherwise `uv tool install` pulls the CPU-only torch wheel and the GUI
  refuses to Submit.

### Fixed
- **Non-finite pixels no longer poison passes.** The reader now sanitises
  `NaN`/`±Inf` from a plate on read (→ `0.0`) and logs a counted WARNING.
  Comp-work plates routinely carry such pixels (unpremult divides, expression
  nodes, render holes); previously they propagated through linearize → display
  transform → model → every sidecar channel, because `np.clip` does not strip
  non-finite values.
- **Silent colourspace-decode failures now warn.** If the active OCIO config
  can't decode the source colourspace (e.g. a config missing "ARRI LogC4"), the
  reader logs a WARNING once per shot instead of silently passing the plate
  through unlinearised (which looked like wrong/dark passes).
- **VRAM hygiene.** The executor frees each pass's model after `run_shot` and
  empties the CUDA cache. A multi-pass stack was OOM-crashing on the *second*
  shot of a batch (shot-1 models never freed); peak VRAM is now the single
  heaviest pass.
- **Ultrawide preview crop.** Non-16:9 plates (e.g. 3840×1536) no longer crop
  on the right/bottom in the viewport.
- **Inspector horizontal scrollbar** appears as-needed so long labels aren't
  clipped when the log panel is open.
- **Friendly pre-flight dependency checks** — selecting a pass whose optional
  extra isn't installed now gives an actionable `pip install …[<extra>]` hint
  instead of a raw `ModuleNotFoundError`. Now consistent across Video Depth
  Anything (`einops`/`easydict`), DSINE (`geffnet`), DepthCrafter and
  NormalCrafter (`diffusers`).

[0.2.0]: https://github.com/lettidude/LiveActionAOV/releases/tag/v0.2.0
