# Changelog

All notable changes to LiveActionAOV are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses [semantic versioning](https://semver.org/).

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
  `uv tool install "live-action-aov[all]"` = every pass.

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
