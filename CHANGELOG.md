# Changelog

All notable changes to LiveActionAOV are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/); this project
uses [semantic versioning](https://semver.org/).

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
