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

Lands in Phase 5.
