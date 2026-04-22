# LiveActionAOV

**AI-driven VFX plate preprocessor.** Reads EXR image sequences, runs depth/normals/motion/matte passes, writes sidecar EXRs with Nuke/CG AOV channel conventions.

> **Status**: alpha, pre-v1 release

## Quick start

```bash
git clone https://github.com/lettidude/LiveActionAOV
cd LiveActionAOV
./install.sh        # or install.bat on Windows
```

Then:
```bash
uv run liveaov-gui           # preparation GUI
uv run liveaov --help        # CLI reference
```

## What it does

Given a plate like `/shots/sh020/plate/v003/sh020_plt.####.exr`, the tool produces:

- `/shots/sh020/plate/v003/sh020_plt.utility.####.exr` — sidecar with:
  - `Z` depth channel
  - `N.x/N.y/N.z` camera-space normals
  - `motion.x/motion.y` forward motion vectors (pixels)
  - `back.x/back.y` backward motion vectors
  - `matte.r/g/b/a` top-4 soft hero mattes
  - `mask.<concept>` semantic hard masks

Original plate is never modified. See [design notes](docs/architecture.md) for architectural details.

## Documentation

- [User guide](docs/user-guide.md)
- [Developing plugins](docs/developing-plugins.md)
- [Architecture](docs/architecture.md)

## License

Core: MIT. Individual model plugins have their own licenses — see the license matrix in [architecture notes](docs/architecture.md#license-matrix).
