# Local wheels

Wheels dropped here are installed by `install.bat` / `update.bat` (and the
`.sh` versions) after `uv sync`, which would otherwise remove them.

Used for packages that have no PyPI wheel for the tool's torch / CUDA,
today only **gsplat** (Gaussian splat pass). Get it by either:

- downloading `gsplat-*-cp311-cp311-win_amd64.whl` from the GitHub Release
  that matches your LiveActionAOV version, or
- building it: `scripts\build_gsplat_wheel.bat`.

See `docs/gaussian_splat.md`. `*.whl` files are git-ignored.
