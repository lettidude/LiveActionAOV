# Albedo via UniVidX (not available yet)

Status: **the pass contract is in the tree; the model is not.** It cannot be selected in the GUI or
the CLI yet.

This pass will add an **albedo / irradiance** split of the plate using
[UniVidX](https://github.com/houyuanchen111/UniVidX) (Apache-2.0 code and weights, on the
Apache-2.0 Wan2.1-T2V-14B backbone). Albedo is the base colour with the lighting removed; with it a
comp can keep the albedo and change the light.

It is a large diffusion model and will need a card with plenty of VRAM; the pass declares that, so
the GUI can grey it out on smaller cards.

## What is in the tree

- **Channels** (`io/channels.py`): `albedo.r/g/b`, `irradiance.r/g/b`, in linear working space so
  `plate ≈ albedo * irradiance` holds for comp.
- **Pass contract** (`passes/intrinsic/univid_x.py`, `UniVidXIntrinsicPass`): declares the channels
  and the licence; loading the model raises `NotImplementedError` until the backend is added.
- **Prep script** (`scripts/poc_unividx_prep.py`): turns a plate sequence into the frames UniVidX
  reads, with the same reader and display transform as the real passes. It does not run the model.
- **Tests** (`tests/test_passes/test_univid_x_intrinsic.py`): the contract, and guards that the
  pass stays out of the GUI until the backend exists.

For a working albedo pass today, see `marigold_v2_albedo` (Albedo category, experimental).
