# Intrinsics & materials via Marigold-IID

LiveActionAOV's albedo / intrinsic-decomposition backend. Two passes on the
shared PRS-ETH Marigold (SD2, ~1B) diffusers pipelines, plus a Marigold
normals option.

## Passes

| Plugin | Outputs | Channels |
|---|---|---|
| `marigold_iid_lighting` | albedo, diffuse shading, non-diffuse residual (`I = A·S + R`) | `albedo.*`, `irradiance.*`, `residual.*` |
| `marigold_iid_appearance` | albedo, roughness, metallicity (PBR) | `albedo.*`, `material.roughness`, `material.metalness` |
| `marigold_normals` | surface normals | `N.x/y/z` |

All emitted as linear-space sidecar EXR channels (albedo absolute; shading/
residual up-to-scale). Intrinsics are single-select per GUI category, so you
pick Lighting *or* Appearance (no double albedo).

## Why Marigold (and not UniVidX)

We first validated **UniVidX** (Apache, 14B Wan video-native) on a 5090: it
works and gives good albedo, but it's **too heavy** — 11–36 min for 21 frames,
21–29 GB VRAM, 85 GB weights. Its strongest output (normals) we already produce
cheaper.

**Marigold-IID** gives **comparable / arguably cleaner albedo at ~87× less
compute**: ~0.38 s/frame, ~3.4 GB peak VRAM, ~2 GB model — runs on a laptop.
Single-image, so temporal consistency comes from a fixed per-frame seed + the
flow-guided temporal smoother (`smooth: auto`), which held stable on test
footage. Head-to-head frames: see the POC artifacts under `A:/AI/UniVidX/`
(`results/marigold/`).

Licensing: Marigold weights are **CreativeML OpenRAIL++-M** — commercial use
permitted with behavioural-use restrictions (not pure Apache). Fine for this
tool's personal / freelance use. (UniVidX remains the only Apache-clean
*video-native* option, kept as a contract under `passes/intrinsic/univid_x.py`
if a strict-Apache requirement ever returns.)

## Install / use

```bash
pip install -e ".[marigold]"     # diffusers>=0.37 + accelerate
```

Weights download lazily from the HF hub on first run (~2 GB per checkpoint).
In the GUI, pick under **Intrinsics** (Lighting/Appearance) or **Normals**
(Marigold). CLI: enable the plugin name (e.g. `marigold_iid_lighting`).

Key params: `num_inference_steps` (1–4; default 4), `ensemble_size` (≥3 for
extra precision at linear cost), `seed` (fixed for temporal stability),
`precision` (`fp16`/`fp32`). Normals adds `flip_y`/`flip_z` escape hatches
(default off — Marigold already matches the spec's OpenGL axes).

## Validation status

- Contract tests: `tests/test_passes/test_marigold.py`.
- Real end-to-end (pass class, 5090): Lighting on an HD plate frame →
  9 channels at plate res, sane ranges. **Appearance variant: implemented but
  not yet run on hardware** (downloads a separate ~2 GB checkpoint) — first QA
  target.
