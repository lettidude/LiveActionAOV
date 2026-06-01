# Intrinsics & materials via Marigold-IID

LiveActionAOV's albedo / intrinsic-decomposition backend — two passes on the
shared PRS-ETH Marigold (SD2, ~1B) `MarigoldIntrinsicsPipeline`.

## Passes

| Plugin | Outputs | Channels |
|---|---|---|
| `marigold_iid_lighting` | albedo, diffuse shading, non-diffuse residual (`I = A·S + R`) | `albedo.*`, `irradiance.*`, `residual.*` |
| `marigold_iid_appearance` | albedo, roughness, metallicity (PBR) | `albedo.*`, `material.roughness`, `material.metalness` |

All emitted as linear-space sidecar EXR channels (albedo absolute; shading/
residual up-to-scale). Single-select per GUI **Intrinsics** category — pick
Lighting *or* Appearance (no double albedo). (Marigold normals were evaluated
and dropped — NormalCrafter/DSINE cover normals.)

## Why Marigold (and not UniVidX)

We first validated **UniVidX** (Apache, 14B Wan video-native) on a 5090: good
albedo, but **too heavy** — 11–36 min for 21 frames, 21–29 GB VRAM, 85 GB
weights. **Marigold-IID** gives comparable/cleaner albedo at **~87× less
compute** (~0.38 s/frame, ~3.4 GB VRAM, ~2 GB model). License is **CreativeML
OpenRAIL++-M** (commercial use permitted, use-restricted) — fine for this
tool's personal/freelance use. UniVidX's contract stays parked under
`passes/intrinsic/univid_x.py` if a strict-Apache *video-native* need returns.

## Temporal stability (the open question)

Marigold is **single-image**, so naive per-frame inference flickers on moving
plates. Two stabilizers stack:

1. **Latent propagation** (in `run_shot`): each frame's init latent =
   `temporal_blend·anchor + (1-temporal_blend)·prev`, anchor = frame-0 latent.
   Marigold's own recommended video scheme. Measured **~40% reduction** in
   consecutive-frame albedo delta (0.0157 → 0.0095) at `temporal_blend=0.9` vs
   per-frame, on a moving-water test clip.
2. **Flow-guided smoother** (`smooth: auto`) — stacks on top, but **requires a
   Motion/RAFT pass enabled** to have optical flow to warp along.

**Status: not yet confirmed video-ready.** Per-frame Marigold flickers too much
for video (QA verdict, 2026-06). Latent propagation helps measurably; whether
the propagation + flow smoother combination reaches delivery-grade temporal
stability is the open QA question. If it doesn't, Marigold is still a strong
**stills / DMP albedo** tool. Judge on a scrubbed sequence with Motion enabled.

## Install / use

```bash
pip install -e ".[marigold]"     # diffusers>=0.37 + accelerate
```

Weights download lazily from the HF hub on first run (~2 GB per checkpoint).
GUI: **Intrinsics → Lighting/Appearance**, and enable **Motion → RAFT** so the
flow smoother engages. CLI: enable `marigold_iid_lighting`.

Key params: `temporal_blend` (0–1, default 0.9; higher = steadier/more
anchored, 0 = pure per-frame), `num_inference_steps` (1–4; default 4),
`ensemble_size` (≥3 for extra precision), `seed`, `precision` (`fp16`/`fp32`).

## Validation status

- Contract tests: `tests/test_passes/test_marigold.py` (13).
- End-to-end on a 5090: `marigold_iid_lighting.run_shot` over a clip →
  9 channels at plate res, sane ranges; latent propagation ~40% flicker cut.
- **Appearance variant**: implemented, not yet run on hardware (separate ~2 GB
  checkpoint) — a QA target.
