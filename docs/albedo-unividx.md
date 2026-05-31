# Albedo via UniVidX — validation + vendoring plan

Status: **Phase 1 (contract + POC harness) landed. Backend NOT vendored.**

This doc tracks adding an **albedo / irradiance** intrinsic pass to
LiveActionAOV using [UniVidX](https://github.com/houyuanchen111/UniVidX).
It's deliberately staged so we never commit a large, unverifiable model
backend to the tree before proving it works on real hardware.

## Why UniVidX (and not the alternatives)

Intrinsic decomposition splits an image into **albedo** (base colour,
lighting removed) × **shading/irradiance**. It's the foundation for
relighting: keep albedo, swap the light.

As of 2026-05, UniVidX is the **only commercial-safe, video-native**
intrinsic model:

| Model | Video? | Albedo | Code | Weights | Verdict |
|---|---|---|---|---|---|
| **UniVidX** | ✅ | ✅ primary | Apache-2.0 | Apache-2.0 | ✅ **commercial-safe** |
| V-RGBX (CVPR 2026) | ✅ | ✅ | Apache-2.0 | CC-BY-NC | ❌ NC weights |
| UniRelight (NVIDIA) | ✅ | byproduct | NVIDIA NC | NVIDIA NC | ❌ NC + needs target HDRI (it's a relighter) |
| Careaga/Aksoy Ordinal/Colorful | ✗ | ✅ | academic-only | academic-only | ❌ no commercial license |

UniVidX is Apache code + Apache weights on the Apache-2.0
Wan2.1-T2V-14B backbone — the whole chain is clean.

The catch: it's a **14B diffusion model**. Impractical on small cards,
fine on 24–32 GB (Q8 Wan2.1-14B ≈ 15 GB in the field). That's exactly
why this is staged behind a hardware-validation gate.

## Phase 1 — contract + POC (this PR)

Landed:

- **Channels** (`io/channels.py`): `albedo.r/g/b`, `irradiance.r/g/b`,
  added to `CANONICAL_CHANNEL_ORDER`. Linear working space so
  `plate ≈ albedo * irradiance` holds for comp.
- **Pass contract** (`passes/intrinsic/univid_x.py`):
  `UniVidXIntrinsicPass` — Apache-2.0, `VIDEO_CLIP`, `RADIOMETRIC`,
  declares the six channels + a 16 GB VRAM floor. `_load_model` raises
  `NotImplementedError` until the backend is validated + vendored. This
  mirrors the repo's MatAnyone2 precedent (contract shipped, backend
  stubbed, withheld from the GUI catalog).
- **POC harness** (`scripts/poc_unividx_prep.py`): turns a plate
  sequence into the display-space frames UniVidX consumes, using the
  *same* reader + display transform the real passes use, and probes
  VRAM. It does **not** run the model — it prints the upstream commands
  to run by hand.
- **Tests** (`tests/test_passes/test_univid_x_intrinsic.py`): the
  declarative contract + an honest "backend raises until wired" guard +
  "not in GUI catalog yet" guard.

Not done: the model isn't registered as an entry point yet (no
`pyproject.toml` line), so it can't be selected — intentional until the
backend exists.

## Phase 2 — validate on hardware (Leo, on the 5090)

Run the prep harness on one delivered shot:

```bash
uv run python scripts/poc_unividx_prep.py \
    --folder "Y:/path/to/plate/v001" \
    --pattern "shot.####.exr" \
    --first 1009 --last 1031 \
    --proxy 960 \
    --colorspace lin_rec709 \
    --out ./poc_unividx_frames
```

Then run UniVidX upstream inference on the prepped frames (the script
prints the exact commands; **verify them against the current upstream
README** — flags drift). Report back:

1. **Quality** — is the albedo genuinely lighting-free? (No baked
   shadows / highlights in the base colour. Look at a face or a wall
   under directional light — albedo should stay flat.)
2. **Runtime** — wall-clock for a ~23-frame clip at 480p on the 5090.
3. **Peak VRAM** — `nvidia-smi -l 1` during the run.
4. **Temporal stability** — does the albedo flicker frame-to-frame?

## Phase 3 — vendor + wire (next PR, gated on Phase 2)

If Phase 2 looks good:

1. Vendor the Apache-2.0 inference under
   `src/live_action_aov/vendored/univid_x/` (mirrors the
   `video_depth_anything` vendoring pattern). Keep the upstream LICENSE.
2. Implement `_load_model` + `run_shot` + `_infer_clip` in
   `passes/intrinsic/univid_x.py` (pull weights via `hf_hub_download`
   from `houyuanchen/UniVidX`; map UniVidX's albedo/irradiance output
   into our channels; undo the model's sRGB-display on the way out so
   the sidecar lands in linear).
3. Add the entry point in `pyproject.toml`:
   `univid_x_intrinsic = "live_action_aov.passes.intrinsic.univid_x:UniVidXIntrinsicPass"`
4. Add to the GUI catalog (`gui/pass_catalog.py`) under a new
   **"Intrinsics"** group, gated by `meets_vram_requirement(state, 16)`
   (PR #36) so it greys out on cards that can't run it.
5. Bump the pass `version` 0.0.1 → 0.1.0 and the package version.

## Portability note

Per the cross-machine requirement: this pass advertises a 16 GB VRAM
floor via `vram_estimate_gb_fn`. Once the GUI gate from PR #36 is wired
to the catalog (Phase 3 step 4), the pass auto-hides on a 4060/3060 and
stays available on a 3090/4090/5090 — no manual per-machine config.
