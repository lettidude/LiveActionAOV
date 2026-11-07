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

The catch: it's a **14B diffusion model**. Impractical on small cards —
the UniVidX_ComfyUI maintainer (same upstream model) reports a **24 GB
FP8 minimum** (peak 18-20 GB FP8, 32-34 GB BF16). An early ~15 GB Q8
report was optimistic; we treat **24 GB as the honest floor**. That's
exactly why this is staged behind a hardware-validation gate.

## Phase 1 — contract + POC (this PR)

Landed:

- **Channels** (`io/channels.py`): `albedo.r/g/b`, `irradiance.r/g/b`,
  added to `CANONICAL_CHANNEL_ORDER`. Linear working space so
  `plate ≈ albedo * irradiance` holds for comp.
- **Pass contract** (`passes/intrinsic/univid_x.py`):
  `UniVidXIntrinsicPass` — Apache-2.0, `VIDEO_CLIP`, `RADIOMETRIC`,
  declares the six channels + a 24 GB VRAM floor. `_load_model` raises
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

Then run UniVidX upstream inference on the prepped frames. UniVidX is
**config-driven** — you edit `configs/univid_intrinsic_inference.yaml`
(set `mode: R2AIN` for RGB→albedo+irradiance+normal, point
`inference_rgb_path` at the prepped frames, leave the other modality
paths `null`) and run `scripts/inference_univid_intrinsic.py --config …`.
Weights are ~85 GB (Wan2.1-T2V-14B backbone + UniVidX LoRA checkpoint).
The prep script prints the full step list; **verify every path/key
against the current upstream README** — the interface drifts. Report
back:

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
   from `houyuanchen/UniVidX` **lazily** — see "Weights are lazy" below;
   map UniVidX's albedo/irradiance output into our channels; undo the
   model's sRGB-display on the way out so the sidecar lands in linear).
3. Add the entry point in `pyproject.toml`:
   `univid_x_intrinsic = "live_action_aov.passes.intrinsic.univid_x:UniVidXIntrinsicPass"`
4. Add to the GUI catalog (`gui/pass_catalog.py`) under a new
   **"Intrinsics"** group, gated by `meets_vram_requirement(state, 24)`
   (PR #36) so it greys out on cards that can't run it.
5. Bump the pass `version` 0.0.1 → 0.1.0 and the package version.

### Weights are lazy (hard requirement)

The ~85 GB of weights (Wan2.1-T2V-14B backbone ≈ 69 GB + UniVidX LoRA
checkpoint ≈ 1.6 GB) must download **on the first run of the pass**, not
at `install.bat` / `install.sh` time, and only **after the user has
confirmed** the download. LiveActionAOV installs worldwide on modest
connections; a multi-tens-of-GB pull bolted onto general install would
be a killer. Concretely:

- `install.*` and `uv sync` must **not** fetch any UniVidX weights.
- The first time `_load_model` runs, check the HF cache; if missing,
  print/emit a clear "UniVidX needs ~85 GB of weights — download now?"
  warning (size + target path) before calling `hf_hub_download`.
- In the GUI, surface the size up front so a user on a metered link
  isn't surprised by an 85 GB background download.

## Portability note

Per the cross-machine requirement: this pass advertises a 24 GB VRAM
floor via `vram_estimate_gb_fn`. Once the GUI gate from PR #36 is wired
to the catalog (Phase 3 step 4), the pass auto-hides on cards below
24 GB (4060/3060/3070/4070, and the 16 GB 4060 Ti/4080) and stays
available on a 3090/4090/5090 — no manual per-machine config.

## Why not the ComfyUI node (`dreamrec/UniVidX_ComfyUI`)

There is a ComfyUI custom-node pack wrapping the same upstream model. We
**reject it as a backend**, deliberately, so this isn't relitigated:

- **License — disqualifying.** The pack is **GPL-3.0**. LiveActionAOV
  ships commercially (`commercial_tool_resale=True` on this pass), and
  the whole reason we picked UniVidX over V-RGBX / UniRelight was a clean
  Apache-2.0 chain. Vendoring or distributing GPL-3.0 code is copyleft —
  it would force the entire tool to GPL. Non-starter.
- **No actual saving.** The node wraps the *same* Wan2.1-14B weights and
  the *same* inference — identical VRAM, runtime, and ~85 GB download. It
  saves a little glue code, nothing the *user* pays for.
- **Dependency inversion.** It would make our install story "first stand
  up a ComfyUI server + install a custom node + match versions," which is
  strictly worse than our one-click GUI and subordinates us to another
  app's release cadence.

The only legitimate use is **private, local validation** on your own GPU
during Phase 2 (mere use, never distributed) — and even then, running the
upstream Apache repo directly keeps GPL entirely out of the loop, so
that's preferred. The shipped backend is **always** the vendored
Apache-2.0 inference (Phase 3).
