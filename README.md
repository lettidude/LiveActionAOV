<p align="center">
  <img src="docs/img/LogoImage.png" alt="LiveActionAOV — Multi Utility Pass Generator" width="720">
</p>

# LiveActionAOV

**AI-driven AOV pass generator for VFX plates.** Reads EXR image sequences, runs depth / normals / motion / matte passes, writes sidecar EXRs with Nuke-ready channel conventions.

> **Status:** alpha, pre-v1 release

[![Watch the demo](https://img.shields.io/badge/Demo-YouTube-red?style=flat-square&logo=youtube)](https://youtu.be/TODO_REPLACE_AFTER_UPLOAD)
*Demo video — coming soon.*

---

## Quick start

```powershell
# Windows (PowerShell or cmd.exe):
git clone https://github.com/lettidude/LiveActionAOV
cd LiveActionAOV
.\install.bat
```

```bash
# Linux / macOS:
git clone https://github.com/lettidude/LiveActionAOV
cd LiveActionAOV
./install.sh
```

> **PowerShell users:** the `.\` prefix is required — PowerShell doesn't run scripts from the current directory by default. `cmd.exe` accepts both `install.bat` and `.\install.bat`.

Then:
```bash
uv run liveaov-gui           # preparation GUI
uv run liveaov --help        # CLI reference
```

> **First run downloads model checkpoints from Hugging Face.** Expect
> **~1.5 GB** for a minimal stack (Depth Anything V2 + DSINE + RAFT +
> SAM 3 + RVM) up to **~12 GB** for the full video-aware stack
> (DepthCrafter + NormalCrafter + MatAnyone 2). Cached at
> `~/.cache/huggingface/hub` (Linux/macOS) or
> `%USERPROFILE%\.cache\huggingface\hub` (Windows). Subsequent runs are
> offline-capable for any pass whose weights you've already pulled.
>
> **Updating to latest:** from the project root, run `.\update.bat` (Windows) or `./update.sh` (Linux/macOS). Pulls the latest code and re-syncs deps. Idempotent if there's nothing new.
>
> **What is `uv`?** A fast Python package manager from Astral —
> drop-in replacement for `pip` + `venv`. The installer scripts grab
> it for you. See [docs.astral.sh/uv](https://docs.astral.sh/uv/) if
> you'd rather install it yourself first.

---

## What it does

Given a plate like `/shots/sh020/plate/v003/sh020_plt.####.exr`, the tool produces:

- `/shots/sh020/plate/v003/sh020_plt.utility.####.exr` — sidecar with:
  - `Z` depth channel
  - `N.x / N.y / N.z` camera-space normals
  - `motion.x / motion.y` forward motion vectors (pixels)
  - `back.x / back.y` backward motion vectors
  - `matte.r / g / b / a` top-4 soft hero mattes
  - `mask.<concept>` semantic hard masks
  - `P.x / P.y / P.z` world-space position (when depth is present)
  - `ao.a` ambient occlusion (when depth + normals are present)

Original plate is never modified. See [design notes](docs/architecture.md) for architectural details.

---

## The GUI

Drop a plate folder onto the shot list, pick the passes you want, hit **Submit local**. Each model surfaces its license badge inline so you know up-front which combinations are commercial-safe and which require non-commercial confirmation.

<p align="center">
  <img src="docs/img/gui-passes-tab.jpg" alt="LiveActionAOV GUI — Passes tab" width="900">
</p>

A loaded shot with the Output tab open — multi-shot batch queueing, proxy resolution, and the "what the model sees" Transformed view in the centre viewport.

<p align="center">
  <img src="docs/img/gui-loaded-shot.jpg" alt="LiveActionAOV GUI — loaded shot, Output tab" width="900">
</p>

> Test plate above is from the [ActionVFX practice footage library](https://www.actionvfx.com/collections/free-vfx/category) — used here under their license, which permits incorporation into derivative work like this tool demo.

---

## Nuke plugin — UtilityRelight

A companion Nuke node ships in this repo at
[`src/live_action_aov/plugins/nuke/UtilityRelight/`](src/live_action_aov/plugins/nuke/UtilityRelight/).
It consumes a sidecar EXR + a beauty plate and gives Nuke comp artists
live 3D light placement on the subject — six layered light contributions
(key, spec, rim, bounce, glow, fog) computed on the GPU via BlinkScript.

**Install (3 steps):** copy `utility_relight.py` + `UtilityRelightKernel.blink` into `~/.nuke/`, register in `~/.nuke/menu.py`, restart Nuke. Full instructions and quick-test recipe in the [user guide](docs/user-guide.md#nuke-plugin--utilityrelight).

Tested on Nuke 16.0.

---

## Documentation

- [User guide](docs/user-guide.md)
- [Developing plugins](docs/developing-plugins.md)
- [Architecture](docs/architecture.md)
- [Install reference](docs/install.md) — alternate GPU configurations (Pascal/Volta, Apple Silicon, ROCm), troubleshooting, slim install with per-pass extras

---

## Compatibility

| Platform | GPU | Status |
|---|---|---|
| Windows 11 | NVIDIA RTX 20 / 30 / 40 / 50 series | ✅ **Tested at v0.1.0** |
| Linux (Ubuntu 22.04+) | NVIDIA RTX 20 / 30 / 40 / 50 series | ⚠️ Expected to work via `install.sh` (CUDA + uv path identical to Windows) but **untested at v0.1.0**. Bug reports welcome. |
| macOS (Apple Silicon) | M1 / M2 / M3 / M4 (MPS) | ⚠️ **Best-effort.** Some passes work via MPS; fp16 models (DepthCrafter, NormalCrafter, MatAnyone2) are CUDA-only. The GUI refuses Submit and explains when a pass isn't available. |
| Any platform | AMD GPU (ROCm) | ❌ Untested — wheels exist, none of our passes have been validated against them. |
| Any platform | CPU-only | ❌ Not a supported configuration — fp16 kernels don't exist on CPU. |

See [docs/install.md](docs/install.md#alternate-gpu-configurations) for manual overrides on older NVIDIA hardware (Pascal / Volta cu121 wheel) and Apple Silicon notes.

---

## Author

**Leonardo Paolini** — VFX compositor / pipeline TD building tools at the intersection of comp and ML.

- Email: [LeonardoVFX@gmail.com](mailto:LeonardoVFX@gmail.com)
- GitHub: [@lettidude](https://github.com/lettidude)
- LinkedIn: [leonardopaolinivfx](https://www.linkedin.com/in/leonardopaolinivfx/)
- IMDb: [Leonardo Paolini](https://www.imdb.com/name/nm5886055/)
- YouTube: [@LeonardoVFX](https://www.youtube.com/channel/UCZDR5tThQwo9OVVu-NmODbA) — pipeline & VFX-tooling videos

Developed with [Claude](https://claude.com/) (Anthropic).

---

## Support

- **Bug reports / feature requests:** [open an issue](https://github.com/lettidude/LiveActionAOV/issues).
- **Questions about a specific pass / model wiring:** start a [discussion](https://github.com/lettidude/LiveActionAOV/discussions).
- **Direct contact:** [LeonardoVFX@gmail.com](mailto:LeonardoVFX@gmail.com) — fastest for studio-pipeline questions.

---

## License

Core: MIT. Individual model plugins have their own licenses — see the license matrix in [architecture notes](docs/architecture.md#license-matrix). Every sidecar EXR carries a `liveActionAOV/<pass>/license` metadata stamp plus a top-level `liveActionAOV/matte/commercial` flag so downstream QC can audit what's shippable.
