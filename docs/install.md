# Install

## TL;DR

**Windows** — double-click `install.bat` in the project root.
**Linux / macOS** — `./install.sh` from a terminal.

Both scripts will:

1. Install `uv` (the dependency + venv manager) if missing.
2. Check for an NVIDIA driver.
3. Provision Python 3.11 in a project-local `.venv`.
4. Install all Python dependencies.
5. Replace the default PyPI torch wheel with the **CUDA 12.4 build**
   so neural passes actually run on the GPU (see [Why CUDA
   matters](#why-cuda-matters)).
6. Run a smoke test and verify `torch.cuda.is_available()`.

When the script is done you should see `Installation complete.` and
be able to launch:

```
uv run liveaov --help       # CLI
uv run liveaov-gui          # three-panel prep GUI
```

## Requirements

- **Python 3.11** (the installer provisions this via `uv`, no manual
  setup needed).
- **Disk space**: ~15 GB for the venv + torch + transformers cache.
  First-time model downloads add another 10–40 GB depending on which
  passes you run.
- **GPU**: NVIDIA with CUDA 12-capable driver. RTX 20-series and newer
  all work; RTX 30 / 40 / 50-series are covered by the `cu124` wheel.
  RTX 50-series (Blackwell) users: driver must be >= 560.x.
- **macOS**: MPS works for some passes but is materially slower. CPU
  is not a supported configuration — fp16 kernels don't exist there.

## Why CUDA matters

Every pass is a neural model:

- **fp16 passes** (DepthCrafter, NormalCrafter, SAM 3, ...) **hard-
  fail on CPU** — PyTorch doesn't implement fp16 kernels for CPU and
  the pipeline raises immediately.
- **fp32 passes** technically run but at ~1/100 of GPU speed. A
  single plate can take hours instead of seconds.

The default PyPI `pip install torch` gives you the **CPU-only wheel**
on Windows. This is the single most common "why won't it work?"
failure; the installer scripts detect and fix it automatically.

If you installed manually or suspect something's off:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expect `2.x+cu124 True`. If you see `2.x+cpu` or `False`, reinstall
torch:

```powershell
.venv\Scripts\pip.exe install --reinstall torch torchvision `
    --index-url https://download.pytorch.org/whl/cu124
```

Linux/macOS:

```bash
.venv/bin/pip install --reinstall torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124
```

The GUI shows an amber banner and refuses to Submit when CUDA isn't
available — you won't discover the problem 20 minutes into a batch.

## Optional extras (per-pass)

The core install includes the commercial-safe defaults (RAFT,
DepthAnythingV2, DSINE, SAM3, RVM). Non-default backends ship as
optional extras:

```bash
uv pip install 'live-action-aov[depthcrafter]'           # CC-BY-NC-4.0
uv pip install 'live-action-aov[normalcrafter]'          # CC-BY-NC-4.0
uv pip install 'live-action-aov[video_depth_anything]'   # Apache-2.0
uv pip install 'live-action-aov[matte]'                  # default refiners
uv pip install 'live-action-aov[camera]'                 # pycolmap
```

Kitchen-sink:

```bash
uv pip install 'live-action-aov[depthcrafter,normalcrafter,video_depth_anything,matte,camera]'
```

## Non-commercial model licensing

Some models (DepthCrafter, DepthPro, NormalCrafter, MatAnyone2) are
distributed under **CC-BY-NC-4.0**. This restricts the **model
outputs** from being used commercially, regardless of what licence
this tool itself ships under.

The GUI flags these with a warning marker in the Passes tab and pops
a per-submit consent dialog before running a batch that uses them.
The CLI's license gate mirrors the same policy. See individual model
cards on Hugging Face for the exact terms.

## Troubleshooting

### "RemoteEntryNotFoundError: 404 ... model_index.json"

The HF model repo doesn't have a diffusers-pipeline layout (e.g. it
ships only a UNet). If this happens with a built-in pass, it's a bug
— report it. If it's your own plugin, make sure the HF repo has
`model_index.json` at the root, or compose a pipeline from a
separate base model the way `passes/depth/depthcrafter.py` does.

### "Could not open '<plate>/frame_0000.exr'"

The GUI picked up only one EXR because the filename didn't match any
multi-frame pattern. Check that every plate frame has the same
prefix + zero-padded frame number + `.exr`. The sniffer skips files
with `.utility.`, `.hero.`, or `.mask.` in the name (sidecars).

### `uv sync` stalls on torch

The wheel is ~2 GB. Let it finish, or interrupt and re-run — `uv`
resumes cleanly. Consistent mid-download failures = network issue.

### GUI launches but clicks do nothing

Check the OS taskbar for a hidden window — Qt sometimes opens the
window off-screen on multi-monitor setups. `Alt+Space` then `M` then
arrow keys will move a stuck window back into view.

## Uninstall

```bash
./uninstall.sh       # Linux/macOS
uninstall.bat        # Windows
```

Both remove `.venv/` and optionally the cached model weights under
`~/.cache/huggingface/`.
