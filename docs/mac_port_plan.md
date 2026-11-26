# Mac port plan (Apple Silicon)

**Status: groundwork only. Nothing here has been run on a real Mac yet.**
Every pass is marked `mac_verified=False` until it has been.

## What is in the code today

- **One device rule** (`core/device.py`): NVIDIA card (CUDA), then the
  Apple GPU (MPS), then the CPU. `LAOV_DEVICE` forces one (`cuda`,
  `cuda:1`, `mps`, `cpu`, `auto`). With CUDA present and no override the
  choice is exactly what it was before (`torch.device("cuda")`).
- **A per-pass declaration** (`DeviceSupport` on `UtilityPass`,
  `core/pass_base.py`): which devices the pass can be placed on, and what
  is expected on a Mac: `mps` (Apple GPU), `mps_fp32` (Apple GPU, fp32
  only), `cpu` (no GPU model), `cpu_only` (a GPU model kept on the CPU on
  a Mac), `unavailable`, `unknown`. A pass that does not list `mps` never
  lands on the Apple GPU.
- **Reporting**: `liveaov doctor --mac` prints the table (by itself on a
  Mac). The GUI shows an "experimental" banner on a Mac; Submit stops on a
  pass that cannot run there and asks before running unverified ones.
- **fp16 / bf16 stay CUDA-only.** On the Apple GPU every pass runs fp32.

## Target: "Mac light"

What a Mac user should get first, all on the Apple GPU in fp32 unless noted:

| Need | Pass | Expected on Mac |
|---|---|---|
| Depth | Depth Anything v2, Video Depth Anything | Apple GPU |
| Depth (NC) | Depth Pro | Apple GPU |
| Normals (NC) | DSINE | Apple GPU |
| Mattes | SAM 3 (finds and tracks the objects) | CPU for now (the Apple GPU is off for it) |
| Mattes | ViTMatte / RVM / BiRefNet / MatAnyone2 (edges) | Apple GPU |
| Mattes | chroma key, screen pull | CPU |
| Cryptomatte | from the SAM 3 tracks | CPU |
| Motion | RAFT, MegaFlow | Apple GPU |
| Camera | COLMAP solve (pycolmap) | CPU |

Out of scope on Mac: the Gaussian splat background (gsplat is CUDA-only),
DepthCrafter and NormalCrafter (fp16 video diffusion), Marigold and
UniVid-X (very large models in their own CUDA environments), the SAM 3.1
detector (its own CUDA environment). LTX alpha runs in ComfyUI, which can
stay on a PC on the network.

### Gaps found while preparing this

1. **MoGe-3 does not run on a Mac as built.** Its sparse refiner uses
   FlexGEMM, which is pure Triton, and Triton has no macOS build. MoGe-3 is
   the default normals, the default depth (steady, in metres) takes its
   metres from it, and the camera's PnP fallback uses it.
   Options, in order of preference:
   - load MoGe-3 without the sparse refiner on Mac, behind a param (the
     vendored model supports `refiner=None`); output differs, so it needs
     an A/B against the refined result before it is offered;
   - point Mac users at Depth Anything v2 / Video Depth Anything for depth.
2. **No commercial-safe normals on Mac** until (1) is solved: DSINE is
   non-commercial.
3. `install.sh` installs `--extra all`, which includes the MoGe-3 git
   dependencies. Check on the first day that the install does not stop on
   them; if it does, the macOS install needs its own extra set.

## A Mac portable build

`scripts/build_portable.py` is Windows-only today: `python.exe` and the
`Lib/` layout, `.bat` launchers, `robocopy`, the cu128 torch index, a
PowerShell desktop shortcut. A Mac version needs:

- **Python**: the standalone CPython 3.11 for `aarch64-apple-darwin`
  (`uv python install 3.11` on the Mac gives the python-build-standalone
  build). Layout is `bin/python3` and `lib/python3.11/`, so the
  `EXTERNALLY-MANAGED` path and the interpreter path change. arm64 only:
  Intel Macs have no useful GPU for this.
- **torch from PyPI**, not the cu128 index (the PyPI macOS wheel has MPS).
- **Launchers as `.command` files** (double-clickable in Finder):
  `#!/bin/bash`, `cd "$(dirname "$0")"`, the same environment as the
  `.bat` files (`HF_HUB_CACHE`, `TORCH_HOME`, `PYTHONNOUSERSITE`,
  offline unless `LAOV_ONLINE=1`), saved executable (`chmod +x`).
- **`install_local.command`** with `rsync -a --delete` (or `ditto`) in
  place of `robocopy`; an alias on the Desktop is optional.
- **Gatekeeper**: a folder that arrives through a browser, AirDrop or an
  unzipped download carries the `com.apple.quarantine` attribute, and
  macOS then blocks the unsigned launchers and the `.so` / `.dylib` files
  inside Python. Plan: the install step clears it once for the copied
  folder (`xattr -dr com.apple.quarantine <folder>`), and the readme says
  so. Never ask users to switch Gatekeeper off. Signing and notarising
  (an Apple Developer ID) removes the step; that is a later decision.
- **Same model cache layout** (`models/hf/hub`, `models/torch`).
- **Builds on a Mac only**: the wheels are platform-specific, so the Mac
  folder cannot be built from the Windows machine.

## CI

Two draft workflows, run on pushes to the groundwork branch (and by hand
once they are on `main`):

- `.github/workflows/macos-cpu.yml` installs on a hosted Apple Silicon
  runner and runs the CPU tests with `LAOV_DEVICE=cpu`.
- `.github/workflows/camera-check.yml` installs the camera extra on a hosted
  Mac and on Linux and runs a short camera solve on a clip it draws itself.
  The same check runs on any machine: `uv run python scripts/camera_check.py`.

Hosted runners have no GPU to count on, so CI proves the install and the
logic, not MPS and not speed.

## First day with a Mac: checklist

Use a short public practice plate, at a small proxy, so each step takes
minutes.

1. Fresh clone, `./install.sh`. Note every step that fails (likely
   candidates: `oiio-python`, `pycolmap`, the MoGe-3 git dependencies).
2. `python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`.
3. `liveaov doctor --mac`: the GPU line says Apple GPU, the table prints.
4. `uv run pytest -m "not gpu"`: the same result as on Windows.
5. One run per Mac-light pass (Depth Anything v2, Video Depth Anything,
   SAM 3 + ViTMatte, Cryptomatte, RAFT, MegaFlow, camera): record time,
   peak memory (`torch.mps.driver_allocated_memory()` or Activity Monitor)
   and any error.
6. Run the same plate and settings on the PC and compare the EXRs channel
   by channel; call each pass GOOD / WEAK / FAILED and look at a contact
   sheet.
7. One pass with `LAOV_DEVICE=cpu` as a baseline. If a pass stops on an
   operator MPS does not have, retry with `PYTORCH_ENABLE_MPS_FALLBACK=1`
   to find which one (diagnostic only, not a default).
8. GUI: the banner shows; Submit with the splat ticked is stopped; Submit
   with the Mac-light set asks, then runs.
9. Set `mac_verified=True` only for the passes that came out GOOD, in one
   commit, and update the README compatibility row.
10. Then start on gap 1 (MoGe-3 without the refiner), behind a param.
