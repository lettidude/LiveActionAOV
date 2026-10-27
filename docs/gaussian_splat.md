# Camera track + Gaussian splat

Two passes in the **Camera** category of the Passes tab:

| Pass | What you get | Needs |
|---|---|---|
| Camera track (`camera_pycolmap`) | `camera_track/<shot>.camera.nk` (Nuke Camera2), `.camera.json`, `.points.ply`, `<shot>_colmap/` | nothing extra (pycolmap is in the normal install) |
| Gaussian splat (`splat_gsplat`) | EXR layer `splat` = clean plate (`splat.R/G/B`), coverage (`splat.A`), depth (`splat.Z`) + `camera_track/<shot>.splat.ply` | the **gsplat wheel** (below), an NVIDIA GPU |

Ticking *Gaussian splat* also runs the camera track (the splat is trained on
that solve). Turn **Matte** on in the same job and every SAM 3 object
(people, cars...) is masked out of BOTH the camera features and the splat
training — so `splat.rgb` is the scene *without* them: a clean plate built
from the frames where the camera saw behind them.

## Installing gsplat (one time)

gsplat contains CUDA kernels. Its own prebuilt Windows wheels stop at
torch 2.4 / CUDA 12.4 and do not run on RTX 50xx; LiveActionAOV uses
torch 2.11 / CUDA 12.8. So the tool installs gsplat from a wheel built for
*its* torch, kept in the `wheels\` folder. `install.bat` / `update.bat`
re-install every wheel in `wheels\` after syncing dependencies.

**Option A — download (no compilers needed).** Get
`gsplat-1.5.3-cp311-cp311-win_amd64.whl` from the
[GitHub Releases](https://github.com/lettidude/LiveActionAOV/releases) page
of your LiveActionAOV version, put it in `wheels\`, run `update.bat`.
That wheel is built for every NVIDIA architecture from RTX 20xx (Turing)
to RTX 50xx (Blackwell), plus PTX for newer cards — it is not tied to the
machine that built it, only to Windows + this tool's Python 3.11 / torch.

**Option B — build it yourself.** Needs, on the build machine only:
- NVIDIA CUDA Toolkit **12.8** (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`)
- Visual Studio **2022** Build Tools, workload "Desktop development with C++"
- git

Then:

```bat
scripts\build_gsplat_wheel.bat
```

It clones gsplat v1.5.3 into `.build\`, compiles the all-architecture wheel
into `wheels\` (30-60 min; `set TORCH_CUDA_ARCH_LIST=12.0` first for a
~10 min build that only runs on RTX 50xx) and installs it.

**Linux.** `uv pip install --no-build-isolation gsplat==1.5.3` inside the
tool's venv compiles it with the system CUDA toolkit (12.8 to match torch).

Check: `.venv\Scripts\python.exe -c "from gsplat.cuda._backend import _C; print(_C is not None)"`.

## What to expect (measured, ACTIONVFX PF0081, 64 frames 2048x1080 dolly, RTX 5090)

| | |
|---|---|
| Camera solve | 64/64 frames, 0.99 px mean reprojection, checked inside Nuke 16 (`scripts/nuke_check_camera.py`) |
| Splat training | 61 s, 7000 steps at 2048x1080, ~31k gaussians |
| Quality on frames held OUT of training | 33.9 dB PSNR |
| Total job (SAM 3 + camera + splat) | ~4 min |

## Limits (read before trusting a clean plate)

- It only reconstructs what the plate saw. Anything never revealed (behind
  a person who never moves out of the way, outside the camera move) is
  empty or smeared: check `splat.A` and the plate before using a patch.
- Objects that move but are not masked fade out if they are absent in most
  frames (the gun picked up in PF0081 vanishes, leaving a faint smear).
- Needs parallax: locked-off and nodal-pan shots cannot be solved.
- Units are arbitrary (same as the camera); `splat.Z` is camera-space depth
  in those units, 0 where `splat.A` = 0.
- One camera per shot; lens distortion is not modelled (pinhole).
