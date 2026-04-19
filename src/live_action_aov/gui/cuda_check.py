"""CUDA availability probe — called once at GUI startup.

Every neural pass in this tool (RAFT, DA-V2, DepthCrafter, SAM3, …)
needs a CUDA GPU to run at usable speed. The fp16 passes hard-fail on
CPU (PyTorch doesn't implement fp16 kernels for CPU); the fp32 passes
will technically run but take hours per frame. Either way the user is
going to lose time, so we detect early and warn loudly.

This module is deliberately tiny + import-safe:

- `cuda_state()` returns a structured `CudaState` dataclass the UI
  renders as a banner / status bar line / dialog body.
- Importing this file doesn't trigger any torch imports at module-
  load time; the check runs lazily inside `cuda_state()`. Keeps
  `liveaov-gui --help` style probes fast.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CudaState:
    available: bool
    torch_version: str
    torch_built_for_cuda: bool
    device_name: str | None
    device_count: int
    advisory: str  # Human-readable summary for the UI.


def cuda_state() -> CudaState:
    """Probe torch + CUDA without crashing if torch itself is broken.

    Returns a `CudaState` with `available=False` on any failure path
    (no torch installed, CPU-only build, driver missing, etc.) and an
    `advisory` string the UI surfaces verbatim.
    """
    try:
        import torch
    except ImportError:
        return CudaState(
            available=False,
            torch_version="<not installed>",
            torch_built_for_cuda=False,
            device_name=None,
            device_count=0,
            advisory=(
                "PyTorch is not installed. The neural passes (flow, depth, "
                "normals, matte) cannot run without it. See install.bat / "
                "install.sh in the project root."
            ),
        )

    ver = str(getattr(torch, "__version__", "?"))
    # `+cpu` / `+cu124` / `+cu121` etc. is appended by the upstream
    # wheel tag; presence of `+cu` is the reliable cue that the build
    # has CUDA kernels compiled in.
    built_for_cuda = "+cu" in ver

    try:
        available = bool(torch.cuda.is_available())
    except Exception:
        available = False

    if not available:
        if not built_for_cuda:
            advisory = (
                f"CPU-only PyTorch detected ({ver}). Neural passes "
                "cannot run until a CUDA build of torch is installed.\n\n"
                "Fix (Windows PowerShell / cmd, from project root):\n"
                "    .venv\\Scripts\\pip.exe uninstall -y torch torchvision\n"
                "    .venv\\Scripts\\pip.exe install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cu124\n\n"
                "Linux/macOS: same command, use `.venv/bin/pip` instead."
            )
        else:
            advisory = (
                f"PyTorch {ver} has CUDA support compiled in, but no CUDA "
                "device is visible. Check that the NVIDIA driver is "
                "installed (`nvidia-smi`) and that the GPU isn't being "
                "blocked by another process."
            )
        return CudaState(
            available=False,
            torch_version=ver,
            torch_built_for_cuda=built_for_cuda,
            device_name=None,
            device_count=0,
            advisory=advisory,
        )

    try:
        count = int(torch.cuda.device_count())
        name = torch.cuda.get_device_name(0) if count else None
    except Exception:
        count = 0
        name = None

    return CudaState(
        available=True,
        torch_version=ver,
        torch_built_for_cuda=True,
        device_name=name,
        device_count=count,
        advisory=f"CUDA ready — {name} ({count} device{'s' if count != 1 else ''}).",
    )


__all__ = ["CudaState", "cuda_state"]
