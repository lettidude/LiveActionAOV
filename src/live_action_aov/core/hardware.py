# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""What this machine can run: the card, the RAM, and the settings to propose.

Three small things, no model code:

- `detect()` reads the first CUDA card (name, memory) and the host RAM.
- `suggest()` turns that plus the plate into settings a person would
  otherwise have to know: the proxy size, the detector, and a one-line
  reason for each. It PROPOSES; the GUI and `liveaov doctor` show it and
  the user applies it.
- `estimate_ram_gb()` / `plan_windows()`: the executor keeps every pass's
  channels for every frame until the write, so host RAM grows with the
  frame count. A clip that does not fit is run in overlapping windows of
  frames (`executors.local`), sized here from the free RAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Proxy long edges the GUI offers, biggest first. None = the full plate.
PROXY_STEPS: tuple[int, ...] = (2816, 1920, 1280, 960)

#: Channels a full run keeps per frame (mattes, Cryptomatte, Z, N, P, flow,
#: splat...). The RAM estimate uses it when the job is not known yet.
DEFAULT_CHANNELS = 48
#: Models, torch, the reader and the Python process itself.
BASE_RAM_GB = 8.0
#: On top of the channels: mask stacks, the plate stack of the running pass, copies
#: made by the post-processors (measured: 55 GB for 100 frames x 50 channels at 2K).
RAM_OVERHEAD = 1.3
#: Frames repeated on each side of a window so temporal passes (flow, the
#: smoother) have their neighbours; the cut is in the middle of it.
WINDOW_OVERLAP = 16
MIN_WINDOW = 24


@dataclass(frozen=True)
class Hardware:
    gpu_name: str | None = None
    vram_gb: float | None = None  # the card's total memory
    ram_gb: float | None = None  # installed
    ram_free_gb: float | None = None  # available now

    def summary(self) -> str:
        gpu = (
            f"{self.gpu_name} ({self.vram_gb:.0f} GB)"
            if self.gpu_name and self.vram_gb
            else (self.gpu_name or "no CUDA card")
        )
        ram = f"{self.ram_gb:.0f} GB RAM" if self.ram_gb else "RAM unknown"
        return f"{gpu}, {ram}"


@dataclass(frozen=True)
class Suggestion:
    tier: str  # "quality" | "balanced" | "small" | "minimal" | "none"
    title: str
    proxy_long_edge: int | None  # None = run on the full plate
    detector: str  # "sam3.1" | "sam3"
    window_frames: int | None  # frames per window on a long clip; None = one go
    notes: tuple[str, ...] = field(default_factory=tuple)

    def one_line(self) -> str:
        proxy = "full plate" if self.proxy_long_edge is None else f"proxy {self.proxy_long_edge}"
        det = "SAM 3.1" if self.detector == "sam3.1" else "SAM 3"
        win = f", {self.window_frames}-frame windows" if self.window_frames else ""
        return f"{self.title}: {proxy}, {det}{win}"


def detect_ram() -> tuple[float | None, float | None]:
    """(installed, available) host RAM in GB."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return float(vm.total) / 2**30, float(vm.available) / 2**30
    except Exception:  # psutil comes with torch's stack; without it: unknown
        return None, None


def detect() -> Hardware:
    """The first CUDA card and the host RAM. Never raises."""
    name: str | None = None
    vram: float | None = None
    try:
        import torch

        if torch.cuda.is_available():
            name = str(torch.cuda.get_device_name(0))
            vram = float(torch.cuda.get_device_properties(0).total_memory) / 2**30
    except Exception:
        pass
    total, free = detect_ram()
    return Hardware(gpu_name=name, vram_gb=vram, ram_gb=total, ram_free_gb=free)


def proxy_size(resolution: tuple[int, int], long_edge: int | None) -> tuple[int, int]:
    """The size the passes run at: the plate, or its proxy."""
    w, h = int(resolution[0]), int(resolution[1])
    if long_edge is None or max(w, h) <= long_edge:
        return w, h
    s = long_edge / float(max(w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def estimate_ram_gb(
    resolution: tuple[int, int],
    n_frames: int,
    proxy_long_edge: int | None = None,
    channels: int = DEFAULT_CHANNELS,
) -> float:
    """Peak host RAM of a whole-clip run: every channel of every frame is
    float32 at the working size until the write, plus the fixed cost."""
    w, h = proxy_size(resolution, proxy_long_edge)
    per_frame = channels * w * h * 4 * RAM_OVERHEAD / 2**30
    return BASE_RAM_GB + per_frame * max(int(n_frames), 0)


def plan_windows(
    resolution: tuple[int, int],
    n_frames: int,
    ram_free_gb: float | None,
    proxy_long_edge: int | None = None,
    channels: int = DEFAULT_CHANNELS,
    overlap: int = WINDOW_OVERLAP,
) -> int | None:
    """Frames per window so the run fits in the free RAM, or None when the
    whole clip fits (or the RAM cannot be read)."""
    if ram_free_gb is None or n_frames <= MIN_WINDOW:
        return None
    if estimate_ram_gb(resolution, n_frames, proxy_long_edge, channels) <= 0.8 * ram_free_gb:
        return None
    w, h = proxy_size(resolution, proxy_long_edge)
    per_frame = channels * w * h * 4 * RAM_OVERHEAD / 2**30
    budget = 0.6 * ram_free_gb - BASE_RAM_GB
    fit = int(budget / per_frame) - 2 * overlap if per_frame > 0 else n_frames
    win = max(MIN_WINDOW, fit)
    return None if win >= n_frames else win


def window_spans(
    n_frames: int, window: int, overlap: int = WINDOW_OVERLAP
) -> list[tuple[int, int, int, int]]:
    """Cut `n_frames` in windows of `window` core frames. Each entry is
    (run_start, run_end, core_start, core_end), local 0-based, ends
    exclusive: the passes run on [run_start, run_end), which adds up to
    `overlap` frames of context on each side, and only the core frames
    [core_start, core_end) are written. Cores tile the clip exactly."""
    n, window = int(n_frames), max(int(window), 1)
    if n <= window:
        return [(0, n, 0, n)]
    out = []
    for a in range(0, n, window):
        b = min(a + window, n)
        out.append((max(0, a - overlap), min(n, b + overlap), a, b))
    # A short tail is folded into the window before it.
    if len(out) > 1 and out[-1][3] - out[-1][2] < max(overlap, window // 4):
        ra, _, ca, _ = out[-2]
        out[-2:] = [(ra, n, ca, n)]
    return out


def suggest(
    hw: Hardware,
    resolution: tuple[int, int],
    n_frames: int,
    *,
    sam31_installed: bool = False,
    channels: int = DEFAULT_CHANNELS,
) -> Suggestion:
    """Settings for this plate on this machine, with the reason for each."""
    notes: list[str] = []
    vram = hw.vram_gb
    if vram is None:
        return Suggestion(
            "none",
            "No CUDA card found",
            PROXY_STEPS[-1],
            "sam3",
            None,
            ("The passes need an NVIDIA card with CUDA.",),
        )
    if vram >= 30:
        tier, title, cap = "quality", "Quality", 2816
    elif vram >= 16:
        tier, title, cap = "balanced", "Balanced", 1920
    elif vram >= 11:
        tier, title, cap = "small", "Small GPU", 1280
    else:
        tier, title, cap = "minimal", "Minimal", 960
    long_edge = max(int(resolution[0]), int(resolution[1]))
    proxy: int | None = None if long_edge <= cap else cap
    # RAM: a long clip at a big proxy only fits in windows so short that the
    # overlap is most of the work. One proxy step down is the better trade
    # (same cameras, depth and ids; slightly softer matte edges).
    # (a step that barely shrinks the plate is not worth it: 2048 -> 1920).
    for step in [p for p in PROXY_STEPS if 1920 <= p <= 0.8 * min(cap, long_edge)]:
        win = plan_windows(resolution, n_frames, hw.ram_free_gb, proxy, channels)
        if win is None or win >= 4 * WINDOW_OVERLAP:
            break
        proxy = step
        notes.append(
            f"{n_frames} frames: proxy {step} instead of a bigger one, so the clip "
            f"fits the {hw.ram_free_gb:.0f} GB of free RAM without tiny windows."
        )
    if proxy is None:
        notes.append(f"The plate ({long_edge} px) fits the {vram:.0f} GB card at full size.")
    else:
        notes.append(
            f"Proxy {proxy}: a {vram:.0f} GB card runs the models at that size; "
            "the EXRs are still written at plate size."
        )
    if sam31_installed and vram >= 22:
        detector = "sam3.1"
        notes.append("SAM 3.1 is installed: it keeps identities where SAM 3 swaps them.")
    else:
        detector = "sam3"
        notes.append(
            "SAM 3: SAM 3.1 wants a 24 GB card."
            if sam31_installed
            else "SAM 3: SAM 3.1 is not installed (`liveaov setup-sam31`)."
        )
    window = plan_windows(resolution, n_frames, hw.ram_free_gb, proxy, channels)
    if window is not None:
        need = estimate_ram_gb(resolution, n_frames, proxy, channels)
        notes.append(
            f"{n_frames} frames need about {need:.0f} GB of RAM in one go "
            f"({hw.ram_free_gb:.0f} GB free): run in windows of {window} frames."
        )
    return Suggestion(tier, title, proxy, detector, window, tuple(notes))


__all__ = [
    "BASE_RAM_GB",
    "DEFAULT_CHANNELS",
    "MIN_WINDOW",
    "PROXY_STEPS",
    "WINDOW_OVERLAP",
    "Hardware",
    "Suggestion",
    "detect",
    "detect_ram",
    "estimate_ram_gb",
    "plan_windows",
    "proxy_size",
    "suggest",
    "window_spans",
]
