"""VRAM estimation + planning.

v1 is a thin shim: `query_available_gb()` returns CUDA free VRAM via torch
when available, else a large sentinel so tests and CPU-only smoke runs work
unchanged. Per-pass estimates come from each pass's `vram_estimate_gb_fn`.

The scheduler pre-flight checks estimated vs available VRAM (Phase 6 wires
this into an auto-downscale path). Phase 0 just defines the shapes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VramPlan:
    """Result of pre-flight VRAM planning for a job.

    `fits` is the decision; `per_pass_gb` is the breakdown for logging.
    """

    available_gb: float
    per_pass_gb: dict[str, float]
    fits: bool
    suggestion: str = ""


def query_available_gb() -> float:
    """Return available CUDA VRAM in GB.

    Returns a very large sentinel (1024 GB) when CUDA isn't available, so
    CPU-only tests and smoke runs don't trip the planner.
    """
    try:
        import torch
    except ImportError:
        return 1024.0
    if not torch.cuda.is_available():
        return 1024.0
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


def plan(
    per_pass_gb: dict[str, float],
    available_gb: float | None = None,
    headroom_gb: float = 1.0,
) -> VramPlan:
    """Return a VramPlan for running the given passes concurrently (worst case).

    v1 sums all pass estimates; v2+ can refine for passes that don't coexist
    (e.g. flow runs before depth, they don't both hold VRAM at once).
    """
    avail = query_available_gb() if available_gb is None else available_gb
    required = sum(per_pass_gb.values())
    fits = (required + headroom_gb) <= avail
    suggestion = (
        ""
        if fits
        else (
            f"Estimated {required:.1f} GB needed + {headroom_gb:.1f} GB headroom, "
            f"only {avail:.1f} GB available. Downscale via resize.target or split passes."
        )
    )
    return VramPlan(
        available_gb=avail,
        per_pass_gb=dict(per_pass_gb),
        fits=fits,
        suggestion=suggestion,
    )


__all__ = ["VramPlan", "plan", "query_available_gb"]
