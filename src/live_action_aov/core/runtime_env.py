# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Process-wide network defaults, applied by the CLI/GUI entry points.

These MUST run before `huggingface_hub` is first imported — it reads its
timeout constants at import time. The entry points call
`apply_hf_network_defaults()` as their very first statement; the launch
scripts also set the same env vars so the defaults hold no matter how the
tool is started.

Rationale (real failures seen in a batch run):
- The default 10 s download timeout aborts large cold-cache weight
  downloads (BiRefNet, SAM 3) on a slow link — a late, expensive failure.
- The default 10 s ETag/HEAD timeout used on a *warm* cache means a brief
  network blip can abort a job whose weights are already on disk.

For a content machine / unreliable link, set `HF_HUB_OFFLINE=1` once the
cache is warm (after one successful run that pulled every model, incl. the
secondary repos some passes depend on, e.g. NormalCrafter/DepthCrafter
pulling stable-video-diffusion) to run with no network at all.
"""

from __future__ import annotations

import os


def apply_hf_network_defaults() -> None:
    """Set resilient Hugging Face timeouts if the user hasn't overridden them.

    `setdefault` so explicit env (or `HF_HUB_OFFLINE=1`) always wins.
    """
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


__all__ = ["apply_hf_network_defaults"]
