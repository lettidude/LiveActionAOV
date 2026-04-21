# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""EXR sidecar writer.

Writes a multi-channel EXR with channels ordered against
`CANONICAL_CHANNEL_ORDER` (§11.3 trap 8 — Nuke's reader has specific
expectations). Unknown-but-valid channels (e.g. dynamic `mask.<concept>`)
are appended in insertion order after the canonical channels.

Metadata is written as EXR custom attributes with slash-namespaced keys
(e.g. `liveaov/depth/model`), which Nuke displays as a nested tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.channels import CANONICAL_CHANNEL_ORDER
from live_action_aov.io.oiio_io import write_exr
from live_action_aov.io.writers.base import SidecarWriter


class ExrSidecarWriter(SidecarWriter):
    """Write sidecar EXR files matching the Nuke/VFX channel contract."""

    format_tag = "exr"

    def write_frame(
        self,
        out_path: Path,
        channels: dict[str, np.ndarray],
        *,
        attrs: dict[str, Any] | None = None,
        pixel_aspect: float = 1.0,
    ) -> None:
        if not channels:
            raise ValueError("ExrSidecarWriter requires at least one channel")
        ordered_names = _order_channels(list(channels.keys()))
        first = channels[ordered_names[0]]
        h, w = first.shape[:2]

        stack = np.empty((h, w, len(ordered_names)), dtype=np.float32)
        for i, name in enumerate(ordered_names):
            arr = np.asarray(channels[name], dtype=np.float32)
            if arr.shape[:2] != (h, w):
                raise ValueError(f"Channel '{name}' has shape {arr.shape} but expected {(h, w)}")
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            stack[..., i] = arr

        write_exr(
            out_path,
            stack,
            ordered_names,
            attrs=attrs,
            pixel_aspect=pixel_aspect,
            compression="zip",
            dtype="float32",
        )


def _order_channels(names: list[str]) -> list[str]:
    """Order channels against CANONICAL_CHANNEL_ORDER; unknowns follow."""
    canonical_set = set(CANONICAL_CHANNEL_ORDER)
    canonical_kept = [n for n in CANONICAL_CHANNEL_ORDER if n in names]
    extras = [n for n in names if n not in canonical_set]
    return canonical_kept + extras


__all__ = ["ExrSidecarWriter"]
