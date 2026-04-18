"""SidecarWriter ABC.

Writers are single-frame-aware (they get called once per frame with the
channel dict from `postprocess()`) and metadata-aware (per-pass metadata is
merged into the sidecar header / payload).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class SidecarWriter(ABC):
    """Abstract writer for a single sidecar output type."""

    #: Short format tag used in `Shot.sidecars[tag]`.
    format_tag: str = "exr"

    @abstractmethod
    def write_frame(
        self,
        out_path: Path,
        channels: dict[str, np.ndarray],
        *,
        attrs: dict[str, Any] | None = None,
        pixel_aspect: float = 1.0,
    ) -> None:
        """Write a single frame's channels to `out_path`.

        `channels`: {channel_name: (H, W) float32 array}. The writer packs
        these into its native format (layer paths for EXR, nested keys for
        JSON, etc.). Channel names follow the naming contract in
        `live_action_aov.io.channels`.
        """


__all__ = ["SidecarWriter"]
