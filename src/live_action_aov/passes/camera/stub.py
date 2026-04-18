"""CameraPass stub — declared in v1, implemented in v2a.

This class exists so: (1) the architecture flows through a non-EXR pass and
a non-EXR sidecar from day one, and (2) the `PassType.CAMERA` enum value is
exercised by the registry at load time.

Running it raises `NotImplementedError` with a pointer to v2a (§20).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    License,
    PassType,
    SidecarSpec,
    TemporalMode,
    UtilityPass,
)


class CameraPassStub(UtilityPass):
    name = "camera_stub"
    version = "0.0.0"
    license = License(spdx="MIT", commercial_use=True, notes="Stub only; no inference.")
    pass_type = PassType.CAMERA
    temporal_mode = TemporalMode.VIDEO_CLIP
    produces_channels: list = []
    produces_sidecars: list[SidecarSpec] = [
        SidecarSpec(name="camera", format="json"),
        SidecarSpec(name="camera_nuke", format="nk"),
        SidecarSpec(name="camera_abc", format="abc"),
    ]

    def preprocess(self, frames: np.ndarray) -> Any:  # pragma: no cover — stub
        raise NotImplementedError(
            "Camera pass lands in v2a. See utility_passes_design_notes.md §20."
        )

    def infer(self, tensor: Any) -> Any:  # pragma: no cover — stub
        raise NotImplementedError("Camera pass lands in v2a.")

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:  # pragma: no cover — stub
        raise NotImplementedError("Camera pass lands in v2a.")


__all__ = ["CameraPassStub"]
