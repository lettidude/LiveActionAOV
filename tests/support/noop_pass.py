"""NoOpPass — zero-valued depth channel output, no model.

Phase 0's proof that the full pipeline (IO + core + executor + writer) works
end-to-end without any AI inference. Lives under `tests/` rather than
`passes/` so it doesn't ship with the core wheel.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CH_Z


class NoOpPass(UtilityPass):
    name = "noop"
    version = "0.1.0"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only pass.")
    pass_type = PassType.GEOMETRIC
    temporal_mode = TemporalMode.PER_FRAME
    produces_channels = [ChannelSpec(name=CH_Z, description="Zero-valued depth (test).")]

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        # Expect (N, H, W, C). Emit a single (H, W) zero-valued depth.
        if tensor.ndim == 4:
            h, w = tensor.shape[1:3]
        elif tensor.ndim == 3:
            h, w = tensor.shape[:2]
        else:
            raise ValueError(f"Unexpected tensor shape {tensor.shape}")
        return {CH_Z: np.zeros((h, w), dtype=np.float32)}


__all__ = ["NoOpPass"]
