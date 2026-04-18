"""Display transform: clip-wide exposure, tonemap, EOTF."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.io.display_transform import DisplayTransform


def test_auto_exposure_returns_clip_wide_single_ev() -> None:
    # Two frames with identical median luminance; the computed EV should be
    # a single value, not per-frame (trap 4).
    frame = np.full((32, 32, 3), 0.09, dtype=np.float32)  # half a stop under target 0.18
    dt = DisplayTransform()
    analysis = dt.analyze_clip(
        [frame, frame], DisplayTransformParams(), working_space="acescg"
    )
    assert "ev" in analysis
    # 0.18 / 0.09 = 2; log2(2) = 1
    assert analysis["ev"] == pytest.approx(1.0, abs=1e-4)


def test_manual_exposure_overrides_auto() -> None:
    dt = DisplayTransform()
    analysis = dt.analyze_clip(
        [np.full((32, 32, 3), 0.5, dtype=np.float32)],
        DisplayTransformParams(manual_exposure_ev=-0.7),
    )
    assert analysis["ev"] == -0.7
    assert analysis["source"] == "manual"


def test_apply_is_cheap_and_clamps() -> None:
    dt = DisplayTransform()
    frame = np.linspace(0, 4, 1024, dtype=np.float32).reshape(32, 32, 1)
    out = dt.apply(frame, DisplayTransformParams(), analysis={"ev": 0.0})
    assert out.max() <= 1.0 + 1e-5
    assert out.min() >= 0.0
