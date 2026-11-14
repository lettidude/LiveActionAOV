"""Nuke vector aliases use Nuke's axes (u right, v UP).

motion.* / back.* stay image space (y down); forward.v / backward.v are their
negation, so VectorBlur / IDistort move pixels the right way. Fast, no model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.channels import (
    CH_BACK_X,
    CH_BACK_Y,
    CH_BACKWARD_U,
    CH_BACKWARD_V,
    CH_FLOW_CONFIDENCE,
    CH_FORWARD_U,
    CH_FORWARD_V,
    CH_MOTION_X,
    CH_MOTION_Y,
)
from live_action_aov.passes.flow.raft import RAFTPass, _nuke_aliases


def test_nuke_aliases_negate_v_only() -> None:
    fx = np.full((4, 5), 3.0, np.float32)
    fy = np.full((4, 5), 2.0, np.float32)  # image space: moving DOWN
    bx = np.full((4, 5), -3.0, np.float32)
    by = np.full((4, 5), -2.0, np.float32)
    out = _nuke_aliases(fx, fy, bx, by)
    assert np.array_equal(out[CH_FORWARD_U], fx)
    assert np.array_equal(out[CH_FORWARD_V], -fy)  # Nuke: moving down = negative v
    assert np.array_equal(out[CH_BACKWARD_U], bx)
    assert np.array_equal(out[CH_BACKWARD_V], -by)
    assert all(a.dtype == np.float32 for a in out.values())
    # inputs untouched
    assert float(fy[0, 0]) == 2.0 and float(by[0, 0]) == -2.0


class _Reader:
    def read_frame(self, f: int) -> tuple[np.ndarray, dict[str, Any]]:
        return np.zeros((4, 5, 3), np.float32), {}


def test_run_shot_writes_v_up_aliases() -> None:
    """The run_shot path (the one the executor uses) applies the same convention."""
    p = RAFTPass()
    fx = np.full((4, 5), 1.5, np.float32)
    fy = np.full((4, 5), 4.0, np.float32)
    p.preprocess = lambda pair: None  # type: ignore[method-assign]
    p.infer = lambda t: None  # type: ignore[method-assign]
    p.postprocess = lambda t: {  # type: ignore[method-assign]
        CH_MOTION_X: fx,
        CH_MOTION_Y: fy,
        CH_BACK_X: -fx,
        CH_BACK_Y: -fy,
        CH_FLOW_CONFIDENCE: np.ones((4, 5), np.float32),
    }
    out = p.run_shot(_Reader(), (1, 3))
    for f in (1, 2, 3):
        ch = out[f]
        assert np.array_equal(ch[CH_FORWARD_U], ch[CH_MOTION_X])
        assert np.array_equal(ch[CH_FORWARD_V], -ch[CH_MOTION_Y])
        assert np.array_equal(ch[CH_BACKWARD_U], ch[CH_BACK_X])
        assert np.array_equal(ch[CH_BACKWARD_V], -ch[CH_BACK_Y])
    # a frame with real vertical motion: image-space +4 (down) -> Nuke -4 (down)
    assert float(out[1][CH_MOTION_Y][0, 0]) == 4.0
    assert float(out[1][CH_FORWARD_V][0, 0]) == -4.0
    assert float(out[2][CH_BACKWARD_V][0, 0]) == 4.0
