# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Regression tests for the viewport proxy-resize.

The old stride-then-crop dropped the right/bottom of non-stride-multiple
aspect ratios (a 3840×1536 ultrawide lost ~20% off the right, shifting the
frame off-centre). These pin full-extent sampling.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.gui.preview_loader import _proxy_resize


def test_downsample_hits_long_edge_and_keeps_aspect() -> None:
    px = np.zeros((1536, 3840, 3), dtype=np.float32)  # 2.5:1 ultrawide
    out = _proxy_resize(px, long_edge=1024)
    h, w = out.shape[:2]
    assert max(h, w) == 1024
    # Aspect preserved within a pixel of the 2.5:1 source.
    assert abs((w / h) - (3840 / 1536)) < 0.01


def test_no_right_or_bottom_crop() -> None:
    # Mark the far-right column and bottom row; full-extent sampling must
    # carry those edges into the downsample (the old crop lost them).
    px = np.zeros((1536, 3840, 3), dtype=np.float32)
    px[:, -1, 0] = 1.0  # right edge, red
    px[-1, :, 1] = 1.0  # bottom edge, green
    out = _proxy_resize(px, long_edge=1024)
    assert out[:, -1, 0].max() == 1.0, "right edge was cropped"
    assert out[-1, :, 1].max() == 1.0, "bottom edge was cropped"


def test_small_image_passthrough() -> None:
    px = np.zeros((200, 320, 3), dtype=np.float32)
    out = _proxy_resize(px, long_edge=1024)
    assert out.shape == px.shape  # below long_edge -> untouched
