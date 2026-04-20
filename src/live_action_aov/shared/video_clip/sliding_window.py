# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Sliding-window planning + weighted overlap stitching.

Used by VIDEO_CLIP passes (DepthCrafter, NormalCrafter) to handle clips
longer than the model's native window size. The stitching uses a trapezoid
per-window weight — flat=1 in the stable interior, linear ramp across the
overlap region at each edge — so adjacent windows crossfade smoothly in the
overlap and clip endpoints aren't attenuated (the first window has no
ramp-up on its leading edge; symmetrically for the last window).

All math lives here and is side-effect free so tests can assert stitching
behavior without the diffusers/torch stack.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def plan_window_starts(n_frames: int, window: int, overlap: int) -> list[int]:
    """Return the global start indices for a sliding window covering `n_frames`.

    Guarantees:
      - Every frame in `[0, n_frames)` is covered by at least one window.
      - Windows have exact length `window` (clamped to `n_frames` when
        smaller). The last window ends exactly at `n_frames` — i.e. it
        may shift earlier than `stride * k` to avoid running off the end.
      - No duplicate starts (e.g. when stride evenly tiles the clip).

    Example: `plan_window_starts(250, 110, 25)` → `[0, 85, 140]` (windows
    cover frames 0..109, 85..194, 140..249; last window backtracks to
    end exactly at 250).
    """
    if n_frames <= 0:
        return []
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if overlap < 0 or overlap >= window:
        raise ValueError(f"overlap must be in [0, window), got {overlap}")
    if n_frames <= window:
        return [0]
    stride = window - overlap
    starts: list[int] = []
    s = 0
    while s + window < n_frames:
        starts.append(s)
        s += stride
    # Final window ends exactly at n_frames; may overlap the previous by more
    # than `overlap` if the clip length isn't a clean multiple of the stride.
    last = n_frames - window
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def trapezoid_weight(window: int, overlap: int) -> np.ndarray:
    """Per-frame weight for a single window (shape `(window,)`).

    Interior weights are 1. Within `overlap` frames of each edge the weight
    ramps linearly from `1/overlap` at the extreme edge up to 1 at the
    interior. When `overlap == 0`, all weights are 1 (hard handover).

    The rationale: in the overlap between two adjacent windows, one window's
    right-edge ramp-down and the other's left-edge ramp-up sum to a
    constant, so the weighted average is a clean linear crossfade.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    w = np.ones(window, dtype=np.float32)
    if overlap == 0:
        return w
    ramp = max(overlap, 1)
    for k in range(window):
        d_start = k + 1
        d_end = window - k
        w[k] = min(1.0, d_start / ramp, d_end / ramp)
    return w


def stitch_windowed_predictions(
    predictions: Sequence[np.ndarray],
    starts: Sequence[int],
    n_frames: int,
    overlap: int,
    *,
    endpoint_unramped: bool = True,
) -> np.ndarray:
    """Stitch per-window predictions into a single (n_frames, ...) array.

    - `predictions[i]` has leading axis equal to the window length.
    - All windows must share the same trailing shape.
    - When `endpoint_unramped=True` (default), the leading ramp of the first
      window and the trailing ramp of the last window are replaced with 1.0
      so the absolute clip endpoints aren't attenuated.
    - Frame `t` gets the weighted average of every window covering it,
      using the trapezoid weight from `trapezoid_weight()`.
    """
    if len(predictions) == 0:
        raise ValueError("No predictions to stitch")
    if len(predictions) != len(starts):
        raise ValueError(
            f"predictions and starts length mismatch: {len(predictions)} vs {len(starts)}"
        )

    trailing_shape = predictions[0].shape[1:]
    out_shape = (n_frames, *trailing_shape)
    acc = np.zeros(out_shape, dtype=np.float32)
    wt = np.zeros(n_frames, dtype=np.float32)

    n_windows = len(predictions)
    for i, (pred, start) in enumerate(zip(predictions, starts, strict=True)):
        if pred.shape[1:] != trailing_shape:
            raise ValueError(f"Window {i} trailing shape {pred.shape[1:]} != {trailing_shape}")
        window = pred.shape[0]
        w = trapezoid_weight(window, overlap)
        if endpoint_unramped:
            # First window's leading ramp → 1 (nothing overlaps on the left).
            if i == 0:
                ramp = min(overlap, window)
                w[:ramp] = 1.0
            # Last window's trailing ramp → 1 (nothing overlaps on the right).
            if i == n_windows - 1:
                ramp = min(overlap, window)
                if ramp > 0:
                    w[-ramp:] = 1.0
        for k in range(window):
            g = start + k
            if 0 <= g < n_frames:
                acc[g] += pred[k] * w[k]
                wt[g] += w[k]

    # Broadcast weight to the full trailing shape.
    wt_shape = (n_frames,) + (1,) * len(trailing_shape)
    return (acc / np.maximum(wt, 1e-6).reshape(wt_shape)).astype(np.float32)


__all__ = ["plan_window_starts", "stitch_windowed_predictions", "trapezoid_weight"]
