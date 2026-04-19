"""Pass-type-aware resize (design §8).

The critical insight: different output types need different resize rules.
Depth wants bilinear; hard masks want nearest (bilinear invents fake class
IDs — trap 6 in §11.3); normals want bilinear *then renormalize* (trap 2);
flow vectors want bilinear *then vector-scale* (trap 1).

This module provides those rules as explicit `channel_type` keyword choices
so pass authors can never accidentally use the wrong interpolation.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel


class ResizeMode(str, Enum):
    FIT_LONG_EDGE = "fit_long_edge"
    FIT_SHORT_EDGE = "fit_short_edge"
    FRACTION = "fraction"
    EXACT = "exact"


class ResizeParams(BaseModel):
    mode: ResizeMode = ResizeMode.FIT_LONG_EDGE
    target: int | tuple[int, int] = 1920
    max_vram_gb: float = 16.0
    upscale_back: bool = True
    strategy: Literal["downscale", "tile", "auto"] = "downscale"


ChannelType = Literal["continuous", "discrete", "normal_vector", "flow_vector"]


def target_size(src_size: tuple[int, int], params: ResizeParams) -> tuple[int, int]:
    """Compute (w, h) target size given source size and ResizeParams."""
    sw, sh = src_size
    if params.mode is ResizeMode.EXACT:
        t = params.target
        return (t, t) if isinstance(t, int) else t
    if params.mode is ResizeMode.FRACTION:
        f = params.target if isinstance(params.target, (int, float)) else params.target[0]
        return max(1, int(sw * float(f))), max(1, int(sh * float(f)))
    t = params.target if isinstance(params.target, int) else max(params.target)
    long_edge, short_edge = (sw, sh) if sw >= sh else (sh, sw)
    if params.mode is ResizeMode.FIT_LONG_EDGE:
        scale = t / long_edge
    else:  # FIT_SHORT_EDGE
        scale = t / short_edge
    return max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))


def downscale(
    frame: np.ndarray,
    params: ResizeParams,
    model_constraints: dict | None = None,
) -> np.ndarray:
    """Resize `frame` for inference input.

    `model_constraints` may include `multiple_of` (e.g. 14 for DINOv2).
    """
    h, w = frame.shape[:2]
    tw, th = target_size((w, h), params)
    if model_constraints:
        mo = int(model_constraints.get("multiple_of", 1))
        if mo > 1:
            tw = max(mo, (tw // mo) * mo)
            th = max(mo, (th // mo) * mo)
    if (tw, th) == (w, h):
        return frame
    return _bilinear_resize(frame, (tw, th))


def upscale(
    result: np.ndarray,
    target: tuple[int, int],
    channel_type: ChannelType = "continuous",
) -> np.ndarray:
    """Upscale inference output to plate resolution, respecting channel type.

    - continuous      → bilinear (depth, soft alpha)
    - discrete        → nearest (hard masks, semantic IDs — trap 6)
    - normal_vector   → bilinear, then renormalize N/||N|| (trap 2)
    - flow_vector     → bilinear, then scale vectors by upscale ratio (trap 1)
    """
    tw, th = target
    h, w = result.shape[:2]
    if (tw, th) == (w, h):
        return _postprocess(result, channel_type, scale=(1.0, 1.0))

    if channel_type == "discrete":
        resized = _nearest_resize(result, (tw, th))
    else:
        resized = _bilinear_resize(result, (tw, th))
    return _postprocess(resized, channel_type, scale=(tw / w, th / h))


def _postprocess(
    arr: np.ndarray, channel_type: ChannelType, scale: tuple[float, float]
) -> np.ndarray:
    if channel_type == "normal_vector":
        return _renormalize_vectors(arr)
    if channel_type == "flow_vector":
        return _scale_flow_vectors(arr, scale)
    return arr


# ---------------------------------------------------------------------------
# Intrinsics scaling (design §8.4 — THE critical DSINE bug to avoid)
# ---------------------------------------------------------------------------


def scale_intrinsics(
    intrinsics: dict,
    from_res: tuple[int, int],
    to_res: tuple[int, int],
) -> dict:
    """Scale fx, fy, cx, cy when resizing between resolutions.

    Not doing this is the root cause of "results vary with resolution"
    bugs in DSINE-class normal estimators (design §11.3, trap 3).
    """
    scale_x = to_res[0] / from_res[0]
    scale_y = to_res[1] / from_res[1]
    out = dict(intrinsics)
    if "fx" in out:
        out["fx"] = float(out["fx"]) * scale_x
    if "fy" in out:
        out["fy"] = float(out["fy"]) * scale_y
    if "cx" in out:
        out["cx"] = float(out["cx"]) * scale_x
    if "cy" in out:
        out["cy"] = float(out["cy"]) * scale_y
    return out


# ---------------------------------------------------------------------------
# Primitive interpolators (NumPy, no torch dep — Phase 0 needs to work
# without a GPU). Real passes will reach for torch.nn.functional.interpolate
# via their own preprocessing.
# ---------------------------------------------------------------------------


def _bilinear_resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize (H, W, C) → (new_h, new_w, C) in pure numpy."""
    tw, th = size
    if arr.ndim == 2:
        arr_ = arr[..., None]
        squeezed = True
    else:
        arr_ = arr
        squeezed = False
    h, w, c = arr_.shape

    # Normalized grid in source pixel coordinates.
    xs = np.linspace(0, w - 1, tw, dtype=np.float32)
    ys = np.linspace(0, h - 1, th, dtype=np.float32)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (xs - x0).astype(np.float32)
    wy = (ys - y0).astype(np.float32)

    # Gather with broadcasting.
    Ia = arr_[np.ix_(y0, x0)]
    Ib = arr_[np.ix_(y0, x1)]
    Ic = arr_[np.ix_(y1, x0)]
    Id = arr_[np.ix_(y1, x1)]

    wx_b = wx[None, :, None]
    wy_b = wy[:, None, None]

    top = Ia * (1 - wx_b) + Ib * wx_b
    bot = Ic * (1 - wx_b) + Id * wx_b
    out = top * (1 - wy_b) + bot * wy_b
    return out[..., 0] if squeezed else out


def _nearest_resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tw, th = size
    if arr.ndim == 2:
        arr_ = arr[..., None]
        squeezed = True
    else:
        arr_ = arr
        squeezed = False
    h, w, _ = arr_.shape
    xs = np.clip(np.round(np.linspace(0, w - 1, tw)).astype(np.int32), 0, w - 1)
    ys = np.clip(np.round(np.linspace(0, h - 1, th)).astype(np.int32), 0, h - 1)
    out = arr_[np.ix_(ys, xs)]
    return out[..., 0] if squeezed else out


def _renormalize_vectors(arr: np.ndarray) -> np.ndarray:
    """Renormalize per-pixel vectors to unit length. `arr` is (H, W, 3)."""
    if arr.shape[-1] != 3:
        return arr
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return arr / norm


def _scale_flow_vectors(arr: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    """Scale flow magnitudes by upscale ratio (trap 1).

    Expects a 2-channel (u, v) array (H, W, 2) — (u, v) in pixels.
    """
    if arr.shape[-1] != 2:
        return arr
    sx, sy = scale
    out = arr.astype(np.float32, copy=True)
    out[..., 0] *= sx
    out[..., 1] *= sy
    return out


__all__ = [
    "ChannelType",
    "ResizeMode",
    "ResizeParams",
    "downscale",
    "scale_intrinsics",
    "target_size",
    "upscale",
]
