# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Binary masks stored as their bounding-box crop.

Why: SAM 3 keeps one hard mask per object per frame. At 4K that is 8 MB per
mask, so a crowd of 200 people over 100 frames is ~160 GB of host RAM held
at once - while each tiny person really covers a few hundred pixels. A
`BoxMask` keeps only the crop around the non-zero pixels (plus the plate
shape) and still behaves like the dense (H, W) uint8 array wherever code does
plain numpy on it (`np.asarray(m)`, `m > 0`, `m.any()`, `m.sum()`, `m[y, x]`,
`a | m`), materialising the full frame only for that one operation. Hot
helpers use `box` / `crop` directly to stay local.

`MaskDict` is a dict that stores every assigned array as a `BoxMask`, so code
writing `inst.masks[f] = dense` keeps the memory bound for free.
`LazyStack` is the (T, H, W) `stack` of the `sam3_hard_masks` artifact made
of BoxMasks: `np.asarray(stack)` / `stack[k]` still give dense arrays (one
track at a time, transient), `stack.box(k)` gives the BoxMask.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

_EMPTY = np.zeros((0, 0), np.uint8)


class BoxMask:
    """A 0/1 uint8 mask of `shape`, stored as `crop` at (`y0`, `x0`)."""

    __slots__ = ("crop", "shape", "x0", "y0")
    ndim = 2
    dtype = np.dtype(np.uint8)
    # Make numpy defer binary operators (`a | m`, `a & m`) to us.
    __array_priority__ = 1000

    def __init__(
        self, shape: tuple[int, int], y0: int = 0, x0: int = 0, crop: np.ndarray | None = None
    ) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.y0, self.x0 = int(y0), int(x0)
        self.crop = _EMPTY if crop is None else crop

    @classmethod
    def from_dense(cls, arr: Any) -> BoxMask:
        if isinstance(arr, BoxMask):
            return arr
        a = np.asarray(arr)
        if a.ndim != 2:
            raise ValueError(f"BoxMask needs a 2D mask, got shape {a.shape}")
        b = a > 0.5 if a.dtype.kind == "f" else a > 0
        rows, cols = np.flatnonzero(b.any(axis=1)), np.flatnonzero(b.any(axis=0))
        if rows.size == 0:
            return cls((int(a.shape[0]), int(a.shape[1])))
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        return cls(
            (int(a.shape[0]), int(a.shape[1])),
            y0,
            x0,
            np.ascontiguousarray(b[y0:y1, x0:x1]).astype(np.uint8),
        )

    @classmethod
    def from_crop(cls, shape: tuple[int, int], y0: int, x0: int, crop: np.ndarray) -> BoxMask:
        """Wrap a local crop, tightening it to its own non-zero pixels."""
        local = cls.from_dense(crop)
        if not local.crop.size:
            return cls(shape)
        return cls(shape, y0 + local.y0, x0 + local.x0, local.crop)

    # --- geometry --------------------------------------------------------

    @property
    def box(self) -> tuple[int, int, int, int]:
        """(y0, x0, y1, x1); an empty mask has an empty box."""
        h, w = self.crop.shape
        return self.y0, self.x0, self.y0 + h, self.x0 + w

    @property
    def height(self) -> int:
        return int(self.crop.shape[0])

    def dense(self, dtype: Any = np.uint8) -> np.ndarray:
        out = np.zeros(self.shape, dtype=dtype)
        if self.crop.size:
            y0, x0, y1, x1 = self.box
            out[y0:y1, x0:x1] = self.crop
        return out

    def window(self, y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
        """This mask inside the plate window [y0:y1, x0:x1] (uint8)."""
        out = np.zeros((y1 - y0, x1 - x0), np.uint8)
        by0, bx0, by1, bx1 = self.box
        iy0, ix0, iy1, ix1 = max(y0, by0), max(x0, bx0), min(y1, by1), min(x1, bx1)
        if iy1 > iy0 and ix1 > ix0:
            out[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = self.crop[
                iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0
            ]
        return out

    def resized(self, shape: tuple[int, int]) -> BoxMask:
        """This mask on a plate of another size (e.g. full res -> proxy).
        A tiny object that would round away keeps at least its covered px."""
        import cv2

        h0, w0 = self.shape
        h1, w1 = int(shape[0]), int(shape[1])
        if not self.crop.size:
            return BoxMask((h1, w1))
        sy, sx = h1 / h0, w1 / w0
        y0, x0, y1, x1 = self.box
        ny0, nx0 = int(np.floor(y0 * sy)), int(np.floor(x0 * sx))
        ny1 = max(ny0 + 1, min(h1, int(np.ceil(y1 * sy))))
        nx1 = max(nx0 + 1, min(w1, int(np.ceil(x1 * sx))))
        # Resample the crop into the destination box (INTER_AREA = coverage).
        src = self.crop.astype(np.float32)
        cov = cv2.resize(src, (nx1 - nx0, ny1 - ny0), interpolation=cv2.INTER_AREA)
        m = cov >= 0.5
        if not m.any():
            m = cov > 0
        return BoxMask.from_crop((h1, w1), ny0, nx0, m.astype(np.uint8))

    # --- the numpy surface code actually uses ----------------------------

    def any(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return self.dense().any(*args, **kwargs)
        return bool(self.crop.any())

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return self.dense().sum(*args, **kwargs)
        return int(self.crop.sum())

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        d = self.dense()
        return d if dtype is None else d.astype(dtype)

    def astype(self, dtype: Any, copy: bool = True) -> np.ndarray:
        return self.dense(dtype)

    def __getitem__(self, idx: Any) -> Any:
        return self.dense()[idx]

    def __len__(self) -> int:
        return self.shape[0]

    def __getattr__(self, name: str) -> Any:
        # Anything else ndarray has (.max(), .nonzero(), .copy(), ...) runs on
        # the dense frame. Dunders and unset slots stay AttributeErrors
        # (copy/pickle probe for them).
        if name.startswith("__") or name in BoxMask.__slots__:
            raise AttributeError(name)
        return getattr(self.dense(), name)

    def _op(self, other: Any, fn: Any, swap: bool = False) -> Any:
        o = other.dense() if isinstance(other, BoxMask) else other
        return fn(o, self.dense()) if swap else fn(self.dense(), o)

    def __gt__(self, o: Any) -> Any:
        return self._op(o, np.greater)

    def __ge__(self, o: Any) -> Any:
        return self._op(o, np.greater_equal)

    def __lt__(self, o: Any) -> Any:
        return self._op(o, np.less)

    def __le__(self, o: Any) -> Any:
        return self._op(o, np.less_equal)

    def __and__(self, o: Any) -> Any:
        return self._op(o, np.bitwise_and)

    def __rand__(self, o: Any) -> Any:
        return self._op(o, np.bitwise_and, swap=True)

    def __or__(self, o: Any) -> Any:
        return self._op(o, np.bitwise_or)

    def __ror__(self, o: Any) -> Any:
        return self._op(o, np.bitwise_or, swap=True)

    def __mul__(self, o: Any) -> Any:
        return self._op(o, np.multiply)

    def __rmul__(self, o: Any) -> Any:
        return self._op(o, np.multiply, swap=True)

    def __add__(self, o: Any) -> Any:
        return self._op(o, np.add)

    def __radd__(self, o: Any) -> Any:
        return self._op(o, np.add, swap=True)

    def __sub__(self, o: Any) -> Any:
        return self._op(o, np.subtract)

    def __rsub__(self, o: Any) -> Any:
        return self._op(o, np.subtract, swap=True)

    def __repr__(self) -> str:
        return f"BoxMask(shape={self.shape}, box={self.box}, px={self.sum()})"

    def __getstate__(self) -> tuple[Any, ...]:
        return (self.shape, self.y0, self.x0, self.crop)

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        self.shape, self.y0, self.x0, self.crop = state


class MaskDict(dict[Any, Any]):
    """frame -> BoxMask; plain arrays assigned in are compressed."""

    def __init__(self, items: Any = ()) -> None:
        super().__init__()
        src = items.items() if isinstance(items, dict) else items
        for k, v in src:
            self[k] = v

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, BoxMask.from_dense(value))

    def update(self, *args: Any, **kwargs: Any) -> None:
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def setdefault(self, key: Any, default: Any = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]


class LazyStack:
    """(T, H, W) uint8 stack backed by BoxMasks."""

    ndim = 3
    dtype = np.dtype(np.uint8)

    def __init__(self, masks: Iterable[Any], shape_hw: tuple[int, int]) -> None:
        self._m = [BoxMask.from_dense(m) for m in masks]
        self.shape = (len(self._m), int(shape_hw[0]), int(shape_hw[1]))

    def __len__(self) -> int:
        return len(self._m)

    def box(self, k: int) -> BoxMask:
        return self._m[k]

    def boxes(self) -> Iterator[BoxMask]:
        return iter(self._m)

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, (int, np.integer)):
            return self._m[int(idx)].dense()
        return np.asarray(self)[idx]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = np.zeros(self.shape, np.uint8)
        for k, m in enumerate(self._m):
            if m.crop.size:
                y0, x0, y1, x1 = m.box
                out[k, y0:y1, x0:x1] = m.crop
        return out if dtype is None else out.astype(dtype)

    def astype(self, dtype: Any, copy: bool = True) -> np.ndarray:
        return np.asarray(self, dtype=dtype)

    def __getattr__(self, name: str) -> Any:
        # .min(), .max(), .sum(axis=...) ... on the dense stack (one track).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)


def as_box(m: Any) -> BoxMask:
    return BoxMask.from_dense(m)


def stack_box(stack: Any, k: int) -> BoxMask:
    """Frame `k` of a track stack as a BoxMask (LazyStack or dense array)."""
    return (
        stack.box(k) if isinstance(stack, LazyStack) else BoxMask.from_dense(np.asarray(stack[k]))
    )


def boxes_touch(a: BoxMask, b: BoxMask, margin: int = 0) -> bool:
    ay0, ax0, ay1, ax1 = a.box
    by0, bx0, by1, bx1 = b.box
    if ay1 <= ay0 or by1 <= by0:
        return False
    return not (
        ay1 + margin <= by0 or by1 + margin <= ay0 or ax1 + margin <= bx0 or bx1 + margin <= ax0
    )


def inter_area(a: BoxMask, b: BoxMask) -> int:
    """Pixels set in both masks, computed on the box overlap only."""
    ay0, ax0, ay1, ax1 = a.box
    by0, bx0, by1, bx1 = b.box
    y0, x0, y1, x1 = max(ay0, by0), max(ax0, bx0), min(ay1, by1), min(ax1, bx1)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(
        (
            a.crop[y0 - ay0 : y1 - ay0, x0 - ax0 : x1 - ax0]
            & b.crop[y0 - by0 : y1 - by0, x0 - bx0 : x1 - bx0]
        ).sum()
    )


def paint_max(dst: np.ndarray, m: BoxMask, value: Any = 1) -> None:
    """dst = max(dst, m * value), in place, inside m's box only."""
    if m.crop.size:
        y0, x0, y1, x1 = m.box
        np.maximum(dst[y0:y1, x0:x1], m.crop * value, out=dst[y0:y1, x0:x1], casting="unsafe")


def union_box(
    masks: Iterable[BoxMask], pad: int, shape: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """Padded box around several masks, clipped to the plate."""
    boxes = [m.box for m in masks if m.crop.size]
    if not boxes:
        return None
    return (
        max(0, min(b[0] for b in boxes) - pad),
        max(0, min(b[1] for b in boxes) - pad),
        min(shape[0], max(b[2] for b in boxes) + pad),
        min(shape[1], max(b[3] for b in boxes) + pad),
    )


__all__ = [
    "BoxMask",
    "LazyStack",
    "MaskDict",
    "as_box",
    "boxes_touch",
    "inter_area",
    "paint_max",
    "stack_box",
    "union_box",
]
