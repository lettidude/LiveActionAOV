"""EXR read/write via OpenImageIO.

Thin wrapper: keeps OIIO imports and error handling in one place. Everyone
else (readers, writers, display transform) calls into here.

In environments where OIIO isn't installed (some CI matrices, minimal
installs), the functions raise at call-time rather than import-time, so
modules that just import this file still load. A small `imageio` fallback
is provided for the tests/fixtures EXR writer — writing simple synthetic
plates for Phase 0 doesn't need OIIO's full feature set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import OpenImageIO as oiio  # type: ignore[import-not-found]

    HAS_OIIO = True
except ImportError:  # pragma: no cover — environment-dependent
    oiio = None  # type: ignore[assignment]
    HAS_OIIO = False


class OiioError(RuntimeError):
    """Raised when an OIIO operation fails or when OIIO isn't installed."""


def require_oiio() -> None:
    if not HAS_OIIO:
        raise OiioError(
            "OpenImageIO is not available. Install via `uv sync` (oiio-python wheel) "
            "or see https://docs.openimageio.org/en/latest/installation.html"
        )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_exr(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read an EXR file. Returns `(pixels, attrs)`.

    `pixels`: (H, W, C) float32 array, channel order as in the file.
    `attrs`: dict of EXR header attributes (names stringified). Nuke-written
    files carry hundreds of attributes; we expose the full set so downstream
    metadata extraction can pick whatever it needs.
    """
    require_oiio()
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise OiioError(f"Failed to open EXR: {path} ({oiio.geterror()})")
    try:
        spec = inp.spec()
        pixels = inp.read_image(format=oiio.FLOAT)
        if pixels is None:
            raise OiioError(f"Failed to read pixels: {path} ({oiio.geterror()})")
        attrs = _extract_attrs(spec)
        return _ensure_hwc(pixels, spec), attrs
    finally:
        inp.close()


def _ensure_hwc(pixels: Any, spec: Any) -> np.ndarray:
    arr = np.asarray(pixels, dtype=np.float32)
    h, w, c = spec.height, spec.width, spec.nchannels
    if arr.ndim == 2:
        arr = arr.reshape(h, w, 1)
    elif arr.ndim == 3 and arr.shape != (h, w, c):
        arr = arr.reshape(h, w, c)
    return arr


def _extract_attrs(spec: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        "width": spec.width,
        "height": spec.height,
        "nchannels": spec.nchannels,
        "channelnames": list(spec.channelnames),
        "pixelAspectRatio": float(spec.get_float_attribute("PixelAspectRatio", 1.0)),
    }
    # OIIO exposes `extra_attribs` as an iterable of ParamValue objects with
    # .name / .type / .value.
    try:
        for pv in spec.extra_attribs:
            attrs[pv.name] = pv.value
    except AttributeError:  # pragma: no cover
        pass
    return attrs


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_exr(
    path: Path | str,
    pixels: np.ndarray,
    channel_names: list[str],
    *,
    attrs: dict[str, Any] | None = None,
    pixel_aspect: float = 1.0,
    compression: str = "zip",
    dtype: str = "float32",
) -> None:
    """Write a multi-channel EXR.

    `pixels`: (H, W, C) float array; len(channel_names) must equal C.
    `attrs`: flat dict of EXR custom attributes. Keys with '/' are written
    verbatim (OpenEXR supports slash-namespaced attribute names, Nuke
    displays them as hierarchical).
    """
    require_oiio()
    pixels = np.ascontiguousarray(pixels)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    h, w, c = pixels.shape
    if len(channel_names) != c:
        raise OiioError(
            f"channel_names has {len(channel_names)} entries but pixels has {c} channels"
        )

    oiio_dtype = {
        "float32": oiio.FLOAT,
        "float16": oiio.HALF,
    }.get(dtype)
    if oiio_dtype is None:
        raise OiioError(f"Unsupported dtype {dtype!r}; use 'float32' or 'float16'")

    spec = oiio.ImageSpec(w, h, c, oiio_dtype)
    spec.channelnames = tuple(channel_names)
    spec.attribute("compression", compression)
    spec.attribute("PixelAspectRatio", float(pixel_aspect))
    if attrs:
        for key, val in attrs.items():
            _set_attr(spec, key, val)

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise OiioError(f"No ImageOutput available for: {path} ({oiio.geterror()})")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not out.open(str(path), spec):
            raise OiioError(f"Failed to open for write: {path} ({oiio.geterror()})")
        if dtype == "float16":
            pixels = pixels.astype(np.float16, copy=False)
        else:
            pixels = pixels.astype(np.float32, copy=False)
        if not out.write_image(pixels):
            raise OiioError(f"Failed to write pixels: {path} ({oiio.geterror()})")
    finally:
        out.close()


def _set_attr(spec: Any, name: str, value: Any) -> None:
    """Best-effort typed attribute write."""
    if isinstance(value, bool):
        spec.attribute(name, int(value))
    elif isinstance(value, int):
        spec.attribute(name, int(value))
    elif isinstance(value, float):
        spec.attribute(name, float(value))
    elif isinstance(value, (list, tuple)):
        # Store structured/list values as their repr — keeps round-trip simple
        # for header metadata like channel conventions, hero labels, etc.
        spec.attribute(name, repr(value))
    else:
        spec.attribute(name, str(value))


__all__ = ["HAS_OIIO", "OiioError", "read_exr", "require_oiio", "write_exr"]
