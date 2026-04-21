"""OCIO colorspace transforms.

Thin wrapper around PyOpenColorIO. Callers pass in (frames, from_space,
to_space) and get linear-space arrays back. If OCIO is unavailable the
module degrades to a small set of hard-coded sRGB/linear transforms so
Phase 0 can run with just numpy.

The auto-sniff logic reads a colorspace name from an EXR attribute dict; if
it can't find one, it falls back to the Shot's configured colorspace.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:
    import PyOpenColorIO as ocio

    HAS_OCIO = True
except ImportError:  # pragma: no cover
    ocio = None
    HAS_OCIO = False


# Map our canonical short names to the colorspace names the OCIO studio-
# config uses. Keeps the GUI / CLI vocabulary stable even as upstream
# renames the config entries; any alias the user types (e.g. directly
# `"ARRI LogC4"` from a Nuke OCIO node) still passes through unchanged
# because we fall back to the literal string below.
_CANONICAL_TO_OCIO: dict[str, str] = {
    "acescg": "ACEScg",
    "aces2065_1": "ACES2065-1",
    "lin_rec709": "Linear Rec.709 (sRGB)",
    "linear": "Linear Rec.709 (sRGB)",
    "srgb_display": "sRGB - Display",
    "rec709_display": "Rec.1886 Rec.709 - Display",
    # Log sources — OCIO studio-config ships inverse curves + gamut
    # conversion in a single named transform so a single getProcessor
    # call takes us from LogC-encoded AWG4 pixels to ACEScg linear.
    "arri_logc3": "ARRI LogC3 (EI800)",
    "arri_logc4": "ARRI LogC4",
    "sony_slog3": "S-Log3 S-Gamut3",
    "sony_slog3_cine": "S-Log3 S-Gamut3.Cine",
    "sony_slog3_venice": "S-Log3 Venice S-Gamut3",
    "sony_slog3_venice_cine": "S-Log3 Venice S-Gamut3.Cine",
}


def _resolve_ocio_name(name: str) -> str:
    """Translate a canonical-short name to its studio-config name. If the
    input isn't in the table it's passed through verbatim — OCIO will
    raise a clear `ColorSpace not found` error if it's also not a valid
    studio-config entry."""
    return _CANONICAL_TO_OCIO.get(name, name)


def get_config() -> Any:
    """Return the active OCIO config (from $OCIO or built-in default)."""
    if not HAS_OCIO:
        raise RuntimeError("PyOpenColorIO not available")
    if "OCIO" in os.environ:
        return ocio.Config.CreateFromEnv()
    # Fall back to the studio-default built-in if no $OCIO is set.
    try:
        return ocio.Config.CreateFromBuiltinConfig("studio-config-latest")
    except AttributeError:
        return ocio.Config.CreateFromFile(ocio.GetDefaultConfig().getName())


def to_linear(frames: np.ndarray, from_space: str, config: Any | None = None) -> np.ndarray:
    """Convert `frames` (HxWxC or NxHxWxC) from `from_space` to a linear
    working space.

    If OCIO isn't available, falls back to a small hard-coded set
    (linear-in = passthrough; srgb = gamma 2.2 inverse; acescg = passthrough,
    which is not strictly correct but safe for a Phase 0 no-op smoke test).
    """
    if HAS_OCIO:
        cfg = config or get_config()
        # Known scene-linear reference space name varies per config; use the
        # config's declared working space.
        dst = (
            cfg.getCanonicalName("scene_linear")
            or cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR).getName()
        )
        proc = cfg.getProcessor(_resolve_ocio_name(from_space), dst).getDefaultCPUProcessor()
        arr = np.ascontiguousarray(frames.astype(np.float32, copy=False))
        flat = arr.reshape(-1, arr.shape[-1])
        proc.applyRGB(flat) if arr.shape[-1] == 3 else proc.applyRGBA(flat)
        return flat.reshape(arr.shape)
    return _fallback_to_linear(frames, from_space)


def from_linear(frames: np.ndarray, to_space: str, config: Any | None = None) -> np.ndarray:
    """Convert `frames` from scene-linear to `to_space`."""
    if HAS_OCIO:
        cfg = config or get_config()
        src = (
            cfg.getCanonicalName("scene_linear")
            or cfg.getRoleColorSpace(ocio.ROLE_SCENE_LINEAR).getName()
        )
        proc = cfg.getProcessor(src, _resolve_ocio_name(to_space)).getDefaultCPUProcessor()
        arr = np.ascontiguousarray(frames.astype(np.float32, copy=False))
        flat = arr.reshape(-1, arr.shape[-1])
        proc.applyRGB(flat) if arr.shape[-1] == 3 else proc.applyRGBA(flat)
        return flat.reshape(arr.shape)
    return _fallback_from_linear(frames, to_space)


def sniff_colorspace(attrs: dict[str, Any], fallback: str = "acescg") -> str:
    """Try to infer the colorspace from an EXR header.

    Nuke writes an explicit `colorspace` attribute when saving; ACES-aware
    applications write `chromaticities`. If neither is found, fall back to
    the Shot's declared colorspace (default acescg).
    """
    for key in ("colorspace", "OCIO/colorspace", "ocio:colorspace"):
        v = attrs.get(key)
        if isinstance(v, str) and v:
            return v
    # Chromaticities hint: ACES has recognisable primaries.
    chroma = attrs.get("chromaticities")
    if chroma and isinstance(chroma, (list, tuple)) and len(chroma) >= 8:
        # Rough check for ACES AP1 red primary (~0.713, 0.293).
        try:
            if abs(float(chroma[0]) - 0.713) < 0.01:
                return "acescg"
        except (TypeError, ValueError):
            pass
    return fallback


# ---------------------------------------------------------------------------
# No-OCIO fallback — narrow, explicit, safe enough for Phase 0 smoke tests.
# ---------------------------------------------------------------------------

_SRGB_NAMES = {"srgb", "sRGB", "srgb_display", "srgb_texture", "srgb_linear"}
_LINEAR_NAMES = {"linear", "linear_rec709", "scene_linear", "acescg", "linear_srgb"}


def _fallback_to_linear(frames: np.ndarray, from_space: str) -> np.ndarray:
    if from_space in _LINEAR_NAMES:
        return frames.astype(np.float32, copy=False)
    if from_space in _SRGB_NAMES:
        return _srgb_to_linear(frames)
    # Unknown: identity + warn silently. Callers should set OCIO env for
    # anything non-trivial.
    return frames.astype(np.float32, copy=False)


def _fallback_from_linear(frames: np.ndarray, to_space: str) -> np.ndarray:
    if to_space in _LINEAR_NAMES:
        return frames.astype(np.float32, copy=False)
    if to_space in _SRGB_NAMES:
        return _linear_to_srgb(frames)
    return frames.astype(np.float32, copy=False)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    a = 0.055
    lo = x / 12.92
    hi = np.power((x + a) / (1 + a), 2.4)
    return np.where(x <= 0.04045, lo, hi)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32, copy=False), 0.0, None)
    a = 0.055
    lo = x * 12.92
    hi = (1 + a) * np.power(x, 1 / 2.4) - a
    return np.where(x <= 0.0031308, lo, hi)


__all__ = [
    "HAS_OCIO",
    "from_linear",
    "get_config",
    "sniff_colorspace",
    "to_linear",
]
