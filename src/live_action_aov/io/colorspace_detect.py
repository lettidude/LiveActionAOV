"""Colorspace detection with provenance reporting.

A drop-in replacement for `sniff_colorspace` in `io/ocio_color.py` that
returns BOTH the detected value AND a human-readable reason. The GUI
needs to tell the user *why* the tool picked a particular colorspace
so they can spot "lying tag" EXRs (the ACTIONVFX pack ships
`oiio:ColorSpace = lin_rec709` on files whose pixels are clearly
display-referred — the user has to know what the header says before
they can override it).

Detection ladder, first match wins:

  1. Explicit `colorspace` / `OCIO/colorspace` / `ocio:colorspace`
     header attribute → "from oiio:ColorSpace header"
  2. Explicit `oiio:ColorSpace` attribute (OIIO writes this) →
     "from oiio:ColorSpace header"
  3. `chromaticities` matching ACES AP1 primaries →
     "inferred from chromaticities primaries (ACES AP1)"
  4. `chromaticities` matching Rec.709 primaries →
     "inferred from chromaticities primaries (Rec.709)"
  5. Pixel-range heuristic (max < ~2.0 and median near 0.5) →
     "inferred from pixel-range heuristic (display-referred)"
  6. Default fallback → "no signal — defaulting to lin_rec709"

The pixel-range heuristic is the closest we have to the manual visual
check the compositor does today; it's deliberately conservative and
produces a soft-confidence result the UI surfaces as a warning rather
than a hard recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Supported colorspace names. The UI dropdown renders these plus `auto`.
# Log spaces are grouped after the linear/display set so the dropdown
# reads "scene-linear / display / log" in that order.
SUPPORTED_COLORSPACES: tuple[str, ...] = (
    "auto",
    "lin_rec709",
    "acescg",
    "aces2065_1",
    "srgb_display",
    "rec709_display",
    "linear",
    "arri_logc3",  # Alexa Classic / Mini / LF / Mini LF, EI800
    "arri_logc4",  # Alexa 35 (2022+)
    "sony_slog3",  # FX-series, FS-series, original Venice
    "sony_slog3_cine",  # S-Gamut3.Cine variant
    "sony_slog3_venice",  # Venice with S-Gamut3
    "sony_slog3_venice_cine",  # Venice with S-Gamut3.Cine
)

# Primaries used for chromaticity-based inference. Values are
# (rx, ry, gx, gy, bx, by, wx, wy) — the 8-tuple EXR writes.
_ACES_AP1_PRIMARIES = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
_REC709_PRIMARIES = (0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290)


@dataclass(frozen=True)
class DetectedColorspace:
    """Result of auto-detecting a plate's colorspace.

    Attributes:
        detected: One of `SUPPORTED_COLORSPACES` (excluding `auto` — the
                  detector never returns `auto` itself; `auto` is a UI
                  concept meaning "use whatever the detector picked").
        reason:   Human-readable provenance, surfaced verbatim in the
                  inspector's colorspace dropdown label.
        confident: True if the detection is from a positively identified
                   tag or chromaticity match; False if it's a heuristic
                   fallback that the user should double-check.
    """

    detected: str
    reason: str
    confident: bool


def detect_colorspace(
    attrs: dict[str, Any],
    sample_pixels: np.ndarray | None = None,
    *,
    fallback: str = "lin_rec709",
) -> DetectedColorspace:
    """Detect the colorspace of an EXR from its header attrs + optional
    pixel sample. Returns a structured result with the detected value,
    a human-readable reason for the UI, and a confidence flag.
    """
    # Ladder step 0: camera metadata wins over `oiio:ColorSpace`.
    # ARRI Raw / Sony X-OCN exports routinely ship with a LYING
    # `oiio:ColorSpace = 'lin_rec709'` header (OIIO's default stamp)
    # on footage that is actually Log-encoded. The camera metadata
    # attrs (`Arri Raw.Gamma Value`, `uniColor.gamma`,
    # `tech_details_color_space`, Sony equivalents…) are written by
    # the camera / DIT software and do NOT lie. Run this first so a
    # LogC4 AWG4 plate gets decoded correctly even when the header
    # says otherwise.
    camera_hit = _detect_from_camera_metadata(attrs)
    if camera_hit is not None:
        return camera_hit

    # Ladder step 1-2: explicit colorspace attributes (several spellings).
    for key in ("colorspace", "OCIO/colorspace", "ocio:colorspace", "oiio:ColorSpace"):
        v = attrs.get(key)
        if isinstance(v, str) and v:
            normalized = _normalize_colorspace_name(v)
            return DetectedColorspace(
                detected=normalized,
                reason=f"from `{key}` header = {v!r}",
                confident=True,
            )

    # Ladder step 3-4: chromaticity-primary inference.
    chroma = attrs.get("chromaticities")
    if chroma and isinstance(chroma, (list, tuple)) and len(chroma) >= 8:
        match = _match_chromaticities(tuple(float(c) for c in chroma[:8]))
        if match is not None:
            name, primaries_label = match
            return DetectedColorspace(
                detected=name,
                reason=f"inferred from chromaticities primaries ({primaries_label})",
                confident=True,
            )

    # Ladder step 5: pixel-range heuristic — soft confidence.
    if sample_pixels is not None and sample_pixels.size > 0:
        guess = _guess_from_pixel_range(sample_pixels)
        if guess is not None:
            return guess

    # Ladder step 6: fallback.
    return DetectedColorspace(
        detected=fallback,
        reason=f"no signal in header — defaulting to {fallback}",
        confident=False,
    )


def _normalize_colorspace_name(raw: str) -> str:
    """Map whatever the header says into one of `SUPPORTED_COLORSPACES`.

    The EXR ecosystem uses a grab-bag of spellings:
      `linear`, `Linear`, `scene_linear`, `lin_rec709`, `Rec709 Linear`,
      `srgb`, `sRGB`, `srgb_texture`, `ACEScg`, `ACES - ACEScg`, …
    We map all of these onto our canonical names so the UI has a stable
    set to render. Anything we don't recognise passes through verbatim;
    the dropdown renders it as-is and the user picks an override.
    """
    norm = raw.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    table = {
        "linear": "linear",
        "scene_linear": "linear",
        "lin_rec709": "lin_rec709",
        "linear_rec709": "lin_rec709",
        "linear_rec_709": "lin_rec709",
        "rec709_linear": "lin_rec709",
        "acescg": "acescg",
        "aces___acescg": "acescg",
        "aces_acescg": "acescg",
        "aces2065_1": "aces2065_1",
        "aces2065": "aces2065_1",
        "srgb": "srgb_display",
        "srgb_display": "srgb_display",
        "srgb_texture": "srgb_display",
        "srgb_encoded": "srgb_display",
        "rec709": "rec709_display",
        "rec709_display": "rec709_display",
        "gamma_2_2": "srgb_display",
        "gamma_22": "srgb_display",
        # Log encodings — many spellings in the wild.
        "arri_logc3": "arri_logc3",
        "arri_logc3_ei800": "arri_logc3",
        "logc3": "arri_logc3",
        "arri_logc": "arri_logc3",  # "LogC" with no version = LogC3 historically
        "arri_logc4": "arri_logc4",
        "logc4": "arri_logc4",
        "sony_slog3": "sony_slog3",
        "slog3": "sony_slog3",
        "s_log3": "sony_slog3",
        "sony_slog3_cine": "sony_slog3_cine",
        "slog3_cine": "sony_slog3_cine",
        "s_log3_s_gamut3": "sony_slog3",
        "s_log3_s_gamut3_cine": "sony_slog3_cine",
    }
    return table.get(norm, raw)


# Markers we look for in camera-metadata attr values to infer the
# capture-space. Each entry: (substring to match case-insensitively,
# canonical colorspace, short label for the reason string).
_CAMERA_METADATA_MARKERS: tuple[tuple[str, str, str], ...] = (
    # ARRI — LogC4 is the default on Alexa 35 (2022+). LogC3 order
    # matters: match LogC4 *before* the generic LogC fallback so a
    # "LogC4 AWG4" string doesn't get mis-bucketed as LogC3.
    ("logc4", "arri_logc4", "ARRI LogC4"),
    ("logc3", "arri_logc3", "ARRI LogC3 EI800"),
    ("arri logc", "arri_logc3", "ARRI LogC (legacy, defaulting to LogC3)"),
    ("alexa wide gamut 4", "arri_logc4", "ARRI LogC4 (AWG4 gamut)"),
    ("awg4", "arri_logc4", "ARRI LogC4 (AWG4 gamut)"),
    # Sony — S-Log3 is what modern cameras ship (FX3/FX6/FX9/Venice).
    # S-Log2 is legacy and not supported yet; if we detect it, fall
    # through to the existing ladder rather than lie.
    ("s-log3 s-gamut3.cine", "sony_slog3_cine", "Sony S-Log3 S-Gamut3.Cine"),
    ("slog3 sgamut3.cine", "sony_slog3_cine", "Sony S-Log3 S-Gamut3.Cine"),
    ("s-log3", "sony_slog3", "Sony S-Log3"),
    ("slog3", "sony_slog3", "Sony S-Log3"),
)


def _detect_from_camera_metadata(attrs: dict[str, Any]) -> DetectedColorspace | None:
    """Scan every string-valued attribute for camera-metadata markers.

    This is the single most important detection step for ARRI / Sony
    footage: the `oiio:ColorSpace` header is written by OIIO with no
    domain knowledge, and is routinely wrong on Log-encoded EXRs
    (the ARRI Reference Tool default-exports LogC4 AWG4 with a
    `lin_rec709` tag). The camera manufacturer's own metadata —
    `Arri Raw.Gamma Value`, `uniColor.gamma`, `uniColor.gamut`,
    `tech_details_color_space`, Sony's `CameraColorSpace` / similar
    — is written by the DIT / camera and does not lie.

    Returns None if no marker was found, letting the caller fall
    through to the rest of the detection ladder.
    """
    # Scan key + value together so e.g. a key like `uniColor.gamut`
    # holding "Arri AlexaWideGamut4" still hits the "awg4" marker.
    for key, value in attrs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        haystack = f"{key} {value}".lower()
        for needle, canonical, label in _CAMERA_METADATA_MARKERS:
            if needle in haystack:
                return DetectedColorspace(
                    detected=canonical,
                    reason=f"camera metadata: `{key}` = {value!r} -> {label}",
                    confident=True,
                )
    return None


def _match_chromaticities(
    chroma: tuple[float, float, float, float, float, float, float, float],
) -> tuple[str, str] | None:
    """Return `(colorspace_name, primaries_label)` if chromaticities match
    a known standard. Tolerance 0.01 per channel — tight enough to
    reject noise, loose enough to accept minor rounding variations
    between writers."""
    if _chromaticities_match(chroma, _ACES_AP1_PRIMARIES):
        return ("acescg", "ACES AP1")
    if _chromaticities_match(chroma, _REC709_PRIMARIES):
        return ("lin_rec709", "Rec.709")
    return None


def _chromaticities_match(
    a: tuple[float, ...],
    b: tuple[float, ...],
    *,
    tol: float = 0.01,
) -> bool:
    if len(a) < 8 or len(b) < 8:
        return False
    return all(abs(float(ai) - float(bi)) < tol for ai, bi in zip(a[:8], b[:8], strict=False))


def _guess_from_pixel_range(pixels: np.ndarray) -> DetectedColorspace | None:
    """Pixel-range heuristic for lying-tag detection.

    The test: a display-referred frame has pixels ~entirely in [0, 1]
    with a median luminance around 0.2-0.5. A scene-linear plate has
    either super-whites above 1.0 (exposure headroom) or a low median
    typical of tonemapped-from-linear lows.

    We sample up to ~1% of pixels to keep the call fast even on 4K
    inputs. The result comes back with `confident=False` since this is
    a guess, not an authoritative read.
    """
    flat = pixels.reshape(-1, pixels.shape[-1])[..., :3] if pixels.ndim >= 2 else pixels
    if flat.size == 0:
        return None

    # Subsample for speed — random stride is fine on natural images.
    step = max(1, flat.shape[0] // 10000)
    sample = flat[::step]
    luma = 0.2126 * sample[..., 0] + 0.7152 * sample[..., 1] + 0.0722 * sample[..., 2]
    luma = luma[np.isfinite(luma)]
    if luma.size == 0:
        return None

    p99 = float(np.quantile(luma, 0.99))
    p50 = float(np.quantile(luma, 0.50))

    # Display-referred: no super-whites, median in the midtone band.
    if p99 < 1.1 and 0.15 < p50 < 0.75:
        return DetectedColorspace(
            detected="srgb_display",
            reason=(
                f"inferred from pixel range (p99={p99:.2f}, p50={p50:.2f}) — "
                "looks display-referred but header gave no signal"
            ),
            confident=False,
        )
    # Scene-linear: super-whites or a dark median.
    if p99 > 1.2 or p50 < 0.15:
        return DetectedColorspace(
            detected="lin_rec709",
            reason=(
                f"inferred from pixel range (p99={p99:.2f}, p50={p50:.2f}) — "
                "looks scene-linear but header gave no signal"
            ),
            confident=False,
        )
    return None


__all__ = [
    "SUPPORTED_COLORSPACES",
    "DetectedColorspace",
    "detect_colorspace",
]
