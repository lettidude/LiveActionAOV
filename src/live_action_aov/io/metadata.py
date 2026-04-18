"""CameraMetadata — cinema camera attributes extracted from EXR + sidecars.

Implemented fully in v1 even though only v2a's CameraPass really uses it,
because the shape is load-bearing for later (design §20.10). The extractor
reads EXR header attributes written by Nuke / ARRI ProRes-to-EXR converters
/ ALE sidecars etc.

Anything we can't read is left as `None`. The reconciliation logic for
focal length (design §20.3) lives in the v2a camera pass, not here — this
module is pure extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CameraMetadata:
    """Per-clip camera/lens metadata read from EXR headers and sidecars."""

    # From EXR header
    resolution: tuple[int, int] | None = None
    pixel_aspect: float | None = None
    colorspace: str | None = None

    # Camera body
    camera_make: str | None = None
    camera_model: str | None = None
    sensor_size_mm: tuple[float, float] | None = None
    iso: int | None = None
    shutter_angle: float | None = None

    # Lens — scalar or per-frame list
    focal_length_mm: float | list[float] | None = None
    focus_distance_m: float | list[float] | None = None
    t_stop: float | list[float] | None = None
    lens_model: str | None = None
    lens_serial: str | None = None

    # Clip identity
    timecode_start: str | None = None
    reel_id: str | None = None
    clip_name: str | None = None

    # Sidecar metadata neighbors (ALE, XML, RMD, CDL)
    sidecar_paths: list[Path] = field(default_factory=list)

    # Raw attribute dict for attributes we didn't map to a named field.
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exr_attrs(cls, attrs: dict[str, Any]) -> "CameraMetadata":
        """Build a CameraMetadata from a raw EXR attribute dict.

        Attribute names differ across ARRI / RED / Nuke conventions; we try
        several candidates for each field. When in doubt the raw dict is
        preserved in `raw` so later code can rescue fields we missed.
        """
        m = cls(raw=dict(attrs))

        # Resolution & aspect
        w = _first(attrs, "ImageWidth", "displayWindow.width", "width")
        h = _first(attrs, "ImageHeight", "displayWindow.height", "height")
        if w is not None and h is not None:
            try:
                m.resolution = (int(w), int(h))
            except (TypeError, ValueError):
                pass
        par = _first(attrs, "pixelAspectRatio", "PixelAspectRatio")
        if par is not None:
            try:
                m.pixel_aspect = float(par)
            except (TypeError, ValueError):
                pass
        m.colorspace = _first(attrs, "colorspace", "chromaticities", "OCIO/colorspace")

        # Camera body
        m.camera_make = _first(attrs, "camera/make", "Make", "cameraMake")
        m.camera_model = _first(attrs, "camera/model", "Model", "cameraModel")
        m.iso = _first_int(attrs, "camera/iso", "ISO")
        m.shutter_angle = _first_float(
            attrs, "camera/shutter_angle", "shutterAngle", "ShutterAngle"
        )

        # Lens
        m.focal_length_mm = _first(attrs, "lens/focal_length_mm", "focalLength", "FocalLength")
        m.focus_distance_m = _first(attrs, "lens/focus_distance_m", "focusDistance")
        m.t_stop = _first(attrs, "lens/t_stop", "tStop", "TStop")
        m.lens_model = _first(attrs, "lens/model", "lensModel", "LensModel")
        m.lens_serial = _first(attrs, "lens/serial", "lensSerial")

        # Clip
        m.timecode_start = _first(attrs, "timeCode", "timecode", "TimeCode")
        m.reel_id = _first(attrs, "reel", "reelID", "ReelID")
        m.clip_name = _first(attrs, "clip", "clipName", "ClipName")
        return m


def _first(attrs: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in attrs and attrs[k] not in (None, ""):
            return attrs[k]
    return None


def _first_float(attrs: dict[str, Any], *keys: str) -> float | None:
    v = _first(attrs, *keys)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _first_int(attrs: dict[str, Any], *keys: str) -> int | None:
    v = _first(attrs, *keys)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


__all__ = ["CameraMetadata"]
