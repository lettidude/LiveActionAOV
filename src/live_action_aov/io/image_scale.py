# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Scale sidecar for images that are not at the passes' size.

The EXR passes are always written at the plate size (proxy runs are scaled
back up). Some by-products are not: the COLMAP model under
`camera/<stem>_colmap/` is solved on frames of at most `max_image_size`
pixels, so its 2D observations are in those pixels, not the passes'. A
tool that samples the passes at such a pixel lands off by the ratio.

Next to such an output, `<output>.scale.json` says how its pixels map to the
passes' pixels:

    {"schema": "laov.image_scale/1", "what": "colmap_model",
     "image_size": [1920, 1013], "passes_size": [4096, 2160],
     "scale_to_passes": [2.1333, 2.1323]}

`x_passes = x_image * scale_to_passes[0]` (pixel corners, top-left origin).

Standard library only (Python 3.7 syntax), so Nuke-side tools can copy it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

SCHEMA = "laov.image_scale/1"
SUFFIX = ".scale.json"


def sidecar_path(target: Any) -> str:
    """`<target>.scale.json` for a folder or a file (trailing slash ignored)."""
    return str(target).rstrip("/\\") + SUFFIX


def write_scale_sidecar(
    target: Any,
    image_size: Sequence[int],
    passes_size: Sequence[int],
    what: str = "",
) -> str:
    """Write `<target>.scale.json`; returns its path."""
    iw, ih = int(image_size[0]), int(image_size[1])
    pw, ph = int(passes_size[0]), int(passes_size[1])
    if min(iw, ih, pw, ph) <= 0:
        raise ValueError(f"bad sizes: image {image_size}, passes {passes_size}")
    doc = {
        "schema": SCHEMA,
        "what": what,
        "image_size": [iw, ih],
        "passes_size": [pw, ph],
        "scale_to_passes": [pw / iw, ph / ih],
    }
    path = sidecar_path(target)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return path


def read_scale_sidecar(target: Any) -> dict[str, Any] | None:
    """The sidecar of `target`, or None when there is none (or it is not ours)."""
    path = sidecar_path(target)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        return None
    return doc


def scale_to_passes(
    target: Any,
    image_size: Sequence[int] | None = None,
    passes_size: Sequence[int] | None = None,
) -> tuple[float, float]:
    """(sx, sy) taking a pixel of `target` to the passes' pixels.

    The sidecar wins. Without one, the ratio of the two sizes when the
    caller knows both (an older run), else (1.0, 1.0): same size assumed,
    which is what every reader did before the sidecar existed.
    """
    doc = read_scale_sidecar(target)
    if doc is not None:
        sx, sy = doc["scale_to_passes"]
        return float(sx), float(sy)
    if image_size is not None and passes_size is not None:
        return (
            float(passes_size[0]) / float(image_size[0]),
            float(passes_size[1]) / float(image_size[1]),
        )
    return 1.0, 1.0


__all__ = [
    "SCHEMA",
    "SUFFIX",
    "read_scale_sidecar",
    "scale_to_passes",
    "sidecar_path",
    "write_scale_sidecar",
]
