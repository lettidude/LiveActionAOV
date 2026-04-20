# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""JSON sidecar writer (minimal in v1).

Used today for shot-level metadata dumps from `liveaov analyze`. In v2a the
CameraPass will emit per-frame camera tracks through this writer as well
(`<shot>.utility.camera.json`). The interface is identical — callers pass
serializable dicts; the writer handles pretty-printing and UTF-8 encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.writers.base import SidecarWriter


class JsonSidecarWriter(SidecarWriter):
    format_tag = "json"

    def write_frame(
        self,
        out_path: Path,
        channels: dict[str, np.ndarray],
        *,
        attrs: dict[str, Any] | None = None,
        pixel_aspect: float = 1.0,
    ) -> None:
        # Arrays are serialized as lists; large tensors should not be shipped
        # via JSON — callers that need image data use the EXR writer.
        payload: dict[str, Any] = {
            "attrs": _jsonable(attrs or {}),
            "pixel_aspect": pixel_aspect,
            "channels": {k: _jsonable(v) for k, v in channels.items()},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = ["JsonSidecarWriter"]
