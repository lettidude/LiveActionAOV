# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Cryptomatte pass — SAM 3 tracked instances -> Nuke-readable per-object IDs.

Consumes the `sam3_hard_masks` artifact the SAM 3 detector publishes
(`{track_id: {label, frames, stack (T,H,W)}}`) and encodes it as a
Cryptomatte (Psyop spec) so a compositor can click any object in Nuke.
Same dependent-pass shape as the RVM refiner: `requires_artifacts`,
`ingest_artifacts`, custom `run_shot`.

Each track becomes an object named `<label>_<track_id>` (e.g. `person_1`,
`vehicle_2`). Because SAM 3 track IDs are temporally consistent, the same
object keeps the same Cryptomatte id across the whole clip — no flicker.

Outputs (channels.py CRYPTOMATTE_CHANNELS): a colour preview + ranked
(id, coverage) channel groups, written float32 by the EXR writer. The
manifest + `cryptomatte/<keyhash>/*` header keys are emitted as the
`cryptomatte_header` artifact and stamped onto every sidecar by the executor.

Note: SAM 3 masks are hard (0/1), so coverage has no sub-pixel anti-aliasing
unless `feather` > 0 (a light Gaussian that fakes soft edges for comp).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.core.pass_base import (
    ChannelSpec,
    License,
    PassType,
    TemporalMode,
    UtilityPass,
)
from live_action_aov.io.channels import CRYPTOMATTE_CHANNELS, CRYPTOMATTE_TYPENAME
from live_action_aov.io.cryptomatte import encode, header_metadata, name_to_id


class CryptomattePass(UtilityPass):
    name = "cryptomatte"
    version = "0.1.0"
    license = License(
        spdx="MIT",
        commercial_use=True,
        commercial_tool_resale=True,
        notes=(
            "Cryptomatte encoder is our own code to the open Psyop spec "
            "(reference impl is BSD-3). Commercial safety of the underlying "
            "detection follows the upstream detector pass (e.g. SAM 3's "
            "SAM-License-1.0); this pass only repackages its masks."
        ),
    )
    pass_type = PassType.SEMANTIC
    temporal_mode = TemporalMode.VIDEO_CLIP
    input_colorspace = "srgb_display"

    produces_channels = [
        ChannelSpec(name=c, dtype="float32", description="Cryptomatte id/coverage (float32)")
        for c in CRYPTOMATTE_CHANNELS
    ]
    # Hard DAG dep: a detector emitting `sam3_hard_masks` must run first.
    requires_artifacts = ["sam3_hard_masks"]
    provides_artifacts = ["cryptomatte_header"]
    # NEVER smooth — temporally blending hash-id channels would corrupt them.
    smoothable_channels: list[str] = []

    DEFAULT_PARAMS: dict[str, Any] = {
        # 0 = hard edges (matches SAM 3 masks). >0 = Gaussian feather radius
        # (px) faking sub-pixel coverage for softer comp edges.
        "feather": 0.0,
    }

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        for k, v in self.DEFAULT_PARAMS.items():
            self.params.setdefault(k, v)
        self._hard_masks: dict[int, dict[str, Any]] = {}
        self._manifest: dict[str, str] = {}

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        hard = artifacts.get("sam3_hard_masks") or {}
        self._hard_masks = next(iter(hard.values())) if hard else {}

    # VIDEO_CLIP — single-frame lifecycle unused but required by the ABC.
    def preprocess(self, frames: np.ndarray) -> Any:
        return frames

    def infer(self, tensor: Any) -> Any:
        return tensor

    def postprocess(self, tensor: Any) -> dict[str, np.ndarray]:
        return {}

    def run_shot(
        self,
        reader: Any,
        frame_range: tuple[int, int],
    ) -> dict[int, dict[str, np.ndarray]]:
        first, last = frame_range
        h, w = reader.read_frame(first)[0].shape[:2]

        # Object name per track + the shot-wide manifest (all objects).
        name_of: dict[int, str] = {}
        self._manifest = {}
        for tid, info in self._hard_masks.items():
            nm = f"{info.get('label', 'object')}_{tid}"
            name_of[tid] = nm
            self._manifest[nm] = name_to_id(nm)[1]

        feather = float(self.params.get("feather", 0.0))
        zero_channels = {c: np.zeros((h, w), np.float32) for c in CRYPTOMATTE_CHANNELS}

        out: dict[int, dict[str, np.ndarray]] = {}
        for f in range(first, last + 1):
            instances: list[tuple[str, np.ndarray]] = []
            for tid, info in self._hard_masks.items():
                frames = list(info.get("frames") or [])
                if f not in frames:
                    continue
                # Index first, then cast just this frame — casting the whole
                # (T,H,W) stack to float32 per frame is O(F) wasted work (and
                # the stack is uint8 now). One (H,W) slice is all we need.
                m = np.asarray(info["stack"][frames.index(f)], dtype=np.float32)
                if feather > 0:
                    m = _feather(m, feather)
                if float(m.sum()) > 0:
                    instances.append((name_of[tid], m))
            if instances:
                channels, _ = encode(instances, typename=CRYPTOMATTE_TYPENAME)
                out[f] = channels
            else:
                # No objects this frame — emit the full zero layer so the
                # channel set is consistent across the clip.
                out[f] = {c: a.copy() for c, a in zero_channels.items()}
        return out

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        if not self._manifest:
            return {}
        # Stash the header attrs the executor merges onto every sidecar.
        return {"cryptomatte_header": {0: header_metadata(CRYPTOMATTE_TYPENAME, self._manifest)}}


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Light Gaussian feather to fake sub-pixel coverage from a hard mask."""
    import cv2

    k = max(1, int(round(radius)) * 2 + 1)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), radius), 0.0, 1.0)


__all__ = ["CryptomattePass"]
