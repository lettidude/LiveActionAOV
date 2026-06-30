# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""EXR sidecar writer.

Writes a multi-channel EXR with channels ordered against
`CANONICAL_CHANNEL_ORDER` (§11.3 trap 8 — Nuke's reader has specific
expectations). Unknown-but-valid channels (e.g. dynamic `mask.<concept>`)
are appended in insertion order after the canonical channels.

Metadata is written as EXR custom attributes with slash-namespaced keys
(e.g. `liveaov/depth/model`), which Nuke displays as a nested tree.

### Delivery: compression + bit depth (bandwidth-bound pipelines)

The writer is configurable so a job can trade size against fidelity:

- `compression`: any OIIO codec — `zip`/`piz` (lossless), `dwaa`/`dwab`
  (lossy, 5-20x smaller on continuous AOVs; takes an optional `:level`,
  e.g. `dwab:45`).
- `dtype`: `float32` or `float16` (half, ~2x smaller, the VFX norm for
  utility data).

**Cryptomatte is special.** Its ID channels carry exact float32 hash
bit-patterns; measured, *any* lossy codec OR even a lossless float16 cast
corrupts them (the id->name match in Nuke silently breaks). So when the
chosen delivery would damage Cryptomatte, those channels are split into a
sibling `<name>.crypto.exr` written lossless `zip`/`float32`, while the
rest of the AOVs take the compact delivery. With a lossless float32
delivery nothing is split — one file, as before.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.io.channels import CANONICAL_CHANNEL_ORDER, CRYPTOMATTE_TYPENAME
from live_action_aov.io.oiio_io import write_exr
from live_action_aov.io.writers.base import SidecarWriter

#: Codecs that do not preserve exact float bit-patterns (Cryptomatte-unsafe).
LOSSY_COMPRESSIONS = frozenset({"dwaa", "dwab", "b44", "b44a", "pxr24"})


class ExrSidecarWriter(SidecarWriter):
    """Write sidecar EXR files matching the Nuke/VFX channel contract.

    Args:
        compression: OIIO codec string (default ``"zip"``, lossless). Pass
            ``"dwab"`` / ``"dwab:45"`` etc. for compact lossy delivery.
        dtype: ``"float32"`` (default) or ``"float16"`` (half).
        split_crypto: ``"auto"`` (default) splits Cryptomatte to a lossless
            sibling file only when the chosen delivery would corrupt it;
            ``True`` always splits; ``False`` never does (caller's risk).
    """

    format_tag = "exr"

    def __init__(
        self,
        *,
        compression: str = "zip",
        dtype: str = "float32",
        split_crypto: bool | str = "auto",
    ) -> None:
        self.compression = compression
        self.dtype = dtype
        self.split_crypto = split_crypto

    def _delivery_corrupts_crypto(self) -> bool:
        """True if the configured codec/dtype would break Cryptomatte IDs."""
        base = self.compression.split(":", 1)[0].strip().lower()
        return base in LOSSY_COMPRESSIONS or self.dtype != "float32"

    def write_frame(
        self,
        out_path: Path,
        channels: dict[str, np.ndarray],
        *,
        attrs: dict[str, Any] | None = None,
        pixel_aspect: float = 1.0,
    ) -> None:
        if not channels:
            raise ValueError("ExrSidecarWriter requires at least one channel")

        crypto = {n: a for n, a in channels.items() if n.startswith(CRYPTOMATTE_TYPENAME)}
        rest = {n: a for n, a in channels.items() if not n.startswith(CRYPTOMATTE_TYPENAME)}

        do_split = bool(crypto) and (
            self.split_crypto is True
            or (self.split_crypto == "auto" and self._delivery_corrupts_crypto())
        )

        if not do_split:
            # Single file, current behaviour — write everything with the
            # configured delivery (safe when lossless float32, or when there's
            # no Cryptomatte to protect).
            self._write_one(out_path, channels, attrs, pixel_aspect, self.compression, self.dtype)
            return

        # Compact delivery for the AOVs; Cryptomatte kept lossless float32 in
        # a sibling file so its hash IDs survive intact.
        if rest:
            self._write_one(out_path, rest, attrs, pixel_aspect, self.compression, self.dtype)
        crypto_path = out_path.parent / f"{out_path.stem}.crypto{out_path.suffix}"
        self._write_one(crypto_path, crypto, attrs, pixel_aspect, "zip", "float32")

    @staticmethod
    def _write_one(
        out_path: Path,
        channels: dict[str, np.ndarray],
        attrs: dict[str, Any] | None,
        pixel_aspect: float,
        compression: str,
        dtype: str,
    ) -> None:
        ordered_names = _order_channels(list(channels.keys()))
        first = channels[ordered_names[0]]
        h, w = first.shape[:2]

        stack = np.empty((h, w, len(ordered_names)), dtype=np.float32)
        for i, name in enumerate(ordered_names):
            arr = np.asarray(channels[name], dtype=np.float32)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            if arr.shape[:2] != (h, w):
                raise ValueError(f"Channel '{name}' has shape {arr.shape} but expected {(h, w)}")
            stack[..., i] = arr

        write_exr(
            out_path,
            stack,
            ordered_names,
            attrs=attrs,
            pixel_aspect=pixel_aspect,
            compression=compression,
            dtype=dtype,
        )


def _order_channels(names: list[str]) -> list[str]:
    """Order channels against CANONICAL_CHANNEL_ORDER; unknowns follow."""
    canonical_set = set(CANONICAL_CHANNEL_ORDER)
    canonical_kept = [n for n in CANONICAL_CHANNEL_ORDER if n in names]
    extras = [n for n in names if n not in canonical_set]
    return canonical_kept + extras


__all__ = ["LOSSY_COMPRESSIONS", "ExrSidecarWriter"]
