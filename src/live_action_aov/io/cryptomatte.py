# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Cryptomatte encoder (Psyop spec).

Turns named per-pixel coverage masks into the ranked (hash-id, coverage)
channels Nuke's Cryptomatte node reads natively, plus the manifest +
`cryptomatte/<keyhash>/*` header metadata.

Encoding (ref: github.com/Psyop/Cryptomatte, nuke/cryptomatte_utilities.py):
- id = MurmurHash3_x86_32(name) reinterpreted as float32, with bit 23
  toggled when the exponent is 0/255 (avoids denormal/inf/nan).
- channels: a colour preview (`<type>.RGB`) + `levels` ranked groups
  (`<type>NN.R/G/B/A` = id/cov/id/cov), ranks sorted by coverage desc.
- header: `cryptomatte/<keyhash>/{name,hash,conversion,manifest}` where
  keyhash = first 7 hex of the typename's id.

Channels MUST be written float32 (ids carry exact hash bits) — the EXR
sidecar writer already does this.
"""

from __future__ import annotations

import json
import struct

import mmh3
import numpy as np

from live_action_aov.io.channels import CRYPTOMATTE_LEVELS, CRYPTOMATTE_TYPENAME


def name_to_id(name: str) -> tuple[float, str]:
    """Object name -> (id_float, 8-char hex) per the Cryptomatte spec."""
    h32 = mmh3.hash(name) & 0xFFFFFFFF
    exp = (h32 >> 23) & 0xFF
    if exp == 0 or exp == 0xFF:  # avoid denormal / inf / nan
        h32 ^= 1 << 23
    f = struct.unpack("<f", struct.pack("<L", h32))[0]
    return f, f"{h32:08x}"


def keyhash(typename: str) -> str:
    """7-hex metadata key for a layer typename."""
    return name_to_id(typename)[1][:-1]


def _preview_color(idf: np.ndarray) -> np.ndarray:
    """Deterministic per-id pseudo-colour (HxW id-float -> 3xHxW in [0,1])."""
    bits = idf.view(np.uint32).astype(np.uint64)
    r = ((bits * 2654435761) & 0xFF) / 255.0
    g = ((bits * 40503) & 0xFF) / 255.0
    b = ((bits * 2246822519) & 0xFF) / 255.0
    out = np.stack([r, g, b], 0).astype(np.float32)
    out[:, idf == 0.0] = 0.0  # background stays black
    return out


def encode(
    instances: list[tuple[str, np.ndarray]],
    typename: str = CRYPTOMATTE_TYPENAME,
    levels: int = CRYPTOMATTE_LEVELS,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Encode one frame.

    `instances`: [(name, coverage HxW float in [0,1]), ...].
    Returns (channels {name: HxW float32}, manifest {name: hex8}).
    Empty input -> all-zero channels + empty manifest (valid, picks nothing).
    """
    if not instances:
        return {}, {}
    h, w = instances[0][1].shape
    ranks = levels * 2
    ids = {name: name_to_id(name) for name, _ in instances}
    manifest = {name: ids[name][1] for name, _ in instances}

    cov = np.clip(np.stack([c for _, c in instances], 0).astype(np.float32), 0.0, 1.0)
    # Normalize per-pixel coverage so overlaps sum to <=1 (disjoint unaffected).
    tot = cov.sum(0, keepdims=True)
    cov = np.where(tot > 1.0, cov / np.maximum(tot, 1e-8), cov)
    idf = np.array([ids[name][0] for name, _ in instances], np.float32)

    order = np.argsort(-cov, axis=0)
    out_id = np.zeros((ranks, h, w), np.float32)
    out_cov = np.zeros((ranks, h, w), np.float32)
    for r in range(min(ranks, cov.shape[0])):
        sel = order[r]
        c = np.take_along_axis(cov, sel[None], 0)[0]
        out_cov[r] = c
        out_id[r] = np.where(c > 0, idf[sel], 0.0)

    channels: dict[str, np.ndarray] = {}
    pr, pg, pb = _preview_color(out_id[0])
    channels[f"{typename}.R"], channels[f"{typename}.G"], channels[f"{typename}.B"] = pr, pg, pb
    for lvl in range(levels):
        a, b = 2 * lvl, 2 * lvl + 1
        channels[f"{typename}{lvl:02d}.R"] = out_id[a]
        channels[f"{typename}{lvl:02d}.G"] = out_cov[a]
        channels[f"{typename}{lvl:02d}.B"] = out_id[b]
        channels[f"{typename}{lvl:02d}.A"] = out_cov[b]
    return channels, manifest


def header_metadata(typename: str, manifest: dict[str, str]) -> dict[str, str]:
    """The `cryptomatte/<keyhash>/*` EXR header attributes Nuke needs."""
    kh = keyhash(typename)
    return {
        f"cryptomatte/{kh}/name": typename,
        f"cryptomatte/{kh}/hash": "MurmurHash3_32",
        f"cryptomatte/{kh}/conversion": "uint32_to_float32",
        f"cryptomatte/{kh}/manifest": json.dumps(manifest),
    }


__all__ = ["encode", "header_metadata", "keyhash", "name_to_id"]
