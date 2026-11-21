# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""EXR header keys next to a Cryptomatte, safe for every Nuke version.

Nuke's native Cryptomatte node reads the four standard string attributes
`cryptomatte/<keyhash>/{name,hash,conversion,manifest}`. Older Nukes also
choke on any OTHER attribute whose name contains "cryptomatte" when its value
is not a string (an int or a float): the node crashes on evaluation. So:

- the provenance of the Cryptomatte pass is written under
  `liveaov/crypto_pass/...` (older files: `liveaov/cryptomatte_pass/...`, and
  before that `liveaov/cryptomatte/...`);
- `safe_header()` is the last step before a header is written: it moves the
  older provenance keys to the new prefix and writes any other non-string
  value under a "cryptomatte" key as a string.

The readers here accept both header layouts and both file layouts (frames at
the version folder's root, as older runs wrote them, or under `passes/`).

Standard library only (Python 3.7 syntax), so Nuke-side tools can copy it.
"""

from __future__ import annotations

import os
import re
from typing import Any

#: Prefix of the Cryptomatte pass provenance (model, version, licence ...).
CRYPTO_PROVENANCE_PREFIX = "liveaov/crypto_pass/"
#: Older prefixes of the same keys, newest first. Read, never written.
LEGACY_PROVENANCE_PREFIXES = ("liveaov/cryptomatte_pass/", "liveaov/cryptomatte/")

_STANDARD_RE = re.compile(r"^cryptomatte/[0-9a-fA-F]+/(name|hash|conversion|manifest)$")


def is_standard_key(key: str) -> bool:
    """True for the four Cryptomatte spec attributes Nuke reads."""
    return bool(_STANDARD_RE.match(key))


def is_unsafe(key: str, value: Any) -> bool:
    """A key that crashes older Nuke Cryptomatte nodes: its name contains
    "cryptomatte" (any case) and its value is not a string."""
    return "cryptomatte" in key.lower() and not isinstance(value, str)


def unsafe_keys(attrs: dict[str, Any]) -> list[str]:
    """Header keys of `attrs` an older Nuke Cryptomatte node crashes on."""
    return sorted(k for k, v in attrs.items() if isinstance(k, str) and is_unsafe(k, v))


def _new_key(key: str) -> str:
    for old in LEGACY_PROVENANCE_PREFIXES:
        if key.startswith(old):
            return CRYPTO_PROVENANCE_PREFIX + key[len(old) :]
    return key


def _as_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_as_string(v) for v in value)
    return str(value)


def safe_header(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """`attrs` with every key safe next to a Cryptomatte, in any Nuke.

    Older provenance keys move to `liveaov/crypto_pass/...` (a key already
    there wins). Any other non-string value under a name containing
    "cryptomatte" is written as a string. Every other key is unchanged.
    """
    out: dict[str, Any] = {}
    if not attrs:
        return out
    for key, value in attrs.items():
        if isinstance(key, str) and not key.startswith(LEGACY_PROVENANCE_PREFIXES):
            out[key] = value
    for key, value in attrs.items():
        if isinstance(key, str) and key.startswith(LEGACY_PROVENANCE_PREFIXES):
            out.setdefault(_new_key(key), value)
    for key in list(out):
        if isinstance(key, str) and is_unsafe(key, out[key]):
            out[key] = _as_string(out[key])
    return out


def crypto_provenance(attrs: dict[str, Any]) -> dict[str, Any]:
    """Cryptomatte pass provenance (`model`, `version`, `license`,
    `commercial` ...) from a header of either layout; the newest wins."""
    prov: dict[str, Any] = {}
    for prefix in (CRYPTO_PROVENANCE_PREFIX, *LEGACY_PROVENANCE_PREFIXES):
        for key, value in attrs.items():
            if isinstance(key, str) and key.startswith(prefix):
                prov.setdefault(key[len(prefix) :], value)
    return prov


def crypto_layers(attrs: dict[str, Any]) -> dict[str, dict[str, str]]:
    """`{keyhash: {name, hash, conversion, manifest}}` from the standard keys."""
    layers: dict[str, dict[str, str]] = {}
    for key, value in attrs.items():
        if isinstance(key, str) and is_standard_key(key):
            _, kh, field = key.split("/")
            layers.setdefault(kh, {})[field] = value
    return layers


# --- file layouts ---------------------------------------------------------

_FRAME_RE = re.compile(r"^(?P<head>.*?)(?P<frame>\d+)(?P<tail>(\.crypto)?\.exr)$", re.IGNORECASE)


def find_crypto_sequences(run_dir: str) -> list[tuple[str, list[int]]]:
    """Cryptomatte EXR sequences of a run folder, as `(template, frames)`
    with `####` for the frame number.

    Looks in the version folder itself (older runs) and under `passes/`
    (any depth), for every naming LAOV wrote: the lossless sibling
    `<stem>.crypto.####.exr`, the older `<stem>.utility.####.crypto.exr`,
    and the per-pass `passes/crypto*/...`. A run with a lossless delivery
    keeps its Cryptomatte inside the utility EXR: no separate sequence, and
    the caller reads the utility instead.
    """
    roots = [run_dir]
    passes = os.path.join(run_dir, "passes")
    seqs: dict[str, list[int]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            _add(seqs, root, name)
    if os.path.isdir(passes):
        for dirpath, _dirs, files in os.walk(passes):
            for name in sorted(files):
                _add(seqs, dirpath, name)
    return [(t, sorted(f)) for t, f in sorted(seqs.items())]


def _add(seqs: dict[str, list[int]], folder: str, name: str) -> None:
    m = _FRAME_RE.match(name)
    if m is None:
        return
    head, frame, tail = m.group("head"), m.group("frame"), m.group("tail")
    in_crypto_dir = os.path.basename(folder).lower().startswith("crypto")
    # `crypto` is the file-kind token of every naming template
    # (`<stem>.crypto.####.exr`, `<plate>_crypto_v001.####.exr` ...).
    if not ("crypto" in head.lower() or tail.lower() == ".crypto.exr" or in_crypto_dir):
        return
    template = os.path.join(folder, head + "#" * len(frame) + tail)
    seqs.setdefault(template, []).append(int(frame))


__all__ = [
    "CRYPTO_PROVENANCE_PREFIX",
    "LEGACY_PROVENANCE_PREFIXES",
    "crypto_layers",
    "crypto_provenance",
    "find_crypto_sequences",
    "is_standard_key",
    "is_unsafe",
    "safe_header",
    "unsafe_keys",
]
