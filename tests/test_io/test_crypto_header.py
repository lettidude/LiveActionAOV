"""Header keys next to a Cryptomatte: safe for older Nuke Cryptomatte nodes.

Older Nukes crash on any NON-string attribute whose name contains
"cryptomatte" (the four standard `cryptomatte/<id>/*` strings are fine). The
tool writes the Cryptomatte pass provenance under `liveaov/crypto_pass/*`;
readers accept the older `liveaov/cryptomatte_pass/*` and
`liveaov/cryptomatte/*` and both file layouts (version root and `passes/`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.crypto_header import (
    crypto_layers,
    crypto_provenance,
    find_crypto_sequences,
    is_standard_key,
    safe_header,
    unsafe_keys,
)
from live_action_aov.io.cryptomatte import encode, header_metadata

STD = header_metadata("CryptoObject", {"person_1": "3f800000"})


def _legacy_header() -> dict:
    return {
        **STD,
        "liveaov/cryptomatte_pass/model": "cryptomatte",
        "liveaov/cryptomatte_pass/commercial": True,
        "liveaov/cryptomatte/score": 0.5,
        "liveaov/matte/cryptomatte_objects": 3,
        "liveaov/frame": 1001,
        "liveaov/moge3/commercial": True,
    }


def test_standard_keys_are_recognised() -> None:
    assert all(is_standard_key(k) for k in STD)
    assert not is_standard_key("liveaov/cryptomatte_pass/model")


def test_safe_header_moves_old_provenance_and_stringifies_the_rest() -> None:
    out = safe_header(_legacy_header())
    assert unsafe_keys(out) == []
    for k, v in STD.items():  # the spec keys, verbatim
        assert out[k] == v
    assert out["liveaov/crypto_pass/model"] == "cryptomatte"
    assert out["liveaov/crypto_pass/commercial"] is True  # no "cryptomatte" in the name: kept typed
    assert out["liveaov/crypto_pass/score"] == 0.5
    assert out["liveaov/matte/cryptomatte_objects"] == "3"
    assert not any(k.startswith("liveaov/cryptomatte") for k in out)
    assert out["liveaov/frame"] == 1001 and out["liveaov/moge3/commercial"] is True


def test_safe_header_is_a_no_op_on_a_clean_header() -> None:
    clean = {**STD, "liveaov/crypto_pass/commercial": True, "liveaov/frame": 3}
    assert safe_header(clean) == clean
    assert safe_header(None) == {}


def test_new_key_wins_over_an_old_one() -> None:
    out = safe_header({"liveaov/crypto_pass/version": "2", "liveaov/cryptomatte_pass/version": "1"})
    assert out == {"liveaov/crypto_pass/version": "2"}


def test_provenance_reads_every_layout() -> None:
    for prefix in ("liveaov/crypto_pass/", "liveaov/cryptomatte_pass/", "liveaov/cryptomatte/"):
        prov = crypto_provenance({f"{prefix}model": "cryptomatte", f"{prefix}commercial": 1})
        assert prov == {"model": "cryptomatte", "commercial": 1}
    layers = crypto_layers(STD)
    (fields,) = layers.values()
    assert fields["name"] == "CryptoObject" and "manifest" in fields


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


@pytest.mark.parametrize(
    "rel",
    [
        "shot_v001.crypto.{f}.exr",  # older runs: version root
        "passes/shot_v001.crypto.{f}.exr",  # current lossless sibling
        "passes/shot_v001.utility.{f}.crypto.exr",  # older sibling name
        "passes/crypto/shot_v001.crypto.{f}.exr",  # per-pass layout
        "passes/plate_crypto_v001.{f}.exr",  # custom naming template
    ],
)
def test_find_crypto_sequences_in_both_file_layouts(tmp_path: Path, rel: str) -> None:
    for f in (1001, 1002, 1003):
        _touch(tmp_path / rel.format(f=f))
    _touch(tmp_path / "passes" / "shot_v001.utility.1001.exr")  # not a crypto file
    seqs = find_crypto_sequences(str(tmp_path))
    assert len(seqs) == 1
    template, frames = seqs[0]
    assert frames == [1001, 1002, 1003]
    assert "####" in template


def test_written_exr_header_is_safe(tmp_path: Path) -> None:
    from live_action_aov.io.oiio_io import HAS_OIIO, read_exr, write_exr

    if not HAS_OIIO:
        pytest.skip("OpenImageIO not installed")
    a = np.zeros((8, 8), np.float32)
    a[:, :4] = 1.0
    ch, manifest = encode([("person_1", a), ("person_2", 1.0 - a)])
    names = sorted(ch)
    path = tmp_path / "x.exr"
    header = {**_legacy_header(), **header_metadata("CryptoObject", manifest)}
    write_exr(path, np.stack([ch[n] for n in names], -1), names, attrs=header)
    _pix, attrs = read_exr(path)
    assert unsafe_keys(attrs) == []
    for k, v in header_metadata("CryptoObject", manifest).items():
        assert attrs[k] == v
    assert crypto_provenance(attrs)["model"] == "cryptomatte"


def test_inspect_flags_an_old_file(tmp_path: Path) -> None:
    import live_action_aov.io.oiio_io as oio
    from live_action_aov.cli.inspect import build_report, format_json, format_text

    if not oio.HAS_OIIO:
        pytest.skip("OpenImageIO not installed")
    import OpenImageIO as oiio

    path = tmp_path / "old.exr"
    spec = oiio.ImageSpec(4, 4, 1, oiio.FLOAT)
    spec.channelnames = ("CryptoObject00.R",)
    for k, v in _legacy_header().items():  # written raw, the way an old run did
        oio._set_attr(spec, k, v)
    out = oiio.ImageOutput.create(str(path))
    out.open(str(path), spec)
    out.write_image(np.zeros((4, 4, 1), np.float32))
    out.close()

    rep = build_report(path)
    assert rep.cryptomatte is not None
    assert rep.cryptomatte["layers"][0]["objects"] == 1
    assert rep.cryptomatte["provenance"]["model"] == "cryptomatte"
    assert "liveaov/cryptomatte_pass/commercial" in rep.cryptomatte["unsafe_keys"]
    assert "crash older Nuke" in format_text(rep)
    assert format_json(rep)["cryptomatte"]["unsafe_keys"]
