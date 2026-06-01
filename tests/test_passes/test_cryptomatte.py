# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Cryptomatte encoder + pass tests.

The pass needs no model (it repackages an upstream artifact), so these run
the full path end-to-end with an injected `sam3_hard_masks` artifact and
verify the manifest round-trip — the exact lookup Nuke's Cryptomatte node
performs when you click a pixel.
"""

from __future__ import annotations

import struct

import numpy as np

from live_action_aov.core.pass_base import License, PassType, TemporalMode
from live_action_aov.io import channels as ch
from live_action_aov.io.cryptomatte import encode, header_metadata, keyhash, name_to_id
from live_action_aov.passes.matte.cryptomatte import CryptomattePass


def _hex_of_float(f: float) -> str:
    return f"{struct.unpack('<L', struct.pack('<f', float(f)))[0]:08x}"


# --- encoder ---------------------------------------------------------


def test_name_to_id_deterministic_and_finite() -> None:
    f, hx = name_to_id("person_1")
    assert (f, hx) == name_to_id("person_1")  # deterministic
    assert len(hx) == 8
    assert np.isfinite(f)  # never inf/nan (spec bit-toggle)
    # The hex must equal the float's bit pattern (Nuke matches on this).
    assert _hex_of_float(f) == hx


def test_keyhash_is_7_hex() -> None:
    kh = keyhash("CryptoObject")
    assert len(kh) == 7
    assert all(c in "0123456789abcdef" for c in kh)


def test_encode_layout_and_ranking() -> None:
    H, W = 4, 6
    a = np.zeros((H, W), np.float32)
    b = np.zeros((H, W), np.float32)
    a[:, :3] = 1.0  # object A on the left
    b[:, 3:] = 1.0  # object B on the right
    channels, manifest = encode([("A", a), ("B", b)], typename="CryptoObject")
    assert set(manifest) == {"A", "B"}
    # rank-0 id on the left pixel decodes back to A.
    id0 = channels["CryptoObject00.R"]
    cov0 = channels["CryptoObject00.G"]
    assert cov0[0, 0] == 1.0
    assert _hex_of_float(id0[0, 0]) == manifest["A"]
    assert _hex_of_float(id0[0, 5]) == manifest["B"]
    # all declared channels present
    assert set(channels) == set(ch.CRYPTOMATTE_CHANNELS)


def test_header_metadata_keys() -> None:
    md = header_metadata("CryptoObject", {"A": "deadbeef"})
    kh = keyhash("CryptoObject")
    assert md[f"cryptomatte/{kh}/name"] == "CryptoObject"
    assert md[f"cryptomatte/{kh}/hash"] == "MurmurHash3_32"
    assert md[f"cryptomatte/{kh}/conversion"] == "uint32_to_float32"
    assert "deadbeef" in md[f"cryptomatte/{kh}/manifest"]


# --- channels contract ----------------------------------------------


def test_cryptomatte_channels_in_canonical_order_no_dupes() -> None:
    for c in ch.CRYPTOMATTE_CHANNELS:
        assert c in ch.CANONICAL_CHANNEL_ORDER
    assert len(ch.CANONICAL_CHANNEL_ORDER) == len(set(ch.CANONICAL_CHANNEL_ORDER))


# --- pass contract + run_shot (no model) ----------------------------


def test_pass_contract() -> None:
    assert CryptomattePass.declared_license().commercial_use is True
    assert isinstance(CryptomattePass.license, License)
    assert CryptomattePass.pass_type is PassType.SEMANTIC
    assert CryptomattePass.temporal_mode is TemporalMode.VIDEO_CLIP
    assert CryptomattePass.requires_artifacts == ["sam3_hard_masks"]
    assert CryptomattePass.smoothable_channels == []  # ids must never be smoothed
    assert {c.name for c in CryptomattePass.produces_channels} == set(ch.CRYPTOMATTE_CHANNELS)


class _StubReader:
    def __init__(self, h: int, w: int) -> None:
        self._f = np.zeros((h, w, 3), np.float32)

    def read_frame(self, f: int):
        return self._f, {}


def test_run_shot_roundtrip_no_model() -> None:
    H, W = 4, 6
    stack = np.zeros((2, H, W), np.float32)
    stack[:, :, :3] = 1.0  # person occupies left half on both frames
    p = CryptomattePass()
    p.ingest_artifacts(
        {"sam3_hard_masks": {0: {7: {"label": "person", "frames": [0, 1], "stack": stack}}}}
    )
    out = p.run_shot(_StubReader(H, W), (0, 1))
    assert set(out) == {0, 1}
    chans = out[0]
    assert set(chans) == set(ch.CRYPTOMATTE_CHANNELS)
    # manifest round-trip: the left pixel decodes to person_7
    man = p.emit_artifacts()["cryptomatte_header"][0]
    kh = keyhash(ch.CRYPTOMATTE_TYPENAME)
    assert man[f"cryptomatte/{kh}/name"] == ch.CRYPTOMATTE_TYPENAME
    assert "person_7" in man[f"cryptomatte/{kh}/manifest"]
    assert _hex_of_float(chans["CryptoObject00.R"][0, 0]) == name_to_id("person_7")[1]


def test_in_gui_catalog() -> None:
    from live_action_aov.gui.pass_catalog import expand_models, find_entry

    e = find_entry("cryptomatte")
    assert e is not None and e.commercial is True
    # Selecting Cryptomatte runs the detector + the encoder.
    assert expand_models(["cryptomatte"]) == ["sam3_matte", "cryptomatte"]


def test_exr_writer_roundtrip(tmp_path: object) -> None:
    """The tool's actual EXR writer must produce a decodable Cryptomatte:
    float32 id channels + header manifest that round-trips (Nuke's lookup)."""
    import OpenImageIO as oiio

    from live_action_aov.io.writers.exr import ExrSidecarWriter

    H, W = 8, 8
    a = np.zeros((H, W), np.float32)
    a[:, :4] = 1.0
    b = np.zeros((H, W), np.float32)
    b[:, 4:] = 1.0
    channels, manifest = encode([("person_1", a), ("vehicle_2", b)])
    attrs = header_metadata(ch.CRYPTOMATTE_TYPENAME, manifest)
    out = str(tmp_path / "c.exr")  # type: ignore[operator]
    ExrSidecarWriter().write_frame(out, channels, attrs=attrs)

    inp = oiio.ImageInput.open(out)
    s = inp.spec()
    assert set(ch.CRYPTOMATTE_CHANNELS) <= set(s.channelnames)
    md = {p.name: p.value for p in s.extra_attribs}
    kh = keyhash(ch.CRYPTOMATTE_TYPENAME)
    assert md.get(f"cryptomatte/{kh}/name") == ch.CRYPTOMATTE_TYPENAME
    assert "person_1" in str(md.get(f"cryptomatte/{kh}/manifest", ""))
    px = np.array(inp.read_image(format="float")).reshape(s.height, s.width, s.nchannels)
    inp.close()
    ci = {n: i for i, n in enumerate(s.channelnames)}
    id0 = px[..., ci["CryptoObject00.R"]]
    assert _hex_of_float(id0[0, 0]) == name_to_id("person_1")[1]  # survives the EXR float32
