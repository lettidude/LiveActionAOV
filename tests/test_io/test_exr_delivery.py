"""EXR delivery options: compression / bit-depth + Cryptomatte auto-split.

Bandwidth-bound pipelines need compact sidecars (half + DWA). But measured,
lossy codecs *and* a lossless float16 cast both corrupt Cryptomatte's exact
float32 hash IDs (the id->name match in Nuke breaks). The writer therefore
splits Cryptomatte into a lossless float32 sibling whenever the chosen
delivery would damage it, and leaves a single file when the delivery is
lossless float32.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.channels import CRYPTOMATTE_TYPENAME
from live_action_aov.io.oiio_io import HAS_OIIO, read_exr
from live_action_aov.io.writers.exr import ExrSidecarWriter

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")

H, W = 16, 24
_ID_CH = f"{CRYPTOMATTE_TYPENAME}00.R"
_COV_CH = f"{CRYPTOMATTE_TYPENAME}00.G"


def _crypto_id() -> np.float32:
    """A float32 whose bit-pattern matters (mirrors a real hash id)."""
    (f,) = struct.unpack("<f", b"\x9a\x4c\x1d\x3e")
    return np.float32(f)


def _channels_with_crypto() -> dict[str, np.ndarray]:
    return {
        "Z": np.linspace(0, 1, H * W, dtype=np.float32).reshape(H, W),
        "mask.person": np.full((H, W), 1.0, np.float32),
        _ID_CH: np.full((H, W), _crypto_id(), np.float32),
        _COV_CH: np.full((H, W), 0.73, np.float32),
    }


def _by_name(path: Path) -> dict[str, np.ndarray]:
    back, attrs = read_exr(path)
    names = list(attrs["channelnames"])
    return {n: back[..., names.index(n)] for n in names}


def test_default_is_single_lossless_file(tmp_path: Path) -> None:
    """Default delivery (zip/float32) writes one file, no crypto sibling."""
    writer = ExrSidecarWriter()
    out = tmp_path / "shot.utility.1001.exr"
    writer.write_frame(out, _channels_with_crypto())

    assert out.exists()
    assert not (tmp_path / "shot.utility.1001.crypto.exr").exists()
    got = _by_name(out)
    assert _ID_CH in got
    assert got[_ID_CH][0, 0] == _crypto_id()  # bit-exact in one lossless file


def test_dwa_half_splits_crypto_to_lossless_sibling(tmp_path: Path) -> None:
    """Compact lossy delivery: AOVs in the main file, Cryptomatte split to a
    lossless float32 sibling with IDs intact."""
    writer = ExrSidecarWriter(compression="dwab:45", dtype="float16")
    out = tmp_path / "shot.utility.1001.exr"
    writer.write_frame(out, _channels_with_crypto())

    crypto_path = tmp_path / "shot.utility.1001.crypto.exr"
    assert out.exists() and crypto_path.exists()

    main = _by_name(out)
    # Crypto must NOT be in the lossy main file.
    assert _ID_CH not in main and _COV_CH not in main
    # AOVs are in the main file.
    assert "Z" in main and "mask.person" in main

    crypto = _by_name(crypto_path)
    # IDs survive bit-exact in the lossless sibling — the whole point.
    assert crypto[_ID_CH][0, 0] == _crypto_id()
    assert np.all(crypto[_ID_CH] == _crypto_id())


def test_dwa_without_crypto_stays_single_file(tmp_path: Path) -> None:
    """No Cryptomatte → nothing to protect → single compact file."""
    writer = ExrSidecarWriter(compression="dwab:45", dtype="float16")
    out = tmp_path / "shot.utility.1001.exr"
    writer.write_frame(
        out,
        {"Z": np.linspace(0, 1, H * W, np.float32).reshape(H, W),
         "mask.person": np.full((H, W), 1.0, np.float32)},
    )
    assert out.exists()
    assert not (tmp_path / "shot.utility.1001.crypto.exr").exists()


def test_half_alone_triggers_split(tmp_path: Path) -> None:
    """Even lossless half corrupts crypto IDs, so it must split too."""
    writer = ExrSidecarWriter(compression="zip", dtype="float16")
    out = tmp_path / "f.1001.exr"
    writer.write_frame(out, _channels_with_crypto())
    assert (tmp_path / "f.1001.crypto.exr").exists()
    assert _by_name(tmp_path / "f.1001.crypto.exr")[_ID_CH][0, 0] == _crypto_id()


def test_compact_delivery_is_smaller(tmp_path: Path) -> None:
    """Sanity: half+DWA really is smaller than the float32+zip baseline.

    Uses a realistic-size smooth frame — at thumbnail sizes EXR header
    overhead swamps the codec win, so this only holds at real resolution
    (the 1080p measurement was ~5x: 0.26 -> 0.05 MB)."""
    bh, bw = 480, 640
    big = np.tile(np.linspace(0, 1, bw, np.float32), (bh, 1))
    chans = {"Z": big, "N.x": big, "motion.x": big}
    base = tmp_path / "base.exr"
    comp = tmp_path / "comp.exr"
    ExrSidecarWriter().write_frame(base, chans)
    ExrSidecarWriter(compression="dwab:45", dtype="float16").write_frame(comp, chans)
    assert comp.stat().st_size < base.stat().st_size
