"""A proxy run end to end through the executor, Cryptomatte in the run.

- `proxy_upscale: "nearest"`: the EXR at plate size holds exactly the IDs of
  the manifest, as many as at proxy size (bilinear invents IDs on edges);
- the header is safe for older Nuke Cryptomatte nodes (no non-string key
  containing "cryptomatte"; provenance under `liveaov/crypto_pass/`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.registry import get_registry
from live_action_aov.io.channels import CRYPTOMATTE_CHANNELS, CRYPTOMATTE_TYPENAME
from live_action_aov.io.crypto_header import unsafe_keys
from live_action_aov.io.cryptomatte import encode, header_metadata
from live_action_aov.io.oiio_io import HAS_OIIO
from live_action_aov.passes.matte.cryptomatte import CryptomattePass

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")

N = 3
W, H = 101, 61  # odd plate: the proxy -> plate factor is not an integer
PROXY = 40


class _Crypto(CryptomattePass):
    """The Cryptomatte pass on three fixed objects, at the size it is given."""

    name = "t_cryptomatte"  # the word in the name: its provenance keys must drop it
    requires_artifacts: list[str] = []

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        pass

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict[int, dict[str, Any]]:
        h, w = reader.read_frame(frame_range[0])[0].shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        a = (xx < w * 0.4).astype(np.float32)
        c = ((xx - w * 0.6) ** 2 + (yy - h * 0.5) ** 2 < (h * 0.3) ** 2).astype(np.float32)
        b = (1.0 - a) * (1.0 - c)
        out = {}
        for f in range(frame_range[0], frame_range[1] + 1):
            ch, self._manifest = encode(
                [("person_1", a), ("person_2", b), ("bag_3", c)], typename=CRYPTOMATTE_TYPENAME
            )
            out[f] = ch
        return out

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        return {"cryptomatte_header": {0: header_metadata(CRYPTOMATTE_TYPENAME, self._manifest)}}


@pytest.fixture(autouse=True)
def _register() -> None:
    get_registry().register_pass(_Crypto.name, _Crypto)


@pytest.fixture(scope="module")
def plate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.fixtures.generate_test_plate import generate

    folder = tmp_path_factory.mktemp("crypto_plate")
    generate(folder, frame_count=N, width=W, height=H)
    return folder


def _run(plate: Path, out: Path, upscale: str | None) -> None:
    shot = Shot(
        name="cx",
        folder=plate,
        output_dir=out,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, N),
        resolution=(W, H),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=[_Crypto.name],
        proxy_long_edge=PROXY,
    )
    params = {} if upscale is None else {"proxy_upscale": upscale}
    job = Job(shot=shot, passes=[PassConfig(name=_Crypto.name, params=params)])
    live_action_aov.run(job)
    assert job.shot.status == "done"


def _read(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import OpenImageIO as oiio

    i = oiio.ImageInput.open(str(path))
    s = i.spec()
    px = np.asarray(i.read_image(0, 0, 0, s.nchannels, oiio.FLOAT)).reshape(
        s.height, s.width, s.nchannels
    )
    attrs = {a.name: a.value for a in s.extra_attribs}
    i.close()
    return {c: px[..., k].copy() for k, c in enumerate(s.channelnames)}, attrs


def _ids(ch: dict[str, np.ndarray]) -> set[int]:
    out: set[int] = set()
    for name in CRYPTOMATTE_CHANNELS:
        if name.endswith((".R", ".B")) and not name.startswith(f"{CRYPTOMATTE_TYPENAME}."):
            out |= set(np.unique(ch[name].view(np.uint32)).tolist())
    return out - {0}


def test_nearest_keeps_the_id_count_at_plate_size(plate: Path, tmp_path: Path) -> None:
    import json

    _run(plate, tmp_path / "near", "nearest")
    files = sorted((tmp_path / "near" / "passes").glob("cx.utility.*.exr"))
    assert len(files) == N
    for p in files:
        ch, attrs = _read(p)
        assert ch[f"{CRYPTOMATTE_TYPENAME}00.R"].shape == (H, W)  # plate size
        manifest = json.loads(next(v for k, v in attrs.items() if k.endswith("/manifest")))
        want = {int(h, 16) for h in manifest.values()}
        assert _ids(ch) == want  # same IDs, same count (3) as at proxy size
        assert len(_ids(ch)) == 3
        # L3: header safe for older Nuke, provenance readable.
        assert unsafe_keys(attrs) == []
        assert attrs["liveaov/t_crypto_pass/model"] == _Crypto.name
        assert not any("cryptomatte" in k for k in attrs if k.startswith("liveaov/"))


def test_default_is_nearest(plate: Path, tmp_path: Path) -> None:
    _run(plate, tmp_path / "def", None)
    ch, _ = _read(sorted((tmp_path / "def" / "passes").glob("cx.utility.*.exr"))[0])
    assert len(_ids(ch)) == 3


def test_linear_still_gives_the_old_output(plate: Path, tmp_path: Path) -> None:
    """`proxy_upscale: linear` is the older behaviour: bilinear, extra IDs."""
    _run(plate, tmp_path / "lin", "linear")
    ch, _ = _read(sorted((tmp_path / "lin" / "passes").glob("cx.utility.*.exr"))[0])
    assert len(_ids(ch)) > 3
