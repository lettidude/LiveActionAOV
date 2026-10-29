"""Output preview: finds a run's layers and turns each into something viewable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.oiio_io import HAS_OIIO, write_exr

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "sh010" / "v002"
    (run / "passes").mkdir(parents=True)
    h, w = 24, 32
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None].repeat(w, 1)
    for f in (1001, 1002):
        px = np.stack(
            [
                yy * 10 + 1,  # Z (unbounded -> normalised)
                (yy > 0.5).astype(np.float32),  # mask.person
                (yy < 0.2).astype(np.float32),  # mask.hand
                yy - 0.5,  # N.x / N.y / N.z
                0.5 - yy,
                np.full_like(yy, 0.7),
                np.full_like(yy, 3.0),  # motion.u / motion.v
                np.full_like(yy, -3.0),
            ],
            axis=-1,
        )
        names = ["depth.Z", "mask.person", "mask.hand", "N.x", "N.y", "N.z", "motion.u", "motion.v"]
        write_exr(run / "passes" / f"sh010_v002.utility.{f}.exr", px, names)
    # Cryptomatte siblings are not displayable layers.
    write_exr(
        run / "passes" / "sh010_v002.utility.1001.crypto.exr",
        np.zeros((h, w, 1), np.float32),
        ["CryptoObject00.r"],
    )
    (run / "camera").mkdir()
    (run / "camera" / "sh010_v002.camera.json").write_text(
        json.dumps(
            {
                "intrinsics": {"fx_px": 1077.6},
                "extrinsics": [{}] * 63,
                "unregistered_frames": [1004],
                "quality": {"aggregate_reprojection_error_px": 0.743},
            }
        )
    )
    (run / "splat").mkdir()
    (run / "splat" / "sh010_v002.splat.json").write_text(
        json.dumps({"gaussians": 36894, "holdout_psnr_db": 35.0, "train_seconds": 32.0})
    )
    return run


def test_scan_finds_layers_and_frames(tmp_path: Path) -> None:
    from live_action_aov.gui.output_preview import scan_run

    layers = {layer.name: layer for layer in scan_run(_run(tmp_path))}
    # Each mask is its own entry (two masks are not a 2-channel image).
    assert set(layers) == {"depth", "mask.person", "mask.hand", "N", "motion"}
    assert layers["N"].channels == ["N.x", "N.y", "N.z"]
    assert layers["depth"].seq.frames == [1001, 1002]


def test_each_layer_displays_in_range(tmp_path: Path) -> None:
    from live_action_aov.gui.output_preview import layer_to_display, read_layer, scan_run

    for layer in scan_run(_run(tmp_path)):
        px = read_layer(layer, 1002)
        assert px.shape == (24, 32, len(layer.channels))
        rgb, matte = layer_to_display(layer, px)
        assert rgb.shape == (24, 32, 3) and rgb.min() >= 0.0 and rgb.max() <= 1.0
        # Only the mask is offered over the plate; depth gets normalised.
        assert (matte is not None) == layer.name.startswith("mask.")
        if layer.name == "depth":
            assert rgb.max() == 1.0 and rgb.min() == 0.0


def test_summary_reads_camera_and_splat_reports(tmp_path: Path) -> None:
    from live_action_aov.gui.output_preview import summary_lines

    cam, splat = summary_lines(_run(tmp_path))
    assert "63 frames solved" in cam and "1 lost" in cam and "0.74 px" in cam
    assert "36,894 gaussians" in splat and "35.0 dB" in splat
