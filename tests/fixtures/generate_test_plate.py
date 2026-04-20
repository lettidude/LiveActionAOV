"""Generate the synthetic test plate used across all phases.

Creates a short sequence at configurable resolution with:
- a smooth depth-varying gradient so depth passes have something to find
- a bright moving blob so flow passes have visible motion
- a uniform background in ACEScg-like working space

Generated on demand by `conftest.py` so the repo doesn't carry binary
fixtures. ~10 frames at 1920×1080 float32 is ~80 MB on disk — skipped if
already present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from live_action_aov.io.oiio_io import HAS_OIIO, write_exr


def generate(
    folder: Path,
    *,
    frame_count: int = 10,
    width: int = 1920,
    height: int = 1080,
    pattern: str = "test_plate.####.exr",
    first_frame: int = 1,
    force: bool = False,
) -> list[Path]:
    """Write a synthetic EXR sequence into `folder`.

    Returns the list of written paths.
    """
    if not HAS_OIIO:
        raise RuntimeError(
            "OpenImageIO is required to generate the test fixture. "
            "Install via `uv sync` (oiio-python wheel)."
        )
    folder.mkdir(parents=True, exist_ok=True)

    # Vertical depth-like gradient + a subtle lateral gradient.
    ys = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    base = 0.15 + 0.35 * ys + 0.1 * xs  # scene-referred mid-gray-ish values

    out: list[Path] = []
    for i in range(frame_count):
        frame_idx = first_frame + i
        t = i / max(1, frame_count - 1)

        # R/G/B with a small cyclic tint so the frames aren't identical.
        r = base + 0.05 * np.sin(2 * np.pi * (xs + t))
        g = base * (0.95 + 0.1 * t)
        b = base * (0.9 + 0.05 * np.cos(2 * np.pi * (ys + t)))

        # A bright circular "subject" sweeping left-to-right so optical flow
        # has unambiguous motion to estimate.
        cy = 0.5
        cx = 0.2 + 0.6 * t
        cy_px, cx_px = cy * height, cx * width
        yy = np.arange(height, dtype=np.float32)[:, None]
        xx = np.arange(width, dtype=np.float32)[None, :]
        r2 = (yy - cy_px) ** 2 + (xx - cx_px) ** 2
        mask = np.exp(-r2 / (2 * (width * 0.04) ** 2)).astype(np.float32)
        r = r + 2.0 * mask  # supra-white highlight, tests HDR path
        g = g + 1.5 * mask
        b = b + 0.5 * mask

        pixels = np.stack([r, g, b], axis=-1).astype(np.float32)

        name = pattern.replace("####", f"{frame_idx:04d}")
        out_path = folder / name
        if out_path.exists() and not force:
            out.append(out_path)
            continue
        write_exr(
            out_path,
            pixels,
            channel_names=["R", "G", "B"],
            attrs={
                "colorspace": "acescg",
                "compression": "zip",
                "liveaov/fixture/index": frame_idx,
            },
            pixel_aspect=1.0,
            compression="zip",
        )
        out.append(out_path)
    return out


__all__ = ["generate"]
