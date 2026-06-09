"""DisplayTransformedReader strips NaN/Inf from plates (don't poison passes).

Comp-work plates routinely carry non-finite pixels. Without a guard they
sail through linearize → transform → model → sidecar (np.clip leaves NaN
alone, manual EV is a no-op, AgX only clips finite values), turning every
pass to NaN or a blank matte. These tests assert the reader sanitizes to
0.0 and logs a counted WARNING so the source is diagnosable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from live_action_aov.core.pass_base import DisplayTransformParams
from live_action_aov.io.readers.base import ImageSequenceReader
from live_action_aov.io.readers.display_transform_reader import (
    DisplayTransformedReader,
    _sanitize_nonfinite,
)


class _FakeReader(ImageSequenceReader):
    """Returns a fixed frame (optionally seeded with NaN/Inf)."""

    def __init__(self, frame: np.ndarray) -> None:
        super().__init__(Path("."), "plate.%04d.exr")
        self._frame = frame

    def frame_range(self) -> tuple[int, int]:
        return (1001, 1001)

    def resolution(self) -> tuple[int, int]:
        h, w = self._frame.shape[:2]
        return (w, h)

    def pixel_aspect(self) -> float:
        return 1.0

    def read_frame(self, frame: int) -> tuple[np.ndarray, dict[str, Any]]:
        # Fresh copy each call — sanitize must not depend on mutating ours.
        return self._frame.copy(), {"oiio:ColorSpace": "lin_rec709"}


def _corrupt_plate() -> np.ndarray:
    f = np.full((8, 8, 3), 0.18, dtype=np.float32)
    f[0, 0, 0] = np.nan
    f[1, 1, 1] = np.inf
    f[2, 2, 2] = -np.inf
    return f


# --- helper ----------------------------------------------------------

def test_sanitize_replaces_nonfinite_with_zero() -> None:
    out = _sanitize_nonfinite(_corrupt_plate(), frame=1001, where="test")
    assert np.isfinite(out).all()
    assert out[0, 0, 0] == 0.0
    assert out[1, 1, 1] == 0.0
    assert out[2, 2, 2] == 0.0
    # Untouched pixels survive.
    assert out[5, 5, 0] == np.float32(0.18)


def test_sanitize_clean_frame_is_noop_no_warning(caplog: Any) -> None:
    clean = np.full((4, 4, 3), 0.5, dtype=np.float32)
    with caplog.at_level(logging.WARNING):
        out = _sanitize_nonfinite(clean, frame=1, where="test")
    assert out is clean  # no copy when nothing to fix
    assert not caplog.records


def test_sanitize_logs_count(caplog: Any) -> None:
    with caplog.at_level(logging.WARNING):
        _sanitize_nonfinite(_corrupt_plate(), frame=1001, where="source plate")
    assert any("3 non-finite" in r.message and "1001" in r.message for r in caplog.records)


# --- reader integration ----------------------------------------------

def test_reader_output_is_finite_on_corrupt_plate() -> None:
    reader = DisplayTransformedReader(
        _FakeReader(_corrupt_plate()),
        params=DisplayTransformParams(manual_exposure_ev=0.0),
        colorspace_override="lin_rec709",
    )
    reader.analyze((1001, 1001))
    out, _ = reader.read_frame(1001)
    assert np.isfinite(out).all()
    assert out.dtype == np.float32


def test_reader_passthrough_path_also_sanitizes() -> None:
    # No analyze() call → passthrough branch. Must still strip NaN so a
    # test/preview path never hands non-finite pixels downstream.
    reader = DisplayTransformedReader(
        _FakeReader(_corrupt_plate()),
        params=DisplayTransformParams(),
        colorspace_override="lin_rec709",
    )
    out, _ = reader.read_frame(1001)
    assert np.isfinite(out).all()


def test_undecodable_colorspace_warns_once(caplog: Any, monkeypatch: Any) -> None:
    # Simulate an OCIO config that lacks the source colorspace: to_linear
    # raises, and the fallback can't decode a Log space → pixels pass
    # through unlinearized. The reader must surface this once.
    from live_action_aov.io import ocio_color

    def _boom(_frames: np.ndarray, _cs: str, *a: Any, **k: Any) -> np.ndarray:
        raise RuntimeError("colorspace 'ARRI LogC4' not found in config")

    monkeypatch.setattr(ocio_color, "to_linear", _boom)
    clean = np.full((4, 4, 3), 0.34, dtype=np.float32)
    reader = DisplayTransformedReader(
        _FakeReader(clean),
        params=DisplayTransformParams(manual_exposure_ev=0.0),
        colorspace_override="arri_logc4",
    )
    with caplog.at_level(logging.WARNING):
        reader.analyze((1001, 1001))
        reader.read_frame(1001)
        reader.read_frame(1001)
    decode_warnings = [r for r in caplog.records if "could not linearize" in r.message]
    assert len(decode_warnings) == 1  # once per shot, not per frame
    assert "arri_logc4" in decode_warnings[0].message
