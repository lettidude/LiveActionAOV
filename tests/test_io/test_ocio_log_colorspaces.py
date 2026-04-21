# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""ARRI LogC / Sony SLog3 → scene-linear round-trip via OCIO.

These tests pin the contract between our canonical short names
(`arri_logc4`, `sony_slog3`…) and the OCIO studio-config's verbose
names (`"ARRI LogC4"`, `"S-Log3 S-Gamut3"`…). If OCIO ships a config
rename upstream, these tests catch it before users do.

OCIO is expected to be available — the M2 GUI requires it at runtime.
If PyOpenColorIO isn't installed the whole suite skips so CI on a
stripped environment keeps working.
"""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.io import ocio_color

pytestmark = pytest.mark.skipif(not ocio_color.HAS_OCIO, reason="PyOpenColorIO not installed")


def _middle_grey_logc4() -> float:
    """Code value a scene-linear 0.18 grey encodes to under LogC4.

    Derived from ARRI's published LogC4 encode formula — middle-grey
    in LogC4 sits at ≈ 0.278 (not 0.41, which is LogC3). Hard-coding
    it here lets the OCIO decode test assert on a specific round-trip
    rather than a loose range.
    """
    return 0.2784


# Gamut conversion from the camera wide-gamut to ACES AP1 adds a few
# percent swing even on perfect neutrals (ARRI AWG3 / AWG4 primaries
# aren't equidistant from AP1's white point the way a chromatically-
# adapted matrix idealises). ±0.05 is wide enough to accommodate the
# real OCIO output while still catching formula regressions — if a
# decode lands outside this band the curve is broken, not just the
# gamut hop.
_MIDDLE_GREY_TOLERANCE = 0.05


def test_logc4_decodes_middle_grey_to_acescg_scene_linear():
    v = _middle_grey_logc4()
    frame = np.full((4, 4, 3), v, dtype=np.float32)
    out = ocio_color.to_linear(frame, "arri_logc4")
    assert out.shape == frame.shape
    assert abs(float(out.mean()) - 0.18) < _MIDDLE_GREY_TOLERANCE


def test_logc3_decodes_middle_grey_to_acescg_scene_linear():
    # LogC3 (EI800) puts middle grey at code value ≈ 0.4105 — the
    # classic value every compositor has memorised.
    v = 0.4105
    frame = np.full((4, 4, 3), v, dtype=np.float32)
    out = ocio_color.to_linear(frame, "arri_logc3")
    assert abs(float(out.mean()) - 0.18) < _MIDDLE_GREY_TOLERANCE


def test_slog3_decodes_middle_grey_to_scene_linear():
    # Sony S-Log3 middle grey is at code value ≈ 0.41 as well (close
    # to LogC3 by design — both target the same 10-bit grading space).
    v = 0.41
    frame = np.full((4, 4, 3), v, dtype=np.float32)
    out = ocio_color.to_linear(frame, "sony_slog3")
    assert abs(float(out.mean()) - 0.18) < _MIDDLE_GREY_TOLERANCE


def test_canonical_names_resolve_to_valid_ocio_colorspaces():
    # Map every canonical name through the resolver and confirm OCIO
    # recognises the result. Catches typos / renames in the studio-
    # config upstream.
    cfg = ocio_color.get_config()
    known = {cs.getName() for cs in cfg.getColorSpaces()}
    for canonical, ocio_name in ocio_color._CANONICAL_TO_OCIO.items():
        assert ocio_name in known, (
            f"Canonical name {canonical!r} maps to {ocio_name!r}, "
            f"which the studio-config does not recognise."
        )


def test_lying_tag_regression_plate_decodes_sanely():
    # Surrogate for the CAT / TRI plates: a uniform grey frame at
    # LogC4 p50 luma (from the real plate metrics) should decode to
    # around 0.35 ACEScg — midtone-ish but brighter than pure 0.18
    # (which is what we actually saw on the real plate pixels).
    v = 0.336  # CAT plate's p50 luma
    frame = np.full((4, 4, 3), v, dtype=np.float32)
    out = ocio_color.to_linear(frame, "arri_logc4")
    assert 0.2 < float(out.mean()) < 0.6
