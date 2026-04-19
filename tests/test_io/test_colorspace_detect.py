"""Tests for colorspace auto-detection + provenance.

The GUI surfaces the `reason` field verbatim to the user, so these
tests pin each branch of the detection ladder to a specific reason
string prefix. If those strings change the UI copy changes with them,
which is fine — but it should be a deliberate change, not a silent one.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.colorspace_detect import (
    SUPPORTED_COLORSPACES,
    detect_colorspace,
)


def test_explicit_header_wins_over_everything():
    attrs = {
        "colorspace": "lin_rec709",
        "chromaticities": [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767],
    }
    d = detect_colorspace(attrs)
    assert d.detected == "lin_rec709"
    assert d.confident is True
    assert "header" in d.reason
    assert "colorspace" in d.reason  # the attribute name is in the reason


def test_oiio_header_variant_detected():
    # OIIO writes `oiio:ColorSpace` — the exact case the ACTIONVFX pack
    # uses, and the one the user needs to see called out in the UI.
    attrs = {"oiio:ColorSpace": "lin_rec709"}
    d = detect_colorspace(attrs)
    assert d.detected == "lin_rec709"
    assert "oiio:ColorSpace" in d.reason


def test_srgb_variants_normalize():
    for spelling in ("srgb", "sRGB", "srgb_texture", "sRGB_encoded"):
        attrs = {"colorspace": spelling}
        d = detect_colorspace(attrs)
        assert d.detected == "srgb_display", f"spelling {spelling!r} did not normalize"


def test_linear_rec709_spellings_normalize():
    for spelling in ("linear_rec709", "lin_rec709", "Rec709 Linear", "Linear Rec.709"):
        attrs = {"colorspace": spelling}
        d = detect_colorspace(attrs)
        assert d.detected == "lin_rec709", f"spelling {spelling!r} did not normalize"


def test_acescg_chromaticities_infer_when_no_explicit_tag():
    # ACES AP1 primaries, no explicit colorspace name.
    attrs = {
        "chromaticities": [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767],
    }
    d = detect_colorspace(attrs)
    assert d.detected == "acescg"
    assert d.confident is True
    assert "ACES AP1" in d.reason


def test_rec709_chromaticities_infer_when_no_explicit_tag():
    attrs = {
        "chromaticities": [0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290],
    }
    d = detect_colorspace(attrs)
    assert d.detected == "lin_rec709"
    assert "Rec.709" in d.reason


def test_fallback_when_no_header_no_pixels():
    # Zero signal — default to lin_rec709 with `confident=False` so the
    # UI can flag it as "tool is guessing".
    d = detect_colorspace({})
    assert d.detected == "lin_rec709"
    assert d.confident is False
    assert "defaulting" in d.reason


def test_pixel_range_flags_display_referred_on_lying_tag():
    # Synthesize display-referred-looking pixels: all in [0, 0.8], no
    # super-whites, median ~0.4. With no header tag, the pixel heuristic
    # should flip to srgb_display and mark itself as not-confident so
    # the UI warns the user.
    rng = np.random.default_rng(0)
    pixels = rng.uniform(0.05, 0.75, size=(64, 64, 3)).astype(np.float32)
    d = detect_colorspace({}, sample_pixels=pixels)
    assert d.detected == "srgb_display"
    assert d.confident is False
    assert "pixel range" in d.reason


def test_pixel_range_flags_scene_linear_on_super_whites():
    # Scene-linear with super-whites (exposure headroom). The p99 will
    # be > 1.2 so the heuristic flips to lin_rec709.
    rng = np.random.default_rng(0)
    pixels = rng.uniform(0.0, 1.5, size=(64, 64, 3)).astype(np.float32)
    d = detect_colorspace({}, sample_pixels=pixels)
    assert d.detected == "lin_rec709"
    assert d.confident is False
    assert "scene-linear" in d.reason


def test_explicit_tag_beats_pixel_heuristic():
    # Even if the pixels look display-referred, an explicit tag should
    # win — we report what the file SAYS, not what the pixels look like.
    # The user visually compares and overrides if needed; that's the
    # whole point of the provenance UI.
    rng = np.random.default_rng(0)
    pixels = rng.uniform(0.05, 0.75, size=(32, 32, 3)).astype(np.float32)
    attrs = {"colorspace": "lin_rec709"}
    d = detect_colorspace(attrs, sample_pixels=pixels)
    assert d.detected == "lin_rec709"
    assert d.confident is True
    assert "header" in d.reason


def test_unknown_colorspace_name_passes_through_verbatim():
    # An exotic / studio-specific tag we don't recognise: don't drop it
    # and don't silently remap — surface it verbatim so the user can
    # see what the plate is claiming and decide.
    attrs = {"colorspace": "my_studio_internal_space"}
    d = detect_colorspace(attrs)
    assert d.detected == "my_studio_internal_space"
    assert "my_studio_internal_space" in d.reason


def test_supported_list_includes_auto_sentinel():
    # The UI dropdown includes an `auto` entry meaning "use whatever the
    # detector picked". Confirm the sentinel is in the list.
    assert SUPPORTED_COLORSPACES[0] == "auto"
    assert "lin_rec709" in SUPPORTED_COLORSPACES
    assert "srgb_display" in SUPPORTED_COLORSPACES
    assert "acescg" in SUPPORTED_COLORSPACES
