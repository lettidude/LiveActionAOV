"""TemporalSmoother — flow-guided EMA warp-and-blend (design §9.1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from live_action_aov.post.temporal_smooth import TemporalSmoother
from live_action_aov.shared.optical_flow.cache import FlowCache


def test_zero_flow_blends_identity_with_configured_alpha() -> None:
    """With zero backward flow AND zero forward flow, warp is identity and
    occlusion is zero. The smoother should produce a clean EMA:
        out[t] = alpha * prev + (1 - alpha) * cur
    """
    h, w = 8, 8
    prev = np.full((h, w), 10.0, dtype=np.float32)
    cur = np.full((h, w), 20.0, dtype=np.float32)

    per_frame = {1: {"Z": prev}, 2: {"Z": cur}}
    cache = FlowCache()
    zero_flow = np.zeros((2, h, w), dtype=np.float32)
    cache.put("shotA", 1, "forward", zero_flow)
    cache.put("shotA", 2, "backward", zero_flow)

    smoother = TemporalSmoother({"applied_to": ["Z"], "alpha": 0.25})
    out = smoother.apply(per_frame, cache, "shotA")

    # Frame 1 untouched.
    assert np.allclose(out[1]["Z"], prev)
    # Frame 2 should be 0.25 * 10 + 0.75 * 20 = 17.5.
    assert np.allclose(out[2]["Z"], 17.5, atol=1e-4)


def test_occluded_pixels_keep_raw_value() -> None:
    """When forward/backward flows disagree beyond threshold, the pixel is
    marked occluded and the smoother must leave the raw value unchanged."""
    h, w = 4, 4
    prev = np.full((h, w), 1.0, dtype=np.float32)
    cur = np.full((h, w), 9.0, dtype=np.float32)
    per_frame = {1: {"Z": prev}, 2: {"Z": cur}}

    # fwd_prev and bwd_cur are both large but in the same direction → their
    # sum is ~2 * flow, not ~0 → F-B error is huge → fully occluded.
    cache = FlowCache()
    large_fwd = np.full((2, h, w), 5.0, dtype=np.float32)
    large_bwd = np.full((2, h, w), 5.0, dtype=np.float32)
    cache.put("shotA", 1, "forward", large_fwd)
    cache.put("shotA", 2, "backward", large_bwd)

    smoother = TemporalSmoother({"applied_to": ["Z"], "alpha": 0.9, "fb_threshold_px": 1.0})
    out = smoother.apply(per_frame, cache, "shotA")

    # With err ≈ 10 px and threshold 1 px, occlusion saturates to 1 → raw.
    assert np.allclose(out[2]["Z"], cur)


def test_first_frame_is_untouched() -> None:
    h, w = 4, 4
    cache = FlowCache()
    per_frame = {
        5: {"Z": np.full((h, w), 3.0, dtype=np.float32)},
        6: {"Z": np.full((h, w), 7.0, dtype=np.float32)},
    }
    zero = np.zeros((2, h, w), dtype=np.float32)
    cache.put("s", 5, "forward", zero)
    cache.put("s", 6, "backward", zero)
    smoother = TemporalSmoother({"applied_to": ["Z"], "alpha": 0.5})
    out = smoother.apply(per_frame, cache, "s")
    assert np.allclose(out[5]["Z"], 3.0)


def test_no_applied_to_is_noop() -> None:
    per_frame = {1: {"Z": np.ones((4, 4), dtype=np.float32)}}
    out = TemporalSmoother().apply(per_frame, FlowCache(), "shot")
    assert out is per_frame


def test_normals_triplet_stays_unit_length_after_blend() -> None:
    """Spec §10.3: |N| must equal 1 per pixel. Bilinear-blending three normal
    channels breaks that, so the smoother renormalizes the triplet when all
    three are in `applied_to`.
    """
    h, w = 4, 4
    # Orthogonal-ish unit vectors on prev / cur.
    prev_nx = np.full((h, w), 1.0, dtype=np.float32)
    prev_ny = np.zeros((h, w), dtype=np.float32)
    prev_nz = np.zeros((h, w), dtype=np.float32)
    cur_nx = np.zeros((h, w), dtype=np.float32)
    cur_ny = np.full((h, w), 1.0, dtype=np.float32)
    cur_nz = np.zeros((h, w), dtype=np.float32)

    per_frame = {
        1: {"N.x": prev_nx, "N.y": prev_ny, "N.z": prev_nz},
        2: {"N.x": cur_nx, "N.y": cur_ny, "N.z": cur_nz},
    }
    cache = FlowCache()
    zero = np.zeros((2, h, w), dtype=np.float32)
    cache.put("s", 1, "forward", zero)
    cache.put("s", 2, "backward", zero)

    smoother = TemporalSmoother(
        {"applied_to": ["N.x", "N.y", "N.z"], "alpha": 0.5, "fb_threshold_px": 1.0}
    )
    out = smoother.apply(per_frame, cache, "s")
    nx, ny, nz = out[2]["N.x"], out[2]["N.y"], out[2]["N.z"]
    mag = np.sqrt(nx**2 + ny**2 + nz**2)
    # Without renormalization, blending (1,0,0) and (0,1,0) with alpha=0.5
    # gives (0.5, 0.5, 0) → |N| = √0.5 ≈ 0.707. Renormalization must restore 1.
    assert np.allclose(mag, 1.0, atol=1e-5)
