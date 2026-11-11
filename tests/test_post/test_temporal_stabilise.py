"""TemporalStabiliser: flow-warped symmetric window (median / mean / blend)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from live_action_aov.post.temporal_stabilise import (
    TemporalStabiliser,
    channel_kind,
    default_stabilisable,
)
from live_action_aov.shared.optical_flow.cache import FlowCache

H, W = 12, 16


def _cache(frames: list[int], dx: float = 0.0) -> FlowCache:
    """Constant translation by `dx` px per frame (content moves right)."""
    c = FlowCache()
    fwd = np.zeros((2, H, W), np.float32)
    fwd[0] = dx
    for f in frames:
        c.put("s", f, "forward", fwd)
        c.put("s", f, "backward", -fwd)
    return c


def _run(per_frame, cache, **params):
    p = {"device": "cpu", **params}
    return TemporalStabiliser(p).apply(per_frame, cache, "s")


def test_empty_applied_to_is_identity() -> None:
    per_frame = {1: {"Z": np.ones((H, W), np.float32)}}
    out = TemporalStabiliser().apply(per_frame, FlowCache(), "s")
    assert out is per_frame


def test_median_kills_a_single_frame_pop() -> None:
    frames = list(range(1, 8))
    per_frame = {f: {"Z": np.full((H, W), 4.0, np.float32)} for f in frames}
    per_frame[4]["Z"] = np.full((H, W), 9.0, np.float32)  # one-frame pop
    out = _run(per_frame, _cache(frames), applied_to=["Z"], window=5, mode="median")
    assert np.allclose(out[4]["Z"], 4.0, atol=1e-4)
    # Neighbours are not dragged towards the pop (it is outside the spread).
    assert np.allclose(out[3]["Z"], 4.0, atol=1e-4)
    # Input untouched.
    assert np.allclose(per_frame[4]["Z"], 9.0)


def test_mean_reduces_noise_and_is_not_recursive() -> None:
    rng = np.random.default_rng(0)
    frames = list(range(1, 12))
    per_frame = {
        f: {"matte.a": (0.5 + 0.02 * rng.standard_normal((H, W))).astype(np.float32)}
        for f in frames
    }
    out = _run(per_frame, _cache(frames), applied_to=["matte.*"], window=7, mode="mean")
    raw_std = np.std([per_frame[f]["matte.a"] for f in frames], axis=0).mean()
    new_std = np.std([out[f]["matte.a"] for f in frames[3:-3]], axis=0).mean()
    assert new_std < raw_std * 0.6


def test_translation_is_followed_by_the_warp() -> None:
    """A ramp moving 1 px/frame: warped neighbours line up, so a stable ramp
    comes out unchanged (no smearing across the motion)."""
    frames = list(range(1, 8))
    xs = np.arange(W, dtype=np.float32)
    per_frame = {f: {"Z": np.tile(1.0 + 0.1 * (xs - f), (H, 1)) + 5.0} for f in frames}
    out = _run(per_frame, _cache(frames, dx=1.0), applied_to=["Z"], window=5, mode="mean")
    core = (slice(None), slice(4, W - 4))
    assert np.allclose(out[4]["Z"][core], per_frame[4]["Z"][core], rtol=1e-4)


def test_inconsistent_flow_keeps_raw() -> None:
    frames = list(range(1, 6))
    per_frame = {f: {"Z": np.full((H, W), float(f), np.float32)} for f in frames}
    c = FlowCache()
    bad = np.full((2, H, W), 3.0, np.float32)  # fwd and bwd both +3: no round trip
    for f in frames:
        c.put("s", f, "forward", bad)
        c.put("s", f, "backward", bad)
    out = _run(per_frame, c, applied_to=["Z"], window=3, mode="mean")
    for f in frames:
        assert np.allclose(out[f]["Z"], per_frame[f]["Z"])


def test_one_sided_disagreement_never_overrides_the_frame() -> None:
    """Only the future side disagrees (as after a cut or a wrong flow on one
    side): the frame keeps its own value."""
    frames = list(range(1, 8))
    per_frame = {f: {"Z": np.full((H, W), 2.0 if f <= 4 else 8.0, np.float32)} for f in frames}
    out = _run(per_frame, _cache(frames), applied_to=["Z"], window=7, mode="median")
    assert np.allclose(out[4]["Z"], 2.0)
    assert np.allclose(out[5]["Z"], 8.0)


def test_normals_stay_unit_length() -> None:
    frames = [1, 2, 3]
    rng = np.random.default_rng(1)
    per_frame = {}
    for f in frames:
        n = np.stack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))]).astype(np.float32)
        n[:2] += 0.1 * rng.standard_normal((2, H, W)).astype(np.float32)
        n /= np.linalg.norm(n, axis=0, keepdims=True)
        per_frame[f] = {"N.x": n[0], "N.y": n[1], "N.z": n[2]}
    out = _run(per_frame, _cache(frames), applied_to=["N.x", "N.y", "N.z"], window=3)
    mag = np.sqrt(out[2]["N.x"] ** 2 + out[2]["N.y"] ** 2 + out[2]["N.z"] ** 2)
    assert np.allclose(mag, 1.0, atol=1e-5)
    assert not np.allclose(out[2]["N.x"], per_frame[2]["N.x"])


def test_matte_clamped_and_depth_holes_pass_through() -> None:
    frames = [1, 2, 3]
    per_frame = {
        f: {
            "matte.a": np.full((H, W), 1.0, np.float32),
            "Z": np.full((H, W), 3.0, np.float32),
        }
        for f in frames
    }
    per_frame[2]["Z"][0, 0] = 0.0  # hole
    per_frame[2]["Z"][0, 1] = np.nan
    out = _run(per_frame, _cache(frames), applied_to=["matte.a", "Z"], window=3, mode="blend")
    assert out[2]["matte.a"].max() <= 1.0
    assert out[2]["Z"][0, 0] == 0.0
    assert np.isnan(out[2]["Z"][0, 1])


def test_bad_params_raise() -> None:
    per_frame = {1: {"Z": np.ones((H, W), np.float32)}}
    with pytest.raises(ValueError):
        _run(per_frame, FlowCache(), applied_to=["Z"], window=4)
    with pytest.raises(ValueError):
        _run(per_frame, FlowCache(), applied_to=["Z"], mode="max")


def test_channel_kinds_and_defaults() -> None:
    assert channel_kind("Z") == "depth"
    assert channel_kind("splat.Z") == "depth"
    assert channel_kind("N.y") == "normal"
    assert channel_kind("mask.person") == "matte"
    assert channel_kind("ao.a") == "other"
    assert default_stabilisable(["Z", "Z_raw", "depth.confidence", "N.x", "motion.x"]) == [
        "Z",
        "N.x",
    ]
