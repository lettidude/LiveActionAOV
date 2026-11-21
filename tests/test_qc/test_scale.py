"""The scale-step referee on synthetic clips: a still depth reads zero, a
whole-frame pump is all scale step, local noise is residual, people are
left out, and a camera move is measured through the motion vectors."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.qc import scale as qs

H, W = 64, 96


def _clip(n: int, pump=(1.0,), noise: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.linspace(2.0, 10.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    out = {}
    for f in range(1, n + 1):
        z = base * pump[f % len(pump)] * (1 + noise * rng.standard_normal((H, W)))
        out[f] = {"Z": z.astype(np.float32), "depth.confidence": np.ones((H, W), np.float32)}
    return out


def test_a_still_depth_reads_zero() -> None:
    c = _clip(5)
    m, w = qs.scale_steps(list(c), c.__getitem__)
    assert not w and m["pairs"] == 4.0
    assert m["scale_step_med"] == 0.0 and m["rel_change_max"] == 0.0 and m["residual_med"] == 0.0


def test_a_pump_is_all_scale_step() -> None:
    c = _clip(7, pump=(1.0, 1.03))
    m, _ = qs.scale_steps(list(c), c.__getitem__)
    # Z_t / Z_t-1 alternates 1.03 and 1/1.03: both about 3 %.
    assert m["scale_step_med"] == pytest.approx(3.0, abs=0.1)
    assert m["residual_med"] == pytest.approx(0.0, abs=1e-6)
    assert qs.grade(m["scale_step_med"]) == "FAILED"


def test_local_noise_is_residual_not_scale() -> None:
    c = _clip(7, noise=0.02)
    m, _ = qs.scale_steps(list(c), c.__getitem__)
    assert m["scale_step_med"] < 0.2
    assert m["residual_med"] > 1.0 and m["rel_change_med"] > 1.0
    assert qs.grade(m["scale_step_med"]) == "GOOD"


def test_people_are_left_out() -> None:
    c = _clip(5)
    for f, fr in c.items():
        fr["Z"] = fr["Z"].copy()
        fr["Z"][:, :60] *= 0.5 if f % 2 else 1.0  # a person close to camera, most of the frame
        fr["mask.person"] = np.zeros((H, W), np.float32)
        fr["mask.person"][:, :60] = 1.0
    m, _ = qs.scale_steps(list(c), c.__getitem__, object_px=2)
    assert m["scale_step_med"] == 0.0
    assert m["static_share"] < 40.0
    bare = {f: {"Z": fr["Z"], "depth.confidence": fr["depth.confidence"]} for f, fr in c.items()}
    m2, _ = qs.scale_steps(list(bare), bare.__getitem__)
    assert m2["rel_change_max"] > m["rel_change_max"]


def test_too_little_background_warns() -> None:
    c = _clip(3)
    for fr in c.values():
        fr["mask.person"] = np.ones((H, W), np.float32)
    m, w = qs.scale_steps(list(c), c.__getitem__)
    assert m["pairs"] == 0.0 and "too little" in w[0]


def test_a_camera_move_is_measured_through_the_vectors() -> None:
    # The scene slides 3 px left per frame; depth itself does not change.
    big = np.linspace(2.0, 10.0, W + 30, dtype=np.float32)[None, :].repeat(H, axis=0)
    c = {}
    for f in range(1, 6):
        x0 = 3 * (f - 1)
        c[f] = {
            "Z": big[:, x0 : x0 + W].copy(),
            "depth.confidence": np.ones((H, W), np.float32),
            "motion.x": np.full((H, W), -3.0, np.float32),
            "motion.y": np.zeros((H, W), np.float32),
            "back.x": np.full((H, W), 3.0, np.float32),
            "back.y": np.zeros((H, W), np.float32),
        }
    same_pixel, _ = qs.scale_steps(list(c), c.__getitem__)
    warped, _ = qs.scale_steps(list(c), c.__getitem__, warp=True)
    assert same_pixel["rel_change_med"] > 1.0  # the slide reads as depth change
    assert warped["rel_change_max"] < 1e-3 and warped["scale_step_max"] < 1e-3
