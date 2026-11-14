"""Stability QC on synthetic clips: both units, the frozen range, jitter,
masks left out, the edge band, and the warnings."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.qc import stability as qc

H, W = 64, 96


def _clip(n: int, noise: float = 0.0, pump: tuple[float, ...] = (1.0,), seed: int = 0):
    rng = np.random.default_rng(seed)
    base = np.linspace(2.0, 10.0, W, dtype=np.float32)[None, :].repeat(H, axis=0)
    zero = np.zeros((H, W), np.float32)
    frames = {}
    for f in range(1, n + 1):
        z = base * pump[f % len(pump)] * (1 + noise * rng.standard_normal((H, W)))
        frames[f] = {
            "Z": z.astype(np.float32),
            "depth.confidence": np.ones((H, W), np.float32),
            "motion.x": zero,
            "motion.y": zero,
            "back.x": zero,
            "back.y": zero,
        }
    return frames


def test_a_still_depth_reads_zero_and_the_range_is_frozen_over_the_clip() -> None:
    clip = _clip(5)
    m, w = qc.depth_stability(list(clip), clip.__getitem__)
    assert not w
    assert m["pairs"] == 4.0 and m["static_dlnz_med"] == 0.0 and m["jitter_med"] == 0.0
    lo, hi = qc.frozen_range(list(clip), clip.__getitem__, 5000.0)
    assert m["range_z"] == pytest.approx(hi - lo, rel=1e-6)
    assert lo == pytest.approx(2.0) and hi == pytest.approx(10.0)


def test_both_units_and_jitter() -> None:
    clip = _clip(6, pump=(1.0, 1.02))  # whole frame pumps by 2 %
    m, _ = qc.depth_stability(list(clip), clip.__getitem__)
    assert m["static_dlnz_med"] == pytest.approx(100 * np.log(1.02), rel=1e-3)
    assert m["jitter_med"] == pytest.approx(100 * np.log(1.02), rel=1e-3)
    # % of range: 2 % of each depth over the clip's ~8 m range, depth-dependent.
    assert 0.5 < m["static_range_med"] < 2.0
    noisy = _clip(6, noise=0.01)
    n, _ = qc.depth_stability(list(noisy), noisy.__getitem__)
    assert n["static_dlnz_med"] == pytest.approx(100 * 0.01 * np.sqrt(2) * 0.6745, rel=0.1)
    assert n["jitter_med"] < 0.05


def test_mattes_are_left_out_of_the_static_set() -> None:
    clip = _clip(4)
    for f, fr in clip.items():
        fr["Z"] = fr["Z"].copy()
        fr["Z"][:, :30] *= 1.5 if f % 2 else 1.0  # a moving thing on the left
        fr["mask.person"] = np.zeros((H, W), np.float32)
        fr["mask.person"][:, :30] = 1.0
    m, _ = qc.depth_stability(list(clip), clip.__getitem__, object_px=2)
    assert m["static_dlnz_med"] == 0.0 and m["static_dlnz_p95"] == 0.0
    assert m["static_share"] < 70.0


def test_the_edge_band_sees_a_depth_step() -> None:
    clip = _clip(4)
    for fr in clip.values():
        fr["Z"] = np.where(np.arange(W)[None, :] < W // 2, 2.0, 8.0).repeat(H, 0).astype(np.float32)
    m, _ = qc.depth_stability(list(clip), clip.__getitem__, edge_px=8)
    assert 0.0 < m["edge_share"] < 30.0
    assert m["edge_dlnz_med"] == 0.0


def test_warnings_without_flow_and_when_vectors_fail() -> None:
    clip = _clip(3)
    bare = {f: {"Z": fr["Z"]} for f, fr in clip.items()}
    m, w = qc.depth_stability(list(bare), bare.__getitem__)
    assert m == {} and "flow pass" in w[0]
    for fr in clip.values():
        fr["back.x"] = np.full((H, W), 5.0, np.float32)  # never closes the round trip
    m, w = qc.depth_stability(list(clip), clip.__getitem__)
    assert m["round_trip_valid"] == 0.0
    assert any("unreliable" in x for x in w) and any("too few" in x for x in w)


def test_sky_and_invalid_pixels_are_not_measured() -> None:
    clip = _clip(4)
    for f, fr in clip.items():
        fr["Z"] = fr["Z"].copy()
        fr["Z"][:10] = 10000.0  # sky fill
        fr["depth.confidence"] = fr["depth.confidence"].copy()
        fr["depth.confidence"][10:20] = 0.0
        fr["Z"][10:20] = float(f)  # junk under zero confidence
    m, _ = qc.depth_stability(list(clip), clip.__getitem__)
    assert m["static_dlnz_p95"] == 0.0
    assert m["range_z"] < 10.0


def test_samples_stay_bounded_on_a_long_shot(monkeypatch) -> None:
    kept: list[int] = []
    real = qc._sub

    def spy(a, cap):
        out = real(a, cap)
        kept.append(out.size)
        return out

    monkeypatch.setattr(qc, "_sub", spy)
    monkeypatch.setattr(qc, "_SAMPLES_TOTAL", 4000)
    monkeypatch.setattr(qc, "_SAMPLES_MIN", 10)
    clip = _clip(41, noise=0.01)
    qc.depth_stability(list(clip), clip.__getitem__)
    # 40 pairs share 4000 samples per set: about 100 per pair, whatever the frame size.
    assert kept and max(kept) <= 4000 // 40 * 1.1
