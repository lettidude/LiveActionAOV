"""Steady metric depth: VDA's shape fitted to MoGe-3's metres, with both
models faked (synthetic scene with known depth)."""

from __future__ import annotations

import numpy as np
import pytest

from live_action_aov.io.channels import CH_DEPTH_CONFIDENCE, CH_P_Z, CH_Z, CH_Z_RAW
from live_action_aov.passes.depth import steady_metric as sm
from live_action_aov.passes.depth.video_depth_anything import VideoDepthAnythingPass
from live_action_aov.passes.normals.moge3 import MoGe3Pass
from live_action_aov.post.position_from_depth import PositionFromDepth

H, W = 24, 32
N = 20

# Ground truth: a ramp from 2 m (bottom) to 8 m (top), the same every frame.
TRUE_Z = np.linspace(8.0, 2.0, H, dtype=np.float32)[:, None].repeat(W, axis=1)


class _Reader:
    def read_frame(self, f: int):
        return np.full((H, W, 3), 0.5, np.float32), {}


def _fake_vda(metric: bool):
    """VDA: steady over time; relative = an affine of disparity, metric =
    depth at the wrong scale."""

    def infer(self, frames_u8):
        n = frames_u8.shape[0]
        if metric:
            one = TRUE_Z * 0.5
        else:
            one = 3.0 * (1.0 / TRUE_Z) + 0.2
        return np.repeat(one[None], n, axis=0).astype(np.float32)

    return infer


def _fake_depth_at(jitter: list[float]):
    """MoGe-3: true depth x a per-frame scale jitter, a moving 'person' that
    is wrong, sky on the top rows."""

    def depth_at(self, reader, frame_range, frames):
        self._intrinsics = (0.8, 1.2, 0.5, 0.5)
        out = {}
        for i, f in enumerate(frames):
            z = TRUE_Z * jitter[i % len(jitter)]
            z = z.copy()
            z[10:16, 4 + i % 8 : 10 + i % 8] = 0.7  # outlier blob
            mask = np.ones((H, W), bool)
            mask[:2] = False
            out[f] = (z.astype(np.float32), mask)
        return out

    return depth_at


@pytest.fixture
def fakes(monkeypatch):
    def install(metric: bool, jitter=(1.0, 1.08, 0.93, 1.04)):
        monkeypatch.setattr(VideoDepthAnythingPass, "_load_model", lambda self: None)
        monkeypatch.setattr(VideoDepthAnythingPass, "_infer_clip", _fake_vda(metric))
        monkeypatch.setattr(MoGe3Pass, "depth_at", _fake_depth_at(list(jitter)))

    return install


def test_license_is_commercial_safe() -> None:
    lic = sm.SteadyMetricDepthPass.declared_license()
    assert lic.commercial_use is True
    with pytest.raises(ValueError):
        sm.SteadyMetricDepthPass({"vda_source": "large"})


def test_reference_frames_include_the_last() -> None:
    assert sm.reference_frames(list(range(10, 20)), 4) == [10, 14, 18, 19]
    assert sm.reference_frames([5], 8) == [5]
    assert sm.reference_frames(list(range(3)), 1) == [0, 1, 2]


def test_robust_fit_recovers_disparity_affine_despite_outliers() -> None:
    rng = np.random.default_rng(0)
    z = rng.uniform(1.0, 20.0, 5000)
    x = (1.0 / z - 0.05) / 2.0  # so 1/z = 2 x + 0.05
    y = 1.0 / z
    y[:500] *= 3.0  # 10 % gross outliers
    a, b = sm.robust_fit(x, y, metric_src=False, keep=0.8)
    assert a == pytest.approx(2.0, rel=1e-3)
    assert b == pytest.approx(0.05, abs=1e-3)


def test_robust_fit_never_sends_far_surfaces_to_infinity() -> None:
    """A best fit with a negative shift would map small disparities to
    'beyond infinity'; the shift is held at zero instead."""
    x = np.array([0.02, 0.05, 0.5, 1.0, 2.0])
    y = 0.9 * x - 0.01  # true shift < 0
    y[:2] = [0.01, 0.03]
    a, b = sm.robust_fit(x, y, metric_src=False, keep=1.0)
    assert b >= 0.0 and a > 0.0
    z, valid = sm.apply_fit(np.array([[0.001, 0.02]]), a, b, metric_src=False, far=1e9)
    assert valid.all() and np.isfinite(z).all()


def test_robust_fit_metric_is_one_scale() -> None:
    x = np.log(np.array([1.0, 2.0, 4.0, 8.0]))
    s, b = sm.robust_fit(x, x + np.log(2.5), metric_src=True, keep=0.8)
    assert (s, b) == (pytest.approx(2.5), 0.0)


def test_window_fits_clip_is_constant_window_follows_a_drift() -> None:
    frames = list(range(0, 41))
    x = np.linspace(0.1, 1.0, 200)
    samples = {r: (x, 2.0 * (1 + r / 40) * x) for r in range(0, 41, 8)}
    clip = sm.window_fits(samples, frames, 0, metric_src=False, keep=1.0)
    assert len({clip[f] for f in frames}) == 1
    win = sm.window_fits(samples, frames, 1, metric_src=False, keep=1.0)
    assert win[0][0] == pytest.approx(2.0, rel=1e-6)
    assert win[40][0] == pytest.approx(4.0, rel=1e-6)
    assert win[4][0] == pytest.approx(2.2, rel=1e-6)  # linear between references


def test_apply_fit_marks_far_and_negative_as_invalid() -> None:
    d = np.array([[0.5, 0.0, -1.0]], np.float32)
    z, valid = sm.apply_fit(d, 1.0, 0.0, metric_src=False, far=100.0)
    assert valid.tolist() == [[True, False, False]]
    assert z[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("source", ["relative", "metric"])
def test_run_shot_gives_steady_metres(fakes, source) -> None:
    fakes(metric=source == "metric")
    p = sm.SteadyMetricDepthPass({"vda_source": source, "ref_every": 4})
    out = p.run_shot(_Reader(), (1, N))
    assert sorted(out) == list(range(1, N + 1))
    zs = np.stack([out[f][CH_Z] for f in out])
    # Metres: the robust pooled fit lands on the truth (MoGe's jitter
    # averages out, its outlier blob is dropped)...
    assert np.median(zs[0] / TRUE_Z) == pytest.approx(1.0, abs=0.03)
    # ...and nothing moves from frame to frame (VDA is steady).
    assert np.abs(zs - zs[0]).max() < 1e-5
    assert out[1][CH_DEPTH_CONFIDENCE].min() == 1.0
    assert CH_Z_RAW not in out[1]
    art = p.emit_artifacts()
    assert art["depth_intrinsics"][1].tolist() == pytest.approx([0.8, 1.2, 0.5, 0.5])
    assert "depth_metric" in art


def test_p_follows_the_delivered_z(fakes) -> None:
    fakes(metric=False)
    p = sm.SteadyMetricDepthPass({"ref_every": 5})
    out = p.run_shot(_Reader(), (1, 6))
    post = PositionFromDepth()
    post.ingest_artifacts(p.emit_artifacts())
    res = post.apply(out, flow_cache=None, shot_id="s")  # type: ignore[arg-type]
    assert np.array_equal(res[3][CH_P_Z], out[3][CH_Z])
    assert post.intrinsics_source == "depth_pass"


def test_vda_metric_weights_give_metres(monkeypatch) -> None:
    monkeypatch.setattr(VideoDepthAnythingPass, "_load_model", lambda self: None)
    monkeypatch.setattr(VideoDepthAnythingPass, "_infer_clip", _fake_vda(metric=True))
    p = VideoDepthAnythingPass({"metric": True})
    out = p.run_shot(_Reader(), (1, 3))
    assert np.allclose(out[2][CH_Z], TRUE_Z * 0.5)
    assert set(p.emit_artifacts()) == {"depth_metric"}


def test_vda_relative_output_is_unchanged(monkeypatch) -> None:
    """The metric switch off: per-clip normalised disparity, as before."""
    monkeypatch.setattr(VideoDepthAnythingPass, "_load_model", lambda self: None)
    monkeypatch.setattr(VideoDepthAnythingPass, "_infer_clip", _fake_vda(metric=False))
    p = VideoDepthAnythingPass({})
    out = p.run_shot(_Reader(), (1, 3))
    raw = 3.0 / TRUE_Z + 0.2
    assert np.allclose(out[1][CH_Z], (raw - raw.min()) / (raw.max() - raw.min()), atol=1e-6)
    assert set(p.emit_artifacts()) == {"depth_norm_min", "depth_norm_max"}


def test_gui_offers_it_without_changing_the_default() -> None:
    from live_action_aov.gui.pass_catalog import PASS_CATALOG, recommended_models

    keys = [e.key for e in PASS_CATALOG["Depth"]]
    assert "video_depth_steady" in keys and "moge3_depth" in keys
    assert "video_depth_steady" not in recommended_models()
    assert "moge3_depth" in recommended_models()
