"""MoGe-3 scale on the static background (no camera needed) and the FOV it
refuses from a camera the referee marked FAILED or nodal."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from live_action_aov.passes.normals import moge3 as m3
from live_action_aov.qc import scale as qs

H, W = 48, 64


def _frames(n: int, jitter, person_near=True, slide_px: float = 0.0):
    """Locked-off (or sliding) background at 4-12 m with a per-frame model scale
    `jitter`, and a person close to camera walking across (in the mask)."""
    big = np.linspace(4.0, 12.0, W + 40, dtype=np.float32)[None, :].repeat(H, axis=0)
    out = []
    for i in range(n):
        x0 = int(round(slide_px * i))
        z = big[:, x0 : x0 + W].copy()
        person = np.zeros((H, W), np.float32)
        if person_near:
            c = 5 + 5 * i
            person[:, c : c + 24] = 1.0
            z[person > 0] = 1.5
        fr = {
            "Z": (z * jitter[i]).astype(np.float32),
            "depth.confidence": np.ones((H, W), np.float32),
            "mask.objects": person,
        }
        if slide_px:
            fr["motion.x"] = np.full((H, W), -slide_px, np.float32)
            fr["motion.y"] = np.zeros((H, W), np.float32)
            fr["back.x"] = np.full((H, W), slide_px, np.float32)
            fr["back.y"] = np.zeros((H, W), np.float32)
        out.append(fr)
    return out


JIT = (1.0, 1.06, 0.95, 1.04, 0.97, 1.08, 0.94, 1.02)


def _score(frames):
    d = dict(enumerate(frames))
    m, _ = qs.scale_steps(list(d), d.__getitem__, object_px=1)
    return m["scale_step_med"]


def test_static_steps_read_the_model_jitter_not_the_person() -> None:
    fr = _frames(8, JIT)
    steps = m3.static_log_steps(fr, warp=False, object_px=1, sky=10000.0)
    want = np.diff(np.log(JIT))
    assert np.allclose(steps, want, atol=1e-6)


def test_whole_shot_window_removes_the_jitter_on_a_locked_off_shot() -> None:
    fr = _frames(8, JIT)
    steps = m3.static_log_steps(fr, warp=False, object_px=1, sky=10000.0)
    gains = m3.static_scale_gains(steps, [0.0] * len(steps), window=0)
    fixed = [{**f, "Z": f["Z"] * g} for f, g in zip(fr, gains, strict=True)]
    assert _score(fr) > 3.0
    assert _score(fixed) < 1e-3
    # One overall level: the median frame's, so the metres stay where they were.
    assert np.median(gains) == pytest.approx(1.0, abs=0.05)


def test_the_old_frame_median_follows_the_person() -> None:
    # No model jitter at all: only the person, walking toward camera (growing).
    fr = _frames(8, (1.0,) * 8, person_near=False)
    for i, f in enumerate(fr):
        f["mask.objects"][:, : 6 + 4 * i] = 1.0
        f["Z"][:, : 6 + 4 * i] = 1.5
    med = [float(np.median(f["Z"])) for f in fr]
    old = m3.scale_gains(med, 9)
    moved = [{**f, "Z": f["Z"] * g} for f, g in zip(fr, old, strict=True)]
    steps = m3.static_log_steps(fr, warp=False, object_px=1, sky=10000.0)
    new = m3.static_scale_gains(steps, [0.0] * len(steps), window=0)
    d = dict(enumerate(moved))
    m, _ = qs.scale_steps(list(d), d.__getitem__, object_px=1)
    assert m["scale_step_max"] > 1.0  # the person crossing half the frame becomes a scale jump
    assert np.allclose(new, 1.0)


def test_a_rolling_window_keeps_a_slow_real_change() -> None:
    # A slow 1 %/frame push-in that is real, plus model jitter.
    n = 24
    real = np.exp(-0.01 * np.arange(n))
    jit = np.array([JIT[i % len(JIT)] for i in range(n)])
    steps = list(np.diff(np.log(real * jit)))
    gains = np.array(m3.static_scale_gains(steps, [0.0] * len(steps), window=9))
    out = real * jit * gains
    mid = slice(5, n - 5)
    trend = np.polyfit(np.arange(n)[mid], np.log(out[mid]), 1)[0]
    assert trend == pytest.approx(-0.01, abs=0.002)
    assert np.std(np.diff(np.log(out[mid]))) < 0.25 * np.std(np.diff(np.log(real * jit)))


def test_unmeasured_pairs_fall_back_to_the_frame_step() -> None:
    gains = m3.static_scale_gains([0.05, None, -0.05], [9.0, 0.02, 9.0], window=0)
    c = np.array([0.0, 0.05, 0.07, 0.02])
    assert np.allclose(np.log(gains), np.median(c) - c)


def test_a_camera_move_is_measured_through_the_vectors() -> None:
    fr = _frames(6, JIT[:6], person_near=False, slide_px=3.0)
    steps = m3.static_log_steps(fr, warp=True, object_px=1, sky=10000.0)
    assert np.allclose(steps, np.diff(np.log(JIT[:6])), atol=1e-4)


def test_background_still() -> None:
    assert m3.background_still(_frames(4, (1.0,) * 4), still_px=0.5) is None  # no vectors
    assert m3.background_still(_frames(4, (1.0,) * 4, slide_px=3.0), still_px=0.5) is False
    fr = _frames(4, (1.0,) * 4, slide_px=3.0)
    for f in fr:
        f["motion.x"] = f["motion.x"] * 0.0
    assert m3.background_still(fr, still_px=0.5) is True


def _solve(verdict: str = "GOOD", nodal: bool | None = False, fx: float = 1000.0):
    intr = SimpleNamespace(source="solve", fx_px=fx, width_px=2000)
    return SimpleNamespace(intrinsics=intr, quality=SimpleNamespace(verdict=verdict, nodal=nodal))


def test_camera_fov_skips_failed_and_nodal_cameras() -> None:
    good = m3.camera_fov(_solve())
    assert good == pytest.approx(math.degrees(2 * math.atan(1.0)))
    assert m3.camera_fov(_solve("FAILED")) is None
    assert m3.camera_fov(_solve("FAILED - DO NOT USE")) is None
    assert m3.camera_fov(_solve(nodal=True)) is None
    assert m3.camera_fov(_solve("WEAK")) == pytest.approx(good)
    # The previous behaviour stays one param away.
    assert m3.camera_fov(_solve("FAILED"), trust="any") == pytest.approx(good)
    assert m3.camera_fov(_solve(nodal=True), trust="any") == pytest.approx(good)
    assert "FAILED" in m3.camera_refused(_solve("FAILED"))
    assert "nodal" in m3.camera_refused(_solve(nodal=True))
    assert m3.camera_refused(_solve()) == ""


def _pass_inputs(n: int = 8):
    fr = _frames(n, JIT)
    out = {
        1001 + i: {"Z": f["Z"], "Z_raw": f["Z"], "depth.confidence": f["depth.confidence"]}
        for i, f in enumerate(fr)
    }
    masks = np.stack([f["mask.objects"] for f in fr]).astype(np.uint8)
    track = {"label": "person", "frames": list(out), "stack": masks}
    medians = [float(np.median(f["Z"])) for f in fr]
    return list(out), out, medians, {"sam3_hard_masks": {1001: {0: track}}}


def test_without_masks_or_vectors_the_scale_is_as_before() -> None:
    frames, out, medians, _ = _pass_inputs()
    p = m3.MoGe3Pass({"object_px": 1})
    assert p._scale_gains(frames, out, medians) == m3.scale_gains(medians, 9)
    assert p.report_fields()["extra"]["scale_ref"] == "frame"


def test_with_masks_the_static_background_sets_the_scale() -> None:
    frames, out, medians, art = _pass_inputs()
    stack = art["sam3_hard_masks"][1001][0]["stack"]

    def score(g):
        fr = [
            {**out[f], "mask.objects": stack[i].astype(np.float32), "Z": out[f]["Z"] * g[i]}
            for i, f in enumerate(frames)
        ]
        return _score(fr)

    raw = score([1.0] * len(frames))
    # Masks only: the shot may move, so a rolling window.
    p = m3.MoGe3Pass({"object_px": 1})
    p.ingest_artifacts(art)
    windowed = score(p._scale_gains(frames, out, medians))
    assert p.report_fields()["extra"]["scale_ref"] == "static, 9 frames"
    assert windowed < 0.5 * raw
    # Motion vectors that say the background is still: the long window.
    zero = np.zeros((2, H, W), np.float32)
    still = {
        **art,
        "forward_flow": dict.fromkeys(frames, zero),
        "backward_flow": dict.fromkeys(frames, zero),
    }
    q = m3.MoGe3Pass({"object_px": 1})
    q.ingest_artifacts(still)
    assert score(q._scale_gains(frames, out, medians)) < 0.01
    assert q.report_fields()["extra"]["scale_ref"] == "static, 33 frames (still background)"
    # The previous behaviour stays one param away, even with masks in the run.
    r = m3.MoGe3Pass({"scale_ref": "frame", "object_px": 1})
    r.ingest_artifacts(still)
    assert r._scale_gains(frames, out, medians) == m3.scale_gains(medians, 9)


def test_too_little_background_is_flagged_and_falls_back() -> None:
    frames, out, medians, art = _pass_inputs()
    art["sam3_hard_masks"][1001][0]["stack"][:] = 1  # the whole frame is a person
    p = m3.MoGe3Pass({"object_px": 1})
    p.ingest_artifacts(art)
    assert p._scale_gains(frames, out, medians) == m3.scale_gains(medians, 9)
    rf = p.report_fields()
    assert rf["extra"]["scale_ref"] == "frame"
    assert any("scale not stabilised" in w for w in rf["warnings"])


def test_on_a_still_shot_everything_that_does_not_move_counts() -> None:
    """A parked car fills most of the frame and SAM masked it: masks alone
    leave too little to measure, but on a still shot it does not move, so it
    counts. The walking person (vectors say so) is still left out."""
    fr = _frames(6, JIT[:6], person_near=False)
    for i, f in enumerate(fr):
        f["Z"] = f["Z"].copy()
        f["mask.objects"][:, 20:] = 1.0  # the parked car, masked
        moving = np.zeros((H, W), np.float32)
        cols = slice(22 + 4 * i, 28 + 4 * i)
        moving[:, cols] = 2.0  # a person walking in front of it
        f["motion.x"], f["motion.y"] = moving, np.zeros((H, W), np.float32)
        f["back.x"], f["back.y"] = -moving, np.zeros((H, W), np.float32)
        f["Z"][:, cols] = 1.5 * (1 + i)  # wrong depth on the person, any value
    assert m3.background_still(fr, 0.5) is True
    masks_only = m3.static_log_steps(fr, warp=False, object_px=1, sky=10000.0)
    assert all(s is None for s in masks_only)
    still = m3.static_log_steps(fr, warp=False, object_px=1, sky=10000.0, still_px=0.5)
    assert np.allclose(still, np.diff(np.log(JIT[:6])), atol=1e-6)


def test_a_clip_run_in_windows_gets_the_same_gains_as_in_one_go() -> None:
    # A long clip is run in windows (cores + context each side): the rolling
    # scale must not step at the window cuts.
    from live_action_aov.core.hardware import WINDOW_OVERLAP, window_spans

    rng = np.random.default_rng(3)
    n = 130
    steps: list[float | None] = list(0.03 * rng.standard_normal(n - 1))
    steps[40] = None
    fallback = list(0.05 * rng.standard_normal(n - 1))
    win = m3.MoGe3Pass.DEFAULT_PARAMS["still_window"]
    assert win // 2 <= WINDOW_OVERLAP
    for w in (win, m3.MoGe3Pass.DEFAULT_PARAMS["scale_window"]):
        one_go = m3.static_scale_gains(steps, fallback, w)
        stitched = [0.0] * n
        for ra, rb, ca, cb in window_spans(n, 24, WINDOW_OVERLAP):
            g = m3.static_scale_gains(steps[ra : rb - 1], fallback[ra : rb - 1], w)
            stitched[ca:cb] = g[ca - ra : cb - ra]
        assert np.allclose(stitched, one_go, rtol=1e-9)
    # One level for the whole run does not survive windows: that is why it
    # is not the default.
    one_go = m3.static_scale_gains(steps, fallback, 0)
    ra, rb, ca, cb = window_spans(n, 24, WINDOW_OVERLAP)[2]
    assert not np.allclose(
        m3.static_scale_gains(steps[ra : rb - 1], fallback[ra : rb - 1], 0)[ca - ra : cb - ra],
        one_go[ca:cb],
    )
