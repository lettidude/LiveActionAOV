"""MoGe-3 pass: shot-level FOV / scale / letterbox logic with a fake model,
the OpenCV -> OpenGL normals, the Depth/Normals slot flags, and the
intrinsics hand-off to PositionFromDepth."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from live_action_aov.cli.app import _moge3_outputs  # noqa: E402
from live_action_aov.gui.submit_worker import _moge3_params  # noqa: E402
from live_action_aov.io.channels import (  # noqa: E402
    CH_DEPTH_CONFIDENCE,
    CH_N_X,
    CH_N_Y,
    CH_N_Z,
    CH_P_X,
    CH_Z,
    CH_Z_RAW,
)
from live_action_aov.passes.normals import moge3  # noqa: E402
from live_action_aov.post.position_from_depth import PositionFromDepth  # noqa: E402

H, W = 40, 64


class _FakeMoGe:
    """Depth 2 m everywhere x a per-call jitter; FOV 50 deg unless given;
    normals face the camera (OpenCV -Z); top rows are sky."""

    def __init__(self) -> None:
        self.calls: list[float | None] = []
        self.jitter = iter([1.0, 1.1, 0.9, 1.05, 0.95] * 20)

    def infer(self, image, fov_x=None, **_kw):
        self.calls.append(fov_x)
        _, h, w = image.shape
        fov = 50.0 if fov_x is None else fov_x
        fx = 0.5 / math.tan(math.radians(fov) / 2)
        mask = torch.ones(h, w, dtype=torch.bool)
        mask[:4] = False
        normal = torch.zeros(h, w, 3)
        normal[..., 2] = -1.0
        return {
            "depth": torch.full((h, w), 2.0) * next(self.jitter),
            "mask": mask,
            "normal": normal,
            "intrinsics": torch.tensor([[fx, 0, 0.5], [0, fx, 0.5], [0, 0, 1.0]]),
        }


class _FakePass(moge3.MoGe3Pass):
    def _load_model(self) -> None:  # type: ignore[override]
        if self._model is None:
            self._model = _FakeMoGe()
            self._device = torch.device("cpu")


class _Reader:
    def __init__(self, letterbox: int = 0) -> None:
        self.letterbox = letterbox

    def read_frame(self, f: int):
        img = np.full((H, W, 3), 0.5, np.float32)
        if self.letterbox:
            img[: self.letterbox] = 0.0
            img[-self.letterbox :] = 0.0
        return img, {}


def test_license_is_commercial_safe() -> None:
    lic = moge3.MoGe3Pass.declared_license()
    assert lic.spdx == "MIT" and lic.commercial_use is True


def test_sample_frames_spread_over_shot() -> None:
    assert moge3.sample_frames(list(range(1001, 1101)), 4) == [1001, 1034, 1067, 1100]
    assert moge3.sample_frames([1, 2], 8) == [1, 2]


def test_letterbox_box_finds_bars_and_ignores_slivers() -> None:
    img = np.full((100, 200, 3), 0.5, np.float32)
    img[:12] = 0
    img[-12:] = 0
    assert moge3.letterbox_box([img]) == (0, 12, 200, 76)
    dark_row = np.full((100, 200, 3), 0.5, np.float32)
    dark_row[0] = 0  # 1% sliver: not a bar
    assert moge3.letterbox_box([dark_row]) == (0, 0, 200, 100)


def test_scale_gains_remove_jitter_keep_a_dolly() -> None:
    jitter = [2.0, 2.2, 1.8, 2.1, 1.9, 2.0, 2.2, 1.8, 2.0]
    g = moge3.scale_gains(jitter, 9)
    fixed = [m * k for m, k in zip(jitter, g, strict=True)]
    assert np.ptp(fixed) < 1e-9
    dolly = list(np.geomspace(2.0, 4.0, 30))
    assert np.allclose(moge3.scale_gains(dolly, 5)[2:-2], 1.0, atol=1e-9)
    assert moge3.scale_gains([2.0, float("nan"), 2.0], 3)[1] == 1.0


def test_camera_fov_skips_guessed_focal() -> None:
    intr = SimpleNamespace(fx_px=1000.0, width_px=2000, source="colmap_solver")
    assert moge3.camera_fov(SimpleNamespace(intrinsics=intr)) == pytest.approx(90.0)
    intr.source = "guess"
    assert moge3.camera_fov(SimpleNamespace(intrinsics=intr)) is None
    assert moge3.camera_fov(None) is None


def test_run_shot_fixes_one_fov_and_stabilises_scale() -> None:
    p = _FakePass({"fov_samples": 3, "scale_window": 5})
    out = p.run_shot(_Reader(), (1, 10))
    model = p._model
    # 3 FOV-estimation calls without a FOV, then every frame with the median.
    assert model.calls[:3] == [None, None, None]
    assert all(c == pytest.approx(50.0) for c in model.calls[3:]) and len(model.calls) == 13
    assert "MoGe median" in p.fov_source
    zs = [float(np.median(out[f][CH_Z][out[f][CH_DEPTH_CONFIDENCE] > 0])) for f in range(1, 11)]
    raws = [float(out[f][CH_Z_RAW][5, 5]) for f in range(1, 11)]
    assert np.ptp(zs) < np.ptp(raws) / 3
    # Sky = far + invalid; normals in OpenGL (+Z toward camera).
    assert out[1][CH_Z][0, 0] == p.params["sky_depth_m"]
    assert out[1][CH_DEPTH_CONFIDENCE][0, 0] == 0.0
    assert np.allclose(out[1][CH_N_Z], 1.0) and np.allclose(out[1][CH_N_Y], 0.0)
    art = p.emit_artifacts()
    fx_norm = art["depth_intrinsics"][1][0]
    assert fx_norm * W == pytest.approx(W / (2 * math.tan(math.radians(25))), rel=1e-4)


def test_camera_track_fov_wins_and_letterbox_is_cropped() -> None:
    p = _FakePass({"fov_samples": 3})
    intr = SimpleNamespace(fx_px=1000.0, width_px=2000, source="colmap_solver")
    p.ingest_artifacts({"camera_solve": {1: SimpleNamespace(intrinsics=intr)}})
    out = p.run_shot(_Reader(letterbox=4), (1, 4))
    assert p.fov_source == "camera track"
    assert p._model.calls == [pytest.approx(90.0)] * 4  # no estimation runs
    assert out[1][CH_DEPTH_CONFIDENCE][:4].max() == 0.0  # bar = invalid
    assert out[1][CH_DEPTH_CONFIDENCE][8, 5] == 1.0  # picture below the fake sky
    _, _, cx, cy = p.emit_artifacts()["depth_intrinsics"][1]
    assert (cx, cy) == (pytest.approx(0.5), pytest.approx(0.5))


def test_slot_flags_limit_outputs() -> None:
    out = _FakePass({"emit_depth": False, "fov_samples": 2}).run_shot(_Reader(), (1, 2))
    assert CH_Z not in out[1] and CH_N_X in out[1]
    p = _FakePass({"emit_normals": False, "fov_samples": 2})
    out = p.run_shot(_Reader(), (1, 2))
    assert CH_N_X not in out[1] and CH_Z in out[1]


def test_smoother_only_touches_the_slots_it_emits() -> None:
    # Normals-only MoGe next to VDA depth must not smooth VDA's Z.
    cls = moge3.MoGe3Pass
    assert cls.smoothable_for({"emit_depth": False, "emit_normals": True}) == [
        CH_N_X,
        CH_N_Y,
        CH_N_Z,
    ]
    assert cls.smoothable_for({"emit_depth": True, "emit_normals": False}) == [CH_Z]
    assert set(cls.smoothable_for(cls.DEFAULT_PARAMS)) == set(cls.smoothable_channels)


def test_position_from_depth_uses_the_depth_pass_intrinsics() -> None:
    p = _FakePass({"fov_x_deg": 90.0})
    out = p.run_shot(_Reader(), (1, 1))
    post = PositionFromDepth()
    post.ingest_artifacts(p.emit_artifacts())
    res = post.apply(out, None, "s")  # type: ignore[arg-type]
    assert post.intrinsics_source == "depth_pass"
    assert post.resolved_intrinsics[0] == pytest.approx(W / 2)  # 90 deg -> fx = W/2
    assert res[1][CH_P_X][20, -1] > 0


def test_gui_and_cli_pick_the_slots() -> None:
    assert _moge3_params(["moge3_normals"]) == {"emit_depth": False, "emit_normals": True}
    assert _moge3_params(["moge3_depth", "moge3_normals"]) == {
        "emit_depth": True,
        "emit_normals": True,
    }
    assert _moge3_outputs(
        ["depth", "normals"], depth_backend="depth_anything_v2", normals_backend="moge3"
    ) == {"emit_depth": False, "emit_normals": True}
    assert _moge3_outputs(["moge3"], depth_backend="x", normals_backend="y") == {
        "emit_depth": True,
        "emit_normals": True,
    }
