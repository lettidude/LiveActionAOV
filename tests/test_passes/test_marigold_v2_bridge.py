"""Marigold V2 bridge passes — plumbing tests with the external process faked.

The real model runs in another environment (20B backbone, its own pins);
here we fake `_run_infer` to drop npy predictions where the subprocess
would, and check the PNG exchange, resize-back, colour decode, per-clip
depth normalisation, command construction and the missing-env error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from live_action_aov.io.channels import CH_ALBEDO_B, CH_ALBEDO_G, CH_ALBEDO_R, CH_Z, CH_Z_RAW
from live_action_aov.passes.intrinsic import marigold_v2 as mod
from live_action_aov.passes.intrinsic.marigold_v2 import (
    MarigoldV2AlbedoPass,
    MarigoldV2SeeThroughDepthPass,
    _fit_long_edge,
)


class _Reader:
    def __init__(self, frames: list[np.ndarray], first: int = 1) -> None:
        self.f, self.first = frames, first

    def read_frame(self, i: int):
        return self.f[i - self.first], {}


def _fake_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Make _check_env pass: a fake repo with infer.py, python, Qwen dir."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "infer.py").write_text("# fake", encoding="utf-8")
    (repo / "assets" / "checkpoints" / "Qwen-Image-Edit-2509").mkdir(parents=True)
    py = repo / "py.exe"
    py.write_text("", encoding="utf-8")
    return repo


def _params(repo: Path) -> dict:
    return {"repo_dir": str(repo), "env_python": str(repo / "py.exe"), "inference_long_edge": 64}


def test_fit_long_edge_caps_and_rounds_to_16() -> None:
    assert _fit_long_edge(1920, 1080, 1024) == (1024, 576)
    assert _fit_long_edge(640, 360, 1024) == (640, 352)  # never upscales; /16
    assert _fit_long_edge(8, 8, 1024) == (16, 16)


def test_missing_env_is_actionable(tmp_path: Path) -> None:
    p = MarigoldV2AlbedoPass({"repo_dir": str(tmp_path / "nope")})
    with pytest.raises(FileNotFoundError, match=r"setup_win\.sh"):
        p._load_model()


def test_infer_command_selects_modality_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_env(monkeypatch, tmp_path)
    a = MarigoldV2AlbedoPass(_params(repo))
    cmd = a._infer_command(tmp_path / "in", tmp_path / "out")
    assert cmd[0] == str(repo / "py.exe") and cmd[1].endswith("infer.py")
    assert cmd[cmd.index("--modality") + 1] == "albedo"
    assert "--checkpoint" not in cmd  # infer.py default for albedo
    d = MarigoldV2SeeThroughDepthPass(_params(repo))
    cmd = d._infer_command(tmp_path / "in", tmp_path / "out")
    assert cmd[cmd.index("--modality") + 1] == "depth"
    assert (
        cmd[cmd.index("--checkpoint") + 1]
        .replace("\\", "/")
        .endswith("Marigold-V2/depth/Log-layered")
    )


def test_albedo_roundtrip_srgb_to_linear_and_resize_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_env(monkeypatch, tmp_path)
    p = MarigoldV2AlbedoPass(_params(repo))
    n_in: dict[str, int] = {}

    def fake_run(in_dir: Path, out_dir: Path) -> None:
        pngs = sorted(in_dir.glob("*.png"))
        n_in["n"] = len(pngs)
        (out_dir / "predictions_npy").mkdir(parents=True)
        for png in pngs:
            # 3xHxW sRGB prediction at the exchange (downsized) resolution:
            # mid grey 0.5 sRGB -> ~0.214 linear.
            import cv2

            h, w = cv2.imread(str(png)).shape[:2]
            np.save(
                out_dir / "predictions_npy" / f"{png.stem}.npy", np.full((3, h, w), 0.5, np.float32)
            )

    monkeypatch.setattr(p, "_run_infer", fake_run)
    frames = [np.full((90, 160, 3), 0.3, np.float32) for _ in range(3)]
    out = p.run_shot(_Reader(frames), (1, 3))
    assert n_in["n"] == 3
    assert set(out) == {1, 2, 3}
    for ch in (CH_ALBEDO_R, CH_ALBEDO_G, CH_ALBEDO_B):
        assert out[2][ch].shape == (90, 160)  # resized back to the plate
        assert abs(float(out[2][ch].mean()) - 0.214) < 0.01  # sRGB decoded


def test_seethrough_depth_per_clip_normalised_near_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _fake_env(monkeypatch, tmp_path)
    p = MarigoldV2SeeThroughDepthPass(_params(repo))

    def fake_run(in_dir: Path, out_dir: Path) -> None:
        (out_dir / "predictions_npy").mkdir(parents=True)
        for k, png in enumerate(sorted(in_dir.glob("*.png"))):
            import cv2

            h, w = cv2.imread(str(png)).shape[:2]
            # log depth ramps left(near, 0) -> right(far, 1), frame k shifted
            ramp = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1)) + 0.1 * k
            np.save(out_dir / "predictions_npy" / f"{png.stem}.npy", ramp)

    monkeypatch.setattr(p, "_run_infer", fake_run)
    frames = [np.zeros((48, 96, 3), np.float32) for _ in range(3)]
    out = p.run_shot(_Reader(frames), (1, 3))
    z = out[1][CH_Z]
    assert z.shape == (48, 96) and out[1][CH_Z_RAW].shape == (48, 96)
    assert z[0, 0] > z[0, -1]  # near = 1, far = 0 (inverted log depth)
    # Per-CLIP normalisation: the +0.1*k shift must survive as a real
    # difference between frames (a per-frame norm would erase it).
    assert float(out[3][CH_Z].mean()) < float(out[1][CH_Z].mean())


def test_failed_subprocess_surfaces_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = _fake_env(monkeypatch, tmp_path)
    p = MarigoldV2AlbedoPass(_params(repo))

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "boom\nCUDA out of memory"

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        p._run_infer(tmp_path / "in", tmp_path / "out")


def test_registered_and_commercial_clean() -> None:
    for cls in (MarigoldV2AlbedoPass, MarigoldV2SeeThroughDepthPass):
        lic = cls.declared_license()
        assert lic.commercial_use is True and lic.spdx == "Apache-2.0"
        assert cls.requires_artifacts == []


def test_albedo_global_gain_drift_is_stabilised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-image predictions drift by a global gain frame to frame (measured
    on PF0071); the pass matches each frame's mean to the clip median."""
    repo = _fake_env(monkeypatch, tmp_path)
    p = MarigoldV2AlbedoPass(_params(repo))
    gains = [0.9, 1.0, 1.1, 1.0]

    def fake_run(in_dir: Path, out_dir: Path) -> None:
        (out_dir / "predictions_npy").mkdir(parents=True)
        for k, png in enumerate(sorted(in_dir.glob("*.png"))):
            import cv2

            h, w = cv2.imread(str(png)).shape[:2]
            np.save(
                out_dir / "predictions_npy" / f"{png.stem}.npy",
                np.full((3, h, w), 0.5 * gains[k], np.float32),
            )

    monkeypatch.setattr(p, "_run_infer", fake_run)
    frames = [np.zeros((32, 48, 3), np.float32) for _ in gains]
    out = p.run_shot(_Reader(frames), (1, 4))
    means = [float(out[f][CH_ALBEDO_G].mean()) for f in (1, 2, 3, 4)]
    assert max(means) - min(means) < 1e-4, means  # drift gone
    p2 = MarigoldV2AlbedoPass({**_params(repo), "stabilize_gain": False})
    monkeypatch.setattr(p2, "_run_infer", fake_run)
    out2 = p2.run_shot(_Reader(frames), (1, 4))
    m2 = [float(out2[f][CH_ALBEDO_G].mean()) for f in (1, 2, 3, 4)]
    assert m2[0] < m2[2]  # untouched when disabled
