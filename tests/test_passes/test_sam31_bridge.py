"""SAM 3.1 detector: the bridge plumbing and the matte pass wiring, with the
SAM 3.1 environment replaced by fakes (it is a separate Python 3.12 env)."""

from __future__ import annotations

import json
import py_compile
import subprocess
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.passes.matte import sam31_bridge
from live_action_aov.passes.matte.sam3 import SAM3MattePass


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


def _stack(n: int, h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    st = np.zeros((n, h, w), np.uint8)
    st[:, y0:y1, x0:x1] = 1
    return st


class _NoSAM3(SAM3MattePass):
    """SAM 3 must never load in SAM 3.1 mode without clicks."""

    def _load_model(self) -> None:  # type: ignore[override]
        raise AssertionError("SAM 3 loaded")


def test_worker_is_valid_python_without_laov_imports() -> None:
    py_compile.compile(str(sam31_bridge.WORKER), doraise=True)
    src = sam31_bridge.WORKER.read_text(encoding="utf-8")
    assert "live_action_aov" not in src.split('"""', 2)[2]  # runs in another env


def test_find_python_explains_the_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(sam31_bridge.ENV_PYTHON, raising=False)
    with pytest.raises(RuntimeError, match=r"docs/install.md"):
        sam31_bridge.find_python({})
    exe = tmp_path / "python.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv(sam31_bridge.ENV_PYTHON, str(exe))
    assert sam31_bridge.find_python({}) == str(exe)


def test_find_python_uses_the_setup_folder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`liveaov setup-sam31` installs under default_dir(): found with no env var."""
    monkeypatch.delenv(sam31_bridge.ENV_PYTHON, raising=False)
    monkeypatch.setattr(sam31_bridge, "default_dir", lambda: tmp_path)
    exe = sam31_bridge.env_python(tmp_path)
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert sam31_bridge.find_python({}) == str(exe)


def test_keep_object_drops_small_and_brief_hits() -> None:
    assert sam31_bridge.keep_object(_stack(10, 100, 100, 10, 60, 10, 40), 0.005, 3)
    assert not sam31_bridge.keep_object(_stack(10, 100, 100, 0, 3, 0, 3), 0.005, 3)  # shelf item
    brief = _stack(10, 100, 100, 10, 60, 10, 40)
    brief[2:] = 0
    assert not sam31_bridge.keep_object(brief, 0.005, 3)


def test_run_sam31_writes_frames_runs_the_worker_and_reads_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict = {}

    def fake_run(cmd, **kw):  # type: ignore[no-untyped-def]
        a = dict(zip(cmd[3::2], cmd[4::2], strict=False))
        seen.update(cmd=cmd, env=kw["env"], n=len(list(Path(a["--frames"]).glob("*.jpg"))))
        out = Path(a["--out"])
        out.mkdir()
        np.savez_compressed(out / "obj_00_0000.npz", mask=_stack(4, 24, 32, 2, 20, 4, 12))
        (out / "result.json").write_text(
            json.dumps({"objects": [{"file": "obj_00_0000.npz", "concept": "person", "obj_id": 0}]})
        )
        return subprocess.CompletedProcess(cmd, 0, "built", "")

    monkeypatch.setattr(sam31_bridge.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "should-not-leak")
    frames = np.full((4, 24, 32, 3), 0.5, np.float32)
    objs = sam31_bridge.run_sam31(frames, ["person"], 2, "py312")
    assert seen["n"] == 4 and seen["cmd"][:3] == ["py312", "-I", str(sam31_bridge.WORKER)]
    assert json.loads(seen["cmd"][seen["cmd"].index("--concepts") + 1]) == ["person"]
    assert "PYTHONPATH" not in seen["env"]
    assert len(objs) == 1 and objs[0].label == "person" and objs[0].stack.shape == (4, 24, 32)


def test_run_sam31_reports_a_worker_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sam31_bridge.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "CUDA out of memory"),
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        sam31_bridge.run_sam31(np.zeros((2, 8, 8, 3), np.float32), ["person"], 0, "py312")


def test_matte_pass_uses_sam31_masks_without_loading_sam3(monkeypatch: pytest.MonkeyPatch) -> None:
    n, h, w = 5, 48, 64
    found = [
        sam31_bridge.Sam31Object("person", 0, _stack(n, h, w, 8, 44, 4, 24)),
        sam31_bridge.Sam31Object("person", 1, _stack(n, h, w, 8, 44, 36, 60)),
        sam31_bridge.Sam31Object("person", 2, _stack(n, h, w, 0, 1, 0, 2)),  # false hit
        # SAM 3.1 at plate res while the pass runs at a half proxy: resized
        sam31_bridge.Sam31Object("person", 3, _stack(n, 2 * h, 2 * w, 0, 30, 54, 70)),
    ]
    calls: dict = {}

    def fake(frames, concepts, seed, python, **kw):  # type: ignore[no-untyped-def]
        calls.update(n=len(frames), concepts=concepts, seed=seed)
        return found

    monkeypatch.setattr(sam31_bridge, "run_sam31", fake)
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    p = _NoSAM3(
        {
            "detector": "sam3.1",
            "concepts": ["person"],
            "layer_order": "area",
            "small_objects": "off",
        }
    )
    frames = [np.full((h, w, 3), 0.5, np.float32) for _ in range(n)]
    p.run_shot(_Reader(frames), frame_range=(1, n))
    assert calls == {"n": n, "concepts": ["person"], "seed": n // 2}
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    assert sorted(hard) == [1, 2, 3]  # false hit dropped, ids from 1
    assert all(d["label"] == "person" and d["stack"].shape == (n, h, w) for d in hard.values())
    assert hard[1]["stack"][2, 20, 10] == 1 and hard[2]["stack"][2, 20, 48] == 1


# --- detector: auto ----------------------------------------------------------


def test_auto_detector_picks_sam31_only_when_it_can_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from live_action_aov.passes.matte import sam3

    p = SAM3MattePass({})
    assert p.params["detector"] == "auto"
    assert p._resolve_detector()[0] == "sam3"  # not installed (conftest)
    monkeypatch.setattr(sam31_bridge, "available", lambda params=None: (True, "py312"))
    monkeypatch.setattr(sam3, "_total_vram_gb", lambda: 31.8)
    assert p._resolve_detector() == ("sam3.1", "installed")
    monkeypatch.setattr(sam3, "_total_vram_gb", lambda: 15.9)
    det, why = p._resolve_detector()
    assert det == "sam3" and "15.9 GB" in why  # small card: SAM 3, and it says why
    assert SAM3MattePass({"detector": "sam3"})._resolve_detector()[0] == "sam3"
    assert SAM3MattePass({"detector": "sam3.1"})._resolve_detector()[0] == "sam3.1"


class _FakeSAM3(SAM3MattePass):
    """SAM 3 replaced by one fixed detection, tracked as a constant mask."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_frame(self, frames, seed_local, concepts):  # type: ignore[no-untyped-def,override]
        m = np.zeros(frames.shape[1:3], np.uint8)
        m[8:44, 4:24] = 1
        return [(1, "person", m)]

    def _track_instance(  # type: ignore[no-untyped-def,override]
        self, frames, seed_frame_idx, seed_mask=None, *, points=None, labels=None, box=None
    ):
        return np.stack([np.asarray(seed_mask)] * len(frames), axis=0)


def test_auto_detector_falls_back_to_sam3_when_sam31_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from live_action_aov.passes.matte import sam3

    def boom(*a, **kw):  # type: ignore[no-untyped-def]
        raise RuntimeError("SAM 3.1 worker failed (exit 1): CUDA out of memory")

    monkeypatch.setattr(sam31_bridge, "available", lambda params=None: (True, "py312"))
    monkeypatch.setattr(sam31_bridge, "find_python", lambda p: "py312")
    monkeypatch.setattr(sam31_bridge, "run_sam31", boom)
    monkeypatch.setattr(sam3, "_total_vram_gb", lambda: 31.8)
    n, h, w = 5, 48, 64
    frames = [np.full((h, w, 3), 0.5, np.float32) for _ in range(n)]
    kw = {"concepts": ["person"], "layer_order": "area", "small_objects": "off"}
    p = _FakeSAM3({**kw, "joint_tracking": False, "redetect_stride": 0})
    with caplog.at_level("WARNING"):
        p.run_shot(_Reader(frames), frame_range=(1, n))
    assert "using SAM 3 for this shot" in caplog.text
    assert (p.detector_used, p.detector_reason) == ("sam3", "SAM 3.1 failed on this shot")
    hard = next(iter(p.emit_artifacts()["sam3_hard_masks"].values()))
    assert [d["label"] for d in hard.values()] == ["person"]
    # Asked for by name: the failure is the user's to see.
    strict = _FakeSAM3({**kw, "detector": "sam3.1"})
    with pytest.raises(RuntimeError, match="out of memory"):
        strict.run_shot(_Reader(frames), frame_range=(1, n))
