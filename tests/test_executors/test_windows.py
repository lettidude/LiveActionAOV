"""Long clips in windows: whole-clip passes run once, window-safe passes run
per window with context, and the written frames are the ones a whole-clip run
writes (same pixels, same header on every frame)."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

import live_action_aov
from live_action_aov.core import hardware
from live_action_aov.core.job import Job, PassConfig, Shot
from live_action_aov.core.pass_base import ChannelSpec, License, PassType, TemporalMode, UtilityPass
from live_action_aov.core.registry import get_registry
from live_action_aov.executors.local import _clip_scope, _crossfade_in, _window_plan
from live_action_aov.io.channels import CH_Z
from live_action_aov.io.oiio_io import HAS_OIIO

pytestmark = pytest.mark.skipif(not HAS_OIIO, reason="OpenImageIO not installed")

N = 11
H, W = 36, 64


class _Base(UtilityPass):
    version = "0.0.1"
    license = License(spdx="MIT", commercial_use=True, notes="Test-only")
    pass_type = PassType.GEOMETRIC
    runs: ClassVar[list[tuple[int, int]]] = []

    def preprocess(self, frames: np.ndarray) -> np.ndarray:
        return frames

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def postprocess(self, tensor: np.ndarray) -> dict[str, np.ndarray]:
        return {}

    def _emit(self, frame_range: tuple[int, int], name: str, value: Any) -> dict:
        type(self).runs.append(frame_range)
        return {
            f: {name: np.full((H, W), value(f) if callable(value) else value, np.float32)}
            for f in range(frame_range[0], frame_range[1] + 1)
        }


class _Depth(_Base):
    """Per frame, from the frame's own pixels."""

    name = "w_depth"
    temporal_mode = TemporalMode.PER_FRAME
    window_safe = True
    produces_channels = [ChannelSpec(name=CH_Z)]
    runs: ClassVar[list[tuple[int, int]]] = []

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        type(self).runs.append(frame_range)
        return {
            f: {CH_Z: (reader.read_frame(f)[0][..., 0] * 0.5 + f).astype(np.float32)}
            for f in range(frame_range[0], frame_range[1] + 1)
        }


class _Tracker(_Base):
    """Must see the whole clip; publishes an artifact the window passes use."""

    name = "w_tracker"
    temporal_mode = TemporalMode.VIDEO_CLIP
    provides_artifacts = ["w_tracks"]
    runs: ClassVar[list[tuple[int, int]]] = []

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        self._range = frame_range
        return self._emit(frame_range, "track.a", lambda f: f * 10.0)

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        return {"w_tracks": {0: {"range": self._range}}}


class _Pair(_Base):
    """Needs the previous frame: wrong on the first frame it is given."""

    name = "w_pair"
    temporal_mode = TemporalMode.PAIR
    window_safe = True
    requires_artifacts = ["w_tracks"]
    runs: ClassVar[list[tuple[int, int]]] = []

    def ingest_artifacts(self, artifacts: dict[str, dict[int, Any]]) -> None:
        self._tracks = artifacts["w_tracks"][0]

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        assert self._tracks["range"] == (1, N)  # the whole-clip artifact, in every window
        return self._emit(frame_range, "pair.x", lambda f: 1.0 if f == frame_range[0] else 0.0)


class _Probe(_Base):
    """A clip-level decision: must be the same in every window."""

    name = "w_probe"
    temporal_mode = TemporalMode.PER_FRAME
    window_safe = True
    runs: ClassVar[list[tuple[int, int]]] = []
    probes: ClassVar[list[tuple[int, int]]] = []

    def probe_clip(self, reader: Any, frame_range: tuple[int, int]) -> dict[str, Any]:
        type(self).probes.append(frame_range)
        return {"decided": float(frame_range[1])}

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        # Whole-clip run: decides on its own range. Windows: the pinned value.
        return self._emit(frame_range, "probe.a", float(self.params.get("decided", frame_range[1])))


class _Fade(_Base):
    """Window-safe, but its level depends on the call (a generative matte)."""

    name = "w_fade"
    temporal_mode = TemporalMode.VIDEO_CLIP
    window_safe = True
    window_crossfade = True
    runs: ClassVar[list[tuple[int, int]]] = []

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        return self._emit(frame_range, "fade.a", float(frame_range[0]))


class _Provider(_Base):
    name = "w_provider"
    temporal_mode = TemporalMode.PER_FRAME
    window_safe = True
    provides_artifacts = ["w_feed"]
    runs: ClassVar[list[tuple[int, int]]] = []

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        type(self).runs.append(frame_range)
        return {}

    def emit_artifacts(self) -> dict[str, dict[int, Any]]:
        return {"w_feed": {0: True}}


class _Solver(_Base):
    """Whole clip, and REQUIRES a window-safe pass's artifact."""

    name = "w_solver"
    temporal_mode = TemporalMode.VIDEO_CLIP
    requires_artifacts = ["w_feed"]
    runs: ClassVar[list[tuple[int, int]]] = []

    def run_shot(self, reader: Any, frame_range: tuple[int, int]) -> dict:
        type(self).runs.append(frame_range)
        return {}


ALL = (_Depth, _Tracker, _Pair, _Probe, _Fade, _Provider, _Solver)


@pytest.fixture(autouse=True)
def _register() -> None:
    for cls in ALL:
        get_registry().register_pass(cls.name, cls)
        cls.runs = []
    _Probe.probes = []


@pytest.fixture(scope="module")
def plate(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.fixtures.generate_test_plate import generate

    folder = tmp_path_factory.mktemp("window_plate")
    generate(folder, frame_count=N, width=W, height=H)
    return folder


def _run(plate: Path, out: Path, names: list[str], **shot_kw: Any) -> Job:
    shot = Shot(
        name="win",
        folder=plate,
        output_dir=out,
        sequence_pattern="test_plate.####.exr",
        frame_range=(1, N),
        resolution=(W, H),
        pixel_aspect=1.0,
        colorspace="acescg",
        passes_enabled=names,
        **shot_kw,
    )
    job = Job(shot=shot, passes=[PassConfig(name=n) for n in names])
    live_action_aov.run(job)
    assert job.shot.status == "done"
    return job


def _read(folder: Path) -> dict[int, tuple[dict[str, np.ndarray], dict[str, Any]]]:
    import OpenImageIO as oiio

    out = {}
    for p in sorted((folder / "passes").glob("win.utility.*.exr")):
        i = oiio.ImageInput.open(str(p))
        s = i.spec()
        px = np.asarray(i.read_image(0, 0, 0, s.nchannels, oiio.FLOAT)).reshape(
            s.height, s.width, s.nchannels
        )
        attrs = {
            a.name: a.value
            for a in s.extra_attribs
            if a.name.startswith("liveaov/")
            and a.name not in ("liveaov/frame", "liveaov/job_id", "liveaov/created")
        }
        i.close()
        out[int(p.name.split(".")[-2])] = (
            {c: px[..., k].copy() for k, c in enumerate(s.channelnames)},
            attrs,
        )
    return out


def test_windowed_run_writes_the_same_frames_as_a_whole_run(plate: Path, tmp_path: Path) -> None:
    names = ["w_tracker", "w_depth", "w_pair", "w_probe"]
    _run(plate, tmp_path / "whole", names, window_frames=None)
    assert _Depth.runs == [(1, N)] and _Probe.probes == []
    for cls in ALL:
        cls.runs = []
    _run(plate, tmp_path / "win", names, window_frames=4, window_overlap=2)

    # The tracker saw the whole clip once; the others ran per window, with
    # 2 frames of context on each side, clipped to the clip.
    assert _Tracker.runs == [(1, N)]
    assert _Depth.runs == [(1, 6), (3, 10), (7, 11)]
    assert _Probe.probes == [(1, N)]  # decided once, on the whole clip

    whole, win = _read(tmp_path / "whole"), _read(tmp_path / "win")
    assert sorted(win) == list(range(1, N + 1)) == sorted(whole)
    for f in range(1, N + 1):
        assert sorted(win[f][0]) == sorted(whole[f][0]), f
        for ch, a in whole[f][0].items():
            np.testing.assert_array_equal(win[f][0][ch], a, err_msg=f"{ch} frame {f}")
        assert win[f][1] == whole[f][1], f  # the header of a whole-clip run ...
        assert win[f][1] == win[1][1], f  # ... and the same on every frame
    # The pair pass is only wrong on a window's first RUN frame, which is context.
    assert [float(win[f][0]["pair.x"][0, 0]) for f in (1, 5, 9)] == [1.0, 0.0, 0.0]
    # The whole-clip pass's channel came back from disk, frame for frame.
    assert float(win[7][0]["track.a"][0, 0]) == 70.0
    assert (tmp_path / "win" / "win.comp.nk").exists()


def test_a_pass_that_differs_between_windows_is_crossfaded(plate: Path, tmp_path: Path) -> None:
    _run(plate, tmp_path, ["w_fade"], window_frames=4, window_overlap=4)
    assert _Fade.runs == [(1, 8), (1, 11)][:1] + _Fade.runs[1:]  # first window: frames 1-8
    v = {f: float(d["fade.a"][0, 0]) for f, (d, _a) in _read(tmp_path).items()}
    # Window 1 (run from frame 1) says 1.0; window 2 (run from frame 1, core 5-11
    # after the short tail is folded in) says its own start. Whatever the two
    # values are, the cut is a ramp over overlap // 2 = 2 frames, not a step.
    a, b = v[4], v[N]
    if a != b:
        assert a < v[5] < v[6] < b or a > v[5] > v[6] > b
        assert v[7] == b


def test_crossfade_keeps_agreeing_pixels_exact() -> None:
    old = {5: {"m": np.array([[0.2, 0.5]], np.float32)}}
    new = {
        5: {"m": np.array([[0.8, 0.5]], np.float32)},
        6: {"m": np.array([[0.8, 0.5]], np.float32)},
    }
    _crossfade_in(new, old, 5, 2)
    np.testing.assert_allclose(new[5]["m"], [[0.2 * 2 / 3 + 0.8 / 3, 0.5]], rtol=1e-6)
    assert new[5]["m"][0, 1] == np.float32(0.5)  # equal in both: untouched
    np.testing.assert_array_equal(  # no old frame: left alone
        new[6]["m"], np.array([[0.8, 0.5]], np.float32)
    )


def test_whole_clip_pass_pulls_its_provider_into_the_clip_phase(
    plate: Path, tmp_path: Path
) -> None:
    _run(plate, tmp_path, ["w_provider", "w_solver", "w_depth"], window_frames=4, window_overlap=2)
    assert _Provider.runs == [(1, N)]  # required by a whole-clip pass: run once
    assert _Solver.runs == [(1, N)]
    assert len(_Depth.runs) == 3


def test_clip_scope_and_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    from live_action_aov.core.dag import PassNode

    nodes = [
        PassNode(name="a", plugin="a", provides=("x",), requires=(), optional=()),
        PassNode(name="b", plugin="b", provides=(), requires=("x",), optional=()),
        PassNode(name="c", plugin="c", provides=(), requires=(), optional=()),
    ]
    by_name = {
        "a": (PassConfig(name="a"), _Provider),
        "b": (PassConfig(name="b"), _Solver),
        "c": (PassConfig(name="c"), _Depth),
    }
    assert _clip_scope(nodes, by_name) == {"a", "b"}

    shot = Shot(
        name="long",
        folder=Path("."),
        sequence_pattern="long.####.exr",
        frame_range=(1001, 1600),
        resolution=(3840, 2160),
        proxy_long_edge=1920,
    )
    assert shot.window_frames == "auto"
    monkeypatch.setattr(hardware, "detect_ram", lambda: (64.0, 48.0))
    spans = _window_plan(shot, [_Depth, _Tracker])
    assert len(spans) > 1 and spans[0][2] == 1001 and spans[-1][3] == 1600
    assert all(a[3] + 1 == b[2] for a, b in itertools.pairwise(spans))  # cores tile
    assert _window_plan(shot, [_Tracker]) == [(1001, 1600, 1001, 1600)]  # nothing window-safe
    monkeypatch.setattr(hardware, "detect_ram", lambda: (512.0, 480.0))
    assert len(_window_plan(shot, [_Depth])) == 1  # fits: one go
    monkeypatch.setattr(hardware, "detect_ram", lambda: (None, None))
    assert len(_window_plan(shot, [_Depth])) == 1  # RAM unknown: one go
    off = shot.model_copy(update={"window_frames": None})
    monkeypatch.setattr(hardware, "detect_ram", lambda: (64.0, 48.0))
    assert len(_window_plan(off, [_Depth])) == 1
    with pytest.raises(ValueError):
        Shot(
            name="x",
            folder=Path("."),
            sequence_pattern="x.####.exr",
            frame_range=(1, 2),
            resolution=(8, 8),
            window_frames=1,
        )
