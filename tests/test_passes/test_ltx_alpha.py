"""LTX-2.5 Alpha-Gen refiner: chunking, crop boxes, paste-back and the SAM
gate, with ComfyUI replaced by a fake that returns a known matte."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from live_action_aov.passes.matte.ltx_alpha import (
    LTXAlphaRefinerPass,
    chunk_spans,
    crossfade_weight,
    grow_box,
    levels,
    work_size,
)


def test_chunk_spans_cover_the_shot_in_8n_plus_1_calls() -> None:
    assert chunk_spans(20, 49, 16) == [(0, 25)]  # short: one call, rounded up
    assert chunk_spans(49, 49, 16) == [(0, 49)]
    spans = chunk_spans(150, 49, 16)
    assert all((n - 1) % 8 == 0 for _, n in spans)
    assert spans[0][0] == 0 and spans[-1][0] + spans[-1][1] == 150
    for (a, la), (b, _) in pairwise(spans):
        assert b - a <= la - 16  # at least `overlap` frames shared


def test_crossfade_weights_sum_to_one_over_the_overlap() -> None:
    n, ov = 49, 16
    step = n - ov
    for j in range(ov):
        a = crossfade_weight(step + j, n, ov, first=True, last=False)
        b = crossfade_weight(j, n, ov, first=False, last=True)
        assert 0.0 < a <= 1.0 and 0.0 < b <= 1.0
    assert crossfade_weight(0, n, ov, True, True) == 1.0


def test_box_takes_a_cut_neighbour_in_whole_but_not_the_car_behind() -> None:
    own = (100, 100, 300, 200)  # y0, x0, y1, x1
    person = (120, 210, 320, 330)  # cut by the padded box -> taken whole
    car = (0, 0, 1080, 2048)  # huge -> may be cut
    box = grow_box(own, [person, car], 0.25, 1.5, 1080, 2048)
    assert box[3] >= 330 and box[2] >= 320
    assert box != (0, 0, 1080, 2048)


def test_work_size_is_a_multiple_of_32_at_the_long_edge() -> None:
    assert work_size(400, 800, 1280) == (640, 1280)
    w, h = work_size(333, 517, 1280)
    assert w % 32 == 0 and h % 32 == 0 and max(w, h) == 1280


def test_levels_removes_the_haze() -> None:
    raw = np.full((2, 10, 10), 0.1, np.float32)
    raw[:, 3:7, 3:7] = 0.9
    out = levels(raw)
    assert out.min() == 0.0 and out.max() == 1.0


class _FakeComfy(LTXAlphaRefinerPass):
    """ComfyUI replaced: returns 0.1 haze with the crop's bright pixels at 0.9
    (the plate has a white square = the object)."""

    calls: list[tuple[int, int, int]] = []

    def _comfy_ready(self) -> bool:  # type: ignore[override]
        return True

    def _free(self) -> None:  # type: ignore[override]
        pass

    def _run_ltx(self, folder: Path, ww: int, wh: int, nf: int) -> np.ndarray:  # type: ignore[override]
        import cv2

        type(self).calls.append((ww, wh, nf))
        files = sorted(folder.glob("*.png"))
        assert len(files) == nf
        out = []
        for f in files:
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            out.append(np.where(g > 0.5, 0.9, 0.1).astype(np.float32))
        return np.stack(out)


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


def _scene(
    n: int, h: int = 120, w: int = 200, speed: int = 1
) -> tuple[list[np.ndarray], np.ndarray]:
    """A white square moving right; SAM's mask is the square shrunk by 2 px
    (LTX should give back the full square)."""
    frames, hard = [], np.zeros((n, h, w), np.uint8)
    for t in range(n):
        f = np.zeros((h, w, 3), np.float32)
        x = 20 + speed * t
        f[40:80, x : x + 40] = 1.0
        frames.append(f)
        hard[t, 42:78, x + 2 : x + 38] = 1
    return frames, hard


def _run(n: int, speed: int = 1, **params: object) -> tuple[_FakeComfy, dict]:
    frames, hard = _scene(n, w=200 + 5 * n * (speed - 1), speed=speed)
    art = {
        "sam3_hard_masks": {
            0: {1: {"label": "box", "frames": list(range(1, n + 1)), "stack": hard}}
        },
        "sam3_instances": {
            0: [
                {
                    "track_id": 1,
                    "slot": "r",
                    "label": "box",
                    "score": 0.9,
                    "frames": list(range(1, n + 1)),
                }
            ]
        },
    }
    _FakeComfy.calls = []
    p = _FakeComfy({"long_edge": 128, "chunk": 17, "overlap": 8, "gate": "dilate", **params})
    p.ingest_artifacts(art)
    return p, p.run_shot(_Reader(frames), frame_range=(1, n))


@pytest.mark.parametrize("n", [9, 40])
def test_matte_lands_back_in_place_across_chunks(n: int) -> None:
    p, out = _run(n)
    assert all((nf - 1) % 8 == 0 and ww % 32 == 0 and wh % 32 == 0 for ww, wh, nf in p.calls)
    if n > 17:
        assert len(p.calls) > 1  # chunked
    for t in (0, n // 2, n - 1):
        a = out[t + 1]["matte.r"]
        x = 20 + t
        assert a[60, x + 2] > 0.9  # SAM's edge, LTX's full square
        assert a[60, x + 20] > 0.95
        assert a[60, x + 50] < 0.05  # off the object
        assert a[10, 10] < 0.05


def test_overlap_frames_keep_full_alpha_when_the_chunk_boxes_differ() -> None:
    """A fast subject: each chunk's box covers a different part of the frame.
    In the overlap a pixel inside only ONE box must keep that chunk's alpha,
    not be averaged with the other chunk's zero (a hard cut in the hair)."""
    p, out = _run(40, speed=5)
    for t in range(9, 31):  # overlap frames of the 17/8 chunking
        x = 20 + 5 * t
        a = out[t + 1]["matte.r"]
        assert a[60, x + 3 : x + 37].min() > 0.9, t


def test_gate_cuts_foreground_far_from_the_sam_mask() -> None:
    frames, hard = _scene(9)
    for f in frames:
        f[90:110, 150:190] = 1.0  # another bright thing LTX would matte too
    art = {
        "sam3_hard_masks": {0: {1: {"label": "box", "frames": list(range(1, 10)), "stack": hard}}},
        "sam3_instances": {
            0: [
                {
                    "track_id": 1,
                    "slot": "r",
                    "label": "box",
                    "score": 0.9,
                    "frames": list(range(1, 10)),
                }
            ]
        },
    }
    p = _FakeComfy({"long_edge": 128, "pad": 2.0, "gate": "dilate"})  # crop sees it
    p.ingest_artifacts(art)
    a = p.run_shot(_Reader(frames), frame_range=(1, 9))[5]["matte.r"]
    assert a[100, 185] < 0.05


def test_comfyui_down_falls_back_to_vitmatte() -> None:
    class _Down(LTXAlphaRefinerPass):
        def _comfy_ready(self) -> bool:  # type: ignore[override]
            return False

    class _FakeVit:
        def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[no-untyped-def]
            return hard_stack.astype(np.float32) * 0.5

    p = _Down({})
    p._fallback = _FakeVit()  # type: ignore[assignment]
    frames, hard = _scene(3)
    out = p._refine_instance(np.stack(frames), hard)
    assert out.max() == pytest.approx(0.5)


def test_trimap_gate_keeps_the_object_and_drops_far_foreground() -> None:
    class _NoGuide:
        def _guide_alpha(self, rgb, binm):  # type: ignore[no-untyped-def]
            return None

    p = LTXAlphaRefinerPass({})
    p._fallback = _NoGuide()  # type: ignore[assignment]
    frames, hard = _scene(2)
    g = p._trimap_gate(np.stack(frames), hard)
    assert g[0, 60, 40] > 0.99  # on the object
    assert g[0, 60, 190] < 0.01  # far away
