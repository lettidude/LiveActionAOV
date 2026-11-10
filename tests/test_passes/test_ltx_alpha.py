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
    pick_model,
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
    p = _FakeComfy(
        {
            "mode": "per_object",
            "long_edge": 128,
            "chunk": 17,
            "overlap": 8,
            "gate": "dilate",
            **params,
        }
    )
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


def _info(unets: list[str], ggufs: list[str], tes: list[str], te_ggufs: list[str]) -> dict:
    return {
        "UNETLoader": {"input": {"required": {"unet_name": [unets]}}},
        "UnetLoaderGGUF": {"input": {"required": {"unet_name": [ggufs]}}},
        "CLIPLoader": {"input": {"required": {"clip_name": [tes]}}},
        "CLIPLoaderGGUF": {"input": {"required": {"clip_name": [te_ggufs]}}},
    }


def test_pick_model_prefers_the_int8_then_the_best_gguf() -> None:
    from live_action_aov.passes.matte.ltx_alpha import _GGUF_UNET

    want = LTXAlphaRefinerPass.DEFAULT_PARAMS["unet"]
    ggufs = [
        r"LTX\ltx-2.5-22b-distilled-Q4_K_M.gguf",
        r"LTX\ltx-2.5-22b-distilled-Q8_0.gguf",
        r"LTX\ltx-2.5-22b-dev-Q8_0.gguf",  # not the distilled model
        "LTX-2.3-distilled-Q4_K_S.gguf",
    ]
    info = _info([want], ggufs, [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) == (
        "UNETLoader",
        want,
    )
    info = _info([], ggufs, [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) == (
        "UnetLoaderGGUF",
        r"LTX\ltx-2.5-22b-distilled-Q8_0.gguf",
    )
    info = _info([], ggufs[2:], [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) is None


def test_graph_uses_the_gguf_loaders_and_uploads_to_a_remote_comfyui(tmp_path: Path) -> None:
    sent: list[str] = []

    class _Remote(LTXAlphaRefinerPass):
        pass

    p = _Remote({"comfy_url": "http://ai-box:8188"})
    p._models = {
        "unet": ("UnetLoaderGGUF", "ltx-2.5-22b-distilled-Q4_K_M.gguf"),
        "text_encoder": ("CLIPLoaderGGUF", "gemma4-12b-with-proj-ltx-2.5-Q4_K_M.gguf"),
    }
    (tmp_path / "f_0000.png").write_bytes(b"png")
    import live_action_aov.passes.matte.ltx_alpha as mod

    orig = mod.upload_image
    mod.upload_image = lambda url, path, sub: sent.append(f"{url}|{path.name}|{sub}")  # type: ignore[assignment]
    try:
        g = p._graph(tmp_path, 512, 288, 9)
    finally:
        mod.upload_image = orig
    assert g["1"] == {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "ltx-2.5-22b-distilled-Q4_K_M.gguf"},
    }
    assert g["5"]["class_type"] == "CLIPLoaderGGUF"
    assert g["8"]["class_type"] == "VHS_LoadImages"
    assert sent and sent[0].startswith("http://ai-box:8188/upload/image|f_0000.png|laov_ltx_")
    assert g["8"]["inputs"]["directory"] == sent[0].rsplit("|", 1)[1]
    # Mattes come back through ComfyUI's temp folder, never its output/.
    assert g["21"]["class_type"] == "PreviewImage"
    # Same machine: the temp folder is read in place, nothing is uploaded.
    local = LTXAlphaRefinerPass({"comfy_url": "http://127.0.0.1:8188"})
    assert local._graph(tmp_path, 512, 288, 9)["8"]["class_type"] == "VHS_LoadImagesPath"


def test_pick_model_finds_the_same_file_in_another_folder() -> None:
    from live_action_aov.passes.matte.ltx_alpha import _GGUF_UNET

    want = LTXAlphaRefinerPass.DEFAULT_PARAMS["unet"]
    moved = "ltx25\\" + want
    info = _info([moved], [], [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) == (
        "UNETLoader",
        moved,
    )


def test_pick_model_takes_another_precision_before_a_gguf() -> None:
    from live_action_aov.passes.matte.ltx_alpha import _GGUF_UNET

    want = LTXAlphaRefinerPass.DEFAULT_PARAMS["unet"]  # the int8 file, not downloaded here
    bf16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    dev = "ltx-2.5-22b-dev-transformer-bf16.safetensors"  # another model: never picked
    info = _info([dev, bf16], ["LTX-2.5-Distilled-Q4_K_S.gguf"], [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) == (
        "UNETLoader",
        bf16,
    )
    info = _info([dev], [], [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) is None


def test_pick_model_matches_a_gguf_without_22b_in_the_name() -> None:
    from live_action_aov.passes.matte.ltx_alpha import _GGUF_UNET

    want = LTXAlphaRefinerPass.DEFAULT_PARAMS["unet"]
    info = _info([], ["LTX-2.5-Distilled-Q4_K_S.gguf", "LTX-2.3-distilled-Q8_0.gguf"], [], [])
    assert pick_model(info, want, "UNETLoader", "UnetLoaderGGUF", "unet_name", _GGUF_UNET) == (
        "UnetLoaderGGUF",
        "LTX-2.5-Distilled-Q4_K_S.gguf",
    )


def test_resolve_finds_lora_and_vaes_in_sub_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    import json as _json

    from live_action_aov.passes.matte import ltx_alpha as la

    p = LTXAlphaRefinerPass.DEFAULT_PARAMS
    info = _info(
        ["ltx-2.5-22b-distilled-transformer-bf16.safetensors"], [], [p["text_encoder"]], []
    )
    for n in la._NODES:
        info.setdefault(n, {"input": {"required": {}}})
    info["VHS_LoadImagesPath"] = {"input": {"required": {}}}
    lora = r"ltxv\ltx25\ltx-2.5-22b-ic-lora-alpha-gen-0.9.safetensors"
    info["LTXICLoRALoaderModelOnly"] = {"input": {"required": {"lora_name": [[lora]]}}}
    info["VAELoader"] = {"input": {"required": {"vae_name": [[p["video_vae"], p["audio_vae"]]]}}}

    class _R:
        def __init__(self, d: dict) -> None:
            self._b = _json.dumps(d).encode()

        def read(self) -> bytes:
            return self._b

    monkeypatch.setattr(la.urllib.request, "urlopen", lambda *a, **k: _R(info))
    ok, msg, models = la._resolve({})
    assert ok, msg
    assert models["lora"] == ("LTXICLoRALoaderModelOnly", lora)
    assert models["unet"] == ("UNETLoader", "ltx-2.5-22b-distilled-transformer-bf16.safetensors")


def _two_actors(n: int, h: int = 120, w: int = 240) -> tuple[list[np.ndarray], dict]:
    """Two white squares side by side (tracks 1 and 2), SAM shrunk by 3 px
    (a missed shoulder), and a far bright prop nobody tracks."""
    frames = []
    hard1, hard2 = np.zeros((n, h, w), np.uint8), np.zeros((n, h, w), np.uint8)
    for t in range(n):
        f = np.zeros((h, w, 3), np.float32)
        f[30:90, 40:80] = 1.0
        f[30:90, 90:130] = 1.0
        f[100:115, 200:235] = 1.0  # prop, > far_frac from both
        frames.append(f)
        hard1[t, 33:87, 43:77] = 1
        hard2[t, 33:87, 93:127] = 1
    art = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": list(range(1, n + 1)), "stack": hard1},
                2: {"label": "person", "frames": list(range(1, n + 1)), "stack": hard2},
            }
        },
        "sam3_instances": {
            0: [
                {
                    "track_id": 1,
                    "slot": "r",
                    "label": "person",
                    "score": 0.9,
                    "frames": list(range(1, n + 1)),
                },
                {
                    "track_id": 2,
                    "slot": "g",
                    "label": "person",
                    "score": 0.9,
                    "frames": list(range(1, n + 1)),
                },
            ]
        },
    }
    return frames, art


def _run_whole(n: int, **params: object) -> tuple[_FakeComfy, dict]:
    frames, art = _two_actors(n)
    _FakeComfy.calls = []
    p = _FakeComfy(
        {
            "mode": "whole_frame",
            "whole_long_edge": 256,
            "whole_chunk": 17,
            "whole_chunk_max": 0,
            "overlap": 8,
            **params,
        }
    )
    p.ingest_artifacts(art)
    return p, p.run_shot(_Reader(frames), frame_range=(1, n))


def test_whole_frame_runs_ltx_once_for_every_object() -> None:
    p, out = _run_whole(9)
    assert len(p.calls) == 1  # two objects, one call
    ww, wh, nf = p.calls[0]
    assert (ww, wh, nf) == (256, 128, 9)
    a, b, fg = out[5]["matte.r"], out[5]["matte.g"], out[5]["foreground.a"]
    assert a[60, 42] > 0.9 and a[60, 60] > 0.9  # SAM missed 40-42: LTX keeps it
    assert b[60, 92] > 0.9 and b[60, 110] > 0.9
    assert a[60, 110] < 0.05 and b[60, 60] < 0.05  # each keeps only its own
    assert a[107, 220] < 0.05 and b[107, 220] < 0.05  # far prop: no owner ...
    assert fg[107, 220] > 0.9  # ... but in the foreground layer
    assert fg[10, 10] < 0.05


def test_object_ltx_does_not_see_is_matted_without_it(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whole frame: LTX mattes the people, not the parked car. The car's
    share is empty, so it is matted by ViTMatte (a subject) and a wall in a
    hero slot keeps SAM's edge (scenery), instead of two empty slots."""
    n = 9
    frames, art = _two_actors(n)
    h, w = art["sam3_hard_masks"][0][1]["stack"].shape[1:]
    car = np.zeros((n, h, w), np.uint8)
    car[:, 95:118, 20:80] = 1  # dark in the plate: LTX returns haze there
    wall = np.zeros((n, h, w), np.uint8)
    wall[:, 2:20, 150:230] = 1
    art["sam3_hard_masks"][0][3] = {"label": "car", "frames": list(range(1, n + 1)), "stack": car}
    art["sam3_hard_masks"][0][4] = {
        "label": "building",
        "frames": list(range(1, n + 1)),
        "stack": wall,
        "scenery": True,
    }
    for tid, slot, lab in ((3, "b", "car"), (4, "a", "building")):
        art["sam3_instances"][0].append(
            {
                "track_id": tid,
                "slot": slot,
                "label": lab,
                "score": 0.5,
                "frames": list(range(1, n + 1)),
            }
        )

    class _Vit:
        calls = 0

        def _refine_instance(self, plate_stack: np.ndarray, hard_stack: np.ndarray) -> np.ndarray:
            type(self).calls += 1
            return (hard_stack > 0).astype(np.float32) * 0.8

    _FakeComfy.calls = []
    p = _FakeComfy({"whole_long_edge": 256, "whole_chunk": 17, "overlap": 8})
    monkeypatch.setattr(p, "_vitmatte", lambda: _Vit())
    p.ingest_artifacts(art)
    with caplog.at_level("WARNING"):
        out = p.run_shot(_Reader(frames), frame_range=(1, n))
    assert len(p.calls) == 1 and _Vit.calls == 1  # one LTX run; ViTMatte for the car only
    assert out[5]["matte.r"][60, 60] > 0.9  # the actors keep LTX's matte
    assert abs(float(out[5]["matte.b"][105, 50]) - 0.8) < 0.05  # the car: ViTMatte
    assert out[5]["matte.a"][10, 190] > 0.9  # the wall: SAM's mask
    assert sorted(p.unseen) == ["building", "car"]
    assert "not foreground for LTX" in caplog.text and "ViTMatte" in caplog.text
    # Off: the empty shares are kept as they are.
    p2 = _FakeComfy({"whole_long_edge": 256, "whole_chunk": 17, "overlap": 8, "whole_min_cover": 0})
    p2.ingest_artifacts(art)
    assert p2.run_shot(_Reader(frames), frame_range=(1, n))[5]["matte.b"][105, 50] < 0.05


def test_a_clip_just_over_the_chunk_is_one_call() -> None:
    from live_action_aov.passes.matte.ltx_alpha import single_call_chunk

    assert single_call_chunk(100, 97, 121) == 105  # one 8k+1 call, not two of 97
    assert single_call_chunk(121, 97, 121) == 121
    assert single_call_chunk(122, 97, 121) == 97  # too long for one call: chunks
    assert single_call_chunk(60, 97, 121) == 97  # shorter than a chunk: unchanged
    assert single_call_chunk(100, 97, 0) == 97  # off
    p, out = _run_whole(20, whole_chunk=17, whole_chunk_max=25)
    assert [nf for _, _, nf in p.calls] == [25]
    assert out[20]["matte.r"][60, 60] > 0.9


def test_whole_frame_is_the_default_mode() -> None:
    assert LTXAlphaRefinerPass({}).params["mode"] == "whole_frame"
    assert LTXAlphaRefinerPass({}).params["crop_labels"] == []


def test_crop_label_gets_its_own_run_inside_a_whole_frame_job() -> None:
    """`crop_labels`: that object is matted on its own rectangle (more
    pixels), the others keep their share of the whole-frame run."""
    frames, art = _two_actors(9)
    art["sam3_hard_masks"][0][2]["label"] = "girl"
    art["sam3_instances"][0][1]["label"] = "girl"
    _FakeComfy.calls = []
    p = _FakeComfy(
        {
            "whole_long_edge": 256,
            "whole_chunk": 17,
            "long_edge": 128,
            "chunk": 17,
            "overlap": 8,
            "gate": "dilate",
            "crop_labels": ["girl"],
        }
    )
    p.ingest_artifacts(art)
    out = p.run_shot(_Reader(frames), frame_range=(1, 9))
    sizes = sorted((ww, wh) for ww, wh, _ in p.calls)
    assert len(p.calls) == 2  # the whole frame once + the girl's crop
    assert (256, 128) in sizes and any(max(s) == 128 for s in sizes)
    a, b = out[5]["matte.r"], out[5]["matte.g"]
    assert a[60, 60] > 0.9 and a[60, 110] < 0.05  # whole-frame share
    assert b[60, 110] > 0.9 and b[60, 60] < 0.05  # the crop run, SAM-gated
    assert "foreground.a" in out[5]


def test_whole_frame_foreground_never_goes_to_scenery() -> None:
    """A building tracked behind the actor: the hair LTX finds just outside
    SAM's person mask lies INSIDE the building's mask. It is the person's
    (scenery owns no foreground), and the building gets none of it."""
    n = 9
    frames, art = _two_actors(n)
    wall = np.ones_like(art["sam3_hard_masks"][0][1]["stack"])
    wall[:, 33:87, 43:77] = 0  # the building is everything but the two actors
    wall[:, 33:87, 93:127] = 0
    art["sam3_hard_masks"][0][3] = {
        "label": "building",
        "frames": list(range(1, n + 1)),
        "stack": wall,
        "scenery": True,
    }
    _FakeComfy.calls = []
    p = _FakeComfy({"whole_long_edge": 256, "whole_chunk": 17, "overlap": 8})
    p.ingest_artifacts(art)
    out = p.run_shot(_Reader(frames), frame_range=(1, n))
    a = out[5]["matte.r"]
    assert a[60, 42] > 0.9  # SAM missed x 40-42; the wall mask covers it
    assert a[60, 77] > 0.9  # same on the other side
    assert p._owner_ids == [1, 2]  # the wall is not an owner


def test_whole_frame_chunks_long_shots_in_8n_plus_1_calls() -> None:
    p, out = _run_whole(40)
    assert len(p.calls) > 1
    assert all((nf - 1) % 8 == 0 for _, _, nf in p.calls)
    for t in (1, 20, 40):
        assert out[t]["matte.r"][60, 60] > 0.9


def test_ltx_cache_skips_the_second_run(tmp_path: Path) -> None:
    p1, out1 = _run_whole(9, cache_dir=str(tmp_path))
    assert len(p1.calls) == 1 and len(list(tmp_path.glob("ltx_*.npy"))) == 1
    p2, out2 = _run_whole(9, cache_dir=str(tmp_path))
    assert p2.calls == []  # served from the cache
    np.testing.assert_allclose(out1[5]["matte.r"], out2[5]["matte.r"], atol=2e-3)
    p3, _ = _run_whole(9, cache_dir=str(tmp_path), seed=7)
    assert len(p3.calls) == 1  # another seed is another result


# ---------------------------------------------------------------------------
# Scenery and the crop box (A65 couple-car plate, 2026-11-09: with the default
# concepts every object's box grew to the whole 2K proxy, and LTX, one salient
# foreground per crop, gave every object somebody else's alpha).
# ---------------------------------------------------------------------------


def test_scenery_neighbour_of_similar_size_does_not_grow_the_box() -> None:
    """A building box that passes the size ratio is still left out when it
    is scenery; a person of that size is still taken in whole."""
    own = (100, 100, 300, 200)  # y0, x0, y1, x1 -> padded 300 x 150 = 45k px
    building = (60, 180, 310, 430)  # 250 x 250 = 62.5k <= 1.5 x 45k: similar size
    person = (120, 210, 320, 330)
    with_person = grow_box(own, [person], 0.25, 1.5, 1080, 2048)
    assert with_person[3] >= 330  # subject neighbour: taken in whole
    kept_out = grow_box(own, [building], 0.25, 1.5, 1080, 2048, scenery=[True])
    assert kept_out == grow_box(own, [], 0.25, 1.5, 1080, 2048)
    # Same call without the flag reproduces the old blow-up.
    assert grow_box(own, [building], 0.25, 1.5, 1080, 2048)[3] >= 430


def test_box_stays_person_sized_inside_a_frame_sized_building() -> None:
    """Synthetic masks through the pass: a person inside a frame-sized
    building (scenery) with a chain of similar-size trees and a sky band.
    The crop stays the padded person box; a second person next to the first
    is still taken in whole."""
    H, W = 240, 400
    n = 3
    hard = {}
    person = np.zeros((n, H, W), np.uint8)
    person[:, 100:160, 100:130] = 1  # 60 x 30 -> padded 90 x 45
    friend = np.zeros((n, H, W), np.uint8)
    friend[:, 100:160, 135:165] = 1  # cut by the padded box (ends at 138) -> taken whole
    building = np.ones((n, H, W), np.uint8)  # the whole frame
    tree = np.zeros((n, H, W), np.uint8)
    tree[:, 60:180, 60:100] = 1  # 120 x 40 = 4800 <= 1.5 x 4050: "similar size"
    sky = np.zeros((n, H, W), np.uint8)
    sky[:, 0:70, :] = 1
    frames = list(range(1, n + 1))
    hard[1] = {"label": "person", "frames": frames, "stack": person, "scenery": False}
    hard[2] = {"label": "person", "frames": frames, "stack": friend, "scenery": False}
    hard[8] = {"label": "building", "frames": frames, "stack": building, "scenery": True}
    hard[4] = {"label": "tree", "frames": frames, "stack": tree, "scenery": True}
    hard[11] = {"label": "sky", "frames": frames, "stack": sky}  # old artifact: by label
    p = LTXAlphaRefinerPass({"pad": 0.25, "neighbour_max_ratio": 1.5})
    p._hard_masks = hard
    p._first = 1
    p._track_id = 1
    box = p._chunk_box(person, [0, 1, 2], H, W)
    assert box is not None
    y0, x0, y1, x1 = box
    assert (y1 - y0) < 100 and (x1 - x0) < 110, box  # person-sized, not 240 x 400
    assert x1 >= 165  # the friend is inside, whole
    assert y0 >= 85 and x0 >= 90  # neither the tree nor the sky pulled it out


def test_is_scenery_track_prefers_the_flag_over_the_label() -> None:
    from live_action_aov.passes.matte.ltx_alpha import _is_scenery_track

    assert _is_scenery_track({"label": "sky"})  # old artifact: by label
    assert not _is_scenery_track({"label": "sky", "scenery": False})  # a click named sky
    assert _is_scenery_track({"label": "hero", "scenery": True})
    assert not _is_scenery_track({"label": "person"})


def test_refine_all_skips_ltx_for_scenery_and_keeps_sam_mask(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """All-masks mode: the person goes to LTX on a person-sized crop; the sky
    (scenery) is not sent to LTX and comes out as SAM's mask, anti-aliased,
    in mask.sky; one log line says so."""
    n, h, w = 9, 120, 200
    frames, hard = _scene(n, h=h, w=w)
    sky = np.zeros((n, h, w), np.uint8)
    sky[:, 0:30, :] = 1
    for f in frames:
        f[0:30, :] = 0.3  # grey sky band (dark for the fake LTX)
    art = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "box", "frames": list(range(1, n + 1)), "stack": hard},
                2: {
                    "label": "sky",
                    "frames": list(range(1, n + 1)),
                    "stack": sky,
                    "scenery": True,
                },
            }
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
    p = _FakeComfy(
        {
            "mode": "per_object",
            "long_edge": 128,
            "chunk": 17,
            "overlap": 8,
            "gate": "dilate",
            "refine_all_masks": True,
            "refine_min_px": 0,
        }
    )
    p.ingest_artifacts(art)
    with caplog.at_level("INFO", logger="live_action_aov.passes.matte.ltx_alpha"):
        out = p.run_shot(_Reader(frames), frame_range=(1, n))
    assert len(p.calls) == 1  # the box only; nothing for the sky
    ww, wh, _ = p.calls[0]
    assert (ww, wh) == (
        128,
        96,
    )  # the padded moving square (66 x 54), not the 200 x 120 frame (128 x 64)
    m = out[5]["mask.sky"]
    assert m[10, 100] == 1.0 and m[60, 100] == 0.0  # SAM's band, solid
    assert out[5]["matte.r"][60, 40] > 0.9
    skipped = [r for r in caplog.records if "is scenery" in r.getMessage()]
    assert len(skipped) == 1 and "track 2 'sky'" in skipped[0].getMessage()
