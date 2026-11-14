"""Matte ownership options, each behind a param (off = output as before):

- held objects: LTX whole-frame foreground that no track covers goes whole to
  the object whose hand touches it (`held_objects`);
- per-object soft mattes named like their Cryptomatte object
  (`object_channels` -> `mask_object.<label>_<id>`);
- proxy runs: Cryptomatte ID channels scaled back up nearest, never blended
  (`proxy_upscale: "nearest"` on the cryptomatte pass).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from live_action_aov.executors.local import _upscale_channels_to_plate
from live_action_aov.io.channels import CRYPTOMATTE_CHANNELS, object_channel
from live_action_aov.io.cryptomatte import encode
from live_action_aov.passes.matte.cryptomatte import CryptomattePass
from live_action_aov.passes.matte.ltx_alpha import LTXAlphaRefinerPass, assign_held
from live_action_aov.passes.matte.rvm import RVMRefinerPass

# --------------------------------------------------------------------------
# Held objects
# --------------------------------------------------------------------------


class _FakeComfy(LTXAlphaRefinerPass):
    """ComfyUI replaced: the plate's bright pixels are the foreground."""

    def _comfy_ready(self) -> bool:  # type: ignore[override]
        return True

    def _free(self) -> None:  # type: ignore[override]
        pass

    def _run_ltx(self, folder: Path, ww: int, wh: int, nf: int) -> np.ndarray:  # type: ignore[override]
        import cv2

        out = []
        for f in sorted(folder.glob("*.png")):
            g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            out.append(np.where(g > 0.5, 0.9, 0.1).astype(np.float32))
        return np.stack(out)


class _Reader:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._f = frames

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._f[idx - 1], {}


def _actor_with_a_bag(n: int, h: int = 120, w: int = 240) -> tuple[list[np.ndarray], dict]:
    """Two actors side by side; the first one's hand (a part) holds a long bag
    that hangs under BOTH actors. Nobody tracks the bag."""
    frames = []
    a1, a2 = np.zeros((n, h, w), np.uint8), np.zeros((n, h, w), np.uint8)
    hand = np.zeros((n, h, w), np.uint8)
    for t in range(n):
        f = np.zeros((h, w, 3), np.float32)
        f[30:90, 40:80] = 1.0
        f[30:90, 90:130] = 1.0
        f[90:104, 74:128] = 1.0  # the bag, hanging from the first actor's hand
        frames.append(f)
        a1[t, 33:87, 43:77] = 1
        a2[t, 33:87, 93:127] = 1
        hand[t, 80:89, 70:79] = 1
    fr = list(range(1, n + 1))
    art = {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": fr, "stack": a1},
                2: {"label": "person", "frames": fr, "stack": a2},
                3: {"label": "hand", "frames": fr, "stack": hand, "parent": 1},
            }
        },
        "sam3_instances": {
            0: [
                {"track_id": 1, "slot": "r", "label": "person", "score": 0.9, "frames": fr},
                {"track_id": 2, "slot": "g", "label": "person", "score": 0.9, "frames": fr},
            ]
        },
    }
    return frames, art


def _run(n: int, **params: object) -> dict:
    frames, art = _actor_with_a_bag(n)
    p = _FakeComfy(
        {"whole_long_edge": 256, "whole_chunk": 17, "whole_chunk_max": 0, "overlap": 8, **params}
    )
    p.ingest_artifacts(art)
    return p.run_shot(_Reader(frames), frame_range=(1, n))


def test_held_object_goes_whole_to_the_holder() -> None:
    out = _run(9, held_objects=True)
    a, b = out[5]["matte.r"], out[5]["matte.g"]
    assert a[97, 120] > 0.85  # the bag under the second actor ...
    assert b[97, 120] < 0.05  # ... belongs to the first: their hand holds it
    assert a[97, 80] > 0.85
    assert b[60, 110] > 0.85 and a[60, 110] < 0.05  # the second keeps their body


def test_held_objects_off_is_the_nearest_owner_split() -> None:
    off = _run(9)
    explicit = _run(9, held_objects=False)
    for k in ("matte.r", "matte.g", "foreground.a"):
        np.testing.assert_array_equal(off[5][k], explicit[5][k])
    assert off[5]["matte.r"][97, 120] < 0.05  # not the holder's without the rule


def test_assign_held_leaves_untouched_and_tiny_pieces_alone() -> None:
    h, w = 40, 60
    owner = np.full((h, w), -1, np.int16)
    body0 = np.zeros((h, w), np.uint8)
    body0[5:30, 5:20] = 1
    owner[body0 > 0] = 0
    fg = np.zeros((h, w), np.float32)
    fg[5:30, 5:20] = 1.0
    fg[10:20, 21:35] = 1.0  # touches body 0 (1 px gap)
    fg[30:38, 45:58] = 1.0  # touches nobody
    fg[4, 20] = 1.0  # a 1-px speck within reach
    before = owner.copy()
    moved = assign_held(owner, fg, [body0], [None], contact_px=2, min_px=4)
    assert moved == 1
    assert (owner[10:20, 21:35] == 0).all()
    np.testing.assert_array_equal(owner[30:38, 45:58], before[30:38, 45:58])
    assert owner[4, 20] == before[4, 20]


def test_hand_contact_beats_a_longer_body_contact() -> None:
    h, w = 30, 60
    owner = np.full((h, w), -1, np.int16)
    b0, b1 = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    b0[0:10, 0:20] = 1
    b1[0:30, 40:60] = 1
    hand0 = np.zeros((h, w), np.uint8)
    hand0[8:10, 18:20] = 1
    fg = np.zeros((h, w), np.float32)
    fg[10:30, 18:40] = 1.0  # long contact with body 1, a corner with hand 0
    assign_held(owner, fg, [b0, b1], [hand0, None], contact_px=1, min_px=1, max_ratio=0)
    assert (owner[10:30, 18:40] == 0).all()


def test_a_piece_two_people_touch_without_a_hand_keeps_the_nearest_split() -> None:
    """Water spray between two actors: both bodies touch it, no hand holds it."""
    h, w = 30, 60
    b0, b1 = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    b0[0:20, 0:20] = 1
    b1[0:20, 40:60] = 1
    owner = np.full((h, w), -1, np.int16)
    owner[:, :30], owner[:, 30:] = 0, 1  # the nearest split
    before = owner.copy()
    fg = np.zeros((h, w), np.float32)
    fg[20:24, 0:60] = 1.0
    assert assign_held(owner, fg, [b0, b1], [None, None], contact_px=1, min_px=1) == 0
    np.testing.assert_array_equal(owner, before)


def test_a_piece_bigger_than_its_holder_is_not_held() -> None:
    h, w = 40, 40
    b0 = np.zeros((h, w), np.uint8)
    b0[0:5, 0:5] = 1
    owner = np.full((h, w), -1, np.int16)
    fg = np.zeros((h, w), np.float32)
    fg[5:30, 0:30] = 1.0  # the ground under a small figure
    assert assign_held(owner, fg, [b0], [None], contact_px=1, min_px=1) == 0
    assert (owner == -1).all()


# --------------------------------------------------------------------------
# Per-object soft mattes named like the Cryptomatte objects
# --------------------------------------------------------------------------


class _HalfRVM(RVMRefinerPass):
    def __init__(self, params: dict | None = None) -> None:
        merged = {"fallback_feather": False, "refine_min_px": 0}
        merged.update(params or {})
        super().__init__(merged)

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
        return (hard_stack * 0.5).astype(np.float32, copy=False)


def _three_objects(n: int, h: int = 16, w: int = 24) -> dict:
    def rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        s = np.zeros((n, h, w), np.float32)
        s[:, y0:y1, x0:x1] = 1.0
        return s

    fr = list(range(1, n + 1))
    late = rect(0, 3, 18, 24)
    late[0] = 0.0  # the wall is only seen from frame 2
    return {
        "sam3_hard_masks": {
            0: {
                1: {"label": "person", "frames": fr, "stack": rect(4, 12, 8, 16)},
                2: {"label": "person", "frames": fr, "stack": rect(13, 16, 0, 6)},
                7: {"label": "brick wall", "frames": fr, "stack": late},
            }
        },
        "sam3_instances": {
            0: [{"track_id": 1, "slot": "r", "label": "person", "score": 0.9, "frames": fr}]
        },
    }


def _plate(n: int, h: int = 16, w: int = 24) -> _Reader:
    return _Reader([np.full((h, w, 3), 0.5, np.float32) for _ in range(n)])


def test_object_channels_are_named_like_the_cryptomatte_objects() -> None:
    n = 3
    art = _three_objects(n)
    p = _HalfRVM({"refine_all_masks": True, "object_channels": True})
    p.ingest_artifacts(art)
    out = p.run_shot(_plate(n), frame_range=(1, n))
    crypto = CryptomattePass()
    crypto.ingest_artifacts(art)
    crypto.run_shot(_plate(n), frame_range=(1, n))
    names = {object_channel(nm) for nm in crypto._manifest}
    assert names == {"mask_object.person_1", "mask_object.person_2", "mask_object.brick_wall_7"}
    for f in range(1, n + 1):
        assert names <= set(out[f])  # every object on every frame
    np.testing.assert_allclose(out[2]["mask_object.person_1"][4:12, 8:16], 0.5)
    assert out[2]["mask_object.person_1"][14, 2] == 0.0  # only its own pixels
    assert out[2]["mask_object.person_2"][14, 2] == 0.5
    assert not out[1]["mask_object.brick_wall_7"].any()  # not there yet
    # The union-by-label channel is unchanged next to them.
    assert out[2]["mask.person"][14, 2] == 0.5 and out[2]["mask.person"][8, 10] == 0.5


def test_object_channels_off_emits_exactly_what_it_did() -> None:
    n = 3
    a = _HalfRVM({"refine_all_masks": True})
    a.ingest_artifacts(_three_objects(n))
    off = a.run_shot(_plate(n), frame_range=(1, n))
    b = _HalfRVM({"refine_all_masks": True, "object_channels": True})
    b.ingest_artifacts(_three_objects(n))
    on = b.run_shot(_plate(n), frame_range=(1, n))
    for f in off:
        assert not any(c.startswith("mask_object.") for c in off[f])
        for c, arr in off[f].items():
            np.testing.assert_array_equal(arr, on[f][c])


def test_whole_frame_ltx_object_channels_follow_the_owner_split() -> None:
    out = _run(9, object_channels=True, held_objects=True)
    np.testing.assert_array_equal(out[5]["mask_object.person_1"], out[5]["matte.r"])
    assert "mask_object.person_2" in out[5]
    assert "mask_object.hand_3" not in out[5]  # heroes only without refine_all


# --------------------------------------------------------------------------
# Nearest upscale of the ID channels
# --------------------------------------------------------------------------


def _crypto_frame(h: int, w: int) -> dict[str, np.ndarray]:
    a = np.zeros((h, w), np.float32)
    a[:, : w // 2] = 1.0
    b = 1.0 - a
    c = np.zeros((h, w), np.float32)
    c[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 1.0
    b = b * (1.0 - c)
    a = a * (1.0 - c)
    ch, _ = encode([("person_1", a), ("person_2", b), ("bag_3", c)])
    return ch


def _ids(ch: dict[str, np.ndarray]) -> set[int]:
    out: set[int] = set()
    for name in CRYPTOMATTE_CHANNELS:
        if name.endswith((".R", ".B")) and not name.startswith("CryptoObject."):
            out |= set(np.unique(ch[name].view(np.uint32)).tolist())
    return out


def test_nearest_upscale_keeps_only_manifest_ids() -> None:
    ch = _crypto_frame(18, 32)
    src = _ids(ch)
    lin = _upscale_channels_to_plate(ch, 54, 96)
    near = _upscale_channels_to_plate(ch, 54, 96, nearest=set(CRYPTOMATTE_CHANNELS))
    assert _ids(near) <= src  # never a new id
    assert len(_ids(lin)) > len(src)  # bilinear invents ids along the edges


def test_cryptomatte_declares_its_id_channels_and_defaults_to_linear() -> None:
    assert set(CryptomattePass.id_channels) == set(CRYPTOMATTE_CHANNELS)
    assert CryptomattePass().params["proxy_upscale"] == "linear"
