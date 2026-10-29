"""Small subjects (crowds): box-crop masks, tiled detection merge, size floor,
crowd-friendly refiner / Cryptomatte paths. No model weights needed."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np

from live_action_aov.io.cryptomatte import encode, encode_crops
from live_action_aov.passes.matte.boxmask import BoxMask, LazyStack, MaskDict, inter_area
from live_action_aov.passes.matte.rvm import RVMRefinerPass
from live_action_aov.passes.matte.sam3 import SAM3MattePass, _merge_detections


def _rect(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


# --- BoxMask ------------------------------------------------------------------


def test_boxmask_behaves_like_the_dense_mask() -> None:
    a = _rect(40, 60, 10, 20, 30, 45)
    m = BoxMask.from_dense(a)
    assert m.box == (10, 30, 20, 45) and m.crop.shape == (10, 15)
    assert m.sum() == a.sum() and m.any()
    assert np.array_equal(np.asarray(m), a)
    assert np.array_equal(m > 0, a > 0)
    taken = np.zeros((40, 60), bool)
    taken |= m > 0
    assert taken.sum() == a.sum()
    u = np.zeros((40, 60), np.float32)
    np.maximum(u, m, out=u)
    assert u.sum() == a.sum()
    assert pickle.loads(pickle.dumps(m)).box == m.box
    assert not BoxMask.from_dense(np.zeros((40, 60))).any()


def test_maskdict_compresses_and_lazystack_round_trips() -> None:
    a, b = _rect(30, 30, 0, 5, 0, 5), _rect(30, 30, 20, 30, 10, 12)
    d = MaskDict({1: a})
    d[2] = b.astype(np.float32)  # float in, binary BoxMask stored
    assert all(isinstance(v, BoxMask) for v in d.values())
    st = LazyStack([d[1], d[2]], (30, 30))
    assert st.shape == (2, 30, 30) and st.dtype == np.uint8
    assert np.array_equal(np.asarray(st), np.stack([a, b]))
    assert np.array_equal(st[1], b) and st.box(1).box == (20, 10, 30, 12)
    assert int(st.max()) == 1


def test_inter_area_only_on_the_overlap() -> None:
    a = BoxMask.from_dense(_rect(50, 50, 0, 20, 0, 20))
    b = BoxMask.from_dense(_rect(50, 50, 10, 30, 15, 40))
    assert inter_area(a, b) == 10 * 5
    assert inter_area(a, BoxMask.from_dense(_rect(50, 50, 40, 50, 40, 50))) == 0


# --- tiled detection: merge, grid, floor --------------------------------------


def _bm(*r: int) -> BoxMask:
    return BoxMask.from_dense(_rect(100, 200, *r))


def test_merge_drops_a_group_mask_and_keeps_the_people() -> None:
    """SAM 3 can return two distant people as ONE full-frame instance; the
    tiles see them apart. Keep the individuals."""
    group = _bm(40, 60, 10, 90)
    p1, p2 = _bm(40, 60, 10, 20), _bm(40, 60, 80, 90)
    kept = _merge_detections([(0.9, group)], [(0.6, p1), (0.6, p2)])
    assert sorted(m.box for _s, m in kept) == sorted([p1.box, p2.box])


def test_merge_dedupes_the_same_person_seen_twice() -> None:
    big = _bm(10, 90, 50, 80)  # whole in the full frame
    same = _bm(12, 90, 51, 80)  # and again in a tile
    small = _bm(5, 15, 150, 155)  # tile-only tiny person
    kept = _merge_detections([(0.8, big)], [(0.7, same), (0.5, small)])
    assert len(kept) == 2
    assert {m.box for _s, m in kept} == {big.box, small.box}


def test_merge_without_tiles_is_the_full_frame_as_is() -> None:
    a, b = _bm(10, 50, 10, 50), _bm(12, 52, 12, 52)  # two overlapping people
    assert len(_merge_detections([(0.9, a), (0.8, b)], [])) == 2


def test_tile_grid_and_size_floor() -> None:
    p = SAM3MattePass({"small_objects": "auto"})
    assert p._tile_grid(2160, 4096) == 2 and p._tile_grid(540, 960) == 1
    assert SAM3MattePass({"small_objects": "max"})._tile_grid(540, 960) == 3
    assert SAM3MattePass({"small_objects": "off"})._tile_grid(2160, 4096) == 1
    tiny = BoxMask.from_dense(_rect(2160, 4096, 100, 130, 100, 110))  # 30 px tall
    assert p._size_ok(tiny, 2160, 4096)  # tiles on: box height floor
    assert not SAM3MattePass({"small_objects": "off"})._size_ok(tiny, 2160, 4096)  # 0.5% area


# --- refiner: small objects skip the engine -----------------------------------


class _CountingRefiner(RVMRefinerPass):
    calls = 0

    def __init__(self, params=None):
        super().__init__({"fallback_feather": False, "refine_all_masks": True, **(params or {})})

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _refine_instance(self, plate_stack, hard_stack):  # type: ignore[override]
        type(self).calls += 1
        return (hard_stack * 0.5).astype(np.float32)


class _Reader:
    def __init__(self, n: int, h: int, w: int) -> None:
        self.f = np.full((h, w, 3), 0.5, np.float32)

    def read_frame(self, idx: int):
        return self.f, {}


def test_small_crowd_members_are_not_refined_but_land_in_the_mask() -> None:
    n, h, w = 2, 200, 300
    hard = {}
    for t in range(6):  # six 10 px tall people, well under 8% of 200 = 16 px
        x = 20 + 40 * t
        hard[t + 1] = {
            "label": "person",
            "frames": [1, 2],
            "stack": LazyStack([_rect(h, w, 150, 160, x, x + 4)] * n, (h, w)),
        }
    _CountingRefiner.calls = 0
    p = _CountingRefiner()
    p.ingest_artifacts({"sam3_hard_masks": {1: hard}, "sam3_instances": {1: []}})
    out = p.run_shot(_Reader(n, h, w), (1, 2))
    assert _CountingRefiner.calls == 0
    m = out[1]["mask.person"]
    assert m[155, 22] == 1.0 and m[155, 222] == 1.0  # every member present
    assert 0.0 < m[149, 22] < 0.5  # anti-aliased edge, not hard


# --- Cryptomatte: crop encoder == dense encoder --------------------------------


def test_encode_crops_matches_encode() -> None:
    rng = np.random.default_rng(1)
    h, w = 24, 32
    inst = []
    for k in range(5):
        y0, x0 = int(rng.integers(0, 12)), int(rng.integers(0, 16))
        c = (rng.random((10, 14)) * (rng.random((10, 14)) > 0.3)).astype(np.float32)
        inst.append((f"obj_{k}", y0, x0, c))
    dense = []
    for name, y0, x0, c in inst:
        full = np.zeros((h, w), np.float32)
        full[y0 : y0 + 10, x0 : x0 + 14] = c
        dense.append((name, full))
    ref, _ = encode(dense)
    got = encode_crops(inst, (h, w))
    assert set(ref) == set(got)
    for k in ref:
        assert np.allclose(ref[k], got[k], atol=1e-6), k


# --- clicks on small subjects: zoomed seed -------------------------------------


def test_click_window_zooms_on_small_clicks_only() -> None:
    from live_action_aov.passes.matte.sam3 import _click_window

    h, w = 2160, 4096
    y0, x0, y1, x1 = _click_window([[1000.0, 800.0]], [1], None, h, w)  # type: ignore[misc]
    assert (y1 - y0, x1 - x0) == (512, 512)  # 1/8 of the long edge
    assert y0 <= 800 < y1 and x0 <= 1000 < x1
    # Near the frame edge the window slides inside.
    assert _click_window([[5.0, 5.0]], [1], None, h, w)[:2] == (0, 0)  # type: ignore[index]
    # A box half the frame wide: nothing to gain.
    assert _click_window([], [], [0.0, 0.0, 2500.0, 1500.0], h, w) is None


class _ZoomFake(SAM3MattePass):
    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        return []

    def _zoom_click_seed(self, frame, points, labels, box):  # type: ignore[override]
        h, w = frame.shape[:2]
        return BoxMask.from_dense(_rect(h, w, 10, 20, 30, 34))

    def _track_instance(self, frames, seed_frame_idx, seed_mask, **kw):  # type: ignore[override]
        assert seed_mask is not None and not kw.get("points"), "zoomed mask must seed the tracker"
        return np.repeat(np.asarray(seed_mask, np.float32)[None], frames.shape[0], 0)


def test_small_click_is_seeded_from_the_zoomed_mask() -> None:
    n, h, w = 2, 64, 96
    p = _ZoomFake(
        {
            "concepts": [],
            "small_objects": "max",
            "prompt_instances": [{"name": "kid", "seed_frame": 1, "points": [[32, 15, 1]]}],
        }
    )
    out = p.run_shot(_Reader(n, h, w), (1, 2))
    m = out[1]["mask.kid"]
    assert m[15, 32] == 1.0 and m.sum() == 10 * 4


# --- re-detection in a crowd ---------------------------------------------------


class _RedetectFake(SAM3MattePass):
    """Frame 0 (seed): one person. Later frames: that person + a second,
    same size, standing right next to them (or a small arm piece)."""

    LATER: list[np.ndarray] = []

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        h, w = seed_frame.shape[:2]
        dets = [(1, "person", _rect(h, w, 10, 50, 10, 30))]
        if float(seed_frame[0, 0, 0]) > 0.6:  # a "later" frame
            dets += [(2, "person", m) for m in self.LATER]
        return dets

    def _track_instance(self, frames, seed_frame_idx, seed_mask=None, **_: Any):  # type: ignore[override]
        return np.stack([np.asarray(seed_mask, np.float32)] * frames.shape[0], 0)


class _SeqReader:
    def __init__(self, n: int, h: int, w: int) -> None:
        self.n, self.h, self.w = n, h, w

    def read_frame(self, idx: int):
        v = 0.5 if idx == 1 else 0.7
        return np.full((self.h, self.w, 3), v, np.float32), {}


def _redetect_run(later: list[np.ndarray]) -> SAM3MattePass:
    _RedetectFake.LATER = later
    p = _RedetectFake(
        {
            "concepts": ["person"],
            "sample_frame": "first",
            "redetect_stride": 2,
            "min_area_fraction": 0.001,
        }
    )
    p.run_shot(_SeqReader(3, 60, 80), (1, 3))
    return p


def test_new_person_next_to_another_is_a_new_track() -> None:
    p = _redetect_run([_rect(60, 80, 10, 50, 27, 47)])  # same size, overlapping a bit
    assert len(p._instances) == 2


def test_lost_arm_is_still_reattached() -> None:
    p = _redetect_run([_rect(60, 80, 20, 26, 30, 40)])  # small piece touching
    assert len(p._instances) == 1


# --- proxy: detect on the original plate ---------------------------------------


def test_boxmask_resized_keeps_tiny_objects() -> None:
    big = BoxMask.from_dense(_rect(400, 800, 100, 140, 200, 220))
    small = big.resized((100, 200))
    assert small.shape == (100, 200) and small.box == (25, 50, 35, 55)
    dot = BoxMask.from_dense(_rect(400, 800, 10, 12, 10, 12)).resized((100, 200))
    assert dot.sum() >= 1  # a 2 px object at 1/4 scale does not vanish


class _ProxyReader:
    """Proxy frames 60x80; the original plate is 240x320."""

    def read_frame(self, idx: int):
        return np.full((60, 80, 3), 0.5, np.float32), {}

    def read_frame_full_res(self, idx: int):
        return np.full((240, 320, 3), 0.5, np.float32), {}


class _ProxyFake(SAM3MattePass):
    seen: list[tuple[int, int]] = []

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        type(self).seen.append(seed_frame.shape[:2])
        h, w = seed_frame.shape[:2]
        return [(1, "person", _rect(h, w, 80, 120, 100, 112))]  # in 240x320 px

    def _track_instance(self, frames, seed_frame_idx, seed_mask=None, **_: Any):  # type: ignore[override]
        return np.stack([np.asarray(seed_mask, np.float32)] * frames.shape[0], 0)


def test_proxy_detection_reads_the_original_plate() -> None:
    _ProxyFake.seen = []
    p = _ProxyFake({"concepts": ["person"], "redetect_stride": None})
    out = p.run_shot(_ProxyReader(), (1, 2))
    assert _ProxyFake.seen == [(240, 320)]
    m = out[1]["mask.person"]
    assert m.shape == (60, 80)
    ys, xs = np.nonzero(m)
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (20, 29, 25, 27)


def test_proxy_detection_off_uses_the_proxy_frames() -> None:
    _ProxyFake.seen = []
    p = _ProxyFake({"concepts": ["person"], "redetect_stride": None, "small_objects": "off"})
    p.run_shot(_ProxyReader(), (1, 2))
    assert _ProxyFake.seen == [(60, 80)]


def test_crowd_sessions_are_split_by_position() -> None:
    from live_action_aov.passes.matte.sam3 import _session_chunks

    objs = [
        {"obj_id": i, "mask": BoxMask.from_dense(_rect(50, 400, 10, 20, x, x + 5))}
        for i, x in enumerate([300, 10, 200, 50, 120])
    ]
    objs.append({"obj_id": 99, "points": [[1, 1]]})  # a click: first session
    chunks = _session_chunks(objs, 3)
    assert [[o["obj_id"] for o in c] for c in chunks] == [[99, 1, 3], [4, 2, 0]]
    assert _session_chunks(objs, 32) == [objs]


def test_proxy_keeps_small_people_found_on_the_plate() -> None:
    """Detected on the 240x320 plate (tiles on), tracked on a 60x80 proxy
    (too small for tiles): a person 12 px tall on the plate (3 px on the
    proxy) must survive the size floor, not hit the area-fraction floor."""

    class _Tiny(_ProxyFake):
        def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
            h, w = seed_frame.shape[:2]
            return [(1, "person", _rect(h, w, 80, 92, 100, 104))]

    p = _Tiny({"concepts": ["person"], "redetect_stride": None, "small_objects": "max"})
    out = p.run_shot(_ProxyReader(), (1, 2))
    assert "mask.person" in out[1] and out[1]["mask.person"].sum() > 0
