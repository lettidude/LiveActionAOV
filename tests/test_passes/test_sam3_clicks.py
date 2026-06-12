"""SAM3MattePass click-to-mask (`prompt_instances`) — model bypassed.

Same fake-subclass convention as test_sam3_matte.py: `_detect_seed` /
`_track_instance` are deterministic overrides, so we can verify the click
plumbing analytically — coordinate rescaling (`ref_size` → actual frames,
the proxy-mode guard), seed-frame plate→local mapping with clamping, the
1000+ track-id block, `mask.<name>` channel emission, and coexistence with
the text-concept path.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from live_action_aov.io.channels import MASK_PREFIX
from live_action_aov.passes.matte.sam3 import SAM3MattePass


class _FakeReader:
    def __init__(self, frames: list[np.ndarray], first: int = 1) -> None:
        self._frames = frames
        self._first = first

    def read_frame(self, idx: int) -> tuple[np.ndarray, dict]:
        return self._frames[idx - self._first], {}


def _plate_frames(n: int, h: int = 32, w: int = 48) -> list[np.ndarray]:
    return [np.full((h, w, 3), 0.5, dtype=np.float32) for _ in range(n)]


def _rect_mask(h: int, w: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


class _ClickFake(SAM3MattePass):
    """Records every click-seeded track call; concept path mirrors the seed
    mask verbatim like the existing fake."""

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self.click_calls: list[dict[str, Any]] = []

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        h, w, _ = seed_frame.shape
        if "person" in concepts:
            return [(1, "person", _rect_mask(h, w, h // 4, 3 * h // 4, w // 4, 3 * w // 4))]
        return []

    def _track_instance(  # type: ignore[override]
        self,
        frames,
        seed_frame_idx,
        seed_mask=None,
        *,
        points=None,
        labels=None,
        box=None,
    ):
        n, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
        if seed_mask is not None:
            return np.stack([seed_mask] * n, axis=0)
        self.click_calls.append(
            {"seed_frame_idx": seed_frame_idx, "points": points, "labels": labels, "box": box}
        )
        # Big deterministic rectangle — comfortably above any area floor.
        return np.stack([_rect_mask(h, w, 4, 28, 4, 44)] * n, axis=0)


def _params(**extra: Any) -> dict[str, Any]:
    p: dict[str, Any] = {"concepts": [], "min_area_fraction": 0.001}
    p.update(extra)
    return p


def test_click_coords_rescaled_from_ref_size() -> None:
    # Frames are 48x32; clicks captured at 96x64 → both axes scale by 0.5.
    frames = _plate_frames(3)
    pass_ = _ClickFake(
        _params(
            prompt_instances=[
                {
                    "name": "hero",
                    "seed_frame": 1,
                    "points": [[20.0, 10.0, 1], [40.0, 30.0, 0]],
                    "box": [0.0, 0.0, 40.0, 20.0],
                    "ref_size": [96, 64],
                }
            ]
        )
    )
    pass_.run_shot(_FakeReader(frames), frame_range=(1, 3))
    assert len(pass_.click_calls) == 1
    call = pass_.click_calls[0]
    assert call["points"] == [[10.0, 5.0], [20.0, 15.0]]
    assert call["labels"] == [1, 0]
    assert call["box"] == [0.0, 0.0, 20.0, 10.0]


def test_seed_frame_plate_to_local_with_clamping() -> None:
    frames = _plate_frames(5)
    pass_ = _ClickFake(
        _params(
            prompt_instances=[
                {"name": "a", "seed_frame": 1003, "points": [[1.0, 1.0, 1]], "box": None},
                # Out of range — clamps to last local index.
                {"name": "b", "seed_frame": 9999, "points": [[1.0, 1.0, 1]], "box": None},
                # Below range — clamps to 0.
                {"name": "c", "seed_frame": 5, "points": [[1.0, 1.0, 1]], "box": None},
            ]
        )
    )
    pass_.run_shot(_FakeReader(frames, first=1001), frame_range=(1001, 1005))
    locals_ = [c["seed_frame_idx"] for c in pass_.click_calls]
    assert locals_ == [2, 4, 0]


def test_click_instances_get_channels_names_and_id_block() -> None:
    frames = _plate_frames(4)
    pass_ = _ClickFake(
        _params(
            prompt_instances=[
                {"name": "hero", "seed_frame": 1, "points": [[5.0, 5.0, 1]], "box": None}
            ]
        )
    )
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 4))
    # mask.<name> channel emitted on every frame.
    assert all(f"{MASK_PREFIX}hero" in out[f] for f in range(1, 5))
    # Instance carries the user's name and lives in the 1000+ id block.
    inst = next(i for i in pass_._instances if i.label == "hero")
    assert inst.track_id >= 1001
    assert "hero" in pass_._concepts_found


def test_no_prompts_changes_nothing_and_empties_skipped() -> None:
    frames = _plate_frames(3)
    # Empty list → no click calls (backward compat).
    p1 = _ClickFake(_params(prompt_instances=[]))
    p1.run_shot(_FakeReader(frames), frame_range=(1, 3))
    assert p1.click_calls == []
    # Instance with neither points nor box → skipped, no crash.
    p2 = _ClickFake(
        _params(prompt_instances=[{"name": "ghost", "seed_frame": 1, "points": [], "box": None}])
    )
    p2.run_shot(_FakeReader(frames), frame_range=(1, 3))
    assert p2.click_calls == []
    assert p2._instances == []


def test_clicks_coexist_with_concept_detection() -> None:
    frames = _plate_frames(4)
    pass_ = _ClickFake(
        _params(
            concepts=["person"],
            prompt_instances=[
                {"name": "prop", "seed_frame": 2, "points": [[6.0, 6.0, 1]], "box": None}
            ],
        )
    )
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 4))
    labels = {i.label for i in pass_._instances}
    assert labels == {"person", "prop"}
    assert f"{MASK_PREFIX}person" in out[1]
    assert f"{MASK_PREFIX}prop" in out[1]
    # Detector ids and click ids never collide.
    ids = [i.track_id for i in pass_._instances]
    assert len(ids) == len(set(ids))
