"""SAM3MattePass — contract tests with the model bypassed.

`_detect_seed` + `_track_instance` are subclass-override hooks (brainstorm
decision #7: deterministic fakes). The fakes emit known rectangles so we
can reason analytically about union masks, area floors, ranking, and the
shape of the emitted artifacts.
"""

from __future__ import annotations

import numpy as np

from live_action_aov.io.channels import MASK_PREFIX
from live_action_aov.passes.matte.sam3 import SAM3MattePass, _pick_seed_frame


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


class _FakeSAM3(SAM3MattePass):
    """Synthetic SAM3: one person (big rectangle) + one vehicle (small)."""

    def _load_model(self) -> None:  # type: ignore[override]
        self._model = object()

    def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
        h, w, _ = seed_frame.shape
        out: list[tuple[int, str, np.ndarray]] = []
        if "person" in concepts:
            # Big rectangle, centered.
            out.append((1, "person", _rect_mask(h, w, h // 4, 3 * h // 4, w // 4, 3 * w // 4)))
        if "vehicle" in concepts:
            # Small rectangle in the corner.
            out.append((2, "vehicle", _rect_mask(h, w, 0, h // 8, 0, w // 8)))
        return out

    def _track_instance(self, frames, seed_frame_idx, seed_mask):  # type: ignore[override]
        # Replicate the seed mask verbatim across every frame — deterministic
        # and easy to reason about.
        n = frames.shape[0]
        return np.stack([seed_mask] * n, axis=0)


def _small_params(**extra):
    p = {
        "concepts": ["person", "vehicle", "sky"],
        "min_area_fraction": 0.001,
    }
    p.update(extra)
    return p


def test_license_is_commercial_safe_with_carveout_note() -> None:
    lic = SAM3MattePass.declared_license()
    assert lic.commercial_use is True
    assert "SAM" in lic.spdx or "sam" in lic.spdx.lower()
    assert "military" in lic.notes.lower() or "itar" in lic.notes.lower()


def test_pick_seed_frame_keywords_and_int() -> None:
    assert _pick_seed_frame(10, "first") == 0
    assert _pick_seed_frame(10, "middle") == 5
    assert _pick_seed_frame(10, "last") == 9
    assert _pick_seed_frame(10, 3) == 3
    # Out-of-range int is clamped.
    assert _pick_seed_frame(10, 99) == 9
    assert _pick_seed_frame(10, -5) == 0


def test_run_shot_emits_mask_channels_per_detected_concept() -> None:
    frames = _plate_frames(5)
    pass_ = _FakeSAM3(_small_params())
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))

    # Person + vehicle were detected; sky was requested but produced no mask
    # (the fake doesn't emit it), so mask.sky must be absent.
    any_frame = out[1]
    assert f"{MASK_PREFIX}person" in any_frame
    assert f"{MASK_PREFIX}vehicle" in any_frame
    assert f"{MASK_PREFIX}sky" not in any_frame


def test_area_floor_drops_tiny_instances() -> None:
    """A vehicle that's below the min_area_fraction must be suppressed, and
    because it's the only instance of that concept, mask.vehicle must not
    appear."""
    frames = _plate_frames(5, h=32, w=48)
    pass_ = _FakeSAM3(_small_params(min_area_fraction=0.5))  # huge floor
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    any_frame = out[1]
    # Big person rectangle (half the frame) should still survive a 0.5 floor.
    # Wait — _rect_mask for person is h/4..3h/4 × w/4..3w/4 = area ~ 1/4 of
    # plate. That's below 0.5, so both get suppressed.
    assert f"{MASK_PREFIX}person" not in any_frame
    assert f"{MASK_PREFIX}vehicle" not in any_frame


def test_union_by_concept_combines_multiple_instances() -> None:
    """Two person instances → mask.person is the OR of both."""
    class _TwoPerson(_FakeSAM3):
        def _detect_seed(self, seed_frame, concepts):  # type: ignore[override]
            h, w, _ = seed_frame.shape
            return [
                (1, "person", _rect_mask(h, w, 0, h // 2, 0, w // 2)),
                (2, "person", _rect_mask(h, w, h // 2, h, w // 2, w)),
            ]

    frames = _plate_frames(3)
    pass_ = _TwoPerson(_small_params())
    out = pass_.run_shot(_FakeReader(frames), frame_range=(1, 3))
    union = out[1][f"{MASK_PREFIX}person"]
    # Both quadrants should be 1.
    assert union[0, 0] == 1.0
    assert union[-1, -1] == 1.0
    # The other two quadrants are zero.
    assert union[0, -1] == 0.0
    assert union[-1, 0] == 0.0


def test_emit_artifacts_shape_for_downstream_refiner() -> None:
    frames = _plate_frames(5)
    pass_ = _FakeSAM3(_small_params())
    pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    artifacts = pass_.emit_artifacts()

    assert "sam3_hard_masks" in artifacts
    assert "sam3_instances" in artifacts
    assert "matte_concepts" in artifacts

    # sam3_hard_masks: dict[track_id, {"label", "frames", "stack"}]
    hard = next(iter(artifacts["sam3_hard_masks"].values()))
    assert set(hard) == {1, 2}
    for tid, data in hard.items():
        assert isinstance(data["label"], str)
        assert isinstance(data["frames"], list)
        assert data["stack"].ndim == 3
        assert data["stack"].dtype == np.float32

    # sam3_instances: ranked hero dicts sorted by slot order.
    heroes = next(iter(artifacts["sam3_instances"].values()))
    assert len(heroes) == 2
    slots = [h["slot"] for h in heroes]
    assert slots == ["r", "g"]          # canonical RGBA order
    # Big rectangle (person) outranks small (vehicle).
    assert heroes[0]["label"] == "person"
    assert heroes[1]["label"] == "vehicle"

    # matte_concepts: the list of concept names actually detected.
    concepts = next(iter(artifacts["matte_concepts"].values()))
    assert set(concepts) == {"person", "vehicle"}


def test_dynamic_mask_channel_not_predeclared_static_list() -> None:
    """The SAM3 pass must NOT statically declare mask.* channels — they're
    dynamic (per-clip, per-concept). The writer appends unknowns after
    canonical order."""
    assert SAM3MattePass.produces_channels == []


def test_ingest_artifacts_accepts_flow_as_soft_dep() -> None:
    """SAM3 can consume forward_flow if a flow pass ran first, but it must
    also work when no flow artifact is present."""
    pass_ = _FakeSAM3(_small_params())
    # Explicitly call ingest_artifacts with no flow — must not raise.
    pass_.ingest_artifacts({})
    # With flow present, the snapshot is retained.
    fake_flow = {1: np.zeros((2, 32, 48), dtype=np.float32)}
    pass_.ingest_artifacts({"forward_flow": fake_flow})
    assert 1 in pass_._forward_flow


def test_heroes_override_routes_to_specific_slot() -> None:
    """User override: vehicle (smaller) pinned to r, person falls to g."""
    frames = _plate_frames(5)
    pass_ = _FakeSAM3(
        _small_params(heroes=[{"track_id": 2, "slot": "r"}])
    )
    pass_.run_shot(_FakeReader(frames), frame_range=(1, 5))
    artifacts = pass_.emit_artifacts()
    heroes = next(iter(artifacts["sam3_instances"].values()))
    by_slot = {h["slot"]: h["label"] for h in heroes}
    assert by_slot["r"] == "vehicle"
    assert by_slot["g"] == "person"
