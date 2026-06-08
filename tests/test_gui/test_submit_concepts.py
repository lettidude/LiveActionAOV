"""SAM 3 concepts free-text → PassConfig injection (Matte + Cryptomatte).

The GUI exposes one comma-separated field telling SAM 3 what to detect.
Both the Matte and the Cryptomatte passes run the shared `sam3_matte`
detector, so the worker injects the parsed list as `params["concepts"]`
on that pass only. Empty field → no params → the pass falls back to its
built-in default concept list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.shot_state import ShotState
from live_action_aov.gui.submit_worker import _build_pass_configs, _parse_concepts


def _shot(**kw: object) -> ShotState:
    base: dict[str, object] = dict(
        name="PF0071",
        folder=Path("."),
        sequence_pattern="PF0071.%04d.exr",
        frame_range=(1, 10),
        resolution=(1920, 1080),
    )
    base.update(kw)
    return ShotState(**base)  # type: ignore[arg-type]


# --- _parse_concepts -------------------------------------------------

def test_parse_concepts_empty_is_empty_list() -> None:
    assert _parse_concepts("") == []
    assert _parse_concepts("   ") == []
    assert _parse_concepts(" , ,") == []


def test_parse_concepts_strips_and_splits_on_comma() -> None:
    assert _parse_concepts("person, vehicle , dog") == ["person", "vehicle", "dog"]


def test_parse_concepts_preserves_multiword_concepts() -> None:
    # Comma (not whitespace) is the separator so "red car" survives.
    assert _parse_concepts("red car, dog") == ["red car", "dog"]


def test_parse_concepts_drops_blank_fields() -> None:
    assert _parse_concepts("person,,, dog,") == ["person", "dog"]


# --- _build_pass_configs --------------------------------------------

def test_concepts_injected_onto_sam3_matte() -> None:
    state = _shot(enabled_models=["sam3_rvm"], sam3_concepts="person, red car")
    configs = _build_pass_configs(state)
    by_name = {c.name: c for c in configs}
    # sam3_rvm expands to sam3_matte + rvm_refiner
    assert by_name["sam3_matte"].params == {"concepts": ["person", "red car"]}
    # The refiner gets no concepts — only the detector is concept-driven.
    assert by_name["rvm_refiner"].params == {}


def test_empty_concepts_leaves_sam3_matte_on_defaults() -> None:
    state = _shot(enabled_models=["sam3_rvm"], sam3_concepts="")
    configs = _build_pass_configs(state)
    sam3 = next(c for c in configs if c.name == "sam3_matte")
    # No params → pass uses its DEFAULT_PARAMS concept list.
    assert sam3.params == {}


def test_concepts_only_touch_sam3_matte_not_other_passes() -> None:
    state = _shot(
        enabled_models=["depth_anything_v2", "sam3_rvm"],
        sam3_concepts="person",
    )
    configs = _build_pass_configs(state)
    by_name = {c.name: c for c in configs}
    assert by_name["depth_anything_v2"].params == {}
    assert by_name["sam3_matte"].params == {"concepts": ["person"]}
