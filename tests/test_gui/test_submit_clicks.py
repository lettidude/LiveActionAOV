"""Interactive click-to-mask prompts → sam3_matte `prompt_instances` param.

The GUI viewport lets the user click an element to seed a mask; each seeded
object is a `ClickInstance` (points + optional box, in plate px, on a seed
frame). The worker serializes these into the `sam3_matte` pass's
`prompt_instances` param, which the pass replays through SAM 3's video tracker
(`add_inputs_to_inference_session(input_points=…, input_boxes=…)`) — the same
seed→propagate→Cryptomatte path the text-concept matte uses. Clicks and text
concepts may coexist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from live_action_aov.gui.shot_state import ClickInstance, ShotState
from live_action_aov.gui.submit_worker import (
    _build_pass_configs,
    _serialize_click_instances,
)


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


# --- _serialize_click_instances --------------------------------------

def test_serialize_points_and_box() -> None:
    inst = ClickInstance(
        name="hero",
        seed_frame=1003,
        points=[(120.0, 240.0, 1), (300.5, 80.0, 0)],
        box=(10.0, 20.0, 200.0, 400.0),
    )
    out = _serialize_click_instances([inst])
    assert out == [
        {
            "name": "hero",
            "seed_frame": 1003,
            "points": [[120.0, 240.0, 1], [300.5, 80.0, 0]],
            "box": [10.0, 20.0, 200.0, 400.0],
        }
    ]


def test_serialize_points_only_box_is_none() -> None:
    inst = ClickInstance(name="rock", seed_frame=1, points=[(5.0, 6.0, 1)])
    out = _serialize_click_instances([inst])
    assert out[0]["box"] is None
    assert out[0]["points"] == [[5.0, 6.0, 1]]


def test_serialize_drops_empty_instances() -> None:
    # No points and no box → nothing to seed the tracker with → dropped.
    empty = ClickInstance(name="ghost", seed_frame=2, points=[], box=None)
    keep = ClickInstance(name="real", seed_frame=2, box=(0.0, 0.0, 1.0, 1.0))
    assert _serialize_click_instances([empty, keep]) == [
        {"name": "real", "seed_frame": 2, "points": [], "box": [0.0, 0.0, 1.0, 1.0]}
    ]


# --- _build_pass_configs ---------------------------------------------

def test_clicks_injected_onto_sam3_matte() -> None:
    state = _shot(
        enabled_models=["sam3_rvm"],
        click_instances=[ClickInstance(name="a", seed_frame=1, points=[(1.0, 2.0, 1)])],
    )
    configs = {c.name: c for c in _build_pass_configs(state)}
    assert "prompt_instances" in configs["sam3_matte"].params
    assert configs["sam3_matte"].params["prompt_instances"][0]["name"] == "a"
    # The refiner is untouched.
    assert configs["rvm_refiner"].params == {}


def test_clicks_and_concepts_coexist() -> None:
    state = _shot(
        enabled_models=["sam3_rvm"],
        sam3_concepts="person, car",
        click_instances=[ClickInstance(name="prop", seed_frame=1, box=(0.0, 0.0, 9.0, 9.0))],
    )
    params = next(c for c in _build_pass_configs(state) if c.name == "sam3_matte").params
    assert params["concepts"] == ["person", "car"]
    assert params["prompt_instances"][0]["name"] == "prop"


def test_no_clicks_no_concepts_leaves_defaults() -> None:
    state = _shot(enabled_models=["sam3_rvm"])
    sam3 = next(c for c in _build_pass_configs(state) if c.name == "sam3_matte")
    assert sam3.params == {}
