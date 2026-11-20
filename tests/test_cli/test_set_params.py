"""`run-shot --set <pass>.<param>=<value>`: pass parameters from the CLI."""

from __future__ import annotations

import pytest

from live_action_aov.cli.app import parse_pass_params


def test_values_parse_as_json_else_text() -> None:
    got = parse_pass_params(
        [
            "cryptomatte.proxy_upscale=nearest",
            "cryptomatte.feather=1.5",
            "ltx_alpha_refiner.held_objects=true",
            'sam3_matte.concepts=["person", "car"]',
        ]
    )
    assert got == {
        "cryptomatte": {"proxy_upscale": "nearest", "feather": 1.5},
        "ltx_alpha_refiner": {"held_objects": True},
        "sam3_matte": {"concepts": ["person", "car"]},
    }


@pytest.mark.parametrize("bad", ["cryptomatte=nearest", "proxy_upscale=nearest", ".x=1", "a.b"])
def test_malformed_items_are_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_pass_params([bad])
