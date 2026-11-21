"""Licence labels shown to the artist must match each pass's own `License`.

The GUI catalog and the CLI help carry hand-written licence strings; the
pass classes carry the declared `License.spdx` that also lands in every
sidecar EXR. These tests stop the two from drifting apart again.
"""

from __future__ import annotations

import re

import click
import pytest
import typer

from live_action_aov.cli.app import app
from live_action_aov.core.registry import get_registry
from live_action_aov.gui.pass_catalog import PASS_CATALOG, ModelEntry

_ALL_ENTRIES: list[ModelEntry] = [e for entries in PASS_CATALOG.values() for e in entries]


def _expected_tag(entry: ModelEntry) -> str:
    registry = get_registry()
    tags: list[str] = []
    for plugin in entry.expansion:
        spdx = registry.get_pass(plugin).license.spdx
        if spdx not in tags:
            tags.append(spdx)
    return " / ".join(tags)


@pytest.mark.parametrize("entry", _ALL_ENTRIES, ids=lambda e: e.key)
def test_catalog_licence_tag_matches_pass_spdx(entry: ModelEntry) -> None:
    expected = _expected_tag(entry)
    tag = entry.license_tag
    # An entry whose submit appends more passes (the Matte switch adds the
    # chosen engine) may say so after " + ".
    assert tag == expected or tag.startswith(expected + " + "), (
        f"{entry.key}: catalog says {tag!r}, the pass declares {expected!r}"
    )


@pytest.mark.parametrize("entry", _ALL_ENTRIES, ids=lambda e: e.key)
def test_catalog_commercial_flag_matches_passes(entry: ModelEntry) -> None:
    registry = get_registry()
    commercial = all(registry.get_pass(p).license.commercial_use for p in entry.expansion)
    assert entry.commercial == commercial, entry.key


_CLI_BACKEND_OPTIONS = ("--depth-backend", "--normals-backend", "--matte-detector", "--refiner")
# "name (LICENCE, ..." / "`name` (LICENCE) ..." in a help string.
_NAMED_LICENCE = re.compile(r"`?([a-z0-9_]+)`?\s+\(([^,;)]+)[,;)]")


def _run_shot_options() -> dict[str, click.Option]:
    group = typer.main.get_command(app)
    assert isinstance(group, click.Group)
    cmd = group.commands["run-shot"]
    out: dict[str, click.Option] = {}
    for param in cmd.params:
        if isinstance(param, click.Option):
            for opt in param.opts:
                out[opt] = param
    return out


@pytest.mark.parametrize("flag", _CLI_BACKEND_OPTIONS)
def test_cli_help_licences_match_pass_spdx(flag: str) -> None:
    registry = get_registry()
    option = _run_shot_options()[flag]
    help_text = " ".join((option.help or "").split())
    named = {m.group(1): m.group(2).strip() for m in _NAMED_LICENCE.finditer(help_text)}
    default = str(option.default)
    assert default in named, f"{flag} help must state the default's licence: {help_text!r}"
    for plugin, label in named.items():
        if plugin not in registry.list_passes():
            continue
        assert label == registry.get_pass(plugin).license.spdx, (
            f"{flag}: help says {plugin} is {label!r}, the pass declares "
            f"{registry.get_pass(plugin).license.spdx!r}"
        )
