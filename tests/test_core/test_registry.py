"""Plugin registry discovery + query."""

from __future__ import annotations

import importlib.metadata as im

import pytest

from live_action_aov.core.registry import GROUP_EXECUTORS, get_registry


def _package_installed() -> bool:
    """Entry-point discovery only finds things when the package is installed
    (editable or otherwise). Tests that require discovery skip when not."""
    try:
        return len(list(im.entry_points(group=GROUP_EXECUTORS))) > 0
    except Exception:
        return False


def test_noop_pass_registered_via_conftest() -> None:
    registry = get_registry()
    names = registry.list_passes()
    assert "noop" in names


def test_list_by_type_filters() -> None:
    registry = get_registry()
    geometric = registry.list_by_type("geometric")
    assert "noop" in geometric


@pytest.mark.skipif(
    not _package_installed(),
    reason="Package not installed; entry-point discovery only works after `uv sync`.",
)
def test_built_in_executors_discovered_via_entry_points() -> None:
    registry = get_registry()
    execs = registry.list_executors()
    assert "local" in execs
    assert "deadline" in execs


@pytest.mark.skipif(
    not _package_installed(),
    reason="Package not installed; entry-point discovery only works after `uv sync`.",
)
def test_built_in_writers_discovered() -> None:
    registry = get_registry()
    writers = registry.list_writers()
    assert "exr" in writers
    assert "json" in writers
