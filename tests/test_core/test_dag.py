"""DAG topological sort + cycle + missing-artifact detection."""

from __future__ import annotations

import pytest

from live_action_aov.core.dag import (
    DagCycleError,
    MissingArtifactError,
    PassNode,
    topological_sort,
)


def _node(
    name: str,
    provides: tuple[str, ...] = (),
    requires: tuple[str, ...] = (),
) -> PassNode:
    return PassNode(name=name, plugin=name, provides=provides, requires=requires)


def test_topological_sort_respects_requires() -> None:
    flow = _node("flow", provides=("forward_flow",))
    depth = _node("depth", requires=("forward_flow",))
    ordered = topological_sort([depth, flow])
    assert [n.name for n in ordered] == ["flow", "depth"]


def test_topological_sort_preserves_input_order_for_independent_nodes() -> None:
    a = _node("a")
    b = _node("b")
    c = _node("c")
    assert [n.name for n in topological_sort([b, a, c])] == ["b", "a", "c"]


def test_missing_artifact_raises() -> None:
    with pytest.raises(MissingArtifactError):
        topological_sort([_node("x", requires=("nowhere",))])


def test_cycle_raises() -> None:
    a = _node("a", provides=("x",), requires=("y",))
    b = _node("b", provides=("y",), requires=("x",))
    with pytest.raises(DagCycleError):
        topological_sort([a, b])
