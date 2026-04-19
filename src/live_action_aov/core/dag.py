"""DAG scheduler: resolves inter-pass artifact dependencies.

Each pass declares `requires_artifacts` and `provides_artifacts`. The
scheduler topologically sorts the pass list so that every required artifact
is produced before it's consumed. Cycles (declared or implicit) raise
`DagCycleError`.

This is the v1 implementation: artifact names must match exactly. v2 can add
wildcards / pattern matching if we need it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


class DagError(RuntimeError):
    """Base class for DAG scheduling errors."""


class DagCycleError(DagError):
    """Raised when the declared dependency graph has a cycle."""


class MissingArtifactError(DagError):
    """Raised when a pass requires an artifact that no scheduled pass provides."""


@dataclass(frozen=True)
class PassNode:
    """One node in the DAG.

    `name` is the job-local pass slot (e.g. "flow"), which need not match
    `plugin` (the entry-point name, e.g. "raft_large"). In v1 the two are
    usually the same, but keeping them separate lets users run two depth
    backends in one job later.
    """

    name: str
    plugin: str
    provides: tuple[str, ...]
    requires: tuple[str, ...]


def topological_sort(nodes: list[PassNode]) -> list[PassNode]:
    """Return `nodes` in execution order.

    Independent nodes preserve their input order so deterministic YAML ->
    deterministic execution. If an artifact is produced by more than one
    node, the first in input order wins (warn upstream if that's actually
    ambiguous — the scheduler does not decide).
    """
    # Map each artifact to the first provider node.
    producer_of: dict[str, PassNode] = {}
    for node in nodes:
        for art in node.provides:
            producer_of.setdefault(art, node)

    # Build the dependency edge set: consumer -> provider.
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    by_name: dict[str, PassNode] = {n.name: n for n in nodes}

    for node in nodes:
        for art in node.requires:
            provider = producer_of.get(art)
            if provider is None:
                raise MissingArtifactError(
                    f"Pass '{node.name}' requires artifact '{art}' but no scheduled "
                    f"pass provides it. Scheduled passes: {[n.name for n in nodes]}"
                )
            if provider.name == node.name:
                # self-dep is fine (provides and requires the same artifact internally)
                continue
            incoming[node.name].add(provider.name)
            outgoing[provider.name].add(node.name)

    # Kahn's algorithm, preserving input order for ties.
    ready: deque[PassNode] = deque(n for n in nodes if not incoming[n.name])
    out: list[PassNode] = []
    seen: set[str] = set()

    while ready:
        node = ready.popleft()
        if node.name in seen:
            continue
        seen.add(node.name)
        out.append(node)
        for dep_name in list(outgoing[node.name]):
            incoming[dep_name].discard(node.name)
            if not incoming[dep_name]:
                ready.append(by_name[dep_name])

    if len(out) != len(nodes):
        remaining = [n.name for n in nodes if n.name not in seen]
        raise DagCycleError(f"Cycle detected in pass dependency graph; unresolved: {remaining}")
    return out


__all__ = [
    "DagCycleError",
    "DagError",
    "MissingArtifactError",
    "PassNode",
    "topological_sort",
]
