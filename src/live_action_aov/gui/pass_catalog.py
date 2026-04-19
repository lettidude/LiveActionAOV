"""The user-facing catalog of models the GUI Passes section exposes.

Central registry so adding a new model to the tool is a one-line edit
in this file. The Inspector reads the catalog and auto-generates a
section + checkboxes; the submit worker reads it to know which
"virtual" entries (matte combos) need to be expanded into multiple
`PassConfig` entries for the executor.

Separating this from the plugin registry itself is deliberate: the
registry knows every pass class that's installed; the catalog knows
which of those should be exposed to the user, how to group them,
what license label to show, and which are composite.

`expansion` is what the submit worker sends to the executor. A plain
single-plugin model expands to one name; matte combos expand to two
(detector + refiner) so the Matte UX stays a single checkbox instead
of asking the user to pair them manually.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    key: str  # stored in ShotState.enabled_models — stable identifier
    label: str  # rendered in the inspector
    license_tag: str  # short SPDX-ish string shown next to the label
    commercial: bool  # gates the non-commercial consent dialog
    expansion: tuple[str, ...]  # concrete plugin names fed to the executor


# Grouped by category; rendering order inside a category is the list
# order, so the commercial-safe default sits at the top of each group.
PASS_CATALOG: dict[str, list[ModelEntry]] = {
    "Flow": [
        ModelEntry("flow", "RAFT", "Apache-2.0", True, ("flow",)),
    ],
    "Depth": [
        ModelEntry(
            "depth_anything_v2",
            "Depth Anything v2",
            "Apache-2.0",
            True,
            ("depth_anything_v2",),
        ),
        ModelEntry(
            "video_depth_anything",
            "Video Depth Anything",
            "Apache-2.0",
            True,
            ("video_depth_anything",),
        ),
        ModelEntry(
            "depthcrafter",
            "DepthCrafter",
            "CC-BY-NC-4.0",
            False,
            ("depthcrafter",),
        ),
        ModelEntry(
            "depthpro",
            "DepthPro",
            "CC-BY-NC-4.0",
            False,
            ("depthpro",),
        ),
    ],
    "Normals": [
        ModelEntry("dsine", "DSINE", "MIT", True, ("dsine",)),
        ModelEntry(
            "normalcrafter",
            "NormalCrafter",
            "CC-BY-NC-4.0",
            False,
            ("normalcrafter",),
        ),
    ],
    "Matte": [
        # Matte combos are virtual entries — one checkbox in the UX
        # but two plugins on the executor side. Keeps the mental
        # model simple ("pick a matte backend") and leaves detector
        # + refiner pairing to the catalog rather than the user.
        ModelEntry(
            "sam3_rvm",
            "SAM3 + RVM",
            "SAM-License / MIT",
            True,
            ("sam3_matte", "rvm_refiner"),
        ),
        ModelEntry(
            "sam3_matanyone2",
            "SAM3 + MatAnyone2",
            "CC-BY-NC-4.0",
            False,
            ("sam3_matte", "matanyone2"),
        ),
    ],
}


def find_entry(key: str) -> ModelEntry | None:
    for entries in PASS_CATALOG.values():
        for e in entries:
            if e.key == key:
                return e
    return None


def expand_models(keys: list[str]) -> list[str]:
    """Turn a list of catalog keys into the concrete plugin-name
    sequence the executor receives. Duplicates collapsed in order (a
    user who enabled both `sam3_rvm` and some other SAM3-based model
    doesn't run SAM3 twice).
    """
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        entry = find_entry(k)
        targets = entry.expansion if entry else (k,)
        for t in targets:
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out


def has_noncommercial(keys: list[str]) -> list[ModelEntry]:
    """Return the subset of enabled models that are non-commercial —
    used by the per-submit consent dialog so we can list them
    specifically rather than waving a generic warning.
    """
    out: list[ModelEntry] = []
    for k in keys:
        e = find_entry(k)
        if e is not None and not e.commercial:
            out.append(e)
    return out


__all__ = [
    "PASS_CATALOG",
    "ModelEntry",
    "expand_models",
    "find_entry",
    "has_noncommercial",
]
