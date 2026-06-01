# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Pre-flight dependency check for passes with optional extras.

Several passes pull heavy backends declared as optional `pip` extras
(`diffusers`, `geffnet`, `einops`, …). If a user enables such a pass without
installing the matching extra, the run used to die mid-submit with a raw
`ModuleNotFoundError` deep in `_load_model`. This module checks up-front —
before any frame is read — and raises one clear, forwardable message naming
the exact `pip install` to run.

Checks use `importlib.util.find_spec` (no import side effects, no model load).
"""

from __future__ import annotations

import importlib.util

# Pass plugin name -> (pip extra, [python modules that must be importable]).
# Mirrors the optional-dependency extras in pyproject.toml. Passes backed only
# by core deps (transformers, torch, opencv) map to ("", []) and are skipped.
_PASS_REQUIREMENTS: dict[str, tuple[str, list[str]]] = {
    "flow": ("", []),
    "depth_anything_v2": ("", []),  # transformers (core)
    "depthpro": ("", []),  # transformers (core)
    "video_depth_anything": ("video_depth_anything", ["einops", "easydict"]),
    "depthcrafter": ("depthcrafter", ["diffusers", "accelerate"]),
    "dsine": ("dsine", ["geffnet"]),
    "normalcrafter": ("normalcrafter", ["diffusers", "accelerate"]),
    "marigold_iid_lighting": ("marigold", ["diffusers", "accelerate"]),
    "marigold_iid_appearance": ("marigold", ["diffusers", "accelerate"]),
    "sam3_matte": ("", []),  # transformers (core)
    "rvm_refiner": ("", []),  # torch.hub (core)
    "matanyone2": ("", []),
}


class MissingDependencyError(RuntimeError):
    """Raised before a run when a selected pass's optional extra isn't installed.

    Carries a ready-to-forward message naming the `pip install` per pass.
    """


def _module_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        # A broken/partial install (find_spec itself raising) is, for our
        # purposes, "not usable" — report it as missing so the user reinstalls.
        return True


def missing_dependencies(pass_names: list[str]) -> list[tuple[str, str, list[str]]]:
    """Return `[(pass_name, pip_extra, [missing_modules]), …]` for selected
    passes whose optional extra isn't fully installed. Empty = all good.

    Unknown pass names (third-party plugins not in the map) are skipped — we
    can't know their deps, and they'll surface their own import error if any.
    """
    out: list[tuple[str, str, list[str]]] = []
    seen: set[str] = set()
    for name in pass_names:
        if name in seen:
            continue
        seen.add(name)
        extra, modules = _PASS_REQUIREMENTS.get(name, ("", []))
        if not modules:
            continue
        missing = [m for m in modules if _module_missing(m)]
        if missing:
            out.append((name, extra, missing))
    return out


def check_dependencies(pass_names: list[str]) -> None:
    """Raise `MissingDependencyError` if any selected pass is missing its extra.

    Call before reading frames / loading models so the user gets one clear
    message instead of a stack trace mid-run.
    """
    missing = missing_dependencies(pass_names)
    if not missing:
        return
    lines = ["Missing dependencies for selected passes — install the matching extra(s):", ""]
    for name, extra, mods in missing:
        target = f".[{extra}]" if extra else "."
        lines.append(f"  - pass '{name}': pip install -e \"{target}\"")
        lines.append(f"      (missing: {', '.join(mods)})")
    lines.append("")
    lines.append("Then re-run. (Or run `uv sync --all-extras` to install every backend.)")
    raise MissingDependencyError("\n".join(lines))


__all__ = ["MissingDependencyError", "check_dependencies", "missing_dependencies"]
