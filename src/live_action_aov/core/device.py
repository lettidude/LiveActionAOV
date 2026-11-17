# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Which torch device a pass runs on: one rule for every pass.

Order: an NVIDIA card (CUDA), then the Apple GPU (MPS), then the CPU.
`LAOV_DEVICE` overrides it (`cuda`, `cuda:1`, `mps`, `cpu`, or `auto`).

On a machine with CUDA and no override, `pick_device()` returns
`torch.device("cuda")`, exactly what every pass did before this module.
MPS is only used when torch reports it (an Apple Silicon Mac) or when
`LAOV_DEVICE=mps` asks for it, and only for passes that list `"mps"` in
their `device_support` (`core.pass_base.DeviceSupport`).

Mac support is experimental and has not been checked on real Apple
hardware yet: see `docs/mac_port_plan.md`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

ENV_VAR = "LAOV_DEVICE"
#: Auto order, best first.
AUTO_ORDER: tuple[str, ...] = ("cuda", "mps", "cpu")
#: What a pass gets when it does not say (today's behaviour).
DEFAULT_DEVICES: tuple[str, ...] = ("cuda", "cpu")

_log = logging.getLogger(__name__)
_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        _log.warning(msg)


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def mps_available() -> bool:
    """True on an Apple Silicon Mac whose torch build has MPS."""
    try:
        import torch

        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def _kind(name: str) -> str:
    """`cuda:1` -> `cuda`."""
    return name.split(":", 1)[0]


def resolve_device_name(
    allowed: Iterable[str] = DEFAULT_DEVICES,
    *,
    override: str | None = None,
    cuda_ok: bool | None = None,
    mps_ok: bool | None = None,
) -> str:
    """The device name a pass should use, as a string (`cuda`, `cuda:1`,
    `mps`, `cpu`). Pure logic: availability can be passed in (tests)."""
    allowed_kinds = {_kind(a) for a in allowed} | {"cpu"}  # the CPU is always a fallback
    if override is None:
        override = os.environ.get(ENV_VAR, "")
    override = override.strip().lower()
    have = {
        "cuda": cuda_available() if cuda_ok is None else cuda_ok,
        "mps": mps_available() if mps_ok is None else mps_ok,
        "cpu": True,
    }
    if override and override != "auto":
        kind = _kind(override)
        if kind not in have:
            _warn_once(f"{ENV_VAR}={override!r} is not a device (cuda, mps, cpu, auto): ignored.")
        elif not have[kind]:
            _warn_once(f"{ENV_VAR}={override!r}: no such device on this machine, using auto.")
        elif kind not in allowed_kinds:
            _warn_once(
                f"{ENV_VAR}={override!r}: a pass does not support {kind}, "
                "it uses the next device that it supports."
            )
        else:
            return override
    for kind in AUTO_ORDER:
        if kind in allowed_kinds and have[kind]:
            return kind
    return "cpu"


def pick_device(allowed: Any = DEFAULT_DEVICES) -> torch.device:
    """The torch device for a pass. `allowed` is the pass's
    `device_support` (or a plain list of device names)."""
    import torch

    devices = getattr(allowed, "devices", allowed)
    return torch.device(resolve_device_name(devices))


def autocast_device_type(device: Any) -> str:
    """The `device_type` string `torch.autocast` takes for `device`."""
    kind = str(getattr(device, "type", device) or "cpu")
    return kind if kind in AUTO_ORDER else "cpu"


def empty_cache() -> None:
    """Hand torch's cached GPU blocks back (CUDA, or MPS on a Mac).
    Never raises; a no-op without a GPU."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif mps_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def mac_table(passes: dict[str, Any]) -> list[tuple[str, str, str]]:
    """`(pass name, Mac status, note)` for each pass class, sorted by name.
    `passes` maps a plugin name to its class."""
    from live_action_aov.core.pass_base import DeviceSupport

    rows = []
    for name in sorted(passes):
        ds = getattr(passes[name], "device_support", None)
        if not isinstance(ds, DeviceSupport):
            ds = DeviceSupport()
        rows.append((name, ds.mac_label(), ds.mac_notes))
    return rows


def mac_check(passes: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Before a run on a Mac: (passes that cannot run there, passes not
    yet checked on a real Mac), each as "name: why" lines."""
    from live_action_aov.core.pass_base import DeviceSupport, MacSupport

    blocked: list[str] = []
    unverified: list[str] = []
    for name in sorted(passes):
        ds = getattr(passes[name], "device_support", None)
        if not isinstance(ds, DeviceSupport):
            ds = DeviceSupport()
        why = f"{name}: {ds.mac_label()}" + (f" - {ds.mac_notes}" if ds.mac_notes else "")
        if ds.mac is MacSupport.UNAVAILABLE:
            blocked.append(why)
        elif not ds.mac_verified:
            unverified.append(why)
    return blocked, unverified


__all__ = [
    "AUTO_ORDER",
    "DEFAULT_DEVICES",
    "ENV_VAR",
    "autocast_device_type",
    "cuda_available",
    "empty_cache",
    "mac_check",
    "mac_table",
    "mps_available",
    "pick_device",
    "resolve_device_name",
]
