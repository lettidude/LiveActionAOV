"""The comp script's matte labels come from what actually ran (run report)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from live_action_aov.core.run_report import Fallback, PassReport
from live_action_aov.executors.local import _matte_info


class _Sam:
    name = "sam3_matte"
    provides_artifacts = ["sam3_hard_masks", "sam3_instances"]
    DEFAULT_PARAMS: dict[str, Any] = {}


class _Ltx:
    name = "ltx_alpha_refiner"
    provides_artifacts = ["matte_heroes"]
    DEFAULT_PARAMS: dict[str, Any] = {"mode": "whole_frame", "refine_all_masks": False}


def _info(reports: dict[str, PassReport], overrides: dict[str, Any] | None = None) -> Any:
    ordered = [SimpleNamespace(name="sam3_matte"), SimpleNamespace(name="ltx_alpha_refiner")]
    resolved = {
        "sam3_matte": (SimpleNamespace(params={}), _Sam),
        "ltx_alpha_refiner": (SimpleNamespace(params={}), _Ltx),
    }
    shot = SimpleNamespace(pass_overrides=overrides or {})
    rr = SimpleNamespace(passes=reports)
    return _matte_info(ordered, resolved, shot, rr)  # type: ignore[arg-type]


def _ran(name: str, engine: str, **kw: Any) -> PassReport:
    return PassReport(name=name, status="ran", engine=engine, **kw)


def test_ltx_whole_frame() -> None:
    info = _info(
        {
            "sam3_matte": _ran("sam3_matte", "sam3.1"),
            "ltx_alpha_refiner": _ran("ltx_alpha_refiner", "ltx", extra={"mode": "whole_frame"}),
        }
    )
    assert info.matte == "LTX Alpha (whole frame)"
    assert info.detector == "SAM 3.1"
    assert not info.soft_masks


def test_ltx_fell_back_to_vitmatte() -> None:
    reason = "ComfyUI not reachable"
    info = _info(
        {
            "ltx_alpha_refiner": _ran(
                "ltx_alpha_refiner",
                "vitmatte",
                extra={"mode": "whole_frame"},
                fallbacks=[Fallback(wanted="ltx", used="vitmatte", reason=reason)],
            )
        }
    )
    assert info.matte == "ViTMatte (LTX Alpha did not run)"


def test_failed_refiner_is_not_named_and_soft_masks_follow_the_shot() -> None:
    failed = PassReport(name="ltx_alpha_refiner", status="failed", engine="ltx")
    assert _info({"ltx_alpha_refiner": failed}).matte == ""
    info = _info(
        {"ltx_alpha_refiner": _ran("ltx_alpha_refiner", "ltx", extra={"mode": "whole_frame"})},
        overrides={"ltx_alpha_refiner": {"refine_all_masks": True}},
    )
    assert info.soft_masks
