"""run-shot flags: --flow-fb-error reaches the flow pass only."""

from __future__ import annotations

from live_action_aov.cli.app import _pass_config


def test_flow_fb_error_flag() -> None:
    assert _pass_config("flow", {}, True).params == {"fb_residual": True}
    assert _pass_config("flow_megaflow", {}, True).params == {"fb_residual": True}
    assert _pass_config("flow", {}, False).params == {}
    assert _pass_config("moge3", {"emit_depth": True}, True).params == {"emit_depth": True}
    assert _pass_config("sam3_matte", {}, True).params == {}
