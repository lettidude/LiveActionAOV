# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: MIT

"""Pre-flight dependency check tests."""

from __future__ import annotations

import pytest

from live_action_aov.core import preflight
from live_action_aov.core.preflight import (
    MissingDependencyError,
    check_dependencies,
    missing_dependencies,
)


def test_module_missing_detection() -> None:
    assert preflight._module_missing("os") is False
    assert preflight._module_missing("definitely_not_a_real_module_xyz") is True


def test_core_pass_has_no_requirements() -> None:
    # `flow` rides on core deps only — never flagged.
    assert missing_dependencies(["flow"]) == []


def test_unknown_pass_is_skipped() -> None:
    # Third-party plugin we don't have a map entry for: can't check, don't guess.
    assert missing_dependencies(["some_third_party_pass"]) == []


def test_missing_extra_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        preflight._PASS_REQUIREMENTS,
        "fake_pass",
        ("fakeextra", ["definitely_not_a_real_module_xyz"]),
    )
    result = missing_dependencies(["fake_pass"])
    assert result == [("fake_pass", "fakeextra", ["definitely_not_a_real_module_xyz"])]


def test_check_raises_with_actionable_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        preflight._PASS_REQUIREMENTS,
        "fake_pass",
        ("fakeextra", ["definitely_not_a_real_module_xyz"]),
    )
    with pytest.raises(MissingDependencyError) as exc:
        check_dependencies(["fake_pass"])
    msg = str(exc.value)
    assert 'pip install -e ".[fakeextra]"' in msg
    assert "definitely_not_a_real_module_xyz" in msg


def test_check_passes_when_all_present() -> None:
    # No exception for core-only passes.
    check_dependencies(["flow", "depth_anything_v2"])
