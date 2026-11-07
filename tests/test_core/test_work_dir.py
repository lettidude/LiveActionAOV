"""Temp folders for plate-derived frames follow LAOV_WORK_ROOT."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from live_action_aov.core.work_dir import ENV_VAR, make_work_dir, set_work_root, work_root


def test_default_is_the_system_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert work_root() is None
    d = make_work_dir("laov_test_")
    try:
        assert d.parent == Path(tempfile.gettempdir())
    finally:
        d.rmdir()


def test_a_set_root_is_used_and_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    root = tmp_path / "project" / "tmp"
    set_work_root(root)
    d = make_work_dir("laov_test_")
    assert d.parent == root and d.name.startswith("laov_test_") and d.is_dir()
    set_work_root("")
    assert work_root() is None


def test_a_relative_root_becomes_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Child processes run elsewhere (Marigold in its own repo): paths must be absolute."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ENV_VAR, "scratch")
    d = make_work_dir("laov_test_")
    assert d.is_absolute() and d.parent == (tmp_path / "scratch").resolve()
