"""`liveaov setup-sam31`: the command plan, with the commands faked."""

from __future__ import annotations

from pathlib import Path

import pytest

from live_action_aov.core import sam31_setup
from live_action_aov.passes.matte import sam31_bridge


def test_setup_runs_every_step_then_skips_what_is_there(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sam31_setup, "missing_tools", lambda: [])
    ran: list[list[str]] = []

    def runner(cmd: list[str]) -> int:
        ran.append(cmd)
        if cmd[:2] == ["uv", "venv"]:  # what uv would create
            py = sam31_bridge.env_python(tmp_path)
            py.parent.mkdir(parents=True)
            py.write_bytes(b"")
        if cmd[:2] == ["git", "clone"]:
            (tmp_path / "sam3_repo" / ".git").mkdir(parents=True)
        return 0

    said: list[str] = []
    py = sam31_setup.setup(tmp_path, echo=said.append, runner=runner)
    assert py == sam31_bridge.env_python(tmp_path) and py.is_file()
    assert [c[:2] for c in ran][:2] == [["uv", "venv"], ["uv", "pip"]]
    assert any(sam31_setup.SAM3_COMMIT in c for c in ran)
    assert ran[-1][-1] == "import sam3"  # ends with the import check
    first = len(ran)
    sam31_setup.setup(tmp_path, echo=said.append, runner=runner)
    again = ran[first:]
    assert not any(c[:2] in (["uv", "venv"], ["git", "clone"]) for c in again)
    assert any("already there" in s for s in said)


def test_setup_names_the_failed_step_and_missing_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sam31_setup, "missing_tools", lambda: ["uv"])
    with pytest.raises(RuntimeError, match="uv not found"):
        sam31_setup.setup(tmp_path, echo=lambda s: None, runner=lambda c: 0)
    monkeypatch.setattr(sam31_setup, "missing_tools", lambda: [])
    with pytest.raises(RuntimeError, match=r"(?s)Python 3.12 environment.*again"):
        sam31_setup.setup(tmp_path, echo=lambda s: None, runner=lambda c: 1)


def test_custom_folder_is_remembered_default_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sam31_setup, "default_dir", lambda: tmp_path / "default")
    ran: list[list[str]] = []
    said: list[str] = []

    def runner(cmd: list[str]) -> int:
        ran.append(cmd)
        return 0

    sam31_setup.remember(
        sam31_bridge.env_python(tmp_path / "default"), echo=said.append, runner=runner
    )
    assert ran == [] and said == []
    sam31_setup.remember(
        sam31_bridge.env_python(tmp_path / "custom"), echo=said.append, runner=runner
    )
    assert said and sam31_bridge.ENV_PYTHON in said[0]
