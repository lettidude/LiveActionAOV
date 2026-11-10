# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Build the SAM 3.1 environment (`liveaov setup-sam31`).

SAM 3.1 ships as Meta's `sam3` package for Python 3.12, with imports it does
not declare, so it lives in its own environment and LiveActionAOV runs it as
a subprocess (`passes.matte.sam31_bridge`). This module is the list of
commands that builds that environment, so nobody has to type them:

    <folder>/env        a Python 3.12 venv (uv) with torch for the card
    <folder>/sam3_repo  Meta's repository at the commit we tested

Every step is skipped when its result is already there, so running the
command again after a failure (or on an existing install) is cheap. The
checkpoint itself is gated on Hugging Face and downloads at the first run;
this never touches tokens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from live_action_aov.passes.matte.sam31_bridge import ENV_PYTHON, default_dir, env_python

SAM3_REPO = "https://github.com/facebookresearch/sam3"
#: The commit LiveActionAOV's worker was written against.
SAM3_COMMIT = "f1647e8"
#: Imported by Meta's package at load time but not declared by it.
UNDECLARED = (
    "pycocotools",
    "psutil",
    "opencv-python",
    "scipy",
    "hydra-core",
    "omegaconf",
    "matplotlib",
    "pandas",
    "python-rapidjson",
)

Runner = Callable[[list[str]], int]


@dataclass(frozen=True)
class Step:
    title: str
    cmd: list[str]
    done: Callable[[], bool] = lambda: False  # True = skip


def steps(root: Path, torch_backend: str = "cu128") -> list[Step]:
    """The commands, in order, for an environment under `root`."""
    py = env_python(root)
    repo = root / "sam3_repo"
    pip = ["uv", "pip", "install", "--python", str(py)]
    extras = list(UNDECLARED) + (["triton-windows"] if os.name == "nt" else [])
    return [
        Step(
            "Python 3.12 environment",
            ["uv", "venv", str(root / "env"), "--python", "3.12"],
            done=py.is_file,
        ),
        Step("PyTorch", [*pip, "torch", "torchvision", f"--torch-backend={torch_backend}"]),
        Step(
            "Meta's sam3 repository",
            ["git", "clone", SAM3_REPO, str(repo)],
            done=lambda: (repo / ".git").exists(),
        ),
        Step("Tested commit", ["git", "-C", str(repo), "checkout", "--quiet", SAM3_COMMIT]),
        Step("sam3 package", [*pip, "-e", str(repo), "huggingface_hub", "einops"]),
        Step("Imports sam3 needs but does not declare", [*pip, *extras]),
        Step("Import check", [str(py), "-I", "-c", "import sam3"]),
    ]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def missing_tools() -> list[str]:
    return [t for t in ("uv", "git") if shutil.which(t) is None]


def setup(
    root: Path | None = None,
    *,
    torch_backend: str = "cu128",
    echo: Callable[[str], None] = print,
    runner: Runner = _run,
) -> Path:
    """Build (or complete) the environment and return its interpreter.
    Raises RuntimeError naming the step that failed."""
    root = Path(root) if root is not None else default_dir()
    lacking = missing_tools()
    if lacking:
        raise RuntimeError(
            f"{' and '.join(lacking)} not found on PATH. LiveActionAOV's installer puts "
            "uv there; git comes from https://git-scm.com."
        )
    root.mkdir(parents=True, exist_ok=True)
    plan = steps(root, torch_backend)
    for i, step in enumerate(plan, 1):
        if step.done():
            echo(f"[{i}/{len(plan)}] {step.title}: already there")
            continue
        echo(f"[{i}/{len(plan)}] {step.title}...")
        rc = runner(step.cmd)
        if rc != 0:
            raise RuntimeError(
                f"'{step.title}' failed (exit {rc}): {' '.join(step.cmd)}\n"
                "Fix the cause and run `liveaov setup-sam31` again: finished steps are skipped."
            )
    return env_python(root)


def remember(python: Path, *, echo: Callable[[str], None] = print, runner: Runner = _run) -> None:
    """A custom folder is not found by itself: store its interpreter in the
    user's environment (Windows), or say what to export."""
    if env_python(default_dir()) == python:
        return
    if os.name == "nt":
        if runner(["setx", ENV_PYTHON, str(python)]) == 0:
            echo(f"{ENV_PYTHON} set for new terminals and for the GUI's next start.")
            return
    echo(f"Set {ENV_PYTHON}={python} in your environment.")


__all__ = ["SAM3_COMMIT", "SAM3_REPO", "Step", "missing_tools", "remember", "setup", "steps"]
