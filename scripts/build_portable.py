# LiveActionAOV
# Copyright (c) 2026 Leonardo Paolini
# Developed with Claude (Anthropic)
# License: BUSL-1.1 (see LICENSE)

"""Build a self-contained, offline LiveActionAOV folder for artist machines.

For studios whose workstations have no internet (TPN): build ONCE on a
machine that has it, put the folder on the file server, and each artist
copies it to a local disk with `install_local.bat` (which also updates).

    uv run python scripts/build_portable.py --out D:\\deploy\\LiveActionAOV

Layout (everything resolves relative to the folder, so it can be moved):

    app/python/      standalone CPython 3.11 + every dependency + liveaov
    models/hf/hub/   Hugging Face weights   (HF_HUB_CACHE)
    models/torch/    torch.hub repos/weights (TORCH_HOME: DSINE, RVM)
    LiveActionAOV.bat   GUI          liveaov.bat   CLI (offline by default)
    install_local.bat   copy/update this folder onto the local disk
    VERSION.txt

No venv: a venv pins the absolute path of its base Python, which breaks the
moment the folder moves. Packages go straight into the standalone CPython.

Re-running against the same --out rebuilds `app/` from scratch but keeps
`models/` (prefetch only downloads what is missing). ComfyUI (LTX alpha)
and the Marigold external env are NOT bundled - they are separate installs.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PY_VERSION = "3.11"

LAUNCH_ENV = r"""set "ROOT=%~dp0"
set "HF_HUB_CACHE=%ROOT%models\hf\hub"
set "TORCH_HOME=%ROOT%models\torch"
set "PYTHONNOUSERSITE=1"
set "PYTHONPATH="
set "PYTHONHOME="
set "PY=%ROOT%app\python\python.exe"
REM Offline unless LAOV_ONLINE=1 (TPN machines must never phone home).
if not "%LAOV_ONLINE%"=="1" set "HF_HUB_OFFLINE=1"
if not "%LAOV_ONLINE%"=="1" set "TRANSFORMERS_OFFLINE=1"
"""

GUI_BAT = (
    "@echo off\r\nREM LiveActionAOV (portable) - GUI. Console stays open for diagnostics.\r\n"
    + LAUNCH_ENV.replace("\n", "\r\n")
    + '"%PY%" -c "from live_action_aov.gui.app import main; main()" %*\r\n'
    + "if errorlevel 1 pause\r\n"
)

CLI_BAT = (
    "@echo off\r\nREM LiveActionAOV (portable) - CLI, e.g.  liveaov.bat run-shot <folder> -p depth,normals\r\n"
    + "setlocal\r\n"
    + LAUNCH_ENV.replace("\n", "\r\n")
    + '"%PY%" -c "from live_action_aov.cli.app import main; main()" %*\r\n'
)

INSTALL_BAT = r"""@echo off
REM Copy (or update) this LiveActionAOV folder onto the local disk.
REM Run it FROM THE SERVER copy. Usage:  install_local.bat [destination]
REM Default destination: %LOCALAPPDATA%\LiveActionAOV
REM Re-run any time to update: only changed files are copied.
setlocal
set "SRC=%~dp0"
if "%SRC:~-1%"=="\" set "SRC=%SRC:~0,-1%"
set "DEST=%~1"
if "%DEST%"=="" set "DEST=%LOCALAPPDATA%\LiveActionAOV"
if /i "%SRC%"=="%DEST%" (
    echo This IS the local copy - run install_local.bat from the server folder.
    pause
    exit /b 1
)
echo Source      : %SRC%
echo Destination : %DEST%
echo.
REM app\ mirrors the server (removes stale files); models\ only adds/updates.
robocopy "%SRC%\app" "%DEST%\app" /MIR /MT:16 /R:2 /W:2 /NFL /NDL /NP
if errorlevel 8 goto :fail
robocopy "%SRC%\models" "%DEST%\models" /E /MT:16 /R:2 /W:2 /NFL /NDL /NP
if errorlevel 8 goto :fail
robocopy "%SRC%" "%DEST%" *.bat VERSION.txt README.txt /R:2 /W:2 /NFL /NDL /NJH /NJS /NP
if errorlevel 8 goto :fail
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\LiveActionAOV.lnk'); $s.TargetPath='%DEST%\LiveActionAOV.bat'; $s.WorkingDirectory='%DEST%'; $s.Save()"
echo.
type "%DEST%\VERSION.txt"
echo [OK] Installed in %DEST%  (desktop shortcut: LiveActionAOV)
pause
exit /b 0
:fail
echo [ERROR] Copy failed - see the robocopy output above.
pause
exit /b 1
"""

README_TXT = """LiveActionAOV - portable build
==============================

Artists:  run install_local.bat from the SERVER folder (copies to
          %LOCALAPPDATA%\\LiveActionAOV + a desktop shortcut). Run it again to
          update. Then start LiveActionAOV from the shortcut.
          Do not run it straight from the server: torch loads thousands of
          files and is very slow over the network.

CLI:      liveaov.bat run-shot <plate folder> -p depth,normals
          liveaov.bat --help

Offline:  the launchers set HF_HUB_OFFLINE=1. Every model must already be in
          models\\ (filled when the folder was built). To allow downloads on
          a machine that has internet: set LAOV_ONLINE=1 before launching.

Not included: ComfyUI (LTX alpha engine) and the Marigold external env.
Needs:    NVIDIA driver recent enough for CUDA 12.8.
"""


def run(cmd: list[str], **kw) -> None:
    print(">", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def clean_env(**extra: str) -> dict[str, str]:
    """Environment for the bundled Python: `uv run` exports PYTHONHOME and
    VIRTUAL_ENV, which would make it load the BUILD machine's stdlib."""
    env = {
        k: v for k, v in os.environ.items()
        if k not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "HF_HUB_OFFLINE"}
        and not k.startswith("UV_INTERNAL")
    }  # fmt: skip
    env["PYTHONNOUSERSITE"] = "1"
    env.update(extra)
    return env


def find_uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("uv not found on PATH (install.bat installs it).")
    return uv


def build_python(uv: str, app: Path) -> Path:
    """Fresh standalone CPython into app/python (never the user's install)."""
    with tempfile.TemporaryDirectory() as tmp:
        run([uv, "python", "install", PY_VERSION, "--install-dir", tmp, "--no-bin"])
        real = [
            d
            for d in Path(tmp).iterdir()
            if d.is_dir() and not d.is_symlink() and d.name.count(".") >= 2
        ]
        if not real:
            sys.exit(f"uv python install produced nothing usable in {tmp}")
        shutil.copytree(real[0], app / "python", symlinks=False)
    # uv marks its Pythons externally managed; this copy is ours to fill.
    (app / "python" / "Lib" / "EXTERNALLY-MANAGED").unlink(missing_ok=True)
    return app / "python" / "python.exe"


def install_packages(uv: str, py: Path, app: Path) -> None:
    req = app / "requirements.txt"
    with req.open("w", encoding="utf-8") as fh:
        subprocess.run(
            [uv, "export", "--frozen", "--no-hashes", "--no-dev", "--extra", "all",
             "--no-emit-project", "--format", "requirements-txt"],
            cwd=REPO, check=True, stdout=fh,
        )  # fmt: skip
    # Versions are pinned exactly by the lock, so best-match across the
    # PyPI + torch-cu128 indexes is safe (and needed: torch==x+cu128).
    base = [uv, "pip", "install", "--python", py, "--link-mode", "copy"]
    run([*base, "-r", req, "--extra-index-url", TORCH_INDEX, "--index-strategy", "unsafe-best-match"])  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        run([uv, "build", "--wheel", "--out-dir", tmp], cwd=REPO)
        wheels = list(Path(tmp).glob("*.whl")) + sorted((REPO / "wheels").glob("*.whl"))
        run([*base, "--no-deps", *wheels])


def prefetch(py: Path, out: Path, models: str) -> None:
    if models == "none":
        return
    # Only the CACHE moves: the HF token (gated repos, e.g. SAM 3) stays
    # where `hf auth login` put it.
    env = clean_env(
        HF_HUB_CACHE=str(out / "models" / "hf" / "hub"),
        TORCH_HOME=str(out / "models" / "torch"),
    )
    args = ["prefetch"]
    if models == "all":
        args.append("--all")
    elif models != "default":
        args += ["--passes", models]
    run([py, "-c", "from live_action_aov.cli.app import main; main()", *args], env=env)


def write_launchers(out: Path) -> None:
    for name, text in (
        ("LiveActionAOV.bat", GUI_BAT),
        ("liveaov.bat", CLI_BAT),
        ("install_local.bat", INSTALL_BAT.replace("\n", "\r\n")),
        ("README.txt", README_TXT.replace("\n", "\r\n")),
    ):
        (out / name).write_bytes(text.encode("ascii"))
    desc = subprocess.run(
        ["git", "describe", "--tags", "--always", "--dirty"],
        cwd=REPO, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    from datetime import datetime

    (out / "VERSION.txt").write_text(
        f"LiveActionAOV {desc or 'unknown'}  (built {datetime.now():%Y-%m-%d %H:%M})\r\n",
        encoding="ascii",
    )


def smoke_test(py: Path, out: Path) -> None:
    env = clean_env(HF_HUB_OFFLINE="1")
    run([py, "-c", "from live_action_aov.cli.app import main; main()", "--version"], env=env)
    run([py, "-c", "import torch, gsplat, flex_gemm, pycolmap, PySide6; "
         "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"], env=env)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="Folder to build into.")
    ap.add_argument(
        "--models", default="default",
        help="'default' (commercial-safe set), 'all', 'none', or a comma list of passes.",
    )  # fmt: skip
    a = ap.parse_args()
    out = a.out.resolve()
    if REPO in out.parents or out == REPO:
        if "_dev" not in out.parts:
            sys.exit("Build outside the repo (or under _dev/ for tests).")
    uv = find_uv()
    app = out / "app"
    if app.exists():
        print(f"Removing old {app} (models/ is kept)")
        shutil.rmtree(app)
    app.mkdir(parents=True)
    (out / "models").mkdir(exist_ok=True)
    py = build_python(uv, app)
    install_packages(uv, py, app)
    prefetch(py, out, a.models)
    write_launchers(out)
    smoke_test(py, out)
    print(f"\n[OK] Portable build ready: {out}\nCopy the whole folder to the server; "
          "artists run install_local.bat from there.")  # fmt: skip


if __name__ == "__main__":
    main()
