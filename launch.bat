@echo off
REM LiveActionAOV — verbose GUI launcher.
REM Double-click this file; the console stays open so you can see everything
REM uv, Python and the GUI print on the way up and the way down.

cd /d "%~dp0"

echo ======================================================================
echo  LiveActionAOV — launcher
echo ======================================================================
echo  Folder      : %CD%
echo  Date / time : %date% %time%
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo         Run install.bat first, or install uv manually:
    echo         https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

REM --- Environment diagnostics --------------------------------------------
echo  uv version  :
uv --version
echo.

echo  Git branch  :
git rev-parse --abbrev-ref HEAD 2>nul
echo  Git commit  :
git log -1 --oneline 2>nul
echo.

echo  Python (via uv):
uv run python --version
echo.

echo  CUDA visibility (torch):
uv run python -c "import torch; print('  torch       :', torch.__version__); print('  cuda.is_available:', torch.cuda.is_available()); print('  device      :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo.

REM --- Stream Python output to the console in real time -------------------
REM  PYTHONUNBUFFERED  — don't hold stdout/stderr in 4KB buffers.
REM  PYTHONFAULTHANDLER — dump a C-level traceback on segfault (GPU drivers).
set PYTHONUNBUFFERED=1
set PYTHONFAULTHANDLER=1

echo ----------------------------------------------------------------------
echo  Launching: uv run liveaov-gui
echo ----------------------------------------------------------------------
echo.

uv run liveaov-gui
set EXITCODE=%ERRORLEVEL%

echo.
echo ----------------------------------------------------------------------
if %EXITCODE%==0 (
    echo  GUI exited cleanly ^(exit code 0^).
) else (
    echo  [ERROR] GUI exited with code %EXITCODE%.
)
echo ----------------------------------------------------------------------
echo.
pause
exit /b %EXITCODE%
