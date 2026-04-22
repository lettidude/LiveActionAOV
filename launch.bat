@echo off
REM LiveActionAOV — quick launcher for the GUI.
REM Double-click this file or run from anywhere; it cd's to its own folder first.

cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo         Run install.bat first, or install uv manually:
    echo         https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

uv run liveaov-gui
if errorlevel 1 (
    echo.
    echo [ERROR] LiveActionAOV GUI exited with an error.
    pause
    exit /b 1
)
