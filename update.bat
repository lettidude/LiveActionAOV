@echo off
REM LiveActionAOV — pull the latest code + re-sync dependencies.
REM Idempotent: if nothing changed, it's a no-op.

cd /d "%~dp0"

echo ======================================================================
echo  LiveActionAOV — update
echo ======================================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo         Run install.bat first.
    pause
    exit /b 1
)

echo  Fetching latest from origin...
git pull origin main
if errorlevel 1 (
    echo.
    echo [ERROR] git pull failed. If you have local commits, push or stash them first.
    pause
    exit /b 1
)
echo.

echo  Re-syncing dependencies (no-op if pyproject is unchanged)...
uv sync --extra all
if errorlevel 1 (
    echo.
    echo [ERROR] uv sync failed. See the output above for details.
    pause
    exit /b 1
)
echo.

echo ----------------------------------------------------------------------
echo  [OK] LiveActionAOV is up to date.
echo ----------------------------------------------------------------------
echo.
pause
