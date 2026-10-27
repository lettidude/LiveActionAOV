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

REM Locally built / downloaded wheels (wheels\*.whl, e.g. gsplat for the
REM Gaussian splat pass) are not on PyPI for this torch/CUDA, so `uv sync`
REM would remove them. Re-install them after every sync.
for %%W in ("%~dp0wheels\*.whl") do (
    echo  Re-installing local wheel %%~nxW
    uv pip install --no-deps "%%W"
)
echo.

echo ----------------------------------------------------------------------
echo  [OK] LiveActionAOV is up to date.
echo ----------------------------------------------------------------------
echo.
pause
