@echo off
setlocal enabledelayedexpansion

set LOG_FILE=install.log
echo. > %LOG_FILE%

call :log "LiveActionAOV installer started"

REM 1. Check / install uv
where uv >nul 2>&1
if errorlevel 1 (
    call :log "uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >> %LOG_FILE% 2>&1
    if errorlevel 1 (
        call :fail "Failed to install uv" "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit /b 1
    )
)
for /f "tokens=*" %%V in ('uv --version 2^>nul') do set UV_VERSION=%%V
call :log "[OK] uv: %UV_VERSION%"

REM 2. Check NVIDIA driver
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :log "[WARN] nvidia-smi not found. GPU passes will not run."
    call :log "        If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/Download/index.aspx"
) else (
    for /f "tokens=*" %%D in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2^>nul') do set DRIVER_VER=%%D
    call :log "[OK] NVIDIA driver: !DRIVER_VER!"
)

REM 3. Provision Python 3.11
uv python install 3.11 >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "Failed to provision Python 3.11" "Check network connectivity and disk space"
    exit /b 1
)
call :log "[OK] Python 3.11 provisioned"

REM 4. Sync dependencies. `[tool.uv.sources]` in pyproject pins torch
REM    + torchvision to the pytorch-cu124 index on Windows/Linux, so
REM    this call pulls the CUDA build deterministically — no separate
REM    reinstall step needed, no risk of a future `uv sync` silently
REM    downgrading to the CPU wheel.
call :log "Installing dependencies (this may take several minutes)..."
uv sync --extra dev >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "uv sync failed" "See %LOG_FILE% for details. Common causes: network issues, incompatible torch wheel"
    exit /b 1
)
call :log "[OK] Dependencies installed"

REM 5. Verify CUDA is actually available. Missing driver or a
REM    mismatched wheel surface here as a loud warning rather than
REM    silently breaking the first submit.
if defined DRIVER_VER (
    uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> %LOG_FILE% 2>&1
    if errorlevel 1 (
        call :log "[WARN] torch.cuda.is_available() is False despite NVIDIA driver present."
        call :log "       Check nvidia-smi output; try re-running after a reboot."
    ) else (
        call :log "[OK] torch.cuda.is_available() confirmed"
    )
) else (
    call :log "[WARN] No NVIDIA driver detected — neural passes will refuse to run."
    call :log "       Install drivers then re-run this script."
)

REM 6. Smoke test
uv run liveaov --version >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "Smoke test failed" "See %LOG_FILE%"
    exit /b 1
)
for /f "tokens=*" %%O in ('uv run liveaov --version 2^>nul') do set VERSION_OUTPUT=%%O
call :log "[OK] Smoke test: %VERSION_OUTPUT%"

call :log ""
call :log "[OK] Installation complete."
call :log ""
call :log "Next steps:"
call :log "  uv run liveaov --help"
call :log "  uv run liveaov-gui"

goto :eof

:log
set TS=%date% %time%
echo [%TS%] %~1
echo [%TS%] %~1 >> %LOG_FILE%
goto :eof

:fail
call :log "[FAIL] %~1"
if not "%~2"=="" call :log "        -> %~2"
call :log "Install incomplete. See %LOG_FILE% for full details."
goto :eof
