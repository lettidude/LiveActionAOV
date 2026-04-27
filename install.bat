@echo off
setlocal enabledelayedexpansion

REM LiveActionAOV — Windows installation script.
REM
REM Why we resolve `uv` as an absolute path: the official installer drops
REM uv.exe at %USERPROFILE%\.local\bin and updates the user PATH in the
REM registry, but those changes don't take effect for the current cmd.exe
REM session. Inside a parenthesized `if` block, `set "PATH=...;%PATH%"`
REM ALSO behaves unreliably because `%PATH%` is expanded at parse time —
REM seen in the wild on a fresh Windows 11 install where uv installed
REM cleanly but `'uv' is not recognized` triggered on the next command.
REM Calling `%UV_EXE%` directly sidesteps the entire PATH dance.

set LOG_FILE=install.log
echo. > %LOG_FILE%

call :log "LiveActionAOV installer started"

REM 1. Check / install uv
set "UV_EXE=uv"
where uv >nul 2>&1
if errorlevel 1 (
    call :log "uv not found, installing..."
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex" >> %LOG_FILE% 2>&1
    if errorlevel 1 (
        call :fail "Failed to install uv" "Install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit /b 1
    )
    REM Use the absolute path the installer wrote to. PATH won't reflect
    REM the install until a new shell starts, but we don't need PATH —
    REM we have the explicit location.
    set "UV_EXE=%USERPROFILE%\.local\bin\uv.exe"
    if not exist "!UV_EXE!" (
        call :fail "uv installer reported success but %USERPROFILE%\.local\bin\uv.exe is missing" "Re-run the installer or install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit /b 1
    )
)

REM Validate uv actually responds — empty/failed --version was previously
REM logged as `[OK] uv:` with nothing after it, hiding the real problem.
set UV_VERSION=
for /f "tokens=*" %%V in ('"%UV_EXE%" --version 2^>nul') do set UV_VERSION=%%V
if "!UV_VERSION!"=="" (
    call :fail "uv is installed but '%UV_EXE% --version' returned no output" "Try opening a new terminal and re-running install.bat. If the issue persists, install uv manually: https://docs.astral.sh/uv/getting-started/installation/"
    exit /b 1
)
call :log "[OK] uv: !UV_VERSION!"

REM 2. Check NVIDIA driver
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :log "[WARN] nvidia-smi not found. GPU passes will not run."
    call :log "        If you have an NVIDIA GPU, install drivers from https://www.nvidia.com/Download/index.aspx"
) else (
    call :probe_driver
)

REM 3. Provision Python 3.11
"%UV_EXE%" python install 3.11 >> %LOG_FILE% 2>&1
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
"%UV_EXE%" sync --extra dev --extra all >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "uv sync failed" "See %LOG_FILE% for details. Common causes: network issues, incompatible torch wheel"
    exit /b 1
)
call :log "[OK] Dependencies installed"

REM 5. Verify CUDA is actually available. Missing driver or a
REM    mismatched wheel surface here as a loud warning rather than
REM    silently breaking the first submit.
if defined DRIVER_VER (
    "%UV_EXE%" run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >> %LOG_FILE% 2>&1
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
"%UV_EXE%" run liveaov --version >> %LOG_FILE% 2>&1
if errorlevel 1 (
    call :fail "Smoke test failed" "See %LOG_FILE%"
    exit /b 1
)
set VERSION_OUTPUT=
for /f "tokens=*" %%O in ('"%UV_EXE%" run liveaov --version 2^>nul') do set VERSION_OUTPUT=%%O
call :log "[OK] Smoke test: !VERSION_OUTPUT!"

call :log ""
call :log "[OK] Installation complete."
call :log ""
call :log "Next steps:"
call :log "  uv run liveaov --help"
call :log "  uv run liveaov-gui"

goto :eof

:probe_driver
REM Probe the NVIDIA driver version with three fallbacks. Some old/stripped
REM driver installs reject `noheader`/`nounits` and `nvidia-smi` writes the
REM rejection text to stdout — which would then be captured into DRIVER_VER
REM and logged as a successful "[OK]" line. Each tier validates errorlevel
REM AND that the captured string doesn't start with "ERROR".
REM
REM Tier 1: modern format (driver_version, no header, no units)
set DRIVER_VER=
for /f "tokens=*" %%D in ('nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2^>nul') do (
    if not defined DRIVER_VER set DRIVER_VER=%%D
)
call :_validate_driver_var
if defined DRIVER_VER (
    call :log "[OK] NVIDIA driver: !DRIVER_VER!"
    goto :eof
)

REM Tier 2: just `--format=csv` (some drivers reject `noheader`); skip the
REM first line which is the "driver_version" header row.
set _drv_line=0
for /f "tokens=*" %%D in ('nvidia-smi --query-gpu=driver_version --format=csv 2^>nul') do (
    set /a _drv_line+=1
    if !_drv_line! GEQ 2 if not defined DRIVER_VER set DRIVER_VER=%%D
)
call :_validate_driver_var
if defined DRIVER_VER (
    call :log "[OK] NVIDIA driver: !DRIVER_VER!"
    goto :eof
)

REM Tier 3: bare `nvidia-smi` — driver is clearly there (the `where` check
REM passed), we just can't pull a clean version string. Don't fail; mark
REM unknown so step 5's CUDA self-test still runs.
set "DRIVER_VER=unknown"
call :log "[OK] NVIDIA driver detected (version probe returned unexpected output; see %LOG_FILE%)"
nvidia-smi >> %LOG_FILE% 2>&1
goto :eof

:_validate_driver_var
REM If DRIVER_VER got polluted with an error string ("ERROR: ...") or is
REM empty, clear it so the caller falls through to the next tier.
if not defined DRIVER_VER goto :eof
echo !DRIVER_VER! | findstr /b /i "ERROR" >nul && set DRIVER_VER=
if defined DRIVER_VER (
    REM Reject anything with non-numeric leading char (real driver versions
    REM look like `560.94` or `551.86`).
    echo !DRIVER_VER! | findstr /r "^[0-9]" >nul || set DRIVER_VER=
)
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
