@echo off
setlocal

echo LiveActionAOV uninstaller
echo.

if exist ".venv" (
    echo Removing .venv\
    rmdir /s /q .venv
    echo [OK] .venv removed
) else (
    echo No .venv found in current directory
)

set CACHE_DIR=%LOCALAPPDATA%\live-action-aov\cache
if exist "%CACHE_DIR%" (
    echo.
    echo Model cache found at: %CACHE_DIR%
    set /p REMOVE_CACHE="Remove model cache? [y/N] "
    if /i "!REMOVE_CACHE!"=="y" (
        rmdir /s /q "%CACHE_DIR%"
        echo [OK] Model cache removed
    ) else (
        echo Model cache kept
    )
)

echo.
echo Uninstall complete. User sessions and sidecars are untouched.
endlocal
