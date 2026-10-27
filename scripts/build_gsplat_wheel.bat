@echo off
REM LiveActionAOV - build the gsplat wheel for the Gaussian splat pass (Windows).
REM
REM Why: gsplat ships CUDA kernels. Its prebuilt Windows wheels stop at
REM torch 2.4 / CUDA 12.4 (no RTX 50xx), and the tool runs torch 2.11 /
REM CUDA 12.8. This builds ONE "fat" wheel for every NVIDIA arch from
REM RTX 20xx (Turing) to RTX 50xx (Blackwell) + PTX for newer GPUs, against
REM the tool's own torch. The result works on any Windows machine running
REM this tool (same uv.lock -> same torch), not only the one that built it.
REM
REM Needs (build machine only - users of the wheel need neither):
REM   - NVIDIA CUDA Toolkit 12.8  (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8)
REM   - Visual Studio 2022 Build Tools with "Desktop development with C++"
REM   - git
REM Output: wheels\gsplat-1.5.3-cp311-cp311-win_amd64.whl, installed into .venv.
REM install.bat / update.bat re-install every wheel in wheels\ after `uv sync`.
REM Build time: ~30-60 min for all archs (set TORCH_CUDA_ARCH_LIST=12.0 for a
REM quick single-GPU build that only runs on RTX 50xx).

setlocal
cd /d "%~dp0\.."
set REPO=%CD%
set GSPLAT_TAG=v1.5.3

if "%CUDA_HOME%"=="" set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
if not exist "%CUDA_HOME%\bin\nvcc.exe" (
    echo [ERROR] CUDA Toolkit 12.8 not found at "%CUDA_HOME%". Install it or set CUDA_HOME.
    exit /b 1
)
set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" set VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat
if not exist "%VCVARS%" (
    echo [ERROR] Visual Studio 2022 C++ build tools not found ^(vcvars64.bat^).
    exit /b 1
)
call "%VCVARS%" >nul

set CUDA_PATH=%CUDA_HOME%
set PATH=%CUDA_HOME%\bin;%PATH%
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6;8.9;9.0;12.0+PTX
set DISTUTILS_USE_SDK=1
if "%MAX_JOBS%"=="" set MAX_JOBS=16

if not exist "%REPO%\.build" mkdir "%REPO%\.build"
if not exist "%REPO%\.build\gsplat" (
    git clone --depth 1 --branch %GSPLAT_TAG% --recurse-submodules https://github.com/nerfstudio-project/gsplat.git "%REPO%\.build\gsplat"
    if errorlevel 1 exit /b 1
)

REM Build-only tools into the tool venv (uv sync removes them again later).
uv pip install --python "%REPO%\.venv\Scripts\python.exe" ninja setuptools wheel
if errorlevel 1 exit /b 1

echo Building gsplat %GSPLAT_TAG% for archs %TORCH_CUDA_ARCH_LIST% ...
pushd "%REPO%\.build\gsplat"
"%REPO%\.venv\Scripts\python.exe" setup.py bdist_wheel -d "%REPO%\wheels"
if errorlevel 1 (
    popd
    echo [ERROR] gsplat build failed - see the output above.
    exit /b 1
)
popd

for %%W in ("%REPO%\wheels\gsplat-*.whl") do uv pip install --python "%REPO%\.venv\Scripts\python.exe" --no-deps --reinstall "%%W"
"%REPO%\.venv\Scripts\python.exe" -c "import gsplat, torch; from gsplat.cuda._backend import _C; print('gsplat', gsplat.__version__, 'OK on', torch.cuda.get_device_name(0))"
endlocal
