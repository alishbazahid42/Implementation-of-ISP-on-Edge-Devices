@echo off
REM Compile syenet.cu for Quadro T1000 (SM 7.5)
REM Requires: VS 2022 Build Tools + CUDA 13.3

set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe"
set VCVARS="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo Setting up MSVC environment ...
call %VCVARS% >nul 2>&1

echo Compiling syenet.cu ...
%NVCC% -O3 -arch=sm_75 -o syenet_isp.exe syenet.cu

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo *** Compilation FAILED ***
    exit /b 1
)

echo.
echo OK: syenet_isp.exe built
echo Usage: syenet_isp.exe weights.bin input.bin output.bin
