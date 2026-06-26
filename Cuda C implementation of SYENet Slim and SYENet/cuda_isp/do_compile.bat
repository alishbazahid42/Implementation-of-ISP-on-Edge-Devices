@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -arch=sm_75 -o syenet_isp.exe syenet.cu
if %ERRORLEVEL% == 0 (
    echo BUILD_SUCCESS
) else (
    echo BUILD_FAILED
)
