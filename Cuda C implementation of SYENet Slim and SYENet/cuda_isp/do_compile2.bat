@echo off
cd /d C:\Users\aujla\Desktop\archive\cuda_isp
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe" -O3 -arch=sm_75 -o syenet_isp.exe syenet.cu > build_out.txt 2>&1
echo EXITCODE=%ERRORLEVEL% >> build_out.txt
type build_out.txt
