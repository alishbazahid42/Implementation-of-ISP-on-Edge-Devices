@echo off
REM ============================================================
REM  SYENet ISP CUDA Benchmark — Windows Build Script
REM  Run from: C:\Users\aujla\Desktop\archive\cuda_benchmark\
REM ============================================================

REM ── Set TensorRT path (edit to match your install) ──────────
set TRT_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT-10.9.0.34

REM ── CMake configuration ─────────────────────────────────────
cmake -S . -B build ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DTRT_DIR="%TRT_DIR%" ^
    -DCMAKE_CUDA_ARCHITECTURES="75"

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed.
    pause
    exit /b 1
)

REM ── Build ────────────────────────────────────────────────────
cmake --build build --config Release --parallel

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Build successful!
echo Executable: build\Release\syenet_bench.exe
pause
