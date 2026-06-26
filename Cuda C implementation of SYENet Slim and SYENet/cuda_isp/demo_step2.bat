@echo off
title SYEISPNetS CPU Demo - Step 2: Run Inference
cd /d C:\Users\aujla\Desktop\archive\cuda_isp

echo.
echo ============================================================
echo   SYEISPNetS  CPU Inference  ^|  Step 2: Run Inference
echo ============================================================
echo.
echo   Command : syenet_cpu.exe weights.bin input.bin output_cpu.bin
echo   Model   : SYEISPNetS slim (5640 weights, 22.1 KB)
echo   Hardware: CPU only (single-threaded C, /O2 /AVX2)
echo   Input   : 4x128x128 RGGB RAW image (Bayer pattern)
echo   Output  : 3x256x256 RGB image (2x upscaled)
echo.
echo ============================================================
echo.
%~dp0syenet_cpu.exe weights.bin input.bin output_cpu.bin
echo.
echo ============================================================
echo   Done. output_cpu.bin written (768 KB, CHW float32)
echo   Run verify_cpu.py to check PSNR vs GPU output.
echo ============================================================
echo.
