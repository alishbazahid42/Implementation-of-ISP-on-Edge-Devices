@echo off
title SYEISPNetS CPU Demo - Step 3: Verify Results
cd /d C:\Users\aujla\Desktop\archive\cuda_isp

echo.
echo ============================================================
echo   SYEISPNetS  CPU Inference  ^|  Step 3: Verify Results
echo ============================================================
echo.
echo   Command : py -3.10 verify_cpu.py
echo   Compares: output_cpu.bin vs output.bin (GPU) vs ref.bin (PyTorch)
echo   Metric  : PSNR - higher = more similar (^>60 dB = bit-identical)
echo.
echo ============================================================
echo.
py -3.10 verify_cpu.py
echo.
echo ============================================================
echo   Saved: comparison_cpu.png (CPU ^| GPU ^| GT side-by-side)
echo   DONE - all steps complete!
echo ============================================================
echo.
