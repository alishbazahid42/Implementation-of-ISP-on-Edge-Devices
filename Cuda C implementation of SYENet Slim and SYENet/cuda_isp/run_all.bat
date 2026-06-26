@echo off
REM Full pipeline: extract weights → compile → run inference → verify
REM Run from inside cuda_isp\ directory

echo ============================================================
echo  SYEISPNetS  CUDA inference pipeline
echo ============================================================
echo.

REM Step 1: Extract weights and prepare test image
echo [1/4] Extracting weights from model_best.pkl ...
py -3.10 extract_weights.py
if %ERRORLEVEL% NEQ 0 (
    echo FAILED at step 1
    exit /b 1
)
echo.

REM Step 2: Compile
echo [2/4] Compiling syenet.cu ...
call compile.bat
if %ERRORLEVEL% NEQ 0 (
    echo FAILED at step 2
    exit /b 1
)
echo.

REM Step 3: Run CUDA inference
echo [3/4] Running CUDA inference ...
syenet_isp.exe weights.bin input.bin output.bin
if %ERRORLEVEL% NEQ 0 (
    echo FAILED at step 3
    exit /b 1
)
echo.

REM Step 4: Verify PSNR
echo [4/4] Verifying output vs PyTorch reference ...
py -3.10 verify.py
echo.

echo ============================================================
echo  Pipeline complete.
echo ============================================================
