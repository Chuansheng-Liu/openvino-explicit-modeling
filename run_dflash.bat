@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  DFlash Benchmark (LLM + VLM)
REM
REM  Usage:
REM    run_dflash.bat                           -- Full benchmark (LLM + VLM)
REM    run_dflash.bat --skip-vlm                -- LLM only
REM    run_dflash.bat --skip-baseline           -- DFlash configs only
REM    run_dflash.bat FP16 FP16                 -- Single run: target=FP16 draft=FP16
REM    run_dflash.bat FP16 FP16 512 16 img.jpg  -- Single run: VL mode
REM
REM  When called with quant args (%1=FP16/INT4_ASYM), runs single .exe.
REM  When called with --flags or no args, runs run_dflash_benchmark.ps1.
REM ============================================================

REM --- Paths ---
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"

REM --- DLL environment ---
set "OPENVINO_BIN=%ROOT_DIR%\openvino\bin\intel64\Release"
set "TBB_BIN=%ROOT_DIR%\openvino\temp\Windows_AMD64\tbb\bin"
set "GENAI_DLL_DIR=%ROOT_DIR%\openvino.genai\build\openvino_genai"
set "GENAI_BIN_DIR=%ROOT_DIR%\openvino.genai\build\bin"
set "GENAI_BIN_DIR_REL=%ROOT_DIR%\openvino.genai\build\bin\Release"
set "PATH=%OPENVINO_BIN%;%TBB_BIN%;%GENAI_DLL_DIR%;%GENAI_BIN_DIR%;%GENAI_BIN_DIR_REL%;%PATH%"

REM --- Python paths ---
set "OV_PYTHON=%ROOT_DIR%\openvino\bin\intel64\Release\python"
set "GENAI_PYTHON=%ROOT_DIR%\openvino.genai\build"
set "PYTHONPATH=%OV_PYTHON%;%GENAI_PYTHON%;%PYTHONPATH%"

REM --- Enable modeling API ---
set "OV_GENAI_USE_MODELING_API=1"

REM --- Disable thinking mode (matches --no-think in dflash_benchmark.py) ---
set "OV_GENAI_DISABLE_THINKING=1"

REM --- Check if first arg is a -- flag (benchmark mode) or quant mode (single run) ---
set "ARG1=%~1"
if "%ARG1%"=="" goto :benchmark_mode
if "%ARG1:~0,2%"=="--" goto :benchmark_mode

REM ============================================================
REM  Single-run mode (legacy: run_dflash.bat FP16 FP16 ...)
REM ============================================================
REM Prefer build\bin\ (current CMake output) over build\bin\Release\ (may be stale)
set "DFLASH_EXE=%GENAI_BIN_DIR%\modeling_qwen3_5_dflash.exe"
if not exist "%DFLASH_EXE%" set "DFLASH_EXE=%GENAI_BIN_DIR_REL%\modeling_qwen3_5_dflash.exe"
set "TARGET_MODEL=E:\data\models\Huggingface\Qwen3.5-4B"
set "DRAFT_MODEL=E:\data\models\Huggingface\Qwen3.5-4B-Dflash"

set "TARGET_QUANT=%~1"
set "DRAFT_QUANT=%~2"
set "MAX_NEW_TOKENS=%~3"
set "BLOCK_SIZE=%~4"
set "IMAGE_PATH=%~5"

if "%TARGET_QUANT%"=="" set "TARGET_QUANT=INT4_ASYM"
if "%DRAFT_QUANT%"=="" set "DRAFT_QUANT=INT4_ASYM"
if "%MAX_NEW_TOKENS%"=="" set "MAX_NEW_TOKENS=512"
if "%BLOCK_SIZE%"=="" set "BLOCK_SIZE=16"

set "DEVICE=GPU"

if not "%IMAGE_PATH%"=="" (
    set "MODE=VL"
    set "PROMPT_TEXT=Describe this image in detail."
) else (
    set "MODE=Text"
)

set "PROMPT_FILE=%SCRIPT_DIR%\scripts\prompt_1k.txt"
if "%MODE%"=="Text" (
    if "!PROMPT_TEXT!"=="" set "PROMPT_TEXT=Tell me a short story about a robot."
    if exist "%PROMPT_FILE%" (
        for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "Get-Content -Raw '%PROMPT_FILE%' | %%{ $_ -replace \"`r`n\", \" \" }"`) do (
            set "PROMPT_TEXT=%%A"
        )
    )
)

echo ============================================================
echo   DFlash [%MODE%]  target=%TARGET_QUANT%  draft=%DRAFT_QUANT%
echo   block_size=%BLOCK_SIZE%  max_tokens=%MAX_NEW_TOKENS%
echo   Device: %DEVICE%
if "%MODE%"=="VL" echo   Image: %IMAGE_PATH%
echo ============================================================
echo.

if "%MODE%"=="VL" (
    "%DFLASH_EXE%" "%TARGET_MODEL%" "%DRAFT_MODEL%" "%PROMPT_TEXT%" "%DEVICE%" "%MAX_NEW_TOKENS%" "%BLOCK_SIZE%" "%TARGET_QUANT%" "%DRAFT_QUANT%" "%IMAGE_PATH%"
) else (
    "%DFLASH_EXE%" "%TARGET_MODEL%" "%DRAFT_MODEL%" "%PROMPT_TEXT%" "%DEVICE%" "%MAX_NEW_TOKENS%" "%BLOCK_SIZE%" "%TARGET_QUANT%" "%DRAFT_QUANT%"
)
set "RC=%ERRORLEVEL%"
echo.
echo [Exit code: %RC%]
exit /b %RC%

REM ============================================================
REM  Benchmark mode (calls run_dflash_benchmark.ps1)
REM ============================================================
:benchmark_mode
echo ============================================================
echo   DFlash EXE Benchmark (LLM + VLM)
echo ============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\run_dflash_benchmark.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo [Exit code: %RC%]
exit /b %RC%
