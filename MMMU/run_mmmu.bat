@echo off
REM ============================================================
REM Run MMMU benchmark (VLM multimodal eval)
REM Data in local MMMU dir, calls run_mmmu_openvino.py
REM ============================================================
REM Usage: run_mmmu.bat [model_path] [options]
REM Dataset: MMMU\MMMU_Pro_dataset\vision or standard (10 options)
REM Output: MMMU\mmmu-pro\output\{model}_{setting}_{mode}.jsonl
REM Requires: pip install pillow pyyaml datasets
REM ============================================================

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
cd ..\..
set "OPENVINO_ROOT=%CD%"
cd /d "%SCRIPT_DIR%"

REM Do not set PYTHONPATH - script imports datasets first, then adds OpenVINO paths
set "OV_GENAI_USE_MODELING_API=1"
if "%OV_GENAI_INFLIGHT_QUANT_MODE%"=="" set "OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym"
if "%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%"=="" set "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128"
if "%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%"=="" set "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym"

echo [INFO] OV_GENAI_USE_MODELING_API=1
echo [INFO] OV_GENAI_INFLIGHT_QUANT_MODE=%OV_GENAI_INFLIGHT_QUANT_MODE%
echo.

set "OPENVINO_LIB_PATHS=%OPENVINO_ROOT%\openvino\bin\intel64\Release"
if exist "%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin" set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin"
if exist "%OPENVINO_ROOT%\openvino\build\install\runtime\bin\intel64\Release" set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%\openvino\build\install\runtime\bin\intel64\Release"

set "PATH=%OPENVINO_ROOT%\openvino.genai\build\openvino_genai;%PATH%"
set "PATH=%OPENVINO_ROOT%\openvino\bin\intel64\Release;%PATH%"
if exist "%OPENVINO_ROOT%\openvino.genai\build\bin\Release" set "PATH=%PATH%;%OPENVINO_ROOT%\openvino.genai\build\bin\Release"
if exist "%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin" set "PATH=%PATH%;%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin"

python -c "import sys; print('[INFO] Python:', sys.executable)"
echo.
echo Running: python run_mmmu_openvino.py %*
echo.
python run_mmmu_openvino.py %*

endlocal
