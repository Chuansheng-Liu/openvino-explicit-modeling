@echo off
REM ============================================================
REM Run IFEval benchmark (LLM instruction-following eval)
REM Data in local IFEVAL dir, calls run_ifeval_openvino.py
REM ============================================================
REM Usage: run_ifeval.bat [model_path] [options]
REM Input: IFEVAL\data\input_data.jsonl
REM ============================================================

setlocal
set "SCRIPT_DIR=%~dp0"
set "OPENVINO_ROOT=%SCRIPT_DIR%..\.."
cd /d "%SCRIPT_DIR%"

set "OV_GENAI_USE_MODELING_API=1"
if "%OV_GENAI_INFLIGHT_QUANT_MODE%"=="" set "OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym"
if "%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%"=="" set "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128"
if "%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%"=="" set "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym"

echo [INFO] OV_GENAI_USE_MODELING_API=1
echo [INFO] OV_GENAI_INFLIGHT_QUANT_MODE=%OV_GENAI_INFLIGHT_QUANT_MODE%
echo.

set "OV_PYTHON="
if exist "%OPENVINO_ROOT%\openvino\bin\intel64\Release\python" set "OV_PYTHON=%OPENVINO_ROOT%\openvino\bin\intel64\Release\python"
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%\openvino\build\lib\Release\python" set "OV_PYTHON=%OPENVINO_ROOT%\openvino\build\lib\Release\python"
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%\openvino\build\lib\python" set "OV_PYTHON=%OPENVINO_ROOT%\openvino\build\lib\python"
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%\openvino\build\bin\Release\python" set "OV_PYTHON=%OPENVINO_ROOT%\openvino\build\bin\Release\python"
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%\openvino\build\install\python" set "OV_PYTHON=%OPENVINO_ROOT%\openvino\build\install\python"
if not "%OV_PYTHON%"=="" set "PYTHONPATH=%OV_PYTHON%;%PYTHONPATH%"

set "PYTHONPATH=%OPENVINO_ROOT%\openvino.genai\build;%PYTHONPATH%"

set "OPENVINO_LIB_PATHS=%OPENVINO_ROOT%\openvino\bin\intel64\Release"
if exist "%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin" set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin"
if exist "%OPENVINO_ROOT%\openvino\build\install\runtime\bin\intel64\Release" set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%\openvino\build\install\runtime\bin\intel64\Release"

set "PATH=%OPENVINO_ROOT%\openvino.genai\build\openvino_genai;%PATH%"
set "PATH=%OPENVINO_ROOT%\openvino\bin\intel64\Release;%PATH%"
if exist "%OPENVINO_ROOT%\openvino.genai\build\bin\Release" set "PATH=%PATH%;%OPENVINO_ROOT%\openvino.genai\build\bin\Release"
if exist "%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin" set "PATH=%PATH%;%OPENVINO_ROOT%\openvino\temp\Windows_AMD64\tbb\bin"

echo Running: python run_ifeval_openvino.py %*
echo.
python run_ifeval_openvino.py %*

endlocal
