@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "PYTHON=python"
set "CSV_PATH=%CD%\simple.csv"
set "OUT_DIR=%CD%\results"

set "MODEL_QWEN34=D:\Data\models\Huggingface\Qwen3-4B"
set "MODEL_QWEN35=D:\Data\models\Huggingface\Qwen3.5-35B-A3B"

set "PRINT_EXE_COMMAND=1"
if "%PRINT_EXE_COMMAND%"=="1" (
    set "PRINT_FLAG=--print_exe_command"
) else (
    set "PRINT_FLAG="
)

set "QWEN34_FP=%OUT_DIR%\qwen3_4b_fp32.csv"
set "QWEN34_INT4=%OUT_DIR%\qwen3_4b_int4.csv"
set "EVAL_QWEN34=%OUT_DIR%\eval_qwen3_4b_int4_vs_fp32.csv"

set "QWEN35_FP=%OUT_DIR%\qwen3_5_35b_fp32.csv"
set "QWEN35_INT4=%OUT_DIR%\qwen3_5_35b_int4.csv"
set "EVAL_QWEN35=%OUT_DIR%\eval_qwen3_5_35b_int4_vs_fp32.csv"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
if not exist "%CSV_PATH%" (
    echo [ERROR] CSV file not found: "%CSV_PATH%"
    goto :fail
)

echo ============================================================
echo Step 1/6: Generate Qwen3-4B FP32
"%PYTHON%" ".\modeling_api_generate.py" ^
  --modeltype exe_causal ^
  --model "%MODEL_QWEN34%" ^
  --csv "%CSV_PATH%" ^
  --save_generations_path "%QWEN34_FP%" ^
  %PRINT_FLAG%
if errorlevel 1 goto :fail

echo ============================================================
echo Step 2/6: Generate Qwen3-4B INT4 (channel-wise)
"%PYTHON%" ".\modeling_api_generate.py" ^
  --modeltype exe_causal ^
  --model "%MODEL_QWEN34%" ^
  --exe_quant_args int4_asym -1 int8_asym ^
  --csv "%CSV_PATH%" ^
  --save_generations_path "%QWEN34_INT4%" ^
  %PRINT_FLAG%
if errorlevel 1 goto :fail

echo ============================================================
echo Step 3/6: Evaluate Qwen3-4B INT4 vs FP32
"%PYTHON%" ".\evaluate.py" ^
  --gold "%QWEN34_FP%" ^
  --prediction "%QWEN34_INT4%" ^
  --metrics similarity divergency ^
  --tokenizer_path "%MODEL_QWEN34%" ^
  --save_evaluation_path "%EVAL_QWEN34%"
if errorlevel 1 goto :fail

echo ============================================================
echo Step 4/6: Generate Qwen3.5-35B-A3B FP32 (modeling_qwen3_5)
"%PYTHON%" ".\modeling_api_generate.py" ^
  --modeltype modeling_qwen3_5 ^
  --model "%MODEL_QWEN35%" ^
  --qwen35_mode text ^
  --qwen35_output_tokens 300 ^
  --csv "%CSV_PATH%" ^
  --save_generations_path "%QWEN35_FP%" ^
  %PRINT_FLAG%
if errorlevel 1 goto :fail

echo ============================================================
echo Step 5/6: Generate Qwen3.5-35B-A3B INT4 (env quant)
"%PYTHON%" ".\modeling_api_generate.py" ^
  --modeltype modeling_qwen3_5 ^
  --model "%MODEL_QWEN35%" ^
  --qwen35_mode text ^
  --qwen35_output_tokens 300 ^
  --qwen35_quant_env int4_asym 128 int4_asym ^
  --csv "%CSV_PATH%" ^
  --save_generations_path "%QWEN35_INT4%" ^
  %PRINT_FLAG%
if errorlevel 1 goto :fail

echo ============================================================
echo Step 6/6: Evaluate Qwen3.5-35B-A3B INT4 vs FP32
"%PYTHON%" ".\evaluate.py" ^
  --gold "%QWEN35_FP%" ^
  --prediction "%QWEN35_INT4%" ^
  --metrics similarity divergency ^
  --tokenizer_path "%MODEL_QWEN35%" ^
  --save_evaluation_path "%EVAL_QWEN35%"
if errorlevel 1 goto :fail

echo.
echo ================= Final Result Files =================
echo Qwen3-4B eval:   "%EVAL_QWEN34%"
echo Qwen3.5 eval:    "%EVAL_QWEN35%"
echo.

echo ================= Qwen3-4B Mean Metrics =============
"%PYTHON%" -c "import pandas as pd; df=pd.read_csv(r'%EVAL_QWEN34%'); cols=[c for c in ['similarity','FDT','SDT','FDT norm','SDT norm'] if c in df.columns]; print(df[cols].mean().to_frame('mean').T.to_string(index=False))"
if errorlevel 1 goto :fail

echo.
echo ================= Qwen3.5 Mean Metrics ==============
"%PYTHON%" -c "import pandas as pd; df=pd.read_csv(r'%EVAL_QWEN35%'); cols=[c for c in ['similarity','FDT','SDT','FDT norm','SDT norm'] if c in df.columns]; print(df[cols].mean().to_frame('mean').T.to_string(index=False))"
if errorlevel 1 goto :fail

echo.
echo Benchmark completed successfully.
pause
exit /b 0

:fail
echo.
echo [ERROR] Benchmark failed. Exit code: %errorlevel%
pause
exit /b %errorlevel%
