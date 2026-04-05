@echo off
REM Launch Qwen3.5 OpenAI-compatible API server
REM Usage: run.bat --model C:\data\models\Huggingface\Qwen3.5-4B [options]

set OV_GENAI_USE_MODELING_API=1

REM Activate the serving venv
call D:\chuansheng\src_code\explicit_modeling\serving_env\Scripts\activate.bat

REM Start server
cd /d %~dp0
python server.py %*
