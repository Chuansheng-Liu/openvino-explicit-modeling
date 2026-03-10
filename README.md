# OpenVINO Explicit Modeling API

This document explains how to clone, build, and run the OpenVINO Modeling POC.

## 1. Clone The Repositories

```bash
cd my_workspace
https://github.com/liangali/openvino-explicit-modeling
cd openvino-explicit-modeling
repo.bat
```

After repo.bat done, it will create below folder structure

```
my_workspace/
|-- openvino/
|-- openvino.genai/
`-- openvino-explicit-modeling/
```

## 2. Build 

It builds both `openvino` and `openvino.genai`, and creates the `build` directories if needed
```bash
cd openvino-explicit-modeling
build.bat
```

## 3. Run Tests

### Download Huggingface safetensors model

```bash
# notes: the HF model files must be put in "Huggingface" folder
cd D:\data\models\Huggingface
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/Qwen/Qwen3.5-35B-A3B
```

### Setup tokenizer/detokenizer

```bash
cd my_workspace
python -m venv .venv
.venv\Scripts\activate
pip install openvino-tokenizers
pip install transformers
```

### Option #1: Use auto_tests.py

```bash
# make sure run tests under the .venv for tokenizer/detokenizer conversion
.venv\Scripts\activate

# Run all tests from the openvino-modeling-api root
cd d:\openvino-modeling-api
python openvino-explicit-modeling\scripts\auto_tests.py --help

# List Available Tests
python openvino-explicit-modeling\scripts\auto_tests.py --list

# Specify the model root directory (default: D:\data\models)
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models

# Run specific test indices: 0, 1, 2
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 0,1,2

# Run an index range (1~5 means 1,2,3,4,5)
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 1~5

# Combine ranges and single indices
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests 1~5,7,8~10

# Run all tests
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models --tests all

```

Test Output
- A Markdown report folder is generated in the my_workspace, which incluldes markdown report file.
- The report includes TTFT, throughput, duration, and related metrics.
- Failed cases are listed in `stderr`.

### Option #2: Use command line

use raw command line
```bash
# make sure run tests under the .venv for tokenizer/detokenizer conversion
.venv\Scripts\activate

set OV_GENAI_SAVE_OV_MODEL=1
set OV_GENAI_USE_MODELING_API=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym

set my_workspace=D:\data\code\Openvino_new_arch_2026
set PATH=%my_workspace%\openvino\temp\Windows_AMD64\tbb\bin;%my_workspace%\openvino\bin\intel64\Release;%my_workspace%\openvino.genai\build\openvino_genai;%PATH%

cd %my_workspace%\openvino.genai\build\bin

modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B --cache-model --mode text --prompt "write opencl gemm kernel and host code" --output-tokens 300
greedy_causal_lm.exe C:\data\models\Huggingface\Qwen3.5-35B-A3B "write opencl gemm kernel and host code" GPU 1 3 300 int4_asym 128 int4_asym
```

use run.bat
```bash
# make sure run tests under the .venv for tokenizer/detokenizer conversion
.venv\Scripts\activate

cd openvino-explicit-modeling
run.bat
greedy_causal_lm.exe D:\data\models\Huggingface\Qwen3.5-35B-A3B "write opencl gemm kernel and host code" GPU 1 3 300 int4_asym 128 int4_asym
```