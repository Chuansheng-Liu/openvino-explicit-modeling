# Scripts

This directory contains the main helper scripts for the OpenVINO Explicit
Modeling workspace.

| Script | Purpose |
|--------|---------|
| `auto_tests.py` | Functional test runner for executable samples |
| `ifeval.py` | IFEval benchmark (instruction following, English) |
| `ceval.py` | C-Eval benchmark (Chinese multiple-choice evaluation) |
| `mmlu_redux.py` | MMLU-Redux benchmark (English multiple-choice evaluation) |
| `wwb.py` | Weight-wise benchmark helper |

This README focuses on the scripts used most often:

- `auto_tests.py`
- `ifeval.py`
- `ceval.py`
- `mmlu_redux.py`

---

## 1. Prerequisites (Common Setup)

Before using the scripts in this directory, complete the setup in the
**[root README.md](../README.md)**:

1. Clone the workspace with `repo.bat` or `repo.sh`
2. Build `openvino` and `openvino.genai` with `build.bat` or `build.sh`
3. Create `.venv` and install `openvino-tokenizers` and `transformers`

Expected workspace layout:

```text
my_workspace\                               e.g. D:\data\code\Openvino_new_arch_2026
|-- openvino\
|-- openvino.genai\
|   `-- build\bin\
|       `-- modeling_qwen3_5.exe
|-- openvino-explicit-modeling\
|   `-- scripts\
`-- .venv\
```

Quick verification:

```bat
dir my_workspace\openvino.genai\build\bin\modeling_qwen3_5.exe
```

Linux:

```bash
ls my_workspace/openvino.genai/build/bin/Release/modeling_qwen3_5*
```

---

## 2. auto_tests.py — Functional Test Runner

`auto_tests.py` runs a fixed set of functional checks for text-generation,
vision-language, and image-generation samples, then writes a markdown report.

### What it needs

- a completed OpenVINO + GenAI build
- models under `D:\data\models` by default on Windows
- an explicit `--models-root` on Linux, because the current default is Windows-specific
- `test.jpg` in this `scripts\` directory for VL tests

### Common commands

The examples below assume you run them from `my_workspace`.

```bat
python openvino-explicit-modeling\scripts\auto_tests.py
python openvino-explicit-modeling\scripts\auto_tests.py --list
python openvino-explicit-modeling\scripts\auto_tests.py --tests 0,1,2
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5,7,8~10
python openvino-explicit-modeling\scripts\auto_tests.py --tests all
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models\custom
```

Linux:

```bash
python openvino-explicit-modeling/scripts/auto_tests.py --models-root /data/models
python openvino-explicit-modeling/scripts/auto_tests.py --list
python openvino-explicit-modeling/scripts/auto_tests.py --tests 0,1,2 --models-root /data/models
python openvino-explicit-modeling/scripts/auto_tests.py --tests all --models-root /data/models
```

### CLI arguments

| Argument | Description |
|----------|-------------|
| `--root PATH` | Workspace root containing `openvino`, `openvino.genai`, and `openvino-explicit-modeling` |
| `--models-root PATH` | Model root directory (default: `D:\data\models`) |
| `--build-type` | Choose `Release` or `RelWithDebInfo`; if omitted, try `Release` first |
| `--list` | Print the numbered test list and exit |
| `--tests INDEX...` | Run only selected indices, ranges, or `all` |

### Test groups

| Range | Tests |
|-------|-------|
| 0–6 | Text-generation samples (Qwen3 0.6B/4B, SmolLM, Youtu, GGUF variants) |
| 7–9 | Qwen3-VL (2B/4B/8B) using `test.jpg` |
| 10 | Z-Image-Turbo (writes `cat2.bmp`) |

### Output

The report is written under:

```text
my_workspace\reports\test_report_<timestamp>.md
```

It includes TTFT, throughput, duration, and related metrics. Failed cases are
reported in `stderr`.

### Model downloads for `auto_tests.py`

```bat
cd D:\data\models
git clone https://huggingface.co/Qwen/Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/HuggingFaceTB/SmolLM3-3B
git clone https://huggingface.co/tencent/Youtu-LLM-2B
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
git clone https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
```

---

## 3. LLM Accuracy Benchmarks — Overview

Three benchmark scripts are provided for LLM accuracy evaluation:

| Benchmark | Script | Language | Questions | Type | Scoring |
|-----------|--------|:--------:|:---------:|------|---------|
| **IFEval** | `ifeval.py` | English | 541 | Open-ended generation | Deterministic format-constraint checking |
| **C-Eval** | `ceval.py` | Chinese | ~1,346 | 4-choice MCQ | Letter matching (`A/B/C/D`) |
| **MMLU-Redux** | `mmlu_redux.py` | English | ~4,472 (OK) / ~4,800 (all) | 4-choice MCQ | Letter matching (`A/B/C/D`) |

### What each benchmark measures

#### IFEval

IFEval checks whether the model follows verifiable output constraints such as:
"write at least 300 words", "do not use commas", or "respond in all caps".

- 25 instruction types
- 541 prompts
- 834 total constraints
- fully deterministic evaluation, no LLM judge
- 4 metrics: prompt-level strict/loose and instruction-level strict/loose

#### C-Eval

C-Eval measures Chinese multiple-choice accuracy across 52 subjects.

| Category | # Subjects | Examples |
|----------|:----------:|---------|
| STEM | 20 | Advanced Math, College Physics, Computer Network |
| Social Science | 13 | Law, Economics, Education, Geography |
| Humanities | 11 | Chinese Literature, History, Marxism |
| Other | 8 | Medicine, Accounting, Civil Servant |

- ~1,346 validation questions
- 5-shot prompting by default
- supports `--n-shot` to change the few-shot count
- reports macro average, micro average, and category breakdown

#### MMLU-Redux

MMLU-Redux re-annotates the original MMLU test set and separates quality-checked
questions from the full set.

- 57 subjects
- ~4,800 total questions
- ~4,472 OK-annotated questions after filtering
- 5-shot prompting by default
- `--filter-ok` uses only quality-verified questions
- `--no-filter-ok` includes all questions

### Benchmark environment setup

#### Python dependencies

Only IFEval needs extra Python packages beyond the common setup.

| Dependency | Required by | Install |
|------------|:-----------:|---------|
| `immutabledict`, `langdetect`, `nltk` | IFEval | `pip install -r requirements.txt` |
| NLTK data: `punkt`, `punkt_tab` | IFEval | `python -m nltk.downloader punkt punkt_tab` |
| *(none)* | C-Eval, MMLU-Redux | Standard library only |

```bat
cd my_workspace
.venv\Scripts\activate

pip install -r openvino-explicit-modeling\scripts\requirements.txt
python -m nltk.downloader punkt punkt_tab
```

Linux:

```bash
cd my_workspace
source .venv/bin/activate

python -m pip install -r openvino-explicit-modeling/scripts/requirements.txt
python -m nltk.downloader punkt punkt_tab
```

#### Download Qwen3.5 models

The benchmark scripts default to:

```text
D:\data\models\Huggingface
```

On Linux, pass the model root explicitly if your models are stored elsewhere,
for example `/data/models/Huggingface`.

Supported model indices:

| Index | Model | Parameters |
|:-----:|-------|:----------:|
| 1 | Qwen3.5-0.8B | 0.8 B |
| 2 | Qwen3.5-2B | 2 B |
| 3 | Qwen3.5-4B | 4 B |
| 4 | Qwen3.5-9B | 9 B |
| 5 | Qwen3.5-35B-A3B | 35 B (MoE, 3 B active) |

```bat
cd D:\data\models\Huggingface
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
git clone https://huggingface.co/Qwen/Qwen3.5-2B
git clone https://huggingface.co/Qwen/Qwen3.5-4B
git clone https://huggingface.co/Qwen/Qwen3.5-9B
git clone https://huggingface.co/Qwen/Qwen3.5-35B-A3B
```

If your models live elsewhere, pass `--model-root <path>`.

#### Download benchmark datasets

| Benchmark | Dataset location | Download command |
|-----------|------------------|------------------|
| **IFEval** | Bundled in `ifeval_lib\input_data.jsonl` | Nothing to download |
| **C-Eval** | `scripts\ceval_data\` | `python openvino-explicit-modeling\scripts\ceval.py --download` |
| **MMLU-Redux** | `scripts\mmlu_redux_data\` and `scripts\mmlu_dev_data\` | `python openvino-explicit-modeling\scripts\mmlu_redux.py --download` |

```bat
cd my_workspace
python openvino-explicit-modeling\scripts\ceval.py --download
python openvino-explicit-modeling\scripts\mmlu_redux.py --download
```

Both scripts can also auto-download missing data on first run.

#### Smoke test

Run a short check before launching a full benchmark:

```bat
cd my_workspace
.venv\Scripts\activate

python openvino-explicit-modeling\scripts\ifeval.py --models 1 --limit 5
python openvino-explicit-modeling\scripts\ceval.py --models 1 --limit 5
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 1 --limit 5
```

### Running benchmarks

#### Shared CLI reference

Selection arguments:

| Argument | Format | Default | Description |
|----------|--------|:-------:|-------------|
| `--models` | `1,3,4` or `1~5` | `2` in IFEval, `3` in C-Eval/MMLU-Redux | Model index selectors |
| `--quant-list` | `1,2` or `all` | `1` | Quantization preset selectors |
| `--think` | `0,1` or `all` | `0` | Think mode selectors |

Sampling arguments:

| Argument | Default | Notes |
|----------|:-------:|-------|
| `--temperature` | `1.0` | `0` = greedy argmax |
| `--top-p` | `0.95` | Nucleus sampling |
| `--top-k` | `20` | Top-K filtering |
| `--presence-penalty` | `1.5` (IFEval), `0.0` (MCQ) | Encourages new tokens |
| `--repetition-penalty` | `1.0` | Multiplicative repeat penalty |
| `--frequency-penalty` | `0.0` | Additive frequency penalty |
| `--rng-seed` | `0` | `0` = random |

Control arguments:

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--max-tokens` | `2048` in IFEval, `512` in MCQ benchmarks | Max output tokens per question |
| `--limit N` | all questions | Run only the first `N` questions |
| `--resume <path>` | none | Resume from `answers.jsonl` or `responses.jsonl` |
| `--output-dir <path>` | auto | Custom output directory for a single run |

MCQ-only arguments:

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--n-shot` | `5` | Number of few-shot examples |
| `--download` | off | Download dataset and exit |
| `--filter-ok` / `--no-filter-ok` | `--filter-ok` | MMLU-Redux only |

Quantization presets:

| Index | Preset | Description |
|:-----:|--------|-------------|
| 1 | `[int4_asym, 128, int4_asym]` | Default, fastest |
| 2 | `[int4_asym, 128, int8_asym]` | Better accuracy |
| 3 | `[none, none, none]` | No quantization, best accuracy |

#### IFEval examples

```bat
cd my_workspace

python openvino-explicit-modeling\scripts\ifeval.py --models 3
python openvino-explicit-modeling\scripts\ifeval.py --models 1~5
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --quant-list 1,2
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --think 0,1
python openvino-explicit-modeling\scripts\ifeval.py --models 1~4 --quant-list 1,2 --think 0,1
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --limit 10
```

#### C-Eval examples

```bat
cd my_workspace

python openvino-explicit-modeling\scripts\ceval.py --models 3
python openvino-explicit-modeling\scripts\ceval.py --models 1~4
python openvino-explicit-modeling\scripts\ceval.py --models 3 --n-shot 0
python openvino-explicit-modeling\scripts\ceval.py --models 3 --limit 10
python openvino-explicit-modeling\scripts\ceval.py --models 3 --temperature 0
```

#### MMLU-Redux examples

```bat
cd my_workspace

python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --no-filter-ok
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 1~4
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --n-shot 0
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --limit 10
```

### Output structure

Each benchmark creates a timestamped batch directory under the workspace root:

```text
my_workspace\
|-- results_ifeval\
|   `-- 20250601_143022\
|       |-- Qwen3.5-4B_int4asym_think0_t1.0_n541\
|       |   |-- responses.jsonl
|       |   |-- results.json
|       |   `-- summary.txt
|       `-- batch_summary.md
|-- results_ceval\
|   `-- 20250601_150000\
|       |-- Qwen3.5-4B_int4asym_think0_t1.0_n1346\
|       |   |-- answers.jsonl
|       |   |-- results.json
|       |   `-- summary.txt
|       `-- batch_summary.md
`-- results_mmlu_redux\
    `-- 20250601_160000\
        |-- Qwen3.5-4B_int4asym_think0_t1.0_n4472\
        |   |-- answers.jsonl
        |   |-- results.json
        |   `-- summary.txt
        `-- batch_summary.md
```

Output files:

| File | Content |
|------|---------|
| `responses.jsonl` / `answers.jsonl` | One JSON line per question, also used by `--resume` |
| `results.json` | Full metrics, timing, and run config |
| `summary.txt` | Human-readable summary |
| `batch_summary.md` | Comparison table for the full batch |

### Resuming interrupted runs

All three benchmark scripts support resume. Re-run the same command and the
script will continue from the next unanswered question.

You can also resume explicitly:

```bat
python openvino-explicit-modeling\scripts\ceval.py --models 3 --resume path\to\answers.jsonl
```

### Expected run times

Approximate times on Intel Arc A770 with quantization preset `--quant-list 1`:

| Benchmark | Questions | Qwen3.5-0.8B | Qwen3.5-4B | Qwen3.5-9B |
|-----------|:---------:|:------------:|:----------:|:----------:|
| IFEval | 541 | ~30 min | ~60 min | ~90 min |
| C-Eval | ~1,346 | ~20 min | ~45 min | ~70 min |
| MMLU-Redux | ~4,472 | ~60 min | ~140 min | ~220 min |

C-Eval and MMLU-Redux are faster per question because MCQ output is short
(`--max-tokens 512`) while IFEval generates full text (`--max-tokens 2048`).
