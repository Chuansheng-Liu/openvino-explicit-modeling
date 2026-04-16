# DFlash Test Suite Design

**Date:** 2026-03-25
**Status:** Approved

## Overview

An independent pytest-based test suite (`test_dflash.py`) for validating the correctness and measuring the performance of the DFlash speculative decoding pipeline (`dflash.py`).

Models:
- Target (base): `C:\data\models\Huggingface\Qwen3.5-4B`
- Draft (DFlash): `C:\data\models\Huggingface\Qwen3.5-4B-DFlash`

## Files

```
scripts/
  dflash.py              (existing — unchanged)
  test_dflash.py         (new — test body)
  conftest.py            (new — pytest fixtures and CLI flags)
```

## Design

### conftest.py

Registers CLI flags via the `pytest_addoption(parser)` hook:
- `--full`: enables full prompt set
- `--model-dir`: override target model path (default: `C:\data\models\Huggingface\Qwen3.5-4B`)
- `--draft-dir`: override draft model path (default: `C:\data\models\Huggingface\Qwen3.5-4B-DFlash`)

Provides fixtures:
- `model_dir(request)` (session-scope): returns target model path from `--model-dir` CLI option
- `draft_dir(request)` (session-scope): returns draft model path from `--draft-dir` CLI option
- `prompts(request)`: returns default or full prompt list based on `--full`
- `session_setup(request)` (session-scope): reads model paths via `request.config.getoption()` directly (avoids pytest fixture scope conflict); validates that both model directories exist; calls `pytest.skip()` with a clear message if missing (e.g. "Model directory not found: C:\... — pass --model-dir to override")

**Note on model loading:** `run_baseline()` and `run_dflash()` in `dflash.py` construct a new `LLMPipeline` internally on each call. There is no pipeline injection API. Tests therefore pay the full model-load cost (~60s) per call. Time estimates below account for this.

### test_dflash.py

#### TestDFlashAccuracy

Two parametrized test classes, one per quantization combo. Each test is `@pytest.mark.parametrize` over the prompts fixture.

**`test_fp16_exact_match(prompt, model_dir, draft_dir)`**
1. Calls `run_baseline(model_dir, prompt, device="GPU", max_tokens=256, no_think=True, quant_mode=None)` → `baseline`
2. Calls `run_dflash(model_dir, draft_dir, prompt, device="GPU", max_tokens=256, no_think=True, target_quant=None, draft_quant=None)` → `result`
3. Asserts `baseline.output_text == result.output_text`
4. On failure: prints first 300 chars of each text for diagnosis

**`test_int4_exact_match(prompt, model_dir, draft_dir)`**
1. Calls `run_baseline(...)` with `quant_mode="INT4_ASYM"` → `baseline`
2. Calls `run_dflash(...)` with `target_quant="INT4_ASYM", draft_quant=None` → `result`
3. Asserts `baseline.output_text == result.output_text`

**Accuracy guarantee:** Both paths (baseline and DFlash) use the same target model weights and the same INT4 quantization parameters (`INT4_ASYM`, `group_size=128`). The DFlash draft model only proposes tokens; the target model verifies and accepts/rejects. Speculative decoding with greedy sampling is mathematically equivalent to greedy autoregressive decoding on the target model — identical quantization means identical outputs. Any mismatch is a bug.

**Estimated runtime:**
- Default (3 prompts × 2 test functions × 2 model loads each): 12 model loads × ~60s = ~12 min
- Full (8 prompts × 2 test functions × 2 model loads each): 32 model loads × ~60s = ~32 min

**Implementation note:** Each parametrized test function (fp16 or int4) for a given prompt calls `run_baseline()` once and `run_dflash()` once, then asserts. The parametrize decorator iterates over prompts only.

#### TestDFlashPerformance

**`test_perf_all_configs(model_dir, draft_dir)`**

Runs all 6 configurations sequentially on a single fixed prompt (`"Joy can read 8 pages in 20 minutes. How many hours to read 120 pages?"`). Collects `RunMetrics` for each config, tracking `baseline_fp16` for the summary table.

After all runs:
1. Calls `build_summary_lines(results, baseline_fp16)` and prints to stdout (run pytest with `-s` to see output)
2. Calls `build_text_diff_lines(results)` and prints
3. Constructs a `types.SimpleNamespace` with required fields (`model_dir`, `draft_model_dir`, `prompt_file=None`, `prompt`, `device`, `max_tokens`, `no_think`, `precision`) to satisfy `save_run_report(args, results, baseline_fp16)` signature
4. Saves report to `dflash优化报告/dflash_benchmark_<timestamp>.txt` (filename is hardcoded in `save_run_report()`; the `command=` field in the report will show the pytest invocation rather than a dflash.py command line — this is expected behavior)
5. Always passes (no performance thresholds)

**Estimated runtime:** ~6 model loads × ~60s = ~6 min

### Prompt Sets

**Default (fast regression, ~3 prompts):**
```python
PROMPTS_DEFAULT = [
    "What is 2 + 2?",
    "Joy can read 8 pages in 20 minutes. How many hours to read 120 pages?",
    "Write a Python function to check if a number is prime.",
]
```

**Full (`--full`, ~8 prompts):**
```python
PROMPTS_FULL = PROMPTS_DEFAULT + [
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "用一句话介绍北京。",
    "Write a Python function that reverses a linked list.",
    "Explain the history of the Roman Empire in detail.",
    "Alice is taller than Bob. Bob is taller than Carol. Who is shortest?",
]
```

All prompts use `no_think=True` and `max_tokens=256`.

Note: The Chinese prompt (`"用一句话介绍北京。"`) uses `no_think=True` which applies the explicit no-think template via `build_prompt()`. This is consistent with all other prompts.

## Usage

```bash
# Quick regression (default prompts), show streaming output
pytest scripts/test_dflash.py -v -s

# Full test set
pytest scripts/test_dflash.py -v -s --full

# Override model paths (quote paths with spaces)
pytest scripts/test_dflash.py --model-dir "C:\data\models\Qwen3.5-4B" --draft-dir "C:\data\models\Qwen3.5-4B-DFlash"

# Accuracy only
pytest scripts/test_dflash.py -v -s -k "accuracy"

# Performance only
pytest scripts/test_dflash.py -v -s -k "perf"
```

Note: `-s` is recommended to see streaming output and the performance summary table. Without `-s`, pytest captures stdout and the table will not be visible during the run.

## Key Decisions

- **No pipeline injection**: `dflash.py` is unchanged; each `run_baseline()`/`run_dflash()` call loads the model fresh. This is the honest cost.
- **`types.SimpleNamespace` for report saving**: constructs a synthetic args object to satisfy `save_run_report()` without modifying `dflash.py`.
- **`no_think=True`**: disables thinking mode for deterministic, shorter, faster outputs.
- **`max_new_tokens=256`**: balances coverage with test speed.
- **No performance thresholds**: metrics are recorded and printed for human review; tests never fail due to performance.
- **Reuse existing helpers**: `run_baseline()`, `run_dflash()`, `build_summary_lines()`, `build_text_diff_lines()`, `save_run_report()` imported directly from `dflash`.
