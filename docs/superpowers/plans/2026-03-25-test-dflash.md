# DFlash Test Suite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `conftest.py` and `test_dflash.py` in `scripts/` to test DFlash speculative decoding for accuracy (exact-match greedy equivalence) and performance (record-only table).

**Architecture:** Two files alongside `dflash.py` in `scripts/`. `conftest.py` registers pytest CLI flags (`--full`, `--model-dir`, `--draft-dir`) and provides session-scoped fixtures + prompt parametrization. `test_dflash.py` imports `run_baseline`/`run_dflash` directly from `dflash.py` and contains two test classes: `TestDFlashAccuracy` (parametrized exact-match assertions) and `TestDFlashPerformance` (all-6-configs table, always passes).

**Tech Stack:** Python 3, pytest, `dflash.py` (existing), `openvino_genai` (existing runtime)

**Spec:** `docs/superpowers/specs/2026-03-25-test-dflash-design.md`

---

## Chunk 1: conftest.py

### Task 1: Create `conftest.py` with CLI flags, fixtures, and prompt parametrization

**Files:**
- Create: `scripts/conftest.py`

`conftest.py` must be in the same directory as `test_dflash.py` so pytest picks it up automatically. It does three things:
1. Registers `--full`, `--model-dir`, `--draft-dir` CLI options
2. Provides session-scoped `model_dir` / `draft_dir` fixtures and a `session_setup` autouse fixture that validates paths early
3. Implements `pytest_generate_tests` to parametrize any test with a `prompt` parameter

- [ ] **Step 1.1: Create `scripts/conftest.py`**

```python
# scripts/conftest.py
import sys
import pytest
from pathlib import Path

# Make scripts/ importable so `import dflash` works regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL_DIR = r"C:\data\models\Huggingface\Qwen3.5-4B"
DEFAULT_DRAFT_DIR = r"C:\data\models\Huggingface\Qwen3.5-4B-DFlash"

PROMPTS_DEFAULT = [
    "What is 2 + 2?",
    "Joy can read 8 pages in 20 minutes. How many hours to read 120 pages?",
    "Write a Python function to check if a number is prime.",
]

PROMPTS_FULL = PROMPTS_DEFAULT + [
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    "用一句话介绍北京。",
    "Write a Python function that reverses a linked list.",
    "Explain the history of the Roman Empire in detail.",
    "Alice is taller than Bob. Bob is taller than Carol. Who is shortest?",
]


def pytest_addoption(parser):
    parser.addoption(
        "--full", action="store_true", default=False,
        help="Run full prompt set (8 prompts) instead of default (3 prompts)"
    )
    parser.addoption(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help=f"Target model directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.addoption(
        "--draft-dir", default=DEFAULT_DRAFT_DIR,
        help=f"DFlash draft model directory (default: {DEFAULT_DRAFT_DIR})"
    )


@pytest.fixture(scope="session")
def model_dir(request):
    return request.config.getoption("--model-dir")


@pytest.fixture(scope="session")
def draft_dir(request):
    return request.config.getoption("--draft-dir")


@pytest.fixture(scope="session", autouse=True)
def session_setup(request):
    """Validate model directories exist before any test runs."""
    model = Path(request.config.getoption("--model-dir"))
    draft = Path(request.config.getoption("--draft-dir"))
    if not model.is_dir():
        pytest.skip(
            f"Target model directory not found: {model}\n"
            f"Pass --model-dir to override. Example:\n"
            f"  pytest scripts/test_dflash.py --model-dir \"C:\\path\\to\\Qwen3.5-4B\""
        )
    if not draft.is_dir():
        pytest.skip(
            f"Draft model directory not found: {draft}\n"
            f"Pass --draft-dir to override. Example:\n"
            f"  pytest scripts/test_dflash.py --draft-dir \"C:\\path\\to\\Qwen3.5-4B-DFlash\""
        )


def pytest_generate_tests(metafunc):
    """Parametrize `prompt` fixture for any test that declares it."""
    if "prompt" in metafunc.fixturenames:
        full = metafunc.config.getoption("--full", default=False)
        prompts = PROMPTS_FULL if full else PROMPTS_DEFAULT
        metafunc.parametrize("prompt", prompts)
```

- [ ] **Step 1.2: Verify conftest.py is collected without errors (no model needed)**

Run from the repo root:
```bash
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
python -m pytest scripts/conftest.py --collect-only 2>&1 | head -20
```

Expected: no import errors (even if dflash bootstrap fails, conftest.py itself should import cleanly because `import dflash` is not at module level in conftest).

---

## Chunk 2: test_dflash.py — accuracy tests

### Task 2: Create `test_dflash.py` with accuracy skeleton + FP16 exact-match test

**Files:**
- Create: `scripts/test_dflash.py`

- [ ] **Step 2.1: Create `scripts/test_dflash.py` with imports and `TestDFlashAccuracy.test_fp16_exact_match`**

```python
# scripts/test_dflash.py
"""
DFlash pytest test suite.

Accuracy tests: greedy speculative decoding must produce output identical to
greedy autoregressive decoding on the same target model. Any mismatch is a bug.

Performance tests: run all 6 configs, print summary table, save report. Always passes.

Usage:
    # Quick regression (3 prompts, ~12 min on Arc 140T)
    pytest scripts/test_dflash.py -v -s

    # Full test set (8 prompts, ~32 min)
    pytest scripts/test_dflash.py -v -s --full

    # Accuracy only
    pytest scripts/test_dflash.py -v -s -k "accuracy"

    # Performance only
    pytest scripts/test_dflash.py -v -s -k "perf"
"""

import sys
import types
import pytest
from pathlib import Path

# conftest.py already inserts scripts/ into sys.path, but guard here too
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import dflash
from dflash import (
    run_baseline,
    run_dflash,
    build_summary_lines,
    build_text_diff_lines,
    save_run_report,
)

DEVICE = "GPU"
MAX_TOKENS = 256


class TestDFlashAccuracy:
    """
    Greedy speculative decoding equivalence tests.

    DFlash uses the same target model weights + quantization as the baseline.
    The draft model only proposes; the target verifies. Under greedy sampling,
    speculative decoding is provably equivalent to autoregressive decoding —
    outputs must be bit-for-bit identical. Any difference is a bug.

    Each test is parametrized over the prompt set (see conftest.py).
    Run with -s to see streaming output during generation.
    """

    def test_fp16_exact_match(self, prompt, model_dir, draft_dir):
        """DFlash FP16/FP16 must produce identical output to Baseline FP16."""
        baseline = run_baseline(
            model_dir, prompt, DEVICE, MAX_TOKENS,
            no_think=True, quant_mode=None,
        )
        result = run_dflash(
            model_dir, draft_dir, prompt, DEVICE, MAX_TOKENS,
            no_think=True, target_quant=None, draft_quant=None,
        )
        assert baseline.output_text == result.output_text, (
            f"\nPrompt: {prompt!r}\n"
            f"\nBaseline FP16 (first 300 chars):\n{baseline.output_text[:300]}\n"
            f"\nDFlash FP16/FP16 (first 300 chars):\n{result.output_text[:300]}\n"
            f"\nFull lengths: baseline={len(baseline.output_text)}, "
            f"dflash={len(result.output_text)}"
        )
```

- [ ] **Step 2.2: Verify collection (no model load yet)**

```bash
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
python -m pytest scripts/test_dflash.py::TestDFlashAccuracy::test_fp16_exact_match --collect-only
```

Expected output includes 3 collected items (one per default prompt):
```
<Function test_fp16_exact_match[What is 2 + 2?]>
<Function test_fp16_exact_match[Joy can read...]>
<Function test_fp16_exact_match[Write a Python...]>
```

- [ ] **Step 2.3: Run FP16 accuracy test against real models**

```bash
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
python -m pytest scripts/test_dflash.py::TestDFlashAccuracy::test_fp16_exact_match -v -s
```

Expected: all 3 tests PASS. If any FAIL, the assertion message shows both outputs — file a bug.

### Task 3: Add INT4 exact-match test

**Files:**
- Modify: `scripts/test_dflash.py` (add method to `TestDFlashAccuracy`)

- [ ] **Step 3.1: Append `test_int4_exact_match` to `TestDFlashAccuracy`**

Add this method inside `class TestDFlashAccuracy:` (after `test_fp16_exact_match`):

```python
    def test_int4_exact_match(self, prompt, model_dir, draft_dir):
        """DFlash INT4/FP16 must produce identical output to Baseline INT4.

        Both paths apply INT4_ASYM quantization (group_size=128) to the same
        target model weights. The draft model is FP16. Greedy equivalence holds.
        """
        baseline = run_baseline(
            model_dir, prompt, DEVICE, MAX_TOKENS,
            no_think=True, quant_mode="INT4_ASYM",
        )
        result = run_dflash(
            model_dir, draft_dir, prompt, DEVICE, MAX_TOKENS,
            no_think=True, target_quant="INT4_ASYM", draft_quant=None,
        )
        assert baseline.output_text == result.output_text, (
            f"\nPrompt: {prompt!r}\n"
            f"\nBaseline INT4 (first 300 chars):\n{baseline.output_text[:300]}\n"
            f"\nDFlash INT4/FP16 (first 300 chars):\n{result.output_text[:300]}\n"
            f"\nFull lengths: baseline={len(baseline.output_text)}, "
            f"dflash={len(result.output_text)}"
        )
```

- [ ] **Step 3.2: Verify INT4 test collects**

```bash
python -m pytest scripts/test_dflash.py::TestDFlashAccuracy::test_int4_exact_match --collect-only
```

Expected: 3 items collected.

- [ ] **Step 3.3: Run INT4 accuracy test**

```bash
python -m pytest scripts/test_dflash.py::TestDFlashAccuracy::test_int4_exact_match -v -s
```

Expected: all 3 PASS.

- [ ] **Step 3.4: Commit accuracy tests**

```bash
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
git add scripts/conftest.py scripts/test_dflash.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(tests): add DFlash accuracy tests (greedy exact-match for FP16 and INT4)"
```

---

## Chunk 3: test_dflash.py — performance test

### Task 4: Add `TestDFlashPerformance.test_perf_all_configs`

**Files:**
- Modify: `scripts/test_dflash.py` (add `TestDFlashPerformance` class)

- [ ] **Step 4.1: Append `TestDFlashPerformance` class to `test_dflash.py`**

Add after the `TestDFlashAccuracy` class:

```python
class TestDFlashPerformance:
    """
    Performance measurement across all 6 DFlash configurations.

    Runs a single fixed prompt through:
      1. Baseline FP16
      2. Baseline INT4
      3. DFlash FP16/FP16
      4. DFlash INT4/FP16
      5. DFlash FP16/INT4
      6. DFlash INT4/INT4

    Prints a summary table and saves a report. Never fails on performance —
    results are for human review only.

    Run with -s to see the summary table printed to stdout.
    Estimated runtime: ~6 min (6 model loads × ~60s each).
    """

    PERF_PROMPT = "Joy can read 8 pages in 20 minutes. How many hours to read 120 pages?"

    def test_perf_all_configs(self, model_dir, draft_dir):
        """Run all 6 configs, print table, save report. Always passes."""
        results = []
        baseline_fp16 = None

        # 1. Baseline FP16
        m = run_baseline(
            model_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, quant_mode=None,
        )
        results.append(m)
        baseline_fp16 = m

        # 2. Baseline INT4
        m = run_baseline(
            model_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, quant_mode="INT4_ASYM",
        )
        m.label = "Baseline INT4"
        results.append(m)

        # 3. DFlash FP16/FP16
        m = run_dflash(
            model_dir, draft_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, target_quant=None, draft_quant=None,
        )
        results.append(m)

        # 4. DFlash INT4/FP16
        m = run_dflash(
            model_dir, draft_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, target_quant="INT4_ASYM", draft_quant=None,
        )
        results.append(m)

        # 5. DFlash FP16/INT4
        m = run_dflash(
            model_dir, draft_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, target_quant=None, draft_quant="INT4_ASYM",
        )
        results.append(m)

        # 6. DFlash INT4/INT4
        m = run_dflash(
            model_dir, draft_dir, self.PERF_PROMPT, DEVICE, MAX_TOKENS,
            no_think=True, target_quant="INT4_ASYM", draft_quant="INT4_ASYM",
        )
        results.append(m)

        # Print summary table (visible with pytest -s)
        print()
        for line in build_summary_lines(results, baseline_fp16):
            print(line)
        for line in build_text_diff_lines(results):
            print(line)

        # Save report using save_run_report() — construct a SimpleNamespace
        # to satisfy its argparse.Namespace signature without modifying dflash.py.
        # Note: the saved file is named dflash_benchmark_<timestamp>.txt
        # (filename is hardcoded in save_run_report). The command= field in the
        # report will show the pytest invocation — expected behavior.
        args = types.SimpleNamespace(
            model_dir=model_dir,
            draft_model_dir=draft_dir,
            prompt_file=None,
            prompt=self.PERF_PROMPT,
            device=DEVICE,
            max_tokens=MAX_TOKENS,
            no_think=True,
            precision="f16",
        )
        try:
            report_path = save_run_report(args, results, baseline_fp16)
            print(f"\n[Report] Saved to: {report_path}")
        except OSError as exc:
            print(f"\n[Report] Warning: failed to save report: {exc}")

        # Always passes — no performance thresholds
```

- [ ] **Step 4.2: Verify collection**

```bash
python -m pytest scripts/test_dflash.py::TestDFlashPerformance::test_perf_all_configs --collect-only
```

Expected: 1 item collected.

- [ ] **Step 4.3: Run performance test**

```bash
python -m pytest scripts/test_dflash.py::TestDFlashPerformance::test_perf_all_configs -v -s
```

Expected: PASS. Summary table printed to stdout. Report file created in `dflash优化报告/`.

- [ ] **Step 4.4: Run full test suite (accuracy + performance)**

```bash
python -m pytest scripts/test_dflash.py -v -s
```

Expected: 7 tests PASS (3 fp16 accuracy + 3 int4 accuracy + 1 perf).

- [ ] **Step 4.5: Verify `--full` flag adds more prompts**

```bash
python -m pytest scripts/test_dflash.py --collect-only --full
```

Expected: 17 items collected (8 fp16 + 8 int4 + 1 perf).

- [ ] **Step 4.6: Commit performance test**

```bash
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
git add scripts/test_dflash.py
git commit --author="Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(tests): add DFlash performance test (all 6 configs, record-only table)"
```

---

## Summary

| File | Status | Responsibility |
|------|--------|---------------|
| `scripts/conftest.py` | New | CLI flags, session fixtures, prompt parametrization |
| `scripts/test_dflash.py` | New | Accuracy exact-match tests + performance table test |
| `scripts/dflash.py` | Unchanged | Imported as-is |

**Run all tests:**
```bash
pytest scripts/test_dflash.py -v -s
```

**Expected final result:** 7 tests PASS (default prompts).
