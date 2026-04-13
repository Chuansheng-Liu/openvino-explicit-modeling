"""
DFlash Benchmark — Quality & Performance across diverse input categories.

Runs the DFlash speculative-decoding exe and/or the baseline (non-speculative)
exe with 10 different prompt categories and collects throughput, latency,
acceptance rate, and generated text.

Usage:
    python benchmark_dflash_inputs.py                       # default: dflash only, all 10 prompts
    python benchmark_dflash_inputs.py --mode both           # run baseline + dflash, compare
    python benchmark_dflash_inputs.py --mode baseline       # baseline only
    python benchmark_dflash_inputs.py --max-tokens 256      # shorter output
    python benchmark_dflash_inputs.py --block-size 8        # different block size
    python benchmark_dflash_inputs.py --runs 3              # 3 runs per prompt (stability)
    python benchmark_dflash_inputs.py --categories 0,2,5    # only specific categories (by index)
    python benchmark_dflash_inputs.py --quant FP16          # use FP16 instead of INT4_SYM
    python benchmark_dflash_inputs.py --dry-run             # print configs without running

Build first:  build.bat (from openvino-explicit-modeling root)
"""

import argparse
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ═════════════════════════════════════════════════════════════════
# Configuration — all paths derived from script location
# ═════════════════════════════════════════════════════════════════

# <repo>/scripts/benchmark_dflash_inputs.py → <repo> → <root>
SCRIPT_DIR = Path(__file__).resolve().parent          # .../openvino-explicit-modeling/scripts
OEM_DIR    = SCRIPT_DIR.parent                        # .../openvino-explicit-modeling
ROOT_DIR   = OEM_DIR.parent                           # .../src_code (workspace root)

DEFAULT_MODEL_DIR = None  # Override with --models-root CLI arg
DEVICE = "GPU"

DLL_DIRS = [
    ROOT_DIR / "openvino" / "bin" / "intel64" / "Release",
    ROOT_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin",
    ROOT_DIR / "openvino.genai" / "build" / "openvino_genai",
    ROOT_DIR / "openvino.genai" / "build" / "bin",
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "Release",
]

DFLASH_EXE_CANDIDATES = [
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "modeling_qwen3_5_dflash.exe",
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "Release" / "modeling_qwen3_5_dflash.exe",
]

BASELINE_EXE_CANDIDATES = [
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "modeling_qwen3_5.exe",
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "Release" / "modeling_qwen3_5.exe",
]

REPORT_DIR = ROOT_DIR / "dflash_exe_reports"

TEST_IMAGE_PATH = SCRIPT_DIR / "test.jpg"

TEXT_QUESTIONS = [
    {
        "category": "Logical Reasoning",
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left? Explain your reasoning step by step.",
    },
    {
        "category": "Factual Knowledge",
        "prompt": "What is the capital of Australia, and why do many people incorrectly think it is Sydney? Provide historical context.",
    },
    {
        "category": "Math Calculation",
        "prompt": "Calculate the result of 347 multiplied by 28, then subtract 1523. Show your work step by step.",
    },
    {
        "category": "Reading Comprehension",
        "prompt": (
            "Read the following passage and answer the question. "
            "Passage: 'The Great Wall of China, built over many centuries, stretches approximately 13,171 miles. "
            "Contrary to popular belief, it is not visible from space with the naked eye. "
            "The wall was primarily built to protect against invasions from northern nomadic groups.' "
            "Question: What is a common misconception about the Great Wall mentioned in this passage?"
        ),
    },
    {
        "category": "Creative Writing",
        "prompt": "Write a short poem (8 lines) about the beauty of sunrise over the ocean. Use vivid imagery and at least one metaphor.",
    },
    {
        "category": "Code Generation",
        "prompt": "Write a Python function called 'fibonacci' that takes an integer n and returns the nth Fibonacci number using dynamic programming. Include a brief docstring.",
    },
    {
        "category": "Causal Reasoning",
        "prompt": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain why or why not using principles of formal logic.",
    },
    {
        "category": "Summarization",
        "prompt": (
            "Summarize the following in 2-3 sentences: "
            "'Artificial intelligence has transformed industries ranging from healthcare to finance. "
            "In healthcare, AI assists in diagnosing diseases, predicting patient outcomes, and personalizing treatment plans. "
            "In finance, AI algorithms detect fraud, automate trading, and assess credit risk. "
            "Despite these advances, concerns about job displacement, bias in algorithms, and data privacy "
            "continue to spark debate among policymakers, technologists, and the public.'"
        ),
    },
    {
        "category": "Translation & Multilingual",
        "prompt": (
            "Translate the following English sentence into French, Spanish, and German: "
            "'The quick brown fox jumps over the lazy dog.' "
            "Then explain any interesting linguistic differences between the three translations."
        ),
    },
    {
        "category": "Commonsense & Analogy",
        "prompt": (
            "Complete the analogy: 'Book is to reading as fork is to ___.' "
            "Explain your reasoning and provide two more analogies following the same pattern."
        ),
    },
]

VL_QUESTIONS = [
    {
        "category": "VL: Image Description",
        "prompt": "describe this picture in details: ",
        "image": str(TEST_IMAGE_PATH),
    },
]


# ═════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    category: str
    prompt: str
    run: int
    tokens: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput: float = 0.0
    decode_ms: float = 0.0
    draft_steps: int = 0
    accept_rate: float = 0.0
    avg_accepted: float = 0.0
    draft_avg_ms: float = 0.0
    verify_avg_ms: float = 0.0
    output_text: str = ""
    raw_stdout: str = ""
    success: bool = True


# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

def setup_env():
    """Set DLL paths and required env vars."""
    extra = ";".join(str(d) for d in DLL_DIRS if d.exists())
    os.environ["PATH"] = extra + ";" + os.environ.get("PATH", "")
    os.environ["OV_GENAI_USE_MODELING_API"] = "1"
    os.environ["OV_GENAI_DISABLE_THINKING"] = "1"


def find_exe(candidates, label) -> Path:
    for p in candidates:
        if p.exists():
            return p
    print(f"ERROR: {label} exe not found. Build first:")
    print(f"  {OEM_DIR / 'build.bat'}")
    sys.exit(1)


def build_baseline_env(quant: str) -> dict:
    """Build env dict for baseline with inflight quantization matching dflash config."""
    env = os.environ.copy()
    env["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant.lower()
    env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = "128"
    env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = "int8_asym"
    env["OV_GPU_MOE_DISABLE_ONEDNN"] = "1"
    return env


def parse_output(stdout: str) -> dict:
    """Extract metrics from exe stdout."""
    m = {}
    patterns = {
        "tokens":       (r"Output token size:\s*(\d+)", int),
        "ttft_ms":      (r"TTFT:\s*([\d.]+)\s*ms", float),
        "tpot_ms":      (r"TPOT:\s*([\d.]+)\s*ms/token", float),
        "decode_ms":    (r"Decode time:\s*([\d.]+)\s*ms", float),
        "draft_steps":  (r"Draft steps:\s*(\d+)", int),
        "accept_rate":  (r"Acceptance rate:\s*([\d.]+)", float),
        "avg_accepted": (r"Avg accepted per step:\s*([\d.]+)", float),
        "draft_avg_ms": (r"Avg draft time:\s*([\d.]+)\s*ms", float),
        "verify_avg_ms":(r"Avg verify time:\s*([\d.]+)\s*ms", float),
    }
    for key, (pat, typ) in patterns.items():
        match = re.search(pat, stdout)
        m[key] = typ(match.group(1)) if match else (0 if typ == int else 0.0)

    # Throughput: prefer direct match, fall back to calculated
    tp_match = re.search(r"Throughput:\s*([\d.]+)\s*tokens/s", stdout)
    if tp_match:
        m["throughput"] = float(tp_match.group(1))
    elif m["tpot_ms"] > 0:
        m["throughput"] = 1000.0 / m["tpot_ms"]
    elif m["decode_ms"] > 0 and m["tokens"] > 1:
        m["throughput"] = (m["tokens"] - 1) * 1000.0 / m["decode_ms"]
    else:
        m["throughput"] = 0.0

    # Output text: try dflash format first, then baseline format
    text_match = re.search(
        r"\[Output\]\s*\n(.*?)\n\s*\n\[Generation Complete\]", stdout, re.DOTALL
    )
    if not text_match:
        text_match = re.search(
            r"Throughput:\s*[\d.]+\s*tokens/s\s*\n(.*)", stdout, re.DOTALL
        )
    m["output_text"] = text_match.group(1).strip() if text_match else ""

    return m


def run_dflash(exe: Path, prompt: str, max_tokens: int, block_size: int,
               target_quant: str, draft_quant: str,
               model_dir: Path = None, draft_dir: Path = None,
               image_path: str = None) -> str:
    """Run the DFlash exe and return stdout."""
    cmd = [
        str(exe),
        str(model_dir), str(draft_dir),
        prompt, DEVICE,
        str(max_tokens), str(block_size),
        target_quant, draft_quant,
    ]
    if image_path:
        cmd.append(image_path)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
    )
    return result.stdout + result.stderr


def run_baseline(exe: Path, prompt: str, max_tokens: int,
                 model_dir: Path, quant: str,
                 image_path: str = None) -> str:
    """Run the baseline (non-speculative) exe and return stdout."""
    mode = "vl" if image_path else "text"
    cmd = [
        str(exe),
        "--model", str(model_dir),
        "--cache-model",
        "--mode", mode,
        "--prompt", prompt,
        "--output-tokens", str(max_tokens),
    ]
    if image_path:
        cmd.extend(["--image", image_path])
    env = build_baseline_env(quant)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace", env=env,
    )
    return result.stdout + result.stderr


def mean_std(vals):
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    avg = sum(vals) / n
    if n < 2:
        return avg, 0.0
    var = sum((v - avg) ** 2 for v in vals) / (n - 1)
    return avg, math.sqrt(var)


# ═════════════════════════════════════════════════════════════════
# Reporting helpers
# ═════════════════════════════════════════════════════════════════

def _get_cats(questions):
    """Return unique category names preserving order."""
    seen = []
    for q in questions:
        if q["category"] not in seen:
            seen.append(q["category"])
    return seen


def _aggregate(results, questions):
    """Compute per-category and overall stats. Returns (cat_stats, overall)."""
    ok = [r for r in results if r.success]
    cats = _get_cats(questions)
    cat_stats = {}
    for cat in cats:
        cr = [r for r in ok if r.category == cat]
        if not cr:
            cat_stats[cat] = None
            continue
        tp_avg, tp_std = mean_std([r.throughput for r in cr])
        ttft_avg, _ = mean_std([r.ttft_ms for r in cr])
        tpot_avg, _ = mean_std([r.tpot_ms for r in cr])
        tok_avg = sum(r.tokens for r in cr) / len(cr)
        acc_vals = [r.accept_rate * 100 for r in cr if r.accept_rate > 0]
        acc_avg, _ = mean_std(acc_vals) if acc_vals else (0, 0)
        avgacc_vals = [r.avg_accepted for r in cr if r.avg_accepted > 0]
        avgacc_avg, _ = mean_std(avgacc_vals) if avgacc_vals else (0, 0)
        draft_vals = [r.draft_avg_ms for r in cr if r.draft_avg_ms > 0]
        draft_avg, _ = mean_std(draft_vals) if draft_vals else (0, 0)
        verify_vals = [r.verify_avg_ms for r in cr if r.verify_avg_ms > 0]
        verify_avg, _ = mean_std(verify_vals) if verify_vals else (0, 0)
        cat_stats[cat] = {
            "tp": tp_avg, "tp_std": tp_std,
            "ttft": ttft_avg, "tpot": tpot_avg,
            "tokens": tok_avg, "acc": acc_avg,
            "avgacc": avgacc_avg, "draft_ms": draft_avg, "verify_ms": verify_avg,
            "prompt_len": len(cr[0].prompt),
            "first": cr[0],
        }

    all_tp = [r.throughput for r in ok if r.throughput > 0]
    all_ttft = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    all_tpot = [r.tpot_ms for r in ok if r.tpot_ms > 0]
    all_tok = [r.tokens for r in ok if r.tokens > 0]
    all_acc = [r.accept_rate * 100 for r in ok if r.accept_rate > 0]
    all_avgacc = [r.avg_accepted for r in ok if r.avg_accepted > 0]
    all_draft = [r.draft_avg_ms for r in ok if r.draft_avg_ms > 0]
    all_verify = [r.verify_avg_ms for r in ok if r.verify_avg_ms > 0]
    tp_avg, tp_std = mean_std(all_tp) if all_tp else (0, 0)
    ttft_avg, _ = mean_std(all_ttft) if all_ttft else (0, 0)
    tpot_avg, _ = mean_std(all_tpot) if all_tpot else (0, 0)
    tok_avg = sum(all_tok) / len(all_tok) if all_tok else 0
    acc_avg, acc_std = mean_std(all_acc) if all_acc else (0, 0)
    avgacc_avg, _ = mean_std(all_avgacc) if all_avgacc else (0, 0)
    draft_avg, _ = mean_std(all_draft) if all_draft else (0, 0)
    verify_avg, _ = mean_std(all_verify) if all_verify else (0, 0)
    overall = {
        "tp": tp_avg, "tp_std": tp_std,
        "ttft": ttft_avg, "tpot": tpot_avg,
        "tokens": tok_avg,
        "acc": acc_avg, "acc_std": acc_std,
        "avgacc": avgacc_avg,
        "draft_ms": draft_avg, "verify_ms": verify_avg,
        "ok": len(ok), "total": len(results),
        "all_tp": all_tp, "all_acc": all_acc,
    }
    return cat_stats, overall


def _print_single_results(questions, results, elapsed, args, label):
    """Print results table for a single mode (dflash or baseline)."""
    ok = [r for r in results if r.success]
    if not ok:
        print("\nAll runs failed. Nothing to report.")
        return

    cats = _get_cats(questions)
    cat_stats, overall = _aggregate(results, questions)

    print()
    print("=" * 140)
    print(f"  {label.upper()} BENCHMARK RESULTS")
    print("=" * 140)
    hdr = (f"{'Category':<28s} {'tok/s':>10s} {'TTFT(ms)':>10s} {'TPOT(ms)':>10s} "
           f"{'Tokens':>7s} {'Accept%':>8s} {'AvgAcc':>7s} "
           f"{'Draft ms':>9s} {'Verify ms':>10s} {'Prompt len':>10s}")
    print(hdr)
    print("-" * 140)

    for cat in cats:
        s = cat_stats.get(cat)
        if s is None:
            print(f"{cat:<28s} {'FAILED':>10s}")
            continue
        acc = f"{s['acc']:.1f}%" if s['acc'] > 0 else "-"
        avg = f"{s['avgacc']:.2f}" if s['avgacc'] > 0 else "-"
        dms = f"{s['draft_ms']:.1f}" if s['draft_ms'] > 0 else "-"
        vms = f"{s['verify_ms']:.1f}" if s['verify_ms'] > 0 else "-"
        print(f"{cat:<28s} {s['tp']:10.2f} {s['ttft']:10.0f} {s['tpot']:10.2f} "
              f"{s['tokens']:7.0f} {acc:>8s} {avg:>7s} {dms:>9s} {vms:>10s} {s['prompt_len']:10d}")

    print("-" * 140)
    o = overall
    acc_s = f"{o['acc']:.1f}%" if o['all_acc'] else "-"
    avgacc_s = f"{o['avgacc']:.2f}" if o['avgacc'] > 0 else "-"
    draft_s = f"{o['draft_ms']:.1f}" if o['draft_ms'] > 0 else "-"
    verify_s = f"{o['verify_ms']:.1f}" if o['verify_ms'] > 0 else "-"
    print(f"{'** OVERALL AVERAGE **':<28s} {o['tp']:10.2f} {o['ttft']:10.0f} {o['tpot']:10.2f} "
          f"{o['tokens']:7.0f} {acc_s:>8s} {avgacc_s:>7s} {draft_s:>9s} {verify_s:>10s} {'':>10s}")
    print("=" * 140)

    if o['all_tp']:
        print(f"\n  Throughput:  avg={o['tp']:.2f} tok/s  std={o['tp_std']:.2f}  "
              f"min={min(o['all_tp']):.2f}  max={max(o['all_tp']):.2f}")
    if o['all_acc']:
        print(f"  Acceptance:  avg={o['acc']:.1f}%  std={o['acc_std']:.1f}%")

    # Generated text output
    print()
    print("=" * 140)
    print(f"  GENERATED OUTPUT — {label} (first run per category)")
    print("=" * 140)
    for cat in cats:
        s = cat_stats.get(cat)
        if s is None:
            continue
        r = s["first"]
        print(f"\n--- [{r.category}] ---")
        print(f"Prompt: {r.prompt[:120]}{'...' if len(r.prompt) > 120 else ''}")
        print(f"Output ({r.tokens} tokens, {r.throughput:.1f} tok/s):")
        text = r.output_text if r.output_text else "(no output captured)"
        if len(text) > 500:
            print(text[:500] + "\n  [...truncated]")
        else:
            print(text)

    print(f"\n[Total]  {elapsed:.0f}s")


def _save_single_report(questions, results, elapsed, args, label):
    """Save markdown report for a single mode."""
    ok = [r for r in results if r.success]
    if not ok:
        return
    cats = _get_cats(questions)
    cat_stats, overall = _aggregate(results, questions)
    o = overall

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "dflash" if label == "DFlash" else "baseline"
    md_path = REPORT_DIR / f"{tag}_input_categories_{ts}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {label} Input Category Benchmark\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config**: {args.quant} | max_tokens={args.max_tokens}")
        if label == "DFlash":
            f.write(f" | block_size={args.block_size}")
        f.write(f" | runs={args.runs}  \n")
        f.write(f"**Device**: {DEVICE} | Model: {args.target_model}")
        if label == "DFlash":
            f.write(f" | Draft: {args.draft_model}")
        f.write("  \n")
        f.write(f"**Total time**: {elapsed:.0f}s\n\n")

        f.write("## Performance Summary\n\n")
        f.write("| Category | tok/s | TTFT (ms) | TPOT (ms) | Tokens | Accept % | Avg Acc | Draft (ms) | Verify (ms) |\n")
        f.write("|----------|------:|----------:|----------:|-------:|---------:|--------:|-----------:|------------:|\n")
        for cat in cats:
            s = cat_stats.get(cat)
            if s is None:
                f.write(f"| {cat} | FAILED | | | | | | | |\n")
                continue
            acc = f"{s['acc']:.1f}" if s['acc'] > 0 else "-"
            avg = f"{s['avgacc']:.2f}" if s['avgacc'] > 0 else "-"
            dms = f"{s['draft_ms']:.1f}" if s['draft_ms'] > 0 else "-"
            vms = f"{s['verify_ms']:.1f}" if s['verify_ms'] > 0 else "-"
            f.write(f"| {cat} | {s['tp']:.2f} | {s['ttft']:.0f} | {s['tpot']:.2f} | "
                    f"{s['tokens']:.0f} | {acc} | {avg} | {dms} | {vms} |\n")

        if o['all_tp']:
            f.write(f"| **OVERALL AVERAGE** | **{o['tp']:.2f}** | {o['ttft']:.0f} | {o['tpot']:.2f} | "
                    f"{o['tokens']:.0f} | {o['acc']:.1f} | {o['avgacc']:.2f} | "
                    f"{o['draft_ms']:.1f} | {o['verify_ms']:.1f} |\n")

        f.write("\n## Aggregate\n\n")
        if o['all_tp']:
            f.write(f"- **Throughput**: avg={o['tp']:.2f} tok/s, std={o['tp_std']:.2f}, "
                    f"min={min(o['all_tp']):.2f}, max={max(o['all_tp']):.2f}\n")
        if o['all_acc']:
            f.write(f"- **Acceptance**: avg={o['acc']:.1f}%, std={o['acc_std']:.1f}%\n")
        f.write(f"- **Total runs**: {o['ok']} succeeded / {o['total']} total\n")

        f.write("\n## Generated Outputs\n")
        for cat in cats:
            s = cat_stats.get(cat)
            if s is None:
                continue
            r = s["first"]
            f.write(f"\n### {r.category}\n\n")
            f.write(f"**Prompt**: {r.prompt}\n\n")
            f.write(f"**Stats**: {r.tokens} tokens, {r.throughput:.1f} tok/s")
            if r.accept_rate > 0:
                f.write(f", accept={r.accept_rate*100:.1f}%")
            f.write("\n\n")
            text = r.output_text if r.output_text else "(no output captured)"
            f.write(f"**Output**:\n\n{text}\n\n---\n")

    print(f"\n[Report] {md_path}")


def _print_comparison(questions, dflash_results, baseline_results, elapsed, args):
    """Print side-by-side comparison table."""
    d_ok = [r for r in dflash_results if r.success]
    b_ok = [r for r in baseline_results if r.success]
    if not d_ok and not b_ok:
        print("\nAll runs failed. Nothing to report.")
        return

    cats = _get_cats(questions)
    d_stats, d_overall = _aggregate(dflash_results, questions)
    b_stats, b_overall = _aggregate(baseline_results, questions)

    print()
    print("=" * 130)
    print("  BASELINE vs DFLASH COMPARISON")
    print("=" * 130)
    hdr = (f"{'Category':<28s} {'Base tok/s':>10s} {'DFlash tok/s':>12s} {'Speedup':>8s} "
           f"{'Base TPOT':>10s} {'DFlash TPOT':>12s} {'Accept%':>8s} {'Tokens':>7s}")
    print(hdr)
    print("-" * 130)

    for cat in cats:
        ds = d_stats.get(cat)
        bs = b_stats.get(cat)
        if ds is None and bs is None:
            print(f"{cat:<28s} {'FAILED':>10s}")
            continue

        b_tp = f"{bs['tp']:.2f}" if bs else "-"
        d_tp = f"{ds['tp']:.2f}" if ds else "-"
        b_tpot = f"{bs['tpot']:.2f}" if bs else "-"
        d_tpot = f"{ds['tpot']:.2f}" if ds else "-"
        acc = f"{ds['acc']:.1f}%" if ds and ds['acc'] > 0 else "-"
        tok = f"{ds['tokens']:.0f}" if ds else (f"{bs['tokens']:.0f}" if bs else "-")

        if ds and bs and bs['tp'] > 0:
            speedup = ds['tp'] / bs['tp']
            sp_str = f"{speedup:.2f}x"
        else:
            sp_str = "-"

        print(f"{cat:<28s} {b_tp:>10s} {d_tp:>12s} {sp_str:>8s} "
              f"{b_tpot:>10s} {d_tpot:>12s} {acc:>8s} {tok:>7s}")

    print("-" * 130)

    # Overall
    b_tp = f"{b_overall['tp']:.2f}" if b_overall['all_tp'] else "-"
    d_tp = f"{d_overall['tp']:.2f}" if d_overall['all_tp'] else "-"
    b_tpot = f"{b_overall['tpot']:.2f}" if b_overall['all_tp'] else "-"
    d_tpot = f"{d_overall['tpot']:.2f}" if d_overall['all_tp'] else "-"
    acc = f"{d_overall['acc']:.1f}%" if d_overall['all_acc'] else "-"
    if d_overall['all_tp'] and b_overall['all_tp'] and b_overall['tp'] > 0:
        sp = d_overall['tp'] / b_overall['tp']
        sp_str = f"{sp:.2f}x"
    else:
        sp_str = "-"
    print(f"{'** OVERALL **':<28s} {b_tp:>10s} {d_tp:>12s} {sp_str:>8s} "
          f"{b_tpot:>10s} {d_tpot:>12s} {acc:>8s} {'':>7s}")
    print("=" * 130)

    if d_overall['all_tp'] and b_overall['all_tp'] and b_overall['tp'] > 0:
        print(f"\n  DFlash speedup over baseline: {d_overall['tp'] / b_overall['tp']:.2f}x "
              f"({b_overall['tp']:.2f} -> {d_overall['tp']:.2f} tok/s)")
    if d_overall['all_acc']:
        print(f"  DFlash acceptance rate: {d_overall['acc']:.1f}%")

    print(f"\n[Total]  {elapsed:.0f}s")


def _save_comparison_report(questions, dflash_results, baseline_results, elapsed, args):
    """Save side-by-side comparison markdown report."""
    cats = _get_cats(questions)
    d_stats, d_overall = _aggregate(dflash_results, questions)
    b_stats, b_overall = _aggregate(baseline_results, questions)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = REPORT_DIR / f"dflash_vs_baseline_{ts}.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DFlash vs Baseline Comparison\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config**: {args.quant} | max_tokens={args.max_tokens} | "
                f"block_size={args.block_size} | runs={args.runs}  \n")
        f.write(f"**Device**: {DEVICE} | Model: {args.target_model} | Draft: {args.draft_model}  \n")
        f.write(f"**Total time**: {elapsed:.0f}s\n\n")

        # Comparison table
        f.write("## Performance Comparison\n\n")
        f.write("| Category | Baseline tok/s | DFlash tok/s | Speedup | "
                "Baseline TPOT (ms) | DFlash TPOT (ms) | Accept % | Tokens |\n")
        f.write("|----------|---------------:|-------------:|--------:|"
                "-------------------:|-----------------:|---------:|-------:|\n")

        for cat in cats:
            ds = d_stats.get(cat)
            bs = b_stats.get(cat)
            if ds is None and bs is None:
                f.write(f"| {cat} | FAILED | FAILED | - | - | - | - | - |\n")
                continue

            b_tp = f"{bs['tp']:.2f}" if bs else "-"
            d_tp = f"{ds['tp']:.2f}" if ds else "-"
            b_tpot = f"{bs['tpot']:.2f}" if bs else "-"
            d_tpot = f"{ds['tpot']:.2f}" if ds else "-"
            acc = f"{ds['acc']:.1f}" if ds and ds['acc'] > 0 else "-"
            tok = f"{ds['tokens']:.0f}" if ds else (f"{bs['tokens']:.0f}" if bs else "-")

            if ds and bs and bs['tp'] > 0:
                speedup = ds['tp'] / bs['tp']
                sp_str = f"**{speedup:.2f}x**"
            else:
                sp_str = "-"

            f.write(f"| {cat} | {b_tp} | {d_tp} | {sp_str} | "
                    f"{b_tpot} | {d_tpot} | {acc} | {tok} |\n")

        # Overall row
        b_tp = f"{b_overall['tp']:.2f}" if b_overall['all_tp'] else "-"
        d_tp = f"{d_overall['tp']:.2f}" if d_overall['all_tp'] else "-"
        b_tpot = f"{b_overall['tpot']:.2f}" if b_overall['all_tp'] else "-"
        d_tpot = f"{d_overall['tpot']:.2f}" if d_overall['all_tp'] else "-"
        acc = f"{d_overall['acc']:.1f}" if d_overall['all_acc'] else "-"
        if d_overall['all_tp'] and b_overall['all_tp'] and b_overall['tp'] > 0:
            sp = d_overall['tp'] / b_overall['tp']
            sp_str = f"**{sp:.2f}x**"
        else:
            sp_str = "-"
        f.write(f"| **OVERALL** | **{b_tp}** | **{d_tp}** | {sp_str} | "
                f"{b_tpot} | {d_tpot} | {acc} | - |\n")

        # Detailed per-mode results
        for label, results in [("DFlash", dflash_results), ("Baseline", baseline_results)]:
            ok = [r for r in results if r.success]
            if not ok:
                continue
            stats, ovr = _aggregate(results, questions)

            f.write(f"\n## {label} Details\n\n")
            f.write("| Category | tok/s | TTFT (ms) | TPOT (ms) | Tokens |")
            if label == "DFlash":
                f.write(" Accept % | Avg Acc | Draft (ms) | Verify (ms) |")
            f.write("\n")
            f.write("|----------|------:|----------:|----------:|-------:|")
            if label == "DFlash":
                f.write("---------:|--------:|-----------:|------------:|")
            f.write("\n")

            for cat in cats:
                s = stats.get(cat)
                if s is None:
                    f.write(f"| {cat} | FAILED | | | |")
                    if label == "DFlash":
                        f.write(" | | | |")
                    f.write("\n")
                    continue
                f.write(f"| {cat} | {s['tp']:.2f} | {s['ttft']:.0f} | {s['tpot']:.2f} | {s['tokens']:.0f} |")
                if label == "DFlash":
                    acc = f"{s['acc']:.1f}" if s['acc'] > 0 else "-"
                    avg = f"{s['avgacc']:.2f}" if s['avgacc'] > 0 else "-"
                    dms = f"{s['draft_ms']:.1f}" if s['draft_ms'] > 0 else "-"
                    vms = f"{s['verify_ms']:.1f}" if s['verify_ms'] > 0 else "-"
                    f.write(f" {acc} | {avg} | {dms} | {vms} |")
                f.write("\n")

        # Summary
        f.write("\n## Summary\n\n")
        if d_overall['all_tp'] and b_overall['all_tp'] and b_overall['tp'] > 0:
            sp = d_overall['tp'] / b_overall['tp']
            f.write(f"- **DFlash speedup**: {sp:.2f}x over baseline "
                    f"({b_overall['tp']:.2f} -> {d_overall['tp']:.2f} tok/s)\n")
        if d_overall['all_acc']:
            f.write(f"- **DFlash acceptance rate**: {d_overall['acc']:.1f}%\n")
        if d_overall['all_tp']:
            f.write(f"- **DFlash throughput range**: {min(d_overall['all_tp']):.2f} - {max(d_overall['all_tp']):.2f} tok/s\n")
        if b_overall['all_tp']:
            f.write(f"- **Baseline throughput range**: {min(b_overall['all_tp']):.2f} - {max(b_overall['all_tp']):.2f} tok/s\n")
        d_ok = sum(1 for r in dflash_results if r.success)
        b_ok = sum(1 for r in baseline_results if r.success)
        f.write(f"- **Runs**: DFlash {d_ok}/{len(dflash_results)}, Baseline {b_ok}/{len(baseline_results)}\n")

    print(f"\n[Report] {md_path}")


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DFlash benchmark across diverse prompts")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens (default: 512)")
    parser.add_argument("--block-size", type=int, default=16, help="Block size (default: 16)")
    parser.add_argument("--quant", default="INT4_SYM", help="Quantization for target+draft (default: INT4_SYM)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt (default: 1)")
    parser.add_argument("--categories", default=None, help="Comma-separated indices (0-9) to run subset")
    parser.add_argument("--models-root", type=str, default=None,
                        help="Root dir for HF models (default: auto-detect from script location)")
    parser.add_argument("--target-model", type=str, default="Qwen3.5-4B",
                        help="Target model directory name (default: Qwen3.5-4B)")
    parser.add_argument("--draft-model", type=str, default="Qwen3.5-4B-Dflash",
                        help="Draft model directory name (default: Qwen3.5-4B-Dflash)")
    parser.add_argument("--mode", choices=["dflash", "baseline", "both"], default="dflash",
                        help="What to benchmark: dflash, baseline, or both (default: dflash)")
    parser.add_argument("--test-type", choices=["text", "vl", "all"], default="text",
                        help="Test category type: text, vl, or all (default: text)")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    args = parser.parse_args()

    # Resolve model paths
    if args.models_root:
        models_root = Path(args.models_root)
    else:
        print("ERROR: --models-root is required (dir containing target and draft model dirs)")
        sys.exit(1)
    model_dir = models_root / args.target_model
    draft_dir = models_root / args.draft_model

    run_dflash_mode = args.mode in ("dflash", "both")
    run_baseline_mode = args.mode in ("baseline", "both")

    setup_env()

    dflash_exe = find_exe(DFLASH_EXE_CANDIDATES, "DFlash") if run_dflash_mode else None
    baseline_exe = find_exe(BASELINE_EXE_CANDIDATES, "Baseline") if run_baseline_mode else None

    # Build question pool based on --test-type
    if args.test_type == "text":
        question_pool = TEXT_QUESTIONS
    elif args.test_type == "vl":
        question_pool = VL_QUESTIONS
    else:
        question_pool = TEXT_QUESTIONS + VL_QUESTIONS

    # Validate VL image exists when running VL tests
    vl_in_pool = any(q.get("image") for q in question_pool)
    if vl_in_pool and not TEST_IMAGE_PATH.exists():
        print(f"ERROR: VL test image not found: {TEST_IMAGE_PATH}")
        sys.exit(1)

    # Select categories
    if args.categories:
        indices = [int(x.strip()) for x in args.categories.split(",")]
        questions = [question_pool[i] for i in indices if 0 <= i < len(question_pool)]
    else:
        questions = question_pool

    total_runs = len(questions) * args.runs
    if args.mode == "both":
        total_runs *= 2

    type_label = {"text": "Text", "vl": "Vision-Language", "all": "Text + VL"}
    mode_label = {"dflash": "DFlash only", "baseline": "Baseline only", "both": "DFlash + Baseline"}
    print()
    print("=" * 80)
    print("  DFlash Input Category Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"  Mode           : {mode_label[args.mode]}")
    print(f"  Test type      : {type_label[args.test_type]}")
    print(f"  Categories     : {len(questions)}")
    print(f"  Model          : {model_dir}")
    if run_dflash_mode:
        print(f"  Draft model    : {draft_dir}")
    print(f"  Max new tokens : {args.max_tokens}")
    if run_dflash_mode:
        print(f"  Block size     : {args.block_size}")
    print(f"  Quantization   : {args.quant}")
    print(f"  Runs per prompt: {args.runs}")
    print(f"  Total runs     : {total_runs}")
    if dflash_exe:
        print(f"  DFlash exe     : {dflash_exe}")
    if baseline_exe:
        print(f"  Baseline exe   : {baseline_exe}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Prompts:")
        for i, q in enumerate(questions):
            print(f"  {i:2d}. [{q['category']:<26s}] {q['prompt'][:80]}...")
        print("\n[DRY RUN] No runs executed.")
        return

    # ── Run benchmark ────────────────────────────────────────────
    dflash_results: list[RunResult] = []
    baseline_results: list[RunResult] = []
    run_idx = 0
    t_start = time.time()

    for q in questions:
        for r in range(1, args.runs + 1):
            label = q["category"]
            if args.runs > 1:
                label += f" (run {r})"
            image_path = q.get("image")

            # --- Baseline run ---
            if run_baseline_mode:
                run_idx += 1
                pct = int(100 * run_idx / total_runs)
                print(f"\n[{run_idx}/{total_runs} {pct}%] [BASELINE] {label}")
                print("-" * 70)
                print(f"  Prompt: {q['prompt'][:100]}...")
                if image_path:
                    print(f"  Image:  {image_path}")

                try:
                    stdout = run_baseline(
                        baseline_exe, q["prompt"], args.max_tokens,
                        model_dir=model_dir, quant=args.quant,
                        image_path=image_path,
                    )
                    m = parse_output(stdout)
                    res = RunResult(
                        category=q["category"], prompt=q["prompt"], run=r,
                        tokens=m["tokens"], ttft_ms=m["ttft_ms"], tpot_ms=m["tpot_ms"],
                        throughput=m["throughput"], decode_ms=m["decode_ms"],
                        draft_steps=m["draft_steps"], accept_rate=m["accept_rate"],
                        avg_accepted=m["avg_accepted"], draft_avg_ms=m["draft_avg_ms"],
                        verify_avg_ms=m["verify_avg_ms"], output_text=m["output_text"],
                        raw_stdout=stdout,
                    )
                    baseline_results.append(res)
                    print(f"  => {res.throughput:.1f} tok/s | TTFT={res.ttft_ms:.0f}ms | "
                          f"TPOT={res.tpot_ms:.2f}ms | tokens={res.tokens}")
                except subprocess.TimeoutExpired:
                    print("  => TIMEOUT (600s)")
                    baseline_results.append(RunResult(
                        category=q["category"], prompt=q["prompt"], run=r, success=False
                    ))
                except Exception as e:
                    print(f"  => FAILED: {e}")
                    baseline_results.append(RunResult(
                        category=q["category"], prompt=q["prompt"], run=r, success=False
                    ))

            # --- DFlash run ---
            if run_dflash_mode:
                run_idx += 1
                pct = int(100 * run_idx / total_runs)
                print(f"\n[{run_idx}/{total_runs} {pct}%] [DFLASH] {label}")
                print("-" * 70)
                print(f"  Prompt: {q['prompt'][:100]}...")
                if image_path:
                    print(f"  Image:  {image_path}")

                try:
                    stdout = run_dflash(
                        dflash_exe, q["prompt"], args.max_tokens, args.block_size,
                        args.quant, args.quant,
                        model_dir=model_dir, draft_dir=draft_dir,
                        image_path=image_path,
                    )
                    m = parse_output(stdout)
                    res = RunResult(
                        category=q["category"], prompt=q["prompt"], run=r,
                        tokens=m["tokens"], ttft_ms=m["ttft_ms"], tpot_ms=m["tpot_ms"],
                        throughput=m["throughput"], decode_ms=m["decode_ms"],
                        draft_steps=m["draft_steps"], accept_rate=m["accept_rate"],
                        avg_accepted=m["avg_accepted"], draft_avg_ms=m["draft_avg_ms"],
                        verify_avg_ms=m["verify_avg_ms"], output_text=m["output_text"],
                        raw_stdout=stdout,
                    )
                    dflash_results.append(res)
                    print(f"  => {res.throughput:.1f} tok/s | TTFT={res.ttft_ms:.0f}ms | "
                          f"TPOT={res.tpot_ms:.2f}ms | tokens={res.tokens}")
                    if res.draft_steps > 0:
                        print(f"     accept={res.accept_rate*100:.1f}% | avg_acc={res.avg_accepted:.2f} | "
                              f"draft={res.draft_avg_ms:.1f}ms | verify={res.verify_avg_ms:.1f}ms")
                except subprocess.TimeoutExpired:
                    print("  => TIMEOUT (600s)")
                    dflash_results.append(RunResult(
                        category=q["category"], prompt=q["prompt"], run=r, success=False
                    ))
                except Exception as e:
                    print(f"  => FAILED: {e}")
                    dflash_results.append(RunResult(
                        category=q["category"], prompt=q["prompt"], run=r, success=False
                    ))

    elapsed = time.time() - t_start

    # Merge results for single-mode reporting
    if args.mode == "dflash":
        all_results = dflash_results
    elif args.mode == "baseline":
        all_results = baseline_results
    else:
        all_results = None  # handled separately in "both" mode

    # ═════════════════════════════════════════════════════════════
    # Print and save results
    # ═════════════════════════════════════════════════════════════
    if args.mode == "both":
        _print_comparison(questions, dflash_results, baseline_results, elapsed, args)
        _save_comparison_report(questions, dflash_results, baseline_results, elapsed, args)
    else:
        label = "DFlash" if args.mode == "dflash" else "Baseline"
        _print_single_results(questions, all_results, elapsed, args, label)
        _save_single_report(questions, all_results, elapsed, args, label)


if __name__ == "__main__":
    main()
