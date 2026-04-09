"""
DFlash Benchmark — Quality & Performance across diverse input categories.

Runs the DFlash speculative-decoding exe with 10 different prompt categories
and collects throughput, latency, acceptance rate, and generated text.

Usage:
    python benchmark_dflash_inputs.py                       # default: all 10 prompts, 512 tokens
    python benchmark_dflash_inputs.py --max-tokens 256      # shorter output
    python benchmark_dflash_inputs.py --block-size 8        # different block size
    python benchmark_dflash_inputs.py --runs 3              # 3 runs per prompt (stability)
    python benchmark_dflash_inputs.py --categories 0,2,5    # only specific categories (by index)
    python benchmark_dflash_inputs.py --quant FP16          # use FP16 instead of INT4_SYM
    python benchmark_dflash_inputs.py --dry-run             # print configs without running

Build first:  C:\\chuansheng\\src_code\\openvino-explicit-modeling\\build.bat
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
# Configuration
# ═════════════════════════════════════════════════════════════════

ROOT_DIR = Path(r"C:\chuansheng\src_code")
MODEL_DIR = Path(r"E:\data\models\Huggingface\Qwen3.5-4B")
DRAFT_DIR = Path(r"E:\data\models\Huggingface\Qwen3.5-4B-Dflash")
DEVICE = "GPU"
REPORT_DIR = ROOT_DIR / "dflash_exe_reports"

DLL_DIRS = [
    ROOT_DIR / "openvino" / "bin" / "intel64" / "Release",
    ROOT_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin",
    ROOT_DIR / "openvino.genai" / "build" / "openvino_genai",
    ROOT_DIR / "openvino.genai" / "build" / "bin",
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "Release",
]

EXE_CANDIDATES = [
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "modeling_qwen3_5_dflash.exe",
    ROOT_DIR / "openvino.genai" / "build" / "bin" / "Release" / "modeling_qwen3_5_dflash.exe",
]

QUESTIONS = [
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
    os.environ["OV_GENAI_DFLASH_VERBOSE"] = "0"
    os.environ["OV_GENAI_DFLASH_INT4_LMHEAD"] = "1"


def find_exe() -> Path:
    for p in EXE_CANDIDATES:
        if p.exists():
            return p
    print("ERROR: DFlash exe not found. Build first:")
    print("  C:\\chuansheng\\src_code\\openvino-explicit-modeling\\build.bat")
    sys.exit(1)


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

    # Throughput
    if m["tpot_ms"] > 0:
        m["throughput"] = 1000.0 / m["tpot_ms"]
    elif m["decode_ms"] > 0 and m["tokens"] > 1:
        m["throughput"] = (m["tokens"] - 1) * 1000.0 / m["decode_ms"]
    else:
        m["throughput"] = 0.0

    # Output text
    text_match = re.search(
        r"\[Output\]\s*\n(.*?)\n\s*\n\[Generation Complete\]", stdout, re.DOTALL
    )
    m["output_text"] = text_match.group(1).strip() if text_match else ""

    return m


def run_dflash(exe: Path, prompt: str, max_tokens: int, block_size: int,
               target_quant: str, draft_quant: str) -> str:
    """Run the DFlash exe and return stdout."""
    cmd = [
        str(exe),
        str(MODEL_DIR), str(DRAFT_DIR),
        prompt, DEVICE,
        str(max_tokens), str(block_size),
        target_quant, draft_quant,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600,
        encoding="utf-8", errors="replace",
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
# Main
# ═════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DFlash benchmark across diverse prompts")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max new tokens (default: 512)")
    parser.add_argument("--block-size", type=int, default=16, help="Block size (default: 16)")
    parser.add_argument("--quant", default="INT4_SYM", help="Quantization for target+draft (default: INT4_SYM)")
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt (default: 1)")
    parser.add_argument("--categories", default=None, help="Comma-separated indices (0-9) to run subset")
    parser.add_argument("--dry-run", action="store_true", help="Print config without running")
    args = parser.parse_args()

    setup_env()
    exe = find_exe()

    # Select categories
    if args.categories:
        indices = [int(x.strip()) for x in args.categories.split(",")]
        questions = [QUESTIONS[i] for i in indices if 0 <= i < len(QUESTIONS)]
    else:
        questions = QUESTIONS

    total_runs = len(questions) * args.runs

    print()
    print("=" * 80)
    print("  DFlash Input Category Benchmark")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"  Categories     : {len(questions)}")
    print(f"  Max new tokens : {args.max_tokens}")
    print(f"  Block size     : {args.block_size}")
    print(f"  Quantization   : {args.quant}")
    print(f"  Runs per prompt: {args.runs}")
    print(f"  Total runs     : {total_runs}")
    print(f"  Exe            : {exe}")
    print("=" * 80)

    if args.dry_run:
        print("\n[DRY RUN] Prompts:")
        for i, q in enumerate(questions):
            print(f"  {i:2d}. [{q['category']:<26s}] {q['prompt'][:80]}...")
        print("\n[DRY RUN] No runs executed.")
        return

    # ── Run benchmark ────────────────────────────────────────────
    results: list[RunResult] = []
    run_idx = 0
    t_start = time.time()

    for q in questions:
        for r in range(1, args.runs + 1):
            run_idx += 1
            pct = int(100 * run_idx / total_runs)
            label = q["category"]
            if args.runs > 1:
                label += f" (run {r})"

            print(f"\n[{run_idx}/{total_runs} {pct}%] {label}")
            print("-" * 70)
            print(f"  Prompt: {q['prompt'][:100]}...")

            try:
                stdout = run_dflash(
                    exe, q["prompt"], args.max_tokens, args.block_size,
                    args.quant, args.quant,
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
                results.append(res)

                print(f"  => {res.throughput:.1f} tok/s | TTFT={res.ttft_ms:.0f}ms | "
                      f"TPOT={res.tpot_ms:.2f}ms | tokens={res.tokens}")
                if res.draft_steps > 0:
                    print(f"     accept={res.accept_rate*100:.1f}% | avg_acc={res.avg_accepted:.2f} | "
                          f"draft={res.draft_avg_ms:.1f}ms | verify={res.verify_avg_ms:.1f}ms")

            except subprocess.TimeoutExpired:
                print("  => TIMEOUT (600s)" )
                results.append(RunResult(
                    category=q["category"], prompt=q["prompt"], run=r, success=False
                ))
            except Exception as e:
                print(f"  => FAILED: {e}")
                results.append(RunResult(
                    category=q["category"], prompt=q["prompt"], run=r, success=False
                ))

    elapsed = time.time() - t_start

    # ═════════════════════════════════════════════════════════════
    # Summary table
    # ═════════════════════════════════════════════════════════════
    ok_results = [r for r in results if r.success]
    if not ok_results:
        print("\nAll runs failed. Nothing to report.")
        return

    print()
    print("=" * 140)
    print("  BENCHMARK RESULTS")
    print("=" * 140)

    hdr = (f"{'Category':<28s} {'tok/s':>10s} {'TTFT(ms)':>10s} {'TPOT(ms)':>10s} "
           f"{'Tokens':>7s} {'Accept%':>8s} {'AvgAcc':>7s} "
           f"{'Draft ms':>9s} {'Verify ms':>10s} {'Prompt len':>10s}")
    print(hdr)
    print("-" * 140)

    # Group by category
    cats_seen = []
    for q in questions:
        cat = q["category"]
        if cat in cats_seen:
            continue
        cats_seen.append(cat)

        cat_results = [r for r in ok_results if r.category == cat]
        if not cat_results:
            print(f"{cat:<28s} {'FAILED':>10s}")
            continue

        if len(cat_results) == 1:
            r = cat_results[0]
            acc = f"{r.accept_rate*100:.1f}%" if r.accept_rate > 0 else "-"
            avg = f"{r.avg_accepted:.2f}" if r.avg_accepted > 0 else "-"
            dms = f"{r.draft_avg_ms:.1f}" if r.draft_avg_ms > 0 else "-"
            vms = f"{r.verify_avg_ms:.1f}" if r.verify_avg_ms > 0 else "-"
            print(f"{cat:<28s} {r.throughput:10.2f} {r.ttft_ms:10.0f} {r.tpot_ms:10.2f} "
                  f"{r.tokens:7d} {acc:>8s} {avg:>7s} {dms:>9s} {vms:>10s} {len(r.prompt):10d}")
        else:
            tp_avg, tp_std = mean_std([r.throughput for r in cat_results])
            ttft_avg, _ = mean_std([r.ttft_ms for r in cat_results])
            tpot_avg, _ = mean_std([r.tpot_ms for r in cat_results])
            tok_avg = sum(r.tokens for r in cat_results) / len(cat_results)
            acc_vals = [r.accept_rate * 100 for r in cat_results if r.accept_rate > 0]
            acc_avg, acc_std = mean_std(acc_vals) if acc_vals else (0, 0)

            tp_str = f"{tp_avg:.1f}±{tp_std:.1f}"
            acc_str = f"{acc_avg:.1f}±{acc_std:.1f}" if acc_vals else "-"
            print(f"{cat:<28s} {tp_str:>10s} {ttft_avg:10.0f} {tpot_avg:10.2f} "
                  f"{tok_avg:7.0f} {acc_str:>8s} {'':>7s} {'':>9s} {'':>10s} {len(cat_results[0].prompt):10d}")

    print("-" * 140)

    # Overall average row
    all_tp = [r.throughput for r in ok_results if r.throughput > 0]
    all_ttft = [r.ttft_ms for r in ok_results if r.ttft_ms > 0]
    all_tpot = [r.tpot_ms for r in ok_results if r.tpot_ms > 0]
    all_tok = [r.tokens for r in ok_results if r.tokens > 0]
    all_acc = [r.accept_rate * 100 for r in ok_results if r.accept_rate > 0]
    all_avgacc = [r.avg_accepted for r in ok_results if r.avg_accepted > 0]
    all_draft = [r.draft_avg_ms for r in ok_results if r.draft_avg_ms > 0]
    all_verify = [r.verify_avg_ms for r in ok_results if r.verify_avg_ms > 0]

    tp_avg, tp_std = mean_std(all_tp) if all_tp else (0, 0)
    ttft_avg, _ = mean_std(all_ttft) if all_ttft else (0, 0)
    tpot_avg, _ = mean_std(all_tpot) if all_tpot else (0, 0)
    tok_avg = sum(all_tok) / len(all_tok) if all_tok else 0
    acc_avg, acc_std = mean_std(all_acc) if all_acc else (0, 0)
    avgacc_avg, _ = mean_std(all_avgacc) if all_avgacc else (0, 0)
    draft_avg, _ = mean_std(all_draft) if all_draft else (0, 0)
    verify_avg, _ = mean_std(all_verify) if all_verify else (0, 0)

    acc_s = f"{acc_avg:.1f}%" if all_acc else "-"
    avgacc_s = f"{avgacc_avg:.2f}" if all_avgacc else "-"
    draft_s = f"{draft_avg:.1f}" if all_draft else "-"
    verify_s = f"{verify_avg:.1f}" if all_verify else "-"
    print(f"{'** OVERALL AVERAGE **':<28s} {tp_avg:10.2f} {ttft_avg:10.0f} {tpot_avg:10.2f} "
          f"{tok_avg:7.0f} {acc_s:>8s} {avgacc_s:>7s} {draft_s:>9s} {verify_s:>10s} {'':>10s}")
    print("=" * 140)

    if all_tp:
        print(f"\n  Throughput:  avg={tp_avg:.2f} tok/s  std={tp_std:.2f}  "
              f"min={min(all_tp):.2f}  max={max(all_tp):.2f}")
    if all_acc:
        print(f"  Acceptance:  avg={acc_avg:.1f}%  std={acc_std:.1f}%")

    # Save global averages for markdown (before per-category loop shadows them)
    overall_tp_avg, overall_tp_std = tp_avg, tp_std
    overall_ttft_avg = ttft_avg
    overall_tpot_avg = tpot_avg
    overall_tok_avg = tok_avg
    overall_acc_avg, overall_acc_std = acc_avg, acc_std
    overall_avgacc_avg = avgacc_avg
    overall_draft_avg = draft_avg
    overall_verify_avg = verify_avg

    # ═════════════════════════════════════════════════════════════
    # Generated text output
    # ═════════════════════════════════════════════════════════════
    print()
    print("=" * 140)
    print("  GENERATED OUTPUT (first run per category)")
    print("=" * 140)

    for cat in cats_seen:
        cat_results = [r for r in ok_results if r.category == cat]
        if not cat_results:
            continue
        r = cat_results[0]
        print(f"\n--- [{r.category}] ---")
        print(f"Prompt: {r.prompt[:120]}{'...' if len(r.prompt) > 120 else ''}")
        print(f"Output ({r.tokens} tokens, {r.throughput:.1f} tok/s):")
        # Show first 500 chars of output
        text = r.output_text if r.output_text else "(no output captured)"
        if len(text) > 500:
            print(text[:500] + "\n  [...truncated]")
        else:
            print(text)

    # ═════════════════════════════════════════════════════════════
    # Save reports
    # ═════════════════════════════════════════════════════════════
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = REPORT_DIR / f"dflash_input_categories_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# DFlash Input Category Benchmark\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Config**: {args.quant} | max_tokens={args.max_tokens} | "
                f"block_size={args.block_size} | runs={args.runs}  \n")
        f.write(f"**Device**: {DEVICE} | Model: Qwen3.5-4B | Draft: Qwen3.5-4B-DFlash  \n")
        f.write(f"**Total time**: {elapsed:.0f}s\n\n")

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Category | tok/s | TTFT (ms) | TPOT (ms) | Tokens | Accept % | Avg Acc | Draft (ms) | Verify (ms) |\n")
        f.write("|----------|------:|----------:|----------:|-------:|---------:|--------:|-----------:|------------:|\n")
        for cat in cats_seen:
            cat_results = [r for r in ok_results if r.category == cat]
            if not cat_results:
                f.write(f"| {cat} | FAILED | | | | | | | |\n")
                continue
            if len(cat_results) == 1:
                r = cat_results[0]
                acc = f"{r.accept_rate*100:.1f}" if r.accept_rate > 0 else "-"
                avg = f"{r.avg_accepted:.2f}" if r.avg_accepted > 0 else "-"
                dms = f"{r.draft_avg_ms:.1f}" if r.draft_avg_ms > 0 else "-"
                vms = f"{r.verify_avg_ms:.1f}" if r.verify_avg_ms > 0 else "-"
                f.write(f"| {cat} | {r.throughput:.2f} | {r.ttft_ms:.0f} | {r.tpot_ms:.2f} | "
                        f"{r.tokens} | {acc} | {avg} | {dms} | {vms} |\n")
            else:
                tp_avg, tp_std = mean_std([r.throughput for r in cat_results])
                ttft_avg, _ = mean_std([r.ttft_ms for r in cat_results])
                tpot_avg, _ = mean_std([r.tpot_ms for r in cat_results])
                tok_avg = sum(r.tokens for r in cat_results) / len(cat_results)
                acc_vals = [r.accept_rate * 100 for r in cat_results if r.accept_rate > 0]
                acc_avg, _ = mean_std(acc_vals) if acc_vals else (0, 0)
                acc_str = f"{acc_avg:.1f}" if acc_vals else "-"
                f.write(f"| {cat} | {tp_avg:.2f}±{tp_std:.1f} | {ttft_avg:.0f} | {tpot_avg:.2f} | "
                        f"{tok_avg:.0f} | {acc_str} | - | - | - |\n")

        # Overall average row in markdown
        f.write(f"| **OVERALL AVERAGE** | **{overall_tp_avg:.2f}** | {overall_ttft_avg:.0f} | {overall_tpot_avg:.2f} | "
                f"{overall_tok_avg:.0f} | {overall_acc_avg:.1f} | {overall_avgacc_avg:.2f} | {overall_draft_avg:.1f} | {overall_verify_avg:.1f} |\n"
                if all_tp else "")

        # Aggregate
        f.write("\n## Aggregate\n\n")
        if all_tp:
            f.write(f"- **Throughput**: avg={overall_tp_avg:.2f} tok/s, std={overall_tp_std:.2f}, "
                    f"min={min(all_tp):.2f}, max={max(all_tp):.2f}\n")
        if all_acc:
            f.write(f"- **Acceptance**: avg={overall_acc_avg:.1f}%, std={overall_acc_std:.1f}%\n")
        f.write(f"- **Total runs**: {len(ok_results)} succeeded / {total_runs} total\n")

        # Generated outputs
        f.write("\n## Generated Outputs\n")
        for cat in cats_seen:
            cat_results = [r for r in ok_results if r.category == cat]
            if not cat_results:
                continue
            r = cat_results[0]
            f.write(f"\n### {r.category}\n\n")
            f.write(f"**Prompt**: {r.prompt}\n\n")
            f.write(f"**Stats**: {r.tokens} tokens, {r.throughput:.1f} tok/s, "
                    f"accept={r.accept_rate*100:.1f}%\n\n")
            text = r.output_text if r.output_text else "(no output captured)"
            f.write(f"**Output**:\n\n{text}\n\n---\n")

    print(f"\n[Report] {md_path}")
    print(f"[Total]  {elapsed:.0f}s ({total_runs} runs)")


if __name__ == "__main__":
    main()
