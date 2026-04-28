#!/usr/bin/env python3
"""
Car Assistant DFlash CLI Test
Runs car assistant intent prompts through modeling_qwen3_5_dflash.exe
to measure acceptance rate and throughput per test case.

Usage:
  python car_assistant_test_cli.py [--target-quant INT4_SYM] [--draft-quant FP16]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Paths ──
DEFAULT_EXE = (
    Path(r"D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release")
    / "modeling_qwen3_5_dflash.exe"
)
DEFAULT_TARGET = Path(r"C:\data\models\Huggingface\Qwen3.5-9B")
DEFAULT_DRAFT = Path(r"C:\data\models\Huggingface\Qwen3.5-9B-DFlash")

# ── System prompt (imported from canonical source) ────────────────────

from car_system_prompt import CAR_SYSTEM_PROMPT, make_car_status

CAR_STATUS = make_car_status()
# ── Test cases (single-turn, text only — no VL for exe) ──
TEST_CASES = [
    ("greeting",           "你好呀",                "chat",               None),
    ("open driver window", "打开主驾车窗",          "vehicle_window",     {"action": "on", "position": "front_left"}),
    ("AC off",             "关闭空调",              "hvac_action",        {"action": "off"}),
    ("play music",         "播放音乐",              "music_play_action",  {"action": "on"}),
    ("sport mode",         "切换到运动模式",        "vehicle_drive_mode", {"mode": "sport"}),
    ("next song",          "下一首",                "music_up_down",      {"action": "next"}),
    ("open YouTube",       "打开YouTube",           "gui_open_app",       {"app_name": "YouTube"}),
    ("raise temp",         "升高温度",              "hvac_temp",          {"area": "all", "temp": 26}),
    ("seat heating on",    "打开座椅加热",          "hvac_seat_heating",  {"area": "front_left", "level": 3}),
    ("go home screen",     "回到桌面",              "gui_go_home",        None),
    ("color to red",       "把车换成红色",          "vehicle_color_change", {"color": "red"}),
    ("close all windows",  "把所有车窗关上",        "vehicle_window",     {"action": "off", "position": "all"}),
    ("seat ventilation",   "打开座椅通风",          "hvac_seat_ventilation", {"area": "front_left", "level": 3}),
    ("set temp to 20",     "把温度调到20度",        "hvac_temp",          {"area": "all", "temp": 20}),
    ("goodbye",            "好的谢谢，再见",        "chat",               None),
]


def build_prompt(user_input: str) -> str:
    """Build a single-turn Qwen3.5 chat-formatted prompt."""
    return (
        f"<|im_start|>system\n{CAR_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{CAR_STATUS}\n<user_input>{user_input}</user_input><|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def parse_output(stdout: str) -> dict:
    """Parse modeling_qwen3_5_dflash.exe output for metrics."""
    result = {}

    m = re.search(r"Output token size:\s*(\d+)", stdout)
    result["tokens"] = int(m.group(1)) if m else 0

    m = re.search(r"Throughput:\s*([\d.]+)\s*tokens/s", stdout)
    result["throughput"] = float(m.group(1)) if m else 0.0

    m = re.search(r"TTFT:\s*([\d.]+)\s*ms", stdout)
    result["ttft"] = float(m.group(1)) if m else 0.0

    m = re.search(r"TPOT:\s*([\d.]+)\s*ms/token", stdout)
    result["tpot"] = float(m.group(1)) if m else 0.0

    m = re.search(r"Acceptance rate:\s*([\d.]+)", stdout)
    result["acceptance"] = float(m.group(1)) if m else 0.0

    m = re.search(r"Avg accepted per step:\s*([\d.]+)", stdout)
    result["avg_per_step"] = float(m.group(1)) if m else 0.0

    m = re.search(r"Draft steps:\s*(\d+)", stdout)
    result["draft_steps"] = int(m.group(1)) if m else 0

    # Extract generated text between [Output] and [Generation Complete]
    m = re.search(r"\[Output\]\s*\n(.*?)\n\[Generation Complete\]", stdout, re.DOTALL)
    result["output"] = m.group(1).strip() if m else ""

    return result


def extract_json(text: str) -> dict | None:
    """Extract first JSON object from text."""
    if "{" not in text:
        return None
    start = text.index("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def run_test(exe: Path, target: Path, draft: Path, device: str,
             target_quant: str, draft_quant: str, prompt: str,
             max_tokens: int = 128) -> dict:
    """Run one test case through the exe."""
    cmd = [
        str(exe),
        str(target),
        str(draft),
        prompt,
        device,
        str(max_tokens),
        "0",  # block_size (use default)
        target_quant,
        draft_quant,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            encoding="utf-8", errors="replace"
        )
        return parse_output(result.stdout + result.stderr)
    except subprocess.TimeoutExpired:
        return {"tokens": 0, "throughput": 0, "ttft": 0, "tpot": 0,
                "acceptance": 0, "avg_per_step": 0, "draft_steps": 0,
                "output": "TIMEOUT"}


def main():
    parser = argparse.ArgumentParser(
        description="Car Assistant DFlash CLI Acceptance Test",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--target-quant", default="INT4_SYM",
                        choices=["FP16", "INT4_SYM", "INT4_ASYM"])
    parser.add_argument("--draft-quant", default="FP16",
                        choices=["FP16", "INT4_SYM", "INT4_ASYM"])
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--think", action="store_true",
                        help="Enable thinking mode (default: disabled)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case indices (0-based) or 'all'")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.cases and args.cases != "all":
        indices = [int(x) for x in args.cases.split(",")]
        cases = [TEST_CASES[i] for i in indices]

    # Disable thinking by default (DFlash exe uses env var)
    if not args.think:
        os.environ["OV_GENAI_DISABLE_THINKING"] = "1"

    print("=" * 72)
    print("  Car Assistant DFlash CLI Test")
    print("=" * 72)
    print(f"  Target quant : {args.target_quant}")
    print(f"  Draft quant  : {args.draft_quant}")
    print(f"  Device       : {args.device}")
    print(f"  Thinking     : {'enabled' if args.think else 'disabled'}")
    print(f"  Cases        : {len(cases)}")
    print(f"  Max tokens   : {args.max_tokens}")
    print("=" * 72)

    results = []
    for i, (name, user_input, expect_intent, expect_fields) in enumerate(cases):
        prompt = build_prompt(user_input)
        is_chat = (expect_intent == "chat")

        print(f"\n[{i+1}/{len(cases)}] {name}")
        print("-" * 60)

        r = run_test(args.exe, args.target, args.draft, args.device,
                     args.target_quant, args.draft_quant, prompt, args.max_tokens)

        # Check correctness
        ok = True
        detail = ""
        if is_chat:
            parsed = extract_json(r["output"])
            if parsed and "intent" in parsed:
                ok = False
                detail = "expected chat, got JSON intent"
        else:
            parsed = extract_json(r["output"])
            if parsed is None:
                ok = False
                detail = "no valid JSON found"
            elif parsed.get("intent") != expect_intent:
                ok = False
                detail = f"intent={parsed.get('intent')}, expected {expect_intent}"
            elif expect_fields:
                args_got = parsed.get("arguments", {})
                for k, v in expect_fields.items():
                    if args_got.get(k) != v:
                        ok = False
                        detail += f" {k}={args_got.get(k)}!={v}"

        status = "\033[92mPASS\033[0m" if ok else "\033[91mFAIL\033[0m"
        print(f"  {status} | {r['throughput']:.1f} t/s | TTFT={r['ttft']:.0f}ms | "
              f"Accept={r['acceptance']*100:.1f}% | Avg/step={r['avg_per_step']:.1f} | "
              f"Steps={r['draft_steps']} | Tokens={r['tokens']}")
        if detail:
            print(f"  Detail: {detail}")

        # Show first 120 chars of output
        out_preview = r["output"][:120].replace("\n", " ")
        print(f"  Output: {out_preview}...")

        results.append({"name": name, "ok": ok, **r})

    # Summary table
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    print(f"{'#':>3} {'Name':<25} {'Pass':>5} {'tok/s':>7} {'TTFT':>8} {'Accept%':>8} "
          f"{'Avg/stp':>8} {'Steps':>6} {'Tokens':>7}")
    print("-" * 100)

    total_accept = []
    total_tps = []
    passed = 0
    for i, r in enumerate(results):
        tag = "\033[92m  OK\033[0m" if r["ok"] else "\033[91mFAIL\033[0m"
        print(f"{i+1:3d} {r['name']:<25} {tag} {r['throughput']:7.1f} "
              f"{r['ttft']:7.0f}ms {r['acceptance']*100:7.1f}% "
              f"{r['avg_per_step']:7.1f} {r['draft_steps']:6d} {r['tokens']:7d}")
        if r["acceptance"] > 0:
            total_accept.append(r["acceptance"])
        if r["throughput"] > 0:
            total_tps.append(r["throughput"])
        if r["ok"]:
            passed += 1

    print("-" * 100)
    avg_accept = sum(total_accept) / len(total_accept) * 100 if total_accept else 0
    avg_tps = sum(total_tps) / len(total_tps) if total_tps else 0
    print(f"    {'AVERAGE':<25}       {avg_tps:7.1f}          {avg_accept:7.1f}%")
    print(f"\n  Quality: {passed}/{len(results)} correct ({passed/len(results)*100:.0f}%)")
    print(f"  Config: target={args.target_quant} draft={args.draft_quant}")


if __name__ == "__main__":
    main()
