#!/usr/bin/env python3
"""Quick curl-style e2e latency benchmark for car assistant intent recognition.

Uses unbuffered socket streaming for accurate per-token timing.
Two modes:
  1. Single-prompt (default): repeat same prompt N times, measure cold/warm TTFT.
  2. Prefix-cache (--prefix-cache): send N different user prompts sequentially,
     all sharing the same system prompt.  Measures prefix cache hit speedup.

Usage:
    python scripts/curl_benchmark.py                          # single-prompt mode
    python scripts/curl_benchmark.py --host 192.168.1.100 --port 8093
    python scripts/curl_benchmark.py --runs 5 --verbose
    python scripts/curl_benchmark.py --prompt "turn on the AC"
    python scripts/curl_benchmark.py --prefix-cache            # prefix-cache mode
    python scripts/curl_benchmark.py --prefix-cache --num-prompts 5
"""

from __future__ import annotations

import argparse
import json
import socket
import time


# ── System prompt (imported from canonical source) ────────────────────

from car_system_prompt import CAR_SYSTEM_PROMPT, make_car_status

# Backward-compat aliases (used by prefix_stress.py)
SYSTEM_PROMPT = CAR_SYSTEM_PROMPT
CAR_STATUS = make_car_status()

def stream_request(host: str, port: int, messages: list,
                   max_tokens: int = 200) -> dict:
    """Send streaming request via raw socket, return per-token timing."""
    payload = json.dumps({
        "model": "default",
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()

    request = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Content-Type: application/json\r\n"
        f"Authorization: Bearer test\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"\r\n"
    ).encode() + payload

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect((host, port))

    t0 = time.perf_counter()
    sock.sendall(request)

    raw = b""
    first_token_time = None
    last_token_time = None
    tokens: list[str] = []
    header_done = False

    while True:
        try:
            chunk = sock.recv(1)
            if not chunk:
                break
        except socket.timeout:
            break

        raw += chunk

        if not header_done:
            if b"\r\n\r\n" in raw:
                header_done = True
                _, raw = raw.split(b"\r\n\r\n", 1)
            continue

        while b"\n" in raw:
            line, raw = raw.split(b"\n", 1)
            line = line.decode(errors="replace").strip()
            if line == "data: [DONE]":
                sock.close()
                total = (time.perf_counter() - t0) * 1000
                ttft = (first_token_time - t0) * 1000 if first_token_time else total
                last_t = (last_token_time - t0) * 1000 if last_token_time else total
                decode_ms = last_t - ttft if len(tokens) > 1 else 0
                tps = (len(tokens) - 1) / (decode_ms / 1000) if decode_ms > 0 else 0
                return {
                    "ttft_ms": ttft, "last_token_ms": last_t, "e2e_ms": total,
                    "tokens": len(tokens), "tps": tps,
                    "output": "".join(tokens),
                }

            if line.startswith("data: "):
                try:
                    obj = json.loads(line[6:])
                    choices = obj.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            now = time.perf_counter()
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                            tokens.append(content)
                except Exception:
                    pass

    sock.close()
    total = (time.perf_counter() - t0) * 1000
    return {
        "ttft_ms": (first_token_time - t0) * 1000 if first_token_time else total,
        "last_token_ms": total, "e2e_ms": total,
        "tokens": len(tokens), "tps": 0,
        "output": "".join(tokens),
    }


PREFIX_CACHE_PROMPTS = [
    ("开灯", "vehicle_light"),
    ("打开空调", "hvac_action"),
    ("把温度调到26度", "hvac_temp"),
    ("打开车门", "vehicle_door"),
    ("播放音乐", "music_play_action"),
    ("切换到运动模式", "vehicle_drive_mode"),
    ("打开后备箱", "vehicle_trunk"),
    ("关闭车窗", "vehicle_window"),
    ("下一首歌", "music_up_down"),
    ("打开YouTube", "gui_open_app"),
]


def run_single_prompt_benchmark(args):
    """Original mode: repeat the same prompt N times, measure cold/warm."""
    user_content = args.prompt
    if not args.no_car_status:
        user_content = f"{CAR_STATUS}\n<user_input>{args.prompt}</user_input>"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    print(f"{'═'*65}")
    print(f"  Car Assistant E2E Latency Benchmark — Single Prompt")
    print(f"  Server: {args.host}:{args.port}  |  Runs: {args.runs}")
    print(f"  Prompt: \"{args.prompt}\"")
    print(f"{'═'*65}")
    print(f"\n  {'Run':<10s} {'TTFT':>8s} {'Last Token':>11s} {'E2E':>8s} {'Tokens':>7s} {'TPS':>7s}")
    print(f"  {'─'*10} {'─'*8} {'─'*11} {'─'*8} {'─'*7} {'─'*7}")

    ttfts, e2es = [], []
    for i in range(args.runs):
        label = "cold" if i == 0 else "warm"
        r = stream_request(args.host, args.port, messages)
        tag = f"Run {i+1} ({label})"
        print(f"  {tag:<10s} {r['ttft_ms']:>7.0f}ms {r['last_token_ms']:>10.0f}ms {r['e2e_ms']:>7.0f}ms {r['tokens']:>7d} {r['tps']:>6.1f}")
        if args.verbose:
            print(f"    Output: {r['output'][:200]}")
        ttfts.append(r["ttft_ms"])
        e2es.append(r["e2e_ms"])
        time.sleep(0.5)

    print(f"\n  {'─'*55}")
    if len(ttfts) > 1:
        warm_ttfts = ttfts[1:]
        warm_e2es = e2es[1:]
        avg_ttft = sum(warm_ttfts) / len(warm_ttfts)
        avg_e2e = sum(warm_e2es) / len(warm_e2es)
        print(f"  Cold:  TTFT={ttfts[0]:.0f}ms  E2E={e2es[0]:.0f}ms")
        print(f"  Warm:  TTFT avg={avg_ttft:.0f}ms  E2E avg={avg_e2e:.0f}ms  (n={len(warm_ttfts)})")
    else:
        print(f"  TTFT={ttfts[0]:.0f}ms  E2E={e2es[0]:.0f}ms")
    print(f"{'═'*65}")


def _run_prefix_phase(args, prompts, phase_name):
    """Run a list of prompts and return results + print table."""
    print(f"\n  {'#':<4s} {'Prompt':<22s} {'TTFT':>8s} {'E2E':>8s} {'Tokens':>7s} {'TPS':>7s} {'Output'}")
    print(f"  {'─'*4} {'─'*22} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*30}")

    results = []
    quality_issues = []
    for i, (prompt, expected_intent) in enumerate(prompts):
        user_content = f"{CAR_STATUS}\n<user_input>{prompt}</user_input>"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        r = stream_request(args.host, args.port, messages)

        # Strip <think>...</think> tags for validation
        import re as _re
        clean = _re.sub(r"<think>.*?</think>\s*", "", r["output"], flags=_re.DOTALL).strip()
        # Remove markdown code fences
        if clean.startswith("```"):
            lines = clean.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            clean = "\n".join(lines).strip()

        output_short = clean.replace("\n", " ")[:40]

        # Proper JSON intent validation
        intent_ok = False
        if not clean:
            quality_issues.append(f"  ⚠ #{i+1} '{prompt}': empty output")
        else:
            try:
                parsed_json = json.loads(clean)
                intent_ok = parsed_json.get("intent") == expected_intent
            except json.JSONDecodeError:
                # Try extracting first {...} block
                if "{" in clean:
                    start = clean.index("{")
                    depth = 0
                    for idx in range(start, len(clean)):
                        if clean[idx] == "{":
                            depth += 1
                        elif clean[idx] == "}":
                            depth -= 1
                            if depth == 0:
                                try:
                                    pj = json.loads(clean[start:idx+1])
                                    intent_ok = pj.get("intent") == expected_intent
                                except Exception:
                                    pass
                                break

        marker = "✅" if intent_ok else "❌"

        print(f"  {i+1:<4d} {prompt:<22s} {r['ttft_ms']:>7.0f}ms {r['e2e_ms']:>7.0f}ms {r['tokens']:>7d} {r['tps']:>6.1f} {marker} {output_short}")
        results.append(r)
        time.sleep(0.3)

    ttfts = [r["ttft_ms"] for r in results]
    print(f"\n  {'─'*60}")
    avg_ttft = sum(ttfts) / len(ttfts)
    min_ttft = min(ttfts)
    max_ttft = max(ttfts)
    print(f"  {phase_name}: TTFT avg={avg_ttft:.0f}ms  min={min_ttft:.0f}ms  max={max_ttft:.0f}ms  (n={len(ttfts)})")

    # Perf consistency warning
    if len(ttfts) >= 3:
        for j, t in enumerate(ttfts):
            if t > avg_ttft * 3 and t > 500:
                print(f"  ⚠ PERF OUTLIER: #{j+1} TTFT={t:.0f}ms > 3x avg ({avg_ttft:.0f}ms)")

    for qi in quality_issues:
        print(qi)

    return results


def run_prefix_cache_benchmark(args):
    """Prefix-cache benchmark with two phases:

    Phase 1 — Batch (one session): All N prompts sent sequentially in one
    invocation.  The first few requests build the prefix snapshot; later
    requests should hit the cache and show lower TTFT.

    Phase 2 — Standalone (one prompt per invocation): Each prompt is sent
    individually.  Since the prefix snapshot persists in the server session,
    every request should hit the cache.
    """
    prompts = PREFIX_CACHE_PROMPTS[:args.num_prompts]

    print(f"{'═'*70}")
    print(f"  Prefix-Cache Benchmark")
    print(f"  Server: {args.host}:{args.port}  |  Prompts: {len(prompts)}")
    print(f"  Shared prefix: system_prompt + car_status (~{len(SYSTEM_PROMPT) + len(CAR_STATUS)} chars)")
    print(f"{'═'*70}")

    # Phase 1: all prompts in one batch
    print(f"\n  ▶ Phase 1: Batch — all {len(prompts)} prompts in one session")
    batch_results = _run_prefix_phase(args, prompts, "Batch")

    # Phase 2: each prompt as a standalone request
    print(f"\n  ▶ Phase 2: Standalone — one prompt per invocation")
    standalone_results = _run_prefix_phase(args, prompts, "Standalone")

    # Combined summary
    batch_ttfts = [r["ttft_ms"] for r in batch_results]
    standalone_ttfts = [r["ttft_ms"] for r in standalone_results]
    # For batch, skip warmup requests (first 3) for "cached" avg
    cached_start = min(3, len(batch_ttfts) - 1)
    batch_cached = batch_ttfts[cached_start:] if cached_start < len(batch_ttfts) else batch_ttfts
    batch_cached_avg = sum(batch_cached) / len(batch_cached) if batch_cached else 0
    standalone_avg = sum(standalone_ttfts) / len(standalone_ttfts)

    print(f"\n  {'━'*60}")
    print(f"  Summary")
    print(f"  {'━'*60}")
    print(f"  Batch cached (req {cached_start+1}+):  TTFT avg={batch_cached_avg:.0f}ms")
    print(f"  Standalone (all {len(prompts)}):      TTFT avg={standalone_avg:.0f}ms")
    print(f"{'═'*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick e2e latency benchmark for car assistant (streaming, unbuffered socket)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for single-prompt mode (default: 3)")
    parser.add_argument("--prompt", default="open the door", help="User prompt for single-prompt mode")
    parser.add_argument("--verbose", action="store_true", help="Print model output")
    parser.add_argument("--no-car-status", action="store_true", help="Skip car_status in user message")
    parser.add_argument("--prefix-cache", action="store_true",
                        help="Run prefix-cache benchmark: sequential different prompts sharing system prompt")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of prompts for prefix-cache mode (default: 10, max: 10)")
    args = parser.parse_args()
    args.num_prompts = min(args.num_prompts, len(PREFIX_CACHE_PROMPTS))

    if args.prefix_cache:
        run_prefix_cache_benchmark(args)
    else:
        run_single_prompt_benchmark(args)


if __name__ == "__main__":
    main()
