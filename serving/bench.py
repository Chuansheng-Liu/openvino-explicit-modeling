#!/usr/bin/env python3
"""
Qwen3.5 Performance Benchmark — OpenAI-compatible API server
Tests: TTFT, generation TPS, prompt processing TPS across context lengths
Also: tool calling capability tests

Adapted from OVMS benchmark for lightweight FastAPI server.
Usage:
  python bench.py [mode] [--all-ctx] [--url URL]
  mode: perf, tools, all (default: all)
"""

import json
import time
import sys
import urllib.request
import urllib.error
import math

# Bypass macOS system proxy
_no_proxy_handler = urllib.request.ProxyHandler({})
_opener = urllib.request.build_opener(_no_proxy_handler)
urllib.request.install_opener(_opener)

BASE_URL = "http://localhost:8000"
API_CHAT = "/v1/chat/completions"

ALL_MODELS = []  # auto-detected from server

# Context sizes: 1K, 2K, 4K, 8K for small models; extend with --all-ctx
CONTEXT_SIZES_SAFE = [1000, 2000, 4000, 8000]
CONTEXT_SIZES_ALL = [1000, 2000, 4000, 8000, 16000, 32000, 64000]
CONTEXT_SIZES = CONTEXT_SIZES_SAFE


def generate_filler_text(target_tokens):
    """Generate filler text targeting a specific token count.
    Approximate: 1 CJK char ~ 0.6-0.7 tokens, 1 English word ~ 1-1.5 tokens.
    Use mixed content for more realistic estimation.
    """
    base = (
        "这是一段用于测试大型语言模型在不同上下文长度下推理性能的填充文本。"
        "The quick brown fox jumps over the lazy dog. "
        "自然语言处理是人工智能领域中一个重要的研究方向，涵盖了文本分类、信息抽取、机器翻译等多个子任务。"
        "OpenVINO Model Server provides high-performance inference serving for AI models. "
        "在深度学习时代，Transformer架构已经成为NLP领域的主流方法，BERT、GPT等模型取得了显著成果。"
        "Intel Panther Lake processors feature integrated Arc GPU for efficient AI inference. "
    )
    # ~1.5 chars per token for mixed CJK/English
    chars_needed = int(target_tokens * 1.5)
    repeats = max(1, chars_needed // len(base) + 1)
    return (base * repeats)[:chars_needed]


def chat_streaming(model, messages, max_tokens=200, temperature=0.7, timeout=600):
    """Send a streaming chat request. Returns timing and token metrics."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{API_CHAT}",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    ttft = None
    content_parts = []
    chunk_count = 0
    gen_tokens = 0
    prompt_tokens = 0
    finish_reason = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8")
                buffer += line
                while "\n" in buffer:
                    line_str, buffer = buffer.split("\n", 1)
                    line_str = line_str.strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue
                    json_str = line_str[5:].strip()
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text and ttft is None:
                        ttft = time.perf_counter() - t_start
                    if text:
                        content_parts.append(text)
                        chunk_count += 1
                    usage = chunk.get("usage")
                    if usage:
                        gen_tokens = usage.get("completion_tokens", 0)
                        prompt_tokens = usage.get("prompt_tokens", 0)
                    fr = chunk.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
    except Exception as e:
        return {"error": str(e), "total_time": time.perf_counter() - t_start}

    total_time = time.perf_counter() - t_start
    content = "".join(content_parts)
    if gen_tokens == 0:
        gen_tokens = chunk_count

    return {
        "ttft": round(ttft, 3) if ttft else None,
        "total_time": round(total_time, 3),
        "gen_tokens": gen_tokens,
        "prompt_tokens": prompt_tokens,
        "content": content,
        "finish_reason": finish_reason,
    }


def chat_nonstreaming(
    model, messages, max_tokens=1, temperature=0.1, timeout=600, tools=None
):
    """Send a non-streaming chat request. Returns timing, content, tool_calls."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{API_CHAT}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
    except Exception as e:
        return {"error": str(e), "total_time": time.perf_counter() - t_start}

    total_time = time.perf_counter() - t_start
    result = json.loads(body)
    usage = result.get("usage", {})
    msg = result.get("choices", [{}])[0].get("message", {})
    return {
        "total_time": round(total_time, 3),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "gen_tokens": usage.get("completion_tokens", 0),
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls", []),
        "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
    }


def chat_streaming_with_tools(
    model, messages, max_tokens=2000, temperature=0.1, timeout=120, tools=None
):
    """Streaming chat that accumulates tool_calls from SSE delta chunks."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{API_CHAT}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t_start = time.perf_counter()
    tc_acc = {}  # index -> {id, function: {name, arguments}}
    content_parts = []
    finish_reason = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = ""
            for raw_line in resp:
                line = raw_line.decode("utf-8")
                buffer += line
                while "\n" in buffer:
                    line_str, buffer = buffer.split("\n", 1)
                    line_str = line_str.strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue
                    json_str = line_str[5:].strip()
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    text = delta.get("content", "")
                    if text:
                        content_parts.append(text)
                    for tcd in delta.get("tool_calls") or []:
                        idx = tcd.get("index", 0)
                        if idx not in tc_acc:
                            tc_acc[idx] = {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tcd.get("id"):
                            tc_acc[idx]["id"] = tcd["id"]
                        fn = tcd.get("function", {})
                        if fn.get("name"):
                            tc_acc[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tc_acc[idx]["function"]["arguments"] += fn["arguments"]
                    fr = chunk.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
    except Exception as e:
        return {"error": str(e), "total_time": time.perf_counter() - t_start}

    total_time = time.perf_counter() - t_start
    tool_calls = [tc_acc[i] for i in sorted(tc_acc.keys())]
    return {
        "total_time": round(total_time, 3),
        "content": "".join(content_parts),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
    }


# ═══════════════════════════════════════════════════════════════
# Part 1: LLM Performance Benchmark
# ═══════════════════════════════════════════════════════════════


def bench_llm(model, context_sizes):
    print(f"\n{'=' * 90}")
    print(f"  LLM Benchmark: {model}")
    print(f"{'=' * 90}")

    # Warm up
    print(f"  Warming up...", end=" ", flush=True)
    r = chat_streaming(
        model, [{"role": "user", "content": "Hi"}], max_tokens=50, timeout=120
    )
    if "error" in r:
        print(f"FAILED: {r['error']}")
        return None
    print("OK")

    print(
        f"\n  {'Target':>8} {'PromptTok':>10} {'TTFT':>8} {'GenTok':>7} {'GenTime':>8} "
        f"{'GenTPS':>8} {'PromptTPS':>10} {'Finish':>8}"
    )
    print(f"  {'-' * 82}")

    results = []
    for target in context_sizes:
        filler = generate_filler_text(target)
        # Use a prompt that reliably triggers long output for throughput measurement.
        # /nothink disables Qwen3.5 thinking mode so ALL generated tokens are content.
        bench_prompt = (
            "Write a comprehensive tutorial about building a REST API with Python Flask. "
            "Cover: project setup, routing, request handling, database integration with SQLAlchemy, "
            "authentication with JWT, error handling, testing with pytest, and deployment. "
            "Include code examples for each section. Be thorough and detailed."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Respond in English. Answer directly and thoroughly."},
            {"role": "user", "content": f"/nothink\nContext for reference:\n{filler}\n\n{bench_prompt}"},
        ]

        # Step 1: Get prompt token count via non-streaming with max_tokens=1
        print(f"  {target:>8} ", end="", flush=True)
        nr = chat_nonstreaming(model, messages, max_tokens=1, timeout=1800)
        if "error" in nr:
            print(f"{'ERROR':>10} {nr['error'][:60]}")
            results.append({"ctx": target, "error": nr["error"]})
            continue
        prompt_tokens = nr["prompt_tokens"]
        print(f"{prompt_tokens:>10} ", end="", flush=True)

        # Step 2: Streaming request — measure TTFT only
        output_tokens = 4096
        r = chat_streaming(
            model, messages, max_tokens=output_tokens, temperature=0.0, timeout=3600
        )

        if "error" in r:
            print(f"{'ERROR':>8} {r['error'][:60]}")
            results.append(
                {"ctx": target, "prompt_tokens": prompt_tokens, "error": r["error"]}
            )
            continue

        ttft = r["ttft"]
        total_time = r["total_time"]
        finish = r["finish_reason"] or "?"

        if ttft is None:
            print(f"{'NO_TTFT':>8} (no content generated, total={total_time:.1f}s)")
            results.append(
                {
                    "ctx": target,
                    "prompt_tokens": prompt_tokens,
                    "ttft": None,
                    "gen_tps": None,
                }
            )
            continue

        # Use total completion_tokens (including thinking) for accurate TPS
        # Streaming chunks filter thinking, so gen_tokens from chunks is unreliable.
        # total_time - ttft = decode time for ALL tokens (think + content)
        gen_tokens = r["gen_tokens"]  # from usage if available, else chunk_count
        gen_time = total_time - ttft

        # If gen_tokens seems too low (think filtering), estimate from total time
        # A rough check: if gen_tokens < 50 and gen_time > 2s, likely undercounted
        if gen_tokens < 50 and gen_time > 2.0:
            # Fallback: use non-streaming to get accurate completion_tokens
            nr2 = chat_nonstreaming(model, messages, max_tokens=output_tokens, temperature=0.0, timeout=3600)
            if "error" not in nr2 and nr2.get("gen_tokens", 0) > gen_tokens:
                gen_tokens = nr2["gen_tokens"]
                gen_time = nr2.get("total_time", gen_time) - ttft

        gen_tps = gen_tokens / gen_time if gen_time > 0.01 else 0
        prompt_tps = prompt_tokens / ttft if ttft > 0.01 else 0

        print(
            f"{ttft:>7.2f}s {gen_tokens:>7} {gen_time:>7.1f}s "
            f"{gen_tps:>7.1f} {prompt_tps:>10.0f} {finish:>8}"
        )
        results.append(
            {
                "ctx": target,
                "prompt_tokens": prompt_tokens,
                "ttft": round(ttft, 3),
                "gen_tokens": gen_tokens,
                "gen_time": round(gen_time, 2),
                "gen_tps": round(gen_tps, 1),
                "prompt_tps": round(prompt_tps, 0),
                "finish": finish,
            }
        )

    return results


# ═══════════════════════════════════════════════════════════════
# Part 2: Tool Calling Test
# ═══════════════════════════════════════════════════════════════


def _parse_tool_calls(tc):
    """Extract function names and parsed arguments from tool_calls list."""
    fns = [t["function"]["name"] for t in tc]
    try:
        args = [
            json.loads(t["function"]["arguments"])
            if isinstance(t["function"]["arguments"], str)
            else t["function"]["arguments"]
            for t in tc
        ]
    except Exception:
        args = [{}] * len(tc)
    return fns, args


def _validate_tool_result(test_name, tc, content, expected_fn, expected_check):
    """Validate tool call result against expectations.
    Returns (status, detail_string).
    """
    content = content or ""
    has_tool_pattern = "<tool_call>" in content or "<function=" in content

    # --- no_tool_needed: must NOT produce any tool call ---
    if test_name == "no_tool_needed":
        if not tc and not has_tool_pattern and content:
            return "PASS", "content=yes (no tool call)"
        if tc or has_tool_pattern:
            return "FAIL", "incorrectly called tool"
        return "FAIL", "no content"

    # --- All other tests: expect at least one tool call ---
    if not tc:
        if has_tool_pattern:
            idx = content.find("<tool_call>")
            raw = content[idx : idx + 200] if idx >= 0 else content[-100:]
            return "PARSE_FAIL", f"parser failed: {raw[:80]}"
        return "FAIL", f"no tool_calls, content={content[:50]}"

    fns, args = _parse_tool_calls(tc)
    detail = f"calls={fns} args={args}"

    # Validate function name
    if expected_fn and expected_fn not in fns:
        return "WRONG_FN", f"expected {expected_fn}, got {fns}"

    # Validate expected checks (min call count, specific arg values)
    if expected_check:
        min_calls = expected_check.get("_min_calls")
        if min_calls and len(tc) < min_calls:
            return "PARTIAL", f"expected >={min_calls} calls, got {len(tc)}: {detail}"
        target = expected_fn or fns[0]
        matched_args = [a for f, a in zip(fns, args) if f == target]
        for key, val in expected_check.items():
            if key.startswith("_"):
                continue
            if matched_args and val is not None:
                actual = matched_args[0].get(key)
                if actual is not None and str(actual) != str(val):
                    return "WRONG_ARG", f"{key}: expected={val}, got={actual} | {detail}"

    return "PASS", detail[:70]


def bench_tools(models):
    print(f"\n{'=' * 90}")
    print(f"  Tool Calling Benchmark")
    print(f"{'=' * 90}")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的当前天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"},
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "在互联网上搜索信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "limit": {"type": "integer", "description": "返回结果数量"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "数学表达式"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    # (name, query, description, expected_fn, expected_check)
    # expected_fn: required function name (None = special handling)
    # expected_check: dict with arg key/value to validate, _min_calls for count
    tests = [
        ("basic_call", "北京今天天气如何？",
         "get_weather(city=北京)", "get_weather", None),
        ("multi_select", "帮我搜索一下最新的Python 3.14教程",
         "search_web", "search_web", None),
        ("no_tool_needed", "1+1等于几？请直接回答",
         "no tool call", None, None),
        ("parallel_tools", "帮我查一下北京和上海的天气",
         "get_weather >=2 times", "get_weather", {"_min_calls": 2}),
        ("complex_params", "搜索 OpenVINO 教程，只返回3条结果",
         "search_web(limit=3)", "search_web", {"limit": 3}),
        ("multi_turn", None,
         "multi-turn tool call", None, None),
        ("state_leak_regr", None,
         "3x basic_call, all must PASS", None, None),
        ("streaming_tool", "What's the weather in London?",
         "streaming tool call", "get_weather", None),
        ("english_call", "What's the weather in Tokyo?",
         "English prompt", "get_weather", None),
    ]

    all_results = {}
    for model in models:
        print(f"\n  --- {model} ---")
        model_results = []

        for test_name, query, desc, expected_fn, expected_check in tests:

            # ── multi_turn: two-turn tool calling ──
            if test_name == "multi_turn":
                msgs1 = [{"role": "user", "content": "北京今天天气如何？"}]
                r1 = chat_nonstreaming(model, msgs1, max_tokens=2000,
                                       temperature=0.1, timeout=120, tools=tools)
                if "error" in r1:
                    status, detail = "ERROR", r1["error"][:60]
                    print(f"  {test_name:20s}  {status:12s}  {detail}")
                    model_results.append({"test": test_name, "status": status})
                    continue

                tc1 = r1.get("tool_calls", [])
                content1 = r1.get("content", "")
                if not tc1:
                    status, detail = _validate_tool_result(
                        "basic_call", tc1, content1, "get_weather", None)
                    print(f"  {test_name:20s}  {status:12s}  turn1 failed: {detail}")
                    model_results.append({"test": test_name, "status": status})
                    continue

                fns1, _ = _parse_tool_calls(tc1)
                msgs2 = msgs1 + [
                    {"role": "assistant", "content": content1, "tool_calls": tc1},
                    {"role": "tool",
                     "content": json.dumps({"temperature": "22°C",
                                            "condition": "晴天", "humidity": "45%"}),
                     "tool_call_id": tc1[0].get("id", "call_1")},
                    {"role": "user", "content": "那上海呢？"},
                ]
                r2 = chat_nonstreaming(model, msgs2, max_tokens=2000,
                                       temperature=0.1, timeout=120, tools=tools)
                if "error" in r2:
                    status = "PARTIAL"
                    detail = f"turn1=OK({fns1}), turn2=ERROR: {r2['error'][:40]}"
                else:
                    tc2 = r2.get("tool_calls", [])
                    if tc2:
                        fns2, _ = _parse_tool_calls(tc2)
                        if "get_weather" in fns2:
                            status = "PASS"
                        else:
                            status = "WRONG_FN"
                        detail = f"turn1={fns1}, turn2={fns2}"
                    else:
                        status = "PARTIAL"
                        detail = f"turn1={fns1}, turn2=text({r2.get('content','')[:40]})"

                print(f"  {test_name:20s}  {status:12s}  {detail[:70]}")
                model_results.append({"test": test_name, "status": status,
                                      "time": r1["total_time"]})
                continue

            # ── state_leak_regr: same request 3 times ──
            if test_name == "state_leak_regr":
                all_pass = True
                details = []
                for i in range(3):
                    msgs = [{"role": "user", "content": "北京今天天气如何？"}]
                    r = chat_nonstreaming(model, msgs, max_tokens=2000,
                                          temperature=0.0, timeout=120, tools=tools)
                    if "error" in r:
                        details.append(f"#{i+1}=ERROR")
                        all_pass = False
                        continue
                    tc = r.get("tool_calls", [])
                    content = r.get("content", "")
                    s, d = _validate_tool_result(
                        "basic_call", tc, content, "get_weather", None)
                    details.append(f"#{i+1}={s}")
                    if s != "PASS":
                        all_pass = False

                status = "PASS" if all_pass else "FAIL"
                detail = ", ".join(details)
                print(f"  {test_name:20s}  {status:12s}  {detail}")
                model_results.append({"test": test_name, "status": status})
                continue

            # ── streaming_tool: streaming mode tool call ──
            if test_name == "streaming_tool":
                msgs = [{"role": "user", "content": query}]
                r = chat_streaming_with_tools(
                    model, msgs, max_tokens=2000, temperature=0.0,
                    timeout=120, tools=tools)
                if "error" in r:
                    status, detail = "ERROR", r["error"][:60]
                else:
                    tc = r.get("tool_calls", [])
                    content = r.get("content", "")
                    status, detail = _validate_tool_result(
                        test_name, tc, content, expected_fn, expected_check)

                print(f"  {test_name:20s}  {status:12s}  {detail[:70]}")
                model_results.append({"test": test_name, "status": status,
                                      "time": r.get("total_time")})
                continue

            # ── Standard tests ──
            messages = [{"role": "user", "content": query}]
            r = chat_nonstreaming(
                model, messages, max_tokens=2000, temperature=0.1,
                timeout=120, tools=tools)
            if "error" in r:
                print(f"  {test_name:20s}  ERROR: {r['error'][:50]}")
                model_results.append(
                    {"test": test_name, "status": "ERROR", "error": r["error"]})
                continue

            tc = r.get("tool_calls", [])
            content = r.get("content", "")
            status, detail = _validate_tool_result(
                test_name, tc, content, expected_fn, expected_check)

            print(f"  {test_name:20s}  {status:12s}  {detail[:70]}")
            model_results.append({"test": test_name, "status": status,
                                  "time": r["total_time"]})

        all_results[model] = model_results

    return all_results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════


def main():
    global BASE_URL
    # Usage: python bench.py [mode] [--all-ctx] [--url URL]

    # Parse args (must be before model detection to set BASE_URL correctly)
    mode = "all"
    ctx_sizes = CONTEXT_SIZES_SAFE
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("perf", "tools", "all"):
            mode = arg
        elif arg == "--all-ctx":
            ctx_sizes = CONTEXT_SIZES_ALL
        elif arg == "--url" and i + 1 < len(argv):
            i += 1
            BASE_URL = argv[i]
        elif arg.startswith("--url="):
            BASE_URL = arg.split("=", 1)[1]
        i += 1

    # Auto-detect which models are loaded
    try:
        req = urllib.request.Request(f"{BASE_URL}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            models_data = json.loads(resp.read().decode())
        available = [m["id"] for m in models_data.get("data", [])]
    except Exception as e:
        print(f"ERROR: Cannot connect to {BASE_URL}: {e}")
        sys.exit(1)

    MODELS = available if available else ALL_MODELS
    if not MODELS:
        print(f"ERROR: No models found at {BASE_URL}")
        sys.exit(1)

    print(f"{'#' * 90}")
    print(f"  Qwen3.5 Performance Benchmark — {BASE_URL}")
    print(f"  Models available: {', '.join(MODELS)}")
    print(f"  Context sizes: {', '.join(f'{s // 1000}K' for s in ctx_sizes)}")
    print(f"  Mode: {mode}")
    print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 90}")

    all_llm = {}
    tool_results = {}

    if mode in ("all", "perf"):
        for model in MODELS:
            all_llm[model] = bench_llm(model, ctx_sizes)

    if mode in ("all", "tools"):
        tool_results = bench_tools(MODELS)

    # ─── Summary ───
    print(f"\n\n{'#' * 90}")
    print("  PERFORMANCE SUMMARY")
    print(f"{'#' * 90}")

    if all_llm:
        models_tested = [m for m in MODELS if m in all_llm and all_llm[m]]

        # TTFT table
        print(f"\n  TTFT (seconds):")
        header = f"  {'Context':>8}"
        for m in models_tested:
            short = m.replace("Qwen3.5-", "").replace("-IR", "")
            header += f"  {short:>14}"
        print(header)
        print(f"  {'-' * (8 + 16 * len(models_tested))}")
        for ctx in ctx_sizes:
            row = f"  {ctx // 1000:>7}K"
            for m in models_tested:
                res = all_llm.get(m, [])
                match = [r for r in res if r.get("ctx") == ctx] if res else []
                if match and match[0].get("ttft") is not None:
                    row += f"  {match[0]['ttft']:>13.2f}s"
                elif match and "error" in match[0]:
                    row += f"  {'ERROR':>14}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)

        # GenTPS table
        print(f"\n  Generation TPS (tok/s):")
        header = f"  {'Context':>8}"
        for m in models_tested:
            short = m.replace("Qwen3.5-", "").replace("-IR", "")
            header += f"  {short:>14}"
        print(header)
        print(f"  {'-' * (8 + 16 * len(models_tested))}")
        for ctx in ctx_sizes:
            row = f"  {ctx // 1000:>7}K"
            for m in models_tested:
                res = all_llm.get(m, [])
                match = [r for r in res if r.get("ctx") == ctx] if res else []
                if match and match[0].get("gen_tps") is not None:
                    row += f"  {match[0]['gen_tps']:>14.1f}"
                elif match and "error" in match[0]:
                    row += f"  {'ERROR':>14}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)

        # Prompt TPS table
        print(f"\n  Prompt Processing TPS (tok/s):")
        header = f"  {'Context':>8}"
        for m in models_tested:
            short = m.replace("Qwen3.5-", "").replace("-IR", "")
            header += f"  {short:>14}"
        print(header)
        print(f"  {'-' * (8 + 16 * len(models_tested))}")
        for ctx in ctx_sizes:
            row = f"  {ctx // 1000:>7}K"
            for m in models_tested:
                res = all_llm.get(m, [])
                match = [r for r in res if r.get("ctx") == ctx] if res else []
                if match and match[0].get("prompt_tps") is not None:
                    row += f"  {match[0]['prompt_tps']:>14.0f}"
                elif match and "error" in match[0]:
                    row += f"  {'ERROR':>14}"
                else:
                    row += f"  {'N/A':>14}"
            print(row)

    if tool_results:
        print(f"\n  Tool Calling Results:")
        for model, results in tool_results.items():
            short = model.replace("Qwen3.5-", "").replace("-IR", "")
            passed = sum(1 for r in results if r.get("status") == "PASS")
            parse_fail = sum(1 for r in results if r.get("status") == "PARSE_FAIL")
            total = len(results)
            print(
                f"    {short}: {passed}/{total} PASS, {parse_fail} PARSE_FAIL"
            )

    print(f"\n  Done. {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
