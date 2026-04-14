#!/usr/bin/env python3
"""Sanity tests for ov_serve: text, VL, mixed sequences, and tool calling.

Usage:
    python scripts/sanity_test.py                          # default localhost:8080
    python scripts/sanity_test.py --base-url http://host:8093
    python scripts/sanity_test.py --image scripts/test.jpg # custom image
    python scripts/sanity_test.py --image2 scripts/test_chart.png  # second image
    python scripts/sanity_test.py --no-proxy               # bypass proxy
    python scripts/sanity_test.py --verbose                 # print full responses

Tests (run sequentially to verify prefix cache / session reuse):
  1. text              — simple text question
  2. vl                — single image VL
  3. vl + vl + text    — two VL requests then text
  4. text + vl + text  — text, then VL, then text
  5. multi-image VL + text + multi-image VL
  6. tool calling      — single, multi-turn, no-tools baseline
  7. text              — final text (verify no state corruption)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "test.jpg"
DEFAULT_IMAGE2 = SCRIPT_DIR / "test_chart.png"

# ── Helpers ──────────────────────────────────────────────────────────

def image_to_data_uri(path: Path) -> str:
    data = path.read_bytes()
    suffix = path.suffix.lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        suffix.lstrip("."), "image/jpeg"
    )
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json",
                                  "Authorization": "Bearer test"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def chat(base_url: str, messages: list, *, tools: list | None = None,
         max_tokens: int = 256, model: str = "qwen3.5") -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    return post_json(f"{base_url}/v1/chat/completions", payload)


def chat_stream(base_url: str, messages: list, *, tools: list | None = None,
                max_tokens: int = 256, model: str = "qwen3.5") -> dict:
    """Streaming chat — reassemble SSE chunks into a single response dict."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer test"},
    )
    content_parts: list[str] = []
    tool_calls_accum: dict[int, dict] = {}  # index → {id, type, function:{name, arguments}}
    finish_reason = None
    usage = {}

    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            chunk = json.loads(data_str)
            if "usage" in chunk:
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                for tc in delta.get("tool_calls", []):
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_accum:
                        tool_calls_accum[idx] = {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    entry = tool_calls_accum[idx]
                    if tc.get("id"):
                        entry["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        entry["function"]["name"] = fn["name"]
                    if fn.get("arguments"):
                        entry["function"]["arguments"] += fn["arguments"]

    tool_calls_list = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
    effective_finish = finish_reason or "stop"
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
                "tool_calls": tool_calls_list if tool_calls_list else None,
            },
            "finish_reason": effective_finish,
        }],
        "usage": usage,
    }


# ── Result helpers ───────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str, resp: dict):
        self.name = name
        self.resp = resp
        c = resp["choices"][0]
        self.finish_reason = c["finish_reason"]
        self.msg = c["message"]
        self.content = self.msg.get("content", "") or ""
        self.tool_calls = self.msg.get("tool_calls") or []
        u = resp.get("usage", {})
        self.prompt_tokens = u.get("prompt_tokens", 0)
        self.completion_tokens = u.get("completion_tokens", 0)
        perf = u.get("performance", {})
        self.tps = perf.get("throughput_tps", 0)
        self.ttft = perf.get("ttft_ms", 0)

    def summary(self, verbose: bool = False) -> str:
        lines = [
            f"  finish_reason : {self.finish_reason}",
            f"  tokens        : {self.prompt_tokens} prompt + {self.completion_tokens} gen",
            f"  throughput    : {self.tps:.1f} t/s, ttft: {self.ttft:.0f}ms",
        ]
        if self.tool_calls:
            for tc in self.tool_calls:
                fn = tc.get("function", {})
                lines.append(f"  tool_call     : {fn.get('name','')}({fn.get('arguments','')})")
        if verbose:
            text = self.content[:300] + ("..." if len(self.content) > 300 else "")
            lines.append(f"  content       : {text}")
        return "\n".join(lines)


# ── Test definitions ─────────────────────────────────────────────────

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}

BOOK_FLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight between two airports",
        "parameters": {
            "type": "object",
            "properties": {
                "departure": {
                    "type": "object",
                    "description": "Departure details",
                    "properties": {
                        "airport": {"type": "string", "description": "IATA airport code"},
                        "date": {"type": "string", "description": "Date in YYYY-MM-DD"},
                    },
                    "required": ["airport", "date"],
                },
                "arrival": {
                    "type": "object",
                    "description": "Arrival details",
                    "properties": {
                        "airport": {"type": "string", "description": "IATA airport code"},
                    },
                    "required": ["airport"],
                },
                "passengers": {"type": "integer", "description": "Number of passengers"},
            },
            "required": ["departure", "arrival", "passengers"],
        },
    },
}

GET_TIME_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current time in a specific timezone",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "Timezone name, e.g. Asia/Shanghai"},
            },
            "required": ["timezone"],
        },
    },
}


def run_tests(base_url: str, img1: Path, img2: Path, verbose: bool) -> list[tuple[str, bool, str]]:
    img1_uri = image_to_data_uri(img1)
    img2_uri = image_to_data_uri(img2)
    results: list[tuple[str, bool, str]] = []
    test_num = 0

    def run(name: str, messages, *, tools=None, max_tokens=256,
            expect_finish="stop", expect_tool_name=None, expect_content_contains=None,
            expect_tool_count=None, stream=False):
        nonlocal test_num
        test_num += 1
        label = f"[{test_num}] {name}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            if stream:
                resp = chat_stream(base_url, messages, tools=tools, max_tokens=max_tokens)
            else:
                resp = chat(base_url, messages, tools=tools, max_tokens=max_tokens)
        except Exception as e:
            msg = f"FAIL: {e}"
            print(f"  {msg}")
            results.append((label, False, msg))
            return None
        elapsed = time.time() - t0
        tr = TestResult(name, resp)
        print(tr.summary(verbose))
        print(f"  wall time     : {elapsed:.1f}s")

        passed = True
        fail_reasons = []

        if tr.finish_reason != expect_finish:
            passed = False
            fail_reasons.append(f"finish_reason={tr.finish_reason}, expected={expect_finish}")
        if expect_tool_name:
            names = [tc.get("function", {}).get("name", "") for tc in tr.tool_calls]
            if expect_tool_name not in names:
                passed = False
                fail_reasons.append(f"tool name '{expect_tool_name}' not in {names}")
        if expect_tool_count is not None:
            actual = len(tr.tool_calls)
            if actual != expect_tool_count:
                passed = False
                fail_reasons.append(f"tool_call count={actual}, expected={expect_tool_count}")
        if expect_content_contains:
            if expect_content_contains.lower() not in tr.content.lower():
                passed = False
                fail_reasons.append(f"content missing '{expect_content_contains}'")
        if tr.completion_tokens < 1:
            passed = False
            fail_reasons.append("0 completion tokens")

        status = "PASS" if passed else f"FAIL: {'; '.join(fail_reasons)}"
        print(f"  result        : {status}")
        results.append((label, passed, status))
        return tr

    # ── 1. Text ──
    run("text: capital question",
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        max_tokens=16)

    # ── 2. VL ──
    run("vl: describe image",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "text", "text": "What do you see in this image? Answer in one sentence."}
        ]}],
        max_tokens=64)

    # ── 3. VL + VL + Text ──
    run("vl+vl+text: first VL (3a)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "text", "text": "List the main objects in one sentence."}
        ]}],
        max_tokens=64)

    run("vl+vl+text: second VL (3b)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img2_uri}},
            {"type": "text", "text": "Describe this image in one sentence."}
        ]}],
        max_tokens=64)

    run("vl+vl+text: text (3c)",
        [{"role": "user", "content": "What is 15 * 17? Answer with just the number."}],
        max_tokens=16)

    # ── 4. Text + VL + Text ──
    run("text+vl+text: text (4a)",
        [{"role": "user", "content": "Name three programming languages. Just list them."}],
        max_tokens=32)

    run("text+vl+text: VL (4b)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "text", "text": "What is the dominant color? One word."}
        ]}],
        max_tokens=32)

    run("text+vl+text: text (4c)",
        [{"role": "user", "content": "What is the boiling point of water in Celsius? Just the number."}],
        max_tokens=16)

    # ── 5. Multi-image VL + Text + Multi-image VL ──
    run("multi-img+text+multi-img: two images (5a)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "image_url", "image_url": {"url": img2_uri}},
            {"type": "text", "text": "Are these two images the same? Answer yes or no."}
        ]}],
        max_tokens=32)

    run("multi-img+text+multi-img: text (5b)",
        [{"role": "user", "content": "What is the speed of light in km/s? Just the number."}],
        max_tokens=16)

    run("multi-img+text+multi-img: two images (5c)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img2_uri}},
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "text", "text": "Which image contains a chart? Answer first or second."}
        ]}],
        max_tokens=32)

    # ── 6. Tool calling ──
    # 6a: Single tool call
    run("tool: single call (6a)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the weather in Beijing?"}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # 6b: Multi-turn tool call (send result back, ask follow-up)
    run("tool: multi-turn (6b)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the weather in Beijing?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"},
         {"role": "user", "content": "And what about Shanghai?"}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # 6c: Tool call → text answer (model uses tool result to answer)
    run("tool: result → answer (6c)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the weather in Beijing?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}"},
         {"role": "user", "content": "Great, tell me the temperature."}],
        tools=[WEATHER_TOOL],
        max_tokens=128)

    # 6d: Multiple tools available
    run("tool: multi-tool choice (6d)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Search the web for latest news about OpenVINO."}],
        tools=[WEATHER_TOOL, SEARCH_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="web_search")

    # 6e: No tools — should not generate tool calls
    run("tool: no tools baseline (6e)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is 2+2?"}],
        max_tokens=64)

    # 6f: Parallel tool calls — ask for two cities at once
    run("tool: parallel calls (6f)",
        [{"role": "system", "content": "You are a helpful assistant. When asked about multiple cities, call get_weather once for EACH city separately."},
         {"role": "user", "content": "Get the weather in Beijing and Shanghai. Call the tool for each city."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # 6g: Streaming tool call
    run("tool: streaming (6g)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the weather in Tokyo?"}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
        stream=True,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # 6h: Nested/complex arguments
    run("tool: nested args (6h)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user",
          "content": "Book a flight from Beijing (PEK) to Shanghai (PVG) on 2026-05-01 for 2 passengers."}],
        tools=[BOOK_FLIGHT_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="book_flight")

    # 6i: Long tool result — feed back a large result and get summary
    run("tool: long result (6i)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Search for OpenVINO 2025."},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "web_search",
                                       "arguments": "{\"query\": \"OpenVINO 2025\"}"}}]},
         {"role": "tool", "content": json.dumps({
             "results": [
                 {"title": f"Result {i}", "snippet": f"OpenVINO 2025 feature {i}: " + "x" * 50}
                 for i in range(10)
             ]
         })},
         {"role": "user", "content": "How many results did you find? Answer with just the number."}],
        tools=[SEARCH_TOOL],
        max_tokens=32,
        expect_content_contains="10")

    # 6j: Three tools available — pick the right one
    run("tool: 3-tool choice (6j)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What time is it in Tokyo?"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL, GET_TIME_TOOL],
        max_tokens=256,
        expect_finish="tool_calls",
        expect_tool_name="get_current_time")

    # 6k: Streaming multi-turn tool call
    run("tool: streaming multi-turn (6k)",
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the weather in Beijing?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 15, \"condition\": \"cloudy\"}"},
         {"role": "user", "content": "Now check London."}],
        tools=[WEATHER_TOOL],
        max_tokens=256,
        stream=True,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # ── 7. Final text (verify no state corruption) ──
    run("text: final check",
        [{"role": "user", "content": "Say hello in Japanese, Chinese, and Korean. Be brief."}],
        max_tokens=64)

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sanity test suite for ov_serve: text, VL, mixed, and tool calling.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080",
                        help="Base URL of ov_serve (default: http://127.0.0.1:8080)")
    parser.add_argument("--image", type=str, default=None,
                        help=f"Primary test image (default: {DEFAULT_IMAGE.name})")
    parser.add_argument("--image2", type=str, default=None,
                        help=f"Secondary test image (default: {DEFAULT_IMAGE2.name})")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Unset proxy environment variables.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full response content.")
    args = parser.parse_args()

    if args.no_proxy:
        for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                     "no_proxy", "NO_PROXY"):
            os.environ.pop(var, None)

    img1 = Path(args.image) if args.image else DEFAULT_IMAGE
    img2 = Path(args.image2) if args.image2 else DEFAULT_IMAGE2
    for img in (img1, img2):
        if not img.exists():
            print(f"ERROR: Image not found: {img}", file=sys.stderr)
            sys.exit(1)

    base = args.base_url.rstrip("/")
    print(f"ov_serve sanity tests — {base}")
    print(f"Images: {img1.name}, {img2.name}")

    # Health check
    try:
        r = post_json(f"{base}/v1/models", {})
        print(f"Server OK: {r.get('data', [{}])[0].get('id', '?')}")
    except Exception:
        # /v1/models may be GET
        try:
            req = urllib.request.Request(f"{base}/v1/models")
            with urllib.request.urlopen(req, timeout=5) as resp:
                r = json.loads(resp.read().decode())
                print(f"Server OK: {r.get('data', [{}])[0].get('id', '?')}")
        except Exception as e:
            print(f"WARNING: Server health check failed: {e}")

    t0 = time.time()
    results = run_tests(base, img1, img2, args.verbose)
    total_time = time.time() - t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(results)} tests in {total_time:.1f}s")
    print(f"{'='*60}")
    passed = 0
    failed = 0
    for label, ok, status in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {label}: {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
