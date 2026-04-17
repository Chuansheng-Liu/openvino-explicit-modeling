#!/usr/bin/env python3
"""Sanity tests for ov_serve: text, VL, mixed sequences, tool calling, and Nebula integration.

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
  7. hermes agent      — multi-step agent simulation
  8. nebula            — Automotive.AI-1.7.1 integration patterns
  9. text              — final text (verify no state corruption)
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "test.jpg"
DEFAULT_IMAGE2 = SCRIPT_DIR / "test_ocr2.png"

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
         max_tokens: int = 256, model: str = "qwen3.5",
         temperature: float | None = None, top_p: float | None = None) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    return post_json(f"{base_url}/v1/chat/completions", payload)


def chat_stream(base_url: str, messages: list, *, tools: list | None = None,
                max_tokens: int = 256, model: str = "qwen3.5",
                temperature: float | None = None, top_p: float | None = None) -> dict:
    """Streaming chat — reassemble SSE chunks into a single response dict."""
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
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

    # Strip stray quote chars that the streaming tokeniser may leak into names
    for entry in tool_calls_accum.values():
        entry["function"]["name"] = entry["function"]["name"].strip('"')
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
        self.prefix_cached_tokens = perf.get("prefix_cached_tokens", 0)

    def summary(self, verbose: bool = False) -> str:
        lines = [
            f"  finish_reason : {self.finish_reason}",
            f"  tokens        : {self.prompt_tokens} prompt + {self.completion_tokens} gen",
            f"  throughput    : {self.tps:.1f} t/s, ttft: {self.ttft:.0f}ms",
        ]
        if self.prefix_cached_tokens > 0:
            lines.append(f"  prefix cache  : {self.prefix_cached_tokens} tokens reused")
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
            expect_tool_count=None, expect_prefix_cache_min=None, stream=False,
            temperature=None, top_p=None):
        nonlocal test_num
        test_num += 1
        label = f"[{test_num}] {name}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
        t0 = time.time()
        try:
            if stream:
                resp = chat_stream(base_url, messages, tools=tools, max_tokens=max_tokens,
                                   temperature=temperature, top_p=top_p)
            else:
                resp = chat(base_url, messages, tools=tools, max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p)
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
        if expect_prefix_cache_min is not None:
            if tr.prefix_cached_tokens < expect_prefix_cache_min:
                passed = False
                fail_reasons.append(
                    f"prefix_cached_tokens={tr.prefix_cached_tokens}, "
                    f"expected>={expect_prefix_cache_min}")
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
            {"type": "text", "text": "What type of content is in this image? Answer in one word."}
        ]}],
        max_tokens=16)

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
        max_tokens=128)

    run("multi-img+text+multi-img: text (5b)",
        [{"role": "user", "content": "What is the speed of light in km/s? Just the number."}],
        max_tokens=16)

    run("multi-img+text+multi-img: two images (5c)",
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img2_uri}},
            {"type": "image_url", "image_url": {"url": img1_uri}},
            {"type": "text", "text": "Which image contains a chart? Answer with one word: first or second."}
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
    # Don't pass tools= so the model cannot re-call; it must answer from the result.
    run("tool: result → answer (6c)",
        [{"role": "system", "content": "You are a helpful assistant. The tool has already been called and the result is provided below. Answer the user's question directly using the tool result."},
         {"role": "user", "content": "What is the weather in Beijing?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\", \"humidity\": 45}"},
         {"role": "user", "content": "Great, tell me the temperature."}],
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
    # Note: model may call a tool or answer directly depending on tool proximity
    run("tool: long result (6i)",
        [{"role": "system", "content": "You are a helpful assistant. Answer the user's question directly based on the tool results provided."},
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
        max_tokens=64,
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

    # ── 7. Hermes agent simulation ──
    # Simulates the multi-turn agent loop pattern used by NousResearch Hermes.
    # These tests verify the server handles realistic agent workloads.

    HERMES_SYSTEM = (
        "You are a function calling AI model. You are provided with function "
        "signatures within <tools></tools> XML tags. You may call one or more "
        "functions to assist with the user query. Don't make assumptions about "
        "what values to plug into functions. Here are the available tools:\n"
    )

    CALC_TOOL = {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"],
            },
        },
    }

    # 7a: Agent first turn — model should call tool, not answer directly
    run("agent: first turn tool call (7a)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "What is the current weather in Tokyo and what time is it there?"}],
        tools=[WEATHER_TOOL, GET_TIME_TOOL],
        max_tokens=512,
        expect_finish="tool_calls")

    # 7b: Agent multi-step — after first tool result, should call second tool
    run("agent: chain second tool (7b)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "What is the current weather in Tokyo and what time is it there?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Tokyo\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"partly cloudy\", \"humidity\": 60}"},
         {"role": "user", "content": "Now get the time."}],
        tools=[WEATHER_TOOL, GET_TIME_TOOL],
        max_tokens=512,
        expect_finish="tool_calls",
        expect_tool_name="get_current_time")

    # 7c: Agent final answer — after all tool results, should give text answer
    run("agent: final answer (7c)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "What is the current weather in Tokyo and what time is it there?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Tokyo\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 18, \"condition\": \"partly cloudy\", \"humidity\": 60}"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_1", "type": "function",
                          "function": {"name": "get_current_time",
                                       "arguments": "{\"timezone\": \"Asia/Tokyo\"}"}}]},
         {"role": "tool", "content": "{\"time\": \"2026-04-14T12:30:00+09:00\", \"timezone\": \"JST\"}"},
         {"role": "user", "content": "Summarize both results in one sentence."}],
        tools=[WEATHER_TOOL, GET_TIME_TOOL],
        max_tokens=256,
        expect_finish="stop",
        expect_content_contains="Tokyo")

    # 7d: Agent with calculation — different tool domain
    run("agent: calculate (7d)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "What is 1234 * 5678?"}],
        tools=[CALC_TOOL, WEATHER_TOOL],
        max_tokens=512,
        expect_finish="tool_calls",
        expect_tool_name="calculate")

    # 7e: Agent — tool error handling (tool returns error, model should explain or retry)
    # With tool definitions near end-of-context, model may retry the tool call — both are valid.
    run("agent: tool error recovery (7e)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "What is the weather in Beijing?"},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"error\": \"Service temporarily unavailable. Please try again later.\"}"},
         {"role": "user", "content": "The tool failed. Explain the error to me in one sentence."}],
        max_tokens=256)

    # 7f: Agent long context — 5-turn conversation with tool calls
    run("agent: long context 5-turn (7f)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "Check weather in Beijing."},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_0", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Beijing\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"},
         {"role": "assistant", "content": "Beijing is 22°C and sunny."},
         {"role": "user", "content": "Now check Shanghai."},
         {"role": "assistant", "content": "",
          "tool_calls": [{"id": "call_1", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": "{\"city\": \"Shanghai\"}"}}]},
         {"role": "tool", "content": "{\"temperature\": 25, \"condition\": \"cloudy\"}"},
         {"role": "assistant", "content": "Shanghai is 25°C and cloudy."},
         {"role": "user", "content": "Which city is warmer? Answer in one sentence based on the information above. Do NOT call any tools."}],
        tools=[WEATHER_TOOL],
        max_tokens=128,
        expect_finish="stop")

    # 7g: Agent streaming — full agent turn in streaming mode
    run("agent: streaming tool call (7g)",
        [{"role": "system", "content": HERMES_SYSTEM},
         {"role": "user", "content": "Search the web for the latest OpenVINO release."}],
        tools=[SEARCH_TOOL, WEATHER_TOOL],
        max_tokens=512,
        stream=True,
        expect_finish="tool_calls",
        expect_tool_name="web_search")

    # 7h: Long system prompt + tools — regression test for Hermes agent hang.
    # When the system prompt is very large (>2K tokens), tool definitions must
    # still be visible to the model.  This failed before the fix that moved
    # <tools> to a separate system message near end-of-context.
    LONG_HERMES_SYSTEM = (
        HERMES_SYSTEM + "\n\n"
        "## Agent Guidelines\n\n"
        + "\n".join(
            f"Guideline {i}: When the user asks you to perform task type {i}, "
            f"always use the most appropriate tool. Be thorough and precise."
            for i in range(30)
        )
        + "\n\n## Memory\n\n"
        + "\n".join(f"- User preference {i}: setting_{i}=value_{i}" for i in range(20))
        + "\n\n## Session Context\n\nThis is a long-running agent session.\n"
    )

    run("agent: long sysprompt + tools (7h)",
        [{"role": "system", "content": LONG_HERMES_SYSTEM},
         {"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=[WEATHER_TOOL, SEARCH_TOOL, GET_TIME_TOOL],
        max_tokens=512,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # ── 8. Nebula Automotive.AI integration ──
    # Mirror exact request patterns from Nebula Automotive.AI-1.7.1 to verify
    # ov_serve works as a drop-in LLM backend for the Nebula agent framework.

    NEBULA_SYSTEM = (
        "你是人工智能助手问问, 你可以根据训练所的的知识, "
        "用中文回答用户的问题, 或者用自然语言描述用户提供的图片. "
        "你可以自行判断是否使用其他工具来获取信息或者完成用户的任务."
    )

    # 8a: Chinese system prompt + streaming text
    run("nebula: chinese streaming (8a)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": "北京是哪个国家的首都？用一句话回答。"}],
        stream=True,
        max_tokens=64)

    # 8b: Multi-turn 5-turn conversation (streaming, Nebula accumulates history)
    run("nebula: multi-turn 5-turn (8b)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": "你好"},
         {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
         {"role": "user", "content": "今天天气怎么样？"},
         {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。"},
         {"role": "user", "content": "我在上海"},
         {"role": "assistant", "content": "上海是个美丽的城市！"},
         {"role": "user", "content": "上海有什么好玩的？"},
         {"role": "assistant", "content": "上海有很多好玩的地方，比如外滩、东方明珠、豫园等。"},
         {"role": "user", "content": "外滩在哪里？用一句话回答。"}],
        stream=True,
        max_tokens=128)

    # 8c: Vision + Chinese description (Nebula's primary VL use case, streaming)
    run("nebula: vision chinese (8c)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": [
             {"type": "image_url", "image_url": {"url": img1_uri}},
             {"type": "text", "text": "用中文描述这张图片，一句话。"}
         ]}],
        stream=True,
        max_tokens=128)

    # 8d: Multi-turn with image mid-conversation
    run("nebula: mid-conversation image (8d)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": "你好，我想了解一些图片。"},
         {"role": "assistant", "content": "好的，请发送你想了解的图片。"},
         {"role": "user", "content": [
             {"type": "image_url", "image_url": {"url": img1_uri}},
             {"type": "text", "text": "这张图片里有什么？一句话回答。"}
         ]}],
        max_tokens=128)

    # 8e: Null content in assistant message (Nebula edge case)
    run("nebula: null assistant content (8e)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": "你好"},
         {"role": "assistant", "content": None},
         {"role": "user", "content": "1加1等于几？只回答数字。"}],
        max_tokens=16)

    # 8f: Nebula exact parameters (temperature=0.7, top_p=0.8)
    run("nebula: temp=0.7 top_p=0.8 (8f)",
        [{"role": "system", "content": NEBULA_SYSTEM},
         {"role": "user", "content": "用中文解释什么是人工智能，一句话。"}],
        temperature=0.7,
        top_p=0.8,
        max_tokens=128)

    # 8g: Concurrent requests — Nebula may serve multiple users simultaneously
    test_num += 1
    label = f"[{test_num}] nebula: concurrent requests (8g)"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()

    def _nebula_req(prompt):
        return chat(base_url,
                    [{"role": "system", "content": NEBULA_SYSTEM},
                     {"role": "user", "content": prompt}],
                    max_tokens=32, temperature=0.7, top_p=0.8)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_nebula_req, "1加1等于几？只回答数字。")
        f2 = pool.submit(_nebula_req, "2加3等于几？只回答数字。")
        try:
            r1 = f1.result(timeout=120)
            r2 = f2.result(timeout=120)
            elapsed = time.time() - t0
            c1 = r1["choices"][0]["message"].get("content", "") or ""
            c2 = r2["choices"][0]["message"].get("content", "") or ""
            ok1 = r1["choices"][0]["finish_reason"] in ("stop", "length")
            ok2 = r2["choices"][0]["finish_reason"] in ("stop", "length")
            passed = ok1 and ok2
            fail_reasons = []
            if not ok1:
                fail_reasons.append(f"req1 finish={r1['choices'][0]['finish_reason']}")
            if not ok2:
                fail_reasons.append(f"req2 finish={r2['choices'][0]['finish_reason']}")
            msg = "PASS" if passed else f"FAIL: {'; '.join(fail_reasons)}"
            print(f"  response 1    : {c1[:80]}")
            print(f"  response 2    : {c2[:80]}")
            print(f"  wall time     : {elapsed:.1f}s")
            print(f"  result        : {msg}")
            results.append((label, passed, msg))
        except Exception as e:
            elapsed = time.time() - t0
            msg = f"FAIL: {e}"
            print(f"  wall time     : {elapsed:.1f}s")
            print(f"  result        : {msg}")
            results.append((label, False, msg))

    # ── 10. Prefix cache reuse ──
    # Test that multi-turn conversations reuse KV cache from previous turns.
    # The second request in each pair should have prefix_cached_tokens > 0.

    # 10a. Simple two-turn text conversation
    run("prefix-cache: text turn 1 (cold)",
        [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=16)

    run("prefix-cache: text turn 2 (should hit cache)",
        [{"role": "user", "content": "What is 2+2? Answer with just the number."},
         {"role": "assistant", "content": "<think>\n\n</think>\n\n4"},
         {"role": "user", "content": "And what is 3+3? Answer with just the number."}],
        max_tokens=16,
        expect_prefix_cache_min=10)

    # 10b. Two-turn with tools (verifies tool_defs at fixed position)
    # Include a system message so tool_defs are anchored after it (not fallback path)
    TOOLS_SYS = "You are a helpful weather assistant."
    tr_tool1 = run("prefix-cache: tools turn 1 (cold)",
        [{"role": "system", "content": TOOLS_SYS},
         {"role": "user", "content": "What is the weather in Paris?"}],
        tools=[WEATHER_TOOL],
        max_tokens=64,
        expect_finish="tool_calls",
        expect_tool_name="get_weather")

    # Build turn 2 from actual turn 1 response so cached tokens match exactly
    tool1_msg = {"role": "assistant", "content": tr_tool1.content if tr_tool1 else ""}
    if tr_tool1 and tr_tool1.tool_calls:
        tool1_msg["tool_calls"] = tr_tool1.tool_calls
    tool1_id = (tr_tool1.tool_calls[0].get("id", "call_1")
                if tr_tool1 and tr_tool1.tool_calls else "call_1")

    run("prefix-cache: tools turn 2 (should hit cache)",
        [{"role": "system", "content": TOOLS_SYS},
         {"role": "user", "content": "What is the weather in Paris?"},
         tool1_msg,
         {"role": "tool", "tool_call_id": tool1_id,
          "content": '{"temp": 18, "condition": "sunny"}'},
         {"role": "user", "content": "And what about London?"}],
        tools=[WEATHER_TOOL],
        max_tokens=64,
        expect_finish="tool_calls",
        expect_tool_name="get_weather",
        expect_prefix_cache_min=10)

    # 10c. Streaming two-turn (verify prefix cache works in stream mode too)
    tr_stream1 = run("prefix-cache: stream turn 1 (cold)",
        [{"role": "user", "content": "Name one color. Be brief."}],
        max_tokens=16, stream=True)

    # Use actual turn 1 response so cached tokens match exactly
    stream1_content = tr_stream1.content if tr_stream1 else "Blue"
    run("prefix-cache: stream turn 2 (should hit cache)",
        [{"role": "user", "content": "Name one color. Be brief."},
         {"role": "assistant", "content": stream1_content},
         {"role": "user", "content": "Name another color. Be brief."}],
        max_tokens=16, stream=True,
        expect_prefix_cache_min=10)

    # ── 11. 10-turn mixed text+VL prefix cache stress test ──
    # Each turn uses the actual model response to build the next request,
    # ensuring the prefix cache can match exactly.  Turns alternate between
    # text-only and VL (image) to verify cache behaviour across modalities.
    TURN_PROMPTS = [
        # (user_content, is_vl, label)
        ("What is the capital of Japan? One word.", False, "text-t1"),
        ([{"type": "image_url", "image_url": {"url": img1_uri}},
          {"type": "text", "text": "Describe this image in one sentence."}], True, "vl-t2"),
        ("What language is most spoken there? One word.", False, "text-t3"),
        ([{"type": "image_url", "image_url": {"url": img2_uri}},
          {"type": "text", "text": "What is in this image? Answer with one word."}], True, "vl-t4"),
        ("Name a famous dish from Japan. One word.", False, "text-t5"),
        ("What continent is Japan in? One word.", False, "text-t6"),
        ([{"type": "image_url", "image_url": {"url": img1_uri}},
          {"type": "text", "text": "Is there any text in this image? Yes or no."}], True, "vl-t7"),
        ("What is 100 + 200? Just the number.", False, "text-t8"),
        ("Name one ocean near Japan. Be brief.", False, "text-t9"),
        ("Say goodbye in Japanese. One word.", False, "text-t10"),
    ]

    messages_so_far = [{"role": "system", "content": "You are a helpful assistant. Be very brief."}]
    for i, (prompt, is_vl, label) in enumerate(TURN_PROMPTS):
        turn_num = i + 1
        user_msg = {"role": "user", "content": prompt}
        messages_so_far.append(user_msg)

        # VL requests currently reset cache (different code path), so only
        # expect cache hits on text turns after another text turn.
        expect_cache = None
        if turn_num > 1 and not is_vl:
            # Check if previous turn was also text (not VL) — VL resets cache
            prev_is_vl = TURN_PROMPTS[i - 1][1]
            if not prev_is_vl:
                expect_cache = 10

        tr = run(f"10-turn: {label} (turn {turn_num})",
                 list(messages_so_far),  # copy so mutations don't affect
                 max_tokens=32,
                 expect_prefix_cache_min=expect_cache)

        # Append actual model response for next turn
        resp_content = tr.content if tr else ""
        messages_so_far.append({"role": "assistant", "content": resp_content})

    # ── 12. Final text (verify no state corruption) ──
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
