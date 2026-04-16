#!/usr/bin/env python3
"""
Test script for OpenAI-compatible tool calling with Qwen3.5 models.
Works with both local ov_serve and remote OVMS server.

Usage:
    python test_tool_calling.py                          # default: local ov_serve
    python test_tool_calling.py --url http://x1d1.sh.intel.com:8000/v3  # remote OVMS
    python test_tool_calling.py --url http://localhost:8080/v1           # local ov_serve
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
import os

# ── Tool definitions (OpenAI function calling format) ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Beijing', 'Shanghai', 'Tokyo'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Search for information on a given topic",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "language": {
                        "type": "string",
                        "enum": ["zh", "en"],
                        "description": "Language of the results"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# ── Simulated tool execution ──

def execute_tool(name: str, arguments: dict) -> str:
    """Simulate tool execution and return fake results."""
    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        unit = arguments.get("unit", "celsius")
        temp = 22 if unit == "celsius" else 72
        return json.dumps({
            "city": city,
            "temperature": temp,
            "unit": unit,
            "condition": "sunny",
            "humidity": 45,
            "wind": "3m/s"
        }, ensure_ascii=False)

    elif name == "calculate":
        expr = arguments.get("expression", "0")
        try:
            result = eval(expr)  # only for testing!
            return json.dumps({"expression": expr, "result": result})
        except Exception as e:
            return json.dumps({"error": str(e)})

    elif name == "search_knowledge":
        query = arguments.get("query", "")
        return json.dumps({
            "query": query,
            "results": [
                {"title": f"About {query}", "snippet": f"{query} is a well-known topic..."},
                {"title": f"{query} - Wikipedia", "snippet": f"Detailed information about {query}..."}
            ]
        }, ensure_ascii=False)

    return json.dumps({"error": f"Unknown tool: {name}"})


# ── API call helper ──

def parse_tool_calls_from_content(content: str) -> list:
    """Parse Qwen3.5-style <tool_call> tags from content text.
    OVMS returns tool calls as raw text in content rather than
    structured tool_calls array. This extracts them."""
    tool_calls = []
    # Match <tool_call>...</tool_call> blocks
    pattern = r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>'
    matches = re.finditer(pattern, content, re.DOTALL)

    for i, m in enumerate(matches):
        func_name = m.group(1)
        params_block = m.group(2)

        # Extract <parameter=name>value</parameter> pairs
        args = {}
        param_pattern = r'<parameter=(\w+)>\s*(.*?)\s*</parameter>'
        for pm in re.finditer(param_pattern, params_block, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()

        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(args, ensure_ascii=False)
            }
        })

    return tool_calls


def normalize_response(msg: dict) -> dict:
    """Normalize model response: if tool_calls is empty but content
    contains <tool_call> tags, parse them out.
    Preserves raw_content for OVMS-compatible message replay."""
    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")

    if not tool_calls and "<tool_call>" in content:
        parsed = parse_tool_calls_from_content(content)
        if parsed:
            msg["tool_calls"] = parsed
            msg["raw_content"] = content  # preserve original for OVMS replay
            # Remove tool_call tags from content, keep any remaining text
            cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
            msg["content"] = cleaned

    return msg


def build_history_messages(msg: dict, tool_results: list) -> list:
    """Build OVMS-compatible message history for tool result round-trip.
    OVMS doesn't support tool_calls array in assistant messages or
    tool_call_id/name in tool messages. Use raw content + simplified tool role."""
    out = []
    # Assistant message: use raw content with <tool_call> tags if available
    raw = msg.get("raw_content", "")
    if raw:
        out.append({"role": "assistant", "content": raw})
    else:
        out.append({"role": "assistant", "content": msg.get("content", "")})

    # Tool results: simplified format without tool_call_id/name
    for tr in tool_results:
        out.append({"role": "tool", "content": tr["content"]})

    return out

def chat_completion(base_url: str, model: str, messages: list,
                    tools: list = None, max_tokens: int = 1024,
                    no_proxy: bool = True) -> dict:
    """Send a chat completion request."""
    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-key"
        },
        method="POST"
    )

    # Bypass proxy for internal hosts
    if no_proxy:
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    t0 = time.time()
    with opener.open(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0

    # Normalize: parse <tool_call> tags from content if needed
    if body.get("choices"):
        body["choices"][0]["message"] = normalize_response(body["choices"][0]["message"])

    return body, elapsed


def print_separator(title: str = ""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print(f"{'=' * 60}")


def print_message(role: str, content: str, indent: int = 2):
    prefix = " " * indent
    tag = f"[{role.upper()}]"
    lines = (content or "(empty)").split("\n")
    print(f"{prefix}{tag} {lines[0]}")
    for line in lines[1:]:
        print(f"{prefix}{'.' * len(tag)} {line}")


# ── Test scenarios ──

def test_simple_chat(base_url: str, model: str):
    """Test 1: Simple chat without tools."""
    print_separator("Test 1: Simple Chat (no tools)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply concisely."},
        {"role": "user", "content": "What is 2+2? Answer in one word."}
    ]

    resp, elapsed = chat_completion(base_url, model, messages)
    content = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})

    print_message("assistant", content)
    print(f"  [INFO] Time: {elapsed:.2f}s | Tokens: {usage.get('total_tokens', '?')}")
    return True


def test_single_tool_call(base_url: str, model: str):
    """Test 2: Single tool call (weather query)."""
    print_separator("Test 2: Single Tool Call (weather)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed."},
        {"role": "user", "content": "What's the weather like in Beijing today?"}
    ]

    print_message("user", messages[-1]["content"])

    # Step 1: Send request with tools
    print("\n  >> Sending request with tools...")
    resp, elapsed = chat_completion(base_url, model, messages, tools=TOOLS)
    msg = resp["choices"][0]["message"]
    usage = resp.get("usage", {})

    print(f"  [INFO] Time: {elapsed:.2f}s | Tokens: {usage.get('total_tokens', '?')}")

    # Check if model wants to call a tool
    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")

    if content:
        print_message("assistant", content)

    if not tool_calls:
        print("  [WARN] Model did not request any tool calls.")
        print(f"  [DEBUG] Full response: {json.dumps(msg, indent=2, ensure_ascii=False)}")
        return False

    # Step 2: Execute tool calls
    print(f"\n  >> Model requested {len(tool_calls)} tool call(s):")

    tool_results = []
    for tc in tool_calls:
        func_name = tc["function"]["name"]
        try:
            func_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            func_args = {"raw": tc["function"]["arguments"]}

        print(f"     -> {func_name}({json.dumps(func_args, ensure_ascii=False)})")

        # Execute and add result
        result = execute_tool(func_name, func_args)
        print(f"     <- {result}")
        tool_results.append({"name": func_name, "content": result})

    # Build OVMS-compatible history
    messages.extend(build_history_messages(msg, tool_results))

    # Step 3: Get final response
    print("\n  >> Sending tool results back to model...")
    resp2, elapsed2 = chat_completion(base_url, model, messages)
    final_content = resp2["choices"][0]["message"]["content"]
    usage2 = resp2.get("usage", {})

    print_message("assistant", final_content)
    print(f"  [INFO] Time: {elapsed2:.2f}s | Tokens: {usage2.get('total_tokens', '?')}")
    return True


def test_multi_tool_call(base_url: str, model: str):
    """Test 3: Query that might trigger multiple tool calls."""
    print_separator("Test 3: Multi-Tool Call")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed. You can call multiple tools."},
        {"role": "user", "content": "帮我查一下北京和上海的天气，然后算一下两地温差。"}
    ]

    print_message("user", messages[-1]["content"])

    # Round 1
    print("\n  >> Round 1: Sending request with tools...")
    resp, elapsed = chat_completion(base_url, model, messages, tools=TOOLS)
    msg = resp["choices"][0]["message"]

    if msg.get("content"):
        print_message("assistant", msg["content"])

    tool_calls = msg.get("tool_calls", [])
    if not tool_calls:
        print("  [WARN] Model did not request any tool calls.")
        print(f"  [DEBUG] Full response: {json.dumps(msg, indent=2, ensure_ascii=False)}")
        return False

    print(f"  >> Model requested {len(tool_calls)} tool call(s):")

    tool_results = []
    for tc in tool_calls:
        func_name = tc["function"]["name"]
        try:
            func_args = json.loads(tc["function"]["arguments"])
        except (json.JSONDecodeError, TypeError):
            func_args = {"raw": tc["function"]["arguments"]}

        print(f"     -> {func_name}({json.dumps(func_args, ensure_ascii=False)})")
        result = execute_tool(func_name, func_args)
        print(f"     <- {result}")
        tool_results.append({"name": func_name, "content": result})

    # Build OVMS-compatible history
    messages.extend(build_history_messages(msg, tool_results))

    # Round 2: Send tool results back
    print("\n  >> Round 2: Sending tool results back...")
    resp2, elapsed2 = chat_completion(base_url, model, messages, tools=TOOLS)
    msg2 = resp2["choices"][0]["message"]

    tool_calls2 = msg2.get("tool_calls", [])
    if tool_calls2:
        print(f"  >> Model requested {len(tool_calls2)} more tool call(s):")

        tool_results2 = []
        for tc in tool_calls2:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                func_args = {"raw": tc["function"]["arguments"]}

            print(f"     -> {func_name}({json.dumps(func_args, ensure_ascii=False)})")
            result = execute_tool(func_name, func_args)
            print(f"     <- {result}")
            tool_results2.append({"name": func_name, "content": result})

        messages.extend(build_history_messages(msg2, tool_results2))

        # Round 3: Final response
        print("\n  >> Round 3: Getting final response...")
        resp3, elapsed3 = chat_completion(base_url, model, messages)
        final_content = resp3["choices"][0]["message"]["content"]
        print_message("assistant", final_content)
    else:
        final_content = msg2.get("content", "")
        print_message("assistant", final_content)

    return True


def test_no_tool_needed(base_url: str, model: str):
    """Test 4: Query that should NOT trigger tool calls."""
    print_separator("Test 4: No Tool Needed (tools available but unnecessary)")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Only use tools when necessary."},
        {"role": "user", "content": "What is the capital of France?"}
    ]

    print_message("user", messages[-1]["content"])

    resp, elapsed = chat_completion(base_url, model, messages, tools=TOOLS)
    msg = resp["choices"][0]["message"]

    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")

    if tool_calls:
        print(f"  [WARN] Model called tools when not needed: {[tc['function']['name'] for tc in tool_calls]}")
    else:
        print("  [OK] Model correctly answered without using tools.")

    print_message("assistant", content)
    return not bool(tool_calls)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Test tool calling with Qwen3.5")
    parser.add_argument("--url", default="http://localhost:8080/v1",
                        help="API base URL (default: http://localhost:8080/v1)")
    parser.add_argument("--model", default=None,
                        help="Model name (auto-detected if not specified)")
    parser.add_argument("--test", type=int, default=0,
                        help="Run specific test (1-4), 0=all")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    # Auto-detect model name
    model = args.model
    if not model:
        try:
            models_url = f"{base_url}/models"
            req = urllib.request.Request(models_url)
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(req, timeout=10) as resp:
                models_resp = json.loads(resp.read().decode("utf-8"))
                if models_resp.get("data"):
                    model = models_resp["data"][0]["id"]
        except Exception:
            pass
        if not model:
            model = "default"

    print_separator(f"Tool Calling Test Suite")
    print(f"  Server:  {base_url}")
    print(f"  Model:   {model}")
    print(f"  Tools:   {len(TOOLS)} defined ({', '.join(t['function']['name'] for t in TOOLS)})")

    tests = {
        1: ("Simple Chat", test_simple_chat),
        2: ("Single Tool Call", test_single_tool_call),
        3: ("Multi-Tool Call", test_multi_tool_call),
        4: ("No Tool Needed", test_no_tool_needed),
    }

    results = {}
    run_tests = [args.test] if args.test else list(tests.keys())

    for tid in run_tests:
        name, func = tests[tid]
        try:
            results[tid] = func(base_url, model)
        except urllib.error.HTTPError as e:
            print(f"\n  [ERROR] HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
            results[tid] = False
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            results[tid] = False

    # Summary
    print_separator("Summary")
    for tid in run_tests:
        name = tests[tid][0]
        status = "PASS ✓" if results.get(tid) else "FAIL ✗"
        print(f"  Test {tid}: {name:30s} [{status}]")
    print()

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"  Result: {passed}/{total} passed")
    print()


if __name__ == "__main__":
    main()
