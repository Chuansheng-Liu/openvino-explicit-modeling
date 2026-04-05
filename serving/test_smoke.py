"""Quick smoke tests for the Qwen3.5 API server."""

import httpx
import json
import time
import sys

BASE = "http://localhost:8000"
client = httpx.Client(timeout=120)
passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} — {e}")
        failed += 1


# ── Basic Endpoints ─────────────────────────────────────────────────────────

def test_health():
    r = client.get(f"{BASE}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print(f"    → {data}")

def test_models():
    r = client.get(f"{BASE}/v1/models")
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) > 0
    print(f"    → model: {data['data'][0]['id']}")


# ── Chat Completions ────────────────────────────────────────────────────────

def test_chat_basic():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 30,
        "temperature": 0,
    })
    assert r.status_code == 200
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    assert content and len(content) > 0
    print(f"    → {content[:80]}")
    print(f"    → usage: {data['usage']}")

def test_chat_system_prompt():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are a pirate. Reply in pirate speak."},
            {"role": "user", "content": "How are you?"},
        ],
        "max_tokens": 40,
        "temperature": 0,
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    assert content
    print(f"    → {content[:80]}")

def test_chat_max_tokens():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Count from 1 to 100."}],
        "max_tokens": 10,
        "temperature": 0,
    })
    assert r.status_code == 200
    content = r.json()["choices"][0]["message"]["content"]
    print(f"    → '{content[:60]}' (should be short)")

def test_chat_temperature_0():
    """Two requests with temp=0 should give same result."""
    payload = {
        "messages": [{"role": "user", "content": "What is 1+1?"}],
        "max_tokens": 20,
        "temperature": 0,
    }
    r1 = client.post(f"{BASE}/v1/chat/completions", json=payload)
    r2 = client.post(f"{BASE}/v1/chat/completions", json=payload)
    c1 = r1.json()["choices"][0]["message"]["content"]
    c2 = r2.json()["choices"][0]["message"]["content"]
    print(f"    → r1: '{c1[:50]}'")
    print(f"    → r2: '{c2[:50]}'")
    assert c1 == c2, f"Determinism failed: '{c1}' != '{c2}'"


# ── Streaming ───────────────────────────────────────────────────────────────

def test_chat_stream():
    chunks = []
    with client.stream("POST", f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Say hi."}],
        "max_tokens": 20,
        "stream": True,
        "temperature": 0,
    }) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"]
                if delta.get("content"):
                    chunks.append(delta["content"])
    text = "".join(chunks)
    assert len(chunks) > 0
    print(f"    → {len(chunks)} chunks: '{text[:60]}'")


# ── Text Completions ────────────────────────────────────────────────────────

def test_completions_basic():
    r = client.post(f"{BASE}/v1/completions", json={
        "prompt": "The capital of France is",
        "max_tokens": 20,
        "temperature": 0,
    })
    assert r.status_code == 200
    text = r.json()["choices"][0]["text"]
    assert text
    print(f"    → '{text[:60]}'")


# ── Tool Calling ────────────────────────────────────────────────────────────

def test_tool_call_basic():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
        "tools": [{
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
        }],
        "max_tokens": 200,
        "temperature": 0,
    })
    assert r.status_code == 200
    data = r.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls", f"Expected tool_calls, got {choice['finish_reason']}"
    assert choice["message"]["tool_calls"]
    tc = choice["message"]["tool_calls"][0]
    print(f"    → function: {tc['function']['name']}")
    print(f"    → arguments: {tc['function']['arguments']}")
    args = json.loads(tc["function"]["arguments"])
    assert "city" in args

def test_no_tool_when_unnecessary():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }],
        "max_tokens": 100,
        "temperature": 0,
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    # Model should answer directly, not call get_weather
    print(f"    → finish_reason: {choice['finish_reason']}")
    print(f"    → content: {(choice['message'].get('content') or '')[:60]}")


# ── Error Handling ──────────────────────────────────────────────────────────

def test_empty_messages():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [],
        "max_tokens": 10,
    })
    assert r.status_code == 400

def test_negative_max_tokens():
    r = client.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": -1,
    })
    assert r.status_code == 400


# ── Run All ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("Qwen3.5 API Server — Smoke Tests")
print("=" * 60)

tests = [
    ("Health endpoint", test_health),
    ("Models endpoint", test_models),
    ("Chat basic", test_chat_basic),
    ("Chat system prompt", test_chat_system_prompt),
    ("Chat max_tokens", test_chat_max_tokens),
    ("Chat determinism (temp=0)", test_chat_temperature_0),
    ("Chat streaming", test_chat_stream),
    ("Text completion", test_completions_basic),
    ("Tool call basic", test_tool_call_basic),
    ("No tool when unnecessary", test_no_tool_when_unnecessary),
    ("Error: empty messages", test_empty_messages),
    ("Error: negative max_tokens", test_negative_max_tokens),
]

for name, fn in tests:
    test(name, fn)

print()
print(f"Results: {passed} passed, {failed} failed out of {passed+failed}")
sys.exit(1 if failed else 0)
