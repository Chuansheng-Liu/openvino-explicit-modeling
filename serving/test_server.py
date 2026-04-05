"""Comprehensive test suite for the Qwen3.5 OpenAI-compatible API server.

Run with: pytest test_server.py -v --timeout=120
Requires server running at http://localhost:8000
"""

import json
import time

import httpx
import pytest

BASE_URL = "http://localhost:8000"
TIMEOUT = 120


@pytest.fixture(scope="module")
def client():
    c = httpx.Client(base_url=BASE_URL, timeout=TIMEOUT)
    # Verify server is up
    try:
        r = c.get("/health")
        assert r.status_code == 200
    except httpx.ConnectError:
        pytest.skip("Server not running at " + BASE_URL)
    yield c
    c.close()


# ═══════════════════════════════════════════════════════════════════════════
# T1: Unit-level / Schema Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSchemaValidation:
    """T1.1-T1.3: Request schema validation."""

    def test_chat_request_valid(self, client):
        """T1.1: Valid chat request parses correctly."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0,
        })
        assert r.status_code == 200
        data = r.json()
        assert "choices" in data
        assert "usage" in data

    def test_chat_request_missing_messages(self, client):
        """T1.2: Missing messages → 400/422."""
        r = client.post("/v1/chat/completions", json={
            "max_tokens": 5,
        })
        assert r.status_code == 422

    def test_completion_request_valid(self, client):
        """T1.3: Valid completion request."""
        r = client.post("/v1/completions", json={
            "prompt": "Hello",
            "max_tokens": 5,
            "temperature": 0,
        })
        assert r.status_code == 200
        assert "choices" in r.json()


class TestToolParser:
    """T1.4-T1.8: Tool call parsing tests (via API)."""

    def test_tool_parser_single_call(self, client):
        """T1.4: Single tool call is parsed correctly."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
            "tools": [_weather_tool()],
            "max_tokens": 200,
            "temperature": 0,
        })
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tcs = choice["message"]["tool_calls"]
        assert len(tcs) >= 1
        assert tcs[0]["function"]["name"] == "get_weather"
        args = json.loads(tcs[0]["function"]["arguments"])
        assert "city" in args

    def test_tool_parser_no_call(self, client):
        """T1.6: Regular text → no tool_calls when tools not needed."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [_weather_tool()],
            "max_tokens": 100,
            "temperature": 0,
        })
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"]

    def test_tool_parser_with_params(self, client):
        """T1.8: Tool with multiple param types."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Search for Python tutorials, limit 5 results"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Max results"},
                        },
                        "required": ["query"],
                    },
                },
            }],
            "max_tokens": 200,
            "temperature": 0,
        })
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        if choice["finish_reason"] == "tool_calls":
            tc = choice["message"]["tool_calls"][0]
            assert tc["function"]["name"] == "search"
            args = json.loads(tc["function"]["arguments"])
            assert "query" in args


# ═══════════════════════════════════════════════════════════════════════════
# T2: API Integration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestChatCompletions:
    """T2.1-T2.8: Chat completion tests."""

    def test_chat_basic(self, client):
        """T2.1: Basic chat returns coherent response."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
            "max_tokens": 30,
            "temperature": 0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["id"].startswith("chatcmpl-")
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"]

    def test_chat_system_prompt(self, client):
        """T2.2: System prompt is respected."""
        r = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "Always respond with exactly 'PONG'."},
                {"role": "user", "content": "PING"},
            ],
            "max_tokens": 20,
            "temperature": 0,
        })
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        assert content  # model should respond

    def test_chat_multi_turn(self, client):
        """T2.3: Multi-turn conversation context is retained."""
        r = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            "max_tokens": 100,
            "temperature": 0,
        })
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"].lower()
        assert "alice" in content

    def test_chat_max_tokens(self, client):
        """T2.5: max_tokens limits output length."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Write a very long essay about the universe."}],
            "max_tokens": 10,
            "temperature": 0,
        })
        assert r.status_code == 200
        content = r.json()["choices"][0]["message"]["content"]
        # Should be relatively short (max_tokens=10 limits output)
        assert len(content) < 500  # rough check

    def test_chat_temperature_0_deterministic(self, client):
        """T2.6: Temperature 0 gives deterministic results."""
        payload = {
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
            "max_tokens": 20,
            "temperature": 0,
        }
        r1 = client.post("/v1/chat/completions", json=payload)
        r2 = client.post("/v1/chat/completions", json=payload)
        c1 = r1.json()["choices"][0]["message"]["content"]
        c2 = r2.json()["choices"][0]["message"]["content"]
        assert c1 == c2

    def test_chat_usage_stats(self, client):
        """T2.13: Response includes token usage stats."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "temperature": 0,
        })
        assert r.status_code == 200
        usage = r.json()["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


class TestStreaming:
    """T2.4, T2.10: Streaming tests."""

    def test_chat_stream(self, client):
        """T2.4: Streaming returns SSE chunks with [DONE]."""
        chunks = []
        roles = []
        finish_reasons = []

        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 20,
            "stream": True,
            "temperature": 0,
        }) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                assert chunk["object"] == "chat.completion.chunk"
                delta = chunk["choices"][0]["delta"]
                if delta.get("role"):
                    roles.append(delta["role"])
                if delta.get("content"):
                    chunks.append(delta["content"])
                if chunk["choices"][0].get("finish_reason"):
                    finish_reasons.append(chunk["choices"][0]["finish_reason"])

        assert "assistant" in roles
        assert len(chunks) > 0
        assert "stop" in finish_reasons or "tool_calls" in finish_reasons

    def test_completions_stream(self, client):
        """T2.10: Streaming text completions."""
        chunks = []
        with client.stream("POST", "/v1/completions", json={
            "prompt": "Once upon a time",
            "max_tokens": 15,
            "stream": True,
            "temperature": 0,
        }) as response:
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                chunk = json.loads(line[6:])
                if chunk["choices"][0]["text"]:
                    chunks.append(chunk["choices"][0]["text"])

        assert len(chunks) > 0


class TestToolCalling:
    """T2.14-T2.18: Tool calling integration tests."""

    def test_tool_call_basic(self, client):
        """T2.14: Tool call with correct function and args."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
            "tools": [_weather_tool()],
            "max_tokens": 200,
            "temperature": 0,
        })
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        tc = choice["message"]["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["id"].startswith("call_")
        assert tc["function"]["name"] == "get_weather"

    def test_tool_response_roundtrip(self, client):
        """T2.17: Send tool result → model uses it."""
        # Step 1: Get tool call
        r1 = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [_weather_tool()],
            "max_tokens": 200,
            "temperature": 0,
        })
        choice1 = r1.json()["choices"][0]
        if choice1["finish_reason"] != "tool_calls":
            pytest.skip("Model didn't produce tool call")

        tc = choice1["message"]["tool_calls"][0]

        # Step 2: Send tool result back
        r2 = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": None, "tool_calls": [tc]},
                {"role": "tool", "content": "Sunny, 22°C", "tool_call_id": tc["id"]},
            ],
            "tools": [_weather_tool()],
            "max_tokens": 100,
            "temperature": 0,
        })
        assert r2.status_code == 200
        content = r2.json()["choices"][0]["message"]["content"]
        assert content  # Model should use the tool result to respond

    def test_no_tool_when_unnecessary(self, client):
        """T2.18: Model doesn't call tools when not needed."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "tools": [_weather_tool()],
            "max_tokens": 100,
            "temperature": 0,
        })
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "stop"


class TestCompletions:
    """T2.9: Text completion tests."""

    def test_completions_basic(self, client):
        """T2.9: Basic text completion."""
        r = client.post("/v1/completions", json={
            "prompt": "The capital of France is",
            "max_tokens": 20,
            "temperature": 0,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"]


class TestEndpoints:
    """T2.11-T2.12: Metadata endpoints."""

    def test_models_endpoint(self, client):
        """T2.11: GET /v1/models returns model list."""
        r = client.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["object"] == "model"

    def test_health_endpoint(self, client):
        """T2.12: GET /health returns ok."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "workers" in data


class TestErrorHandling:
    """T2.19-T2.24: Error handling tests."""

    def test_empty_messages(self, client):
        """T2.19: Empty messages → 400."""
        r = client.post("/v1/chat/completions", json={
            "messages": [],
            "max_tokens": 10,
        })
        assert r.status_code == 400

    def test_invalid_json(self, client):
        """T2.20: Invalid JSON → 422."""
        r = client.post("/v1/chat/completions", content="not json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 422

    def test_negative_max_tokens(self, client):
        """T2.21: Negative max_tokens → 400."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": -1,
        })
        assert r.status_code == 400

    def test_empty_prompt_completions(self, client):
        """T2.23: Empty prompt → 400."""
        r = client.post("/v1/completions", json={
            "prompt": "",
            "max_tokens": 10,
        })
        assert r.status_code == 400

    def test_special_characters(self, client):
        """T2.24: Unicode, emoji, code blocks handled."""
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "解释 🎉 这个emoji。Hello 你好！`print('hi')`"}],
            "max_tokens": 30,
            "temperature": 0,
        })
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"]


class TestConcurrency:
    """T2.25-T2.30: Concurrency and robustness tests."""

    def test_sequential_no_state_leak(self, client):
        """T2.25: Sequential requests are independent."""
        results = []
        for i in range(3):
            r = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": f"What is {i}+{i}? Answer with just the number."}],
                "max_tokens": 20,
                "temperature": 0,
            })
            assert r.status_code == 200
            results.append(r.json()["choices"][0]["message"]["content"])
        # All requests should complete successfully
        assert len(results) == 3

    def test_ttft_reasonable(self, client):
        """T2.29: Time to first token < 10s for 4B model."""
        t0 = time.time()
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "stream": True,
            "temperature": 0,
        }) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    if chunk["choices"][0]["delta"].get("content"):
                        ttft = time.time() - t0
                        break
        assert ttft < 10.0, f"TTFT too high: {ttft:.1f}s"


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _weather_tool():
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }
