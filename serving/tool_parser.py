"""Parse Qwen3.5 <tool_call> XML output into OpenAI tool_calls format.

Qwen3.5 format (same as Qwen3Coder):
  <tool_call>
  <function=function_name>
  <parameter=param_name>value</parameter>
  ...
  </function>
  </tool_call>

OpenAI format:
  {"id": "call_xxx", "type": "function",
   "function": {"name": "function_name", "arguments": "{\"param_name\": \"value\"}"}}
"""

import json
import re
import uuid
from dataclasses import dataclass

from schemas import ToolCall, ToolCallFunction


@dataclass
class ParseResult:
    """Result of parsing model output for tool calls."""
    content: str  # Text content (non-tool-call parts)
    tool_calls: list[ToolCall]  # Extracted tool calls
    finish_reason: str  # "stop", "tool_calls", or "length"


# Regex patterns for Qwen3.5 tool call format
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)
_FUNCTION_BLOCK = re.compile(
    r"<function=([^>]+)>(.*?)</function>",
    re.DOTALL,
)
_PARAMETER = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> ParseResult:
    """Parse model output text for tool calls.

    Returns ParseResult with extracted tool_calls and remaining content.
    """
    tool_calls: list[ToolCall] = []
    content_parts: list[str] = []

    # Split text around <tool_call>...</tool_call> blocks
    last_end = 0
    for match in _TOOL_CALL_BLOCK.finditer(text):
        # Text before this tool call block
        before = text[last_end:match.start()].strip()
        if before:
            content_parts.append(before)
        last_end = match.end()

        block = match.group(1)
        func_match = _FUNCTION_BLOCK.search(block)
        if not func_match:
            # Malformed — treat as content
            content_parts.append(match.group(0))
            continue

        func_name = func_match.group(1).strip()
        func_body = func_match.group(2)

        # Extract parameters
        params: dict = {}
        for param_match in _PARAMETER.finditer(func_body):
            param_name = param_match.group(1).strip()
            param_value = param_match.group(2).strip()
            # Try to parse as JSON value (number, bool, etc.)
            params[param_name] = _parse_param_value(param_value)

        tool_calls.append(ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=ToolCallFunction(
                name=func_name,
                arguments=json.dumps(params, ensure_ascii=False),
            ),
        ))

    # Remaining text after last tool call
    remaining = text[last_end:].strip()
    if remaining:
        content_parts.append(remaining)

    content = "\n".join(content_parts).strip()

    # Remove <think>...</think> from content (Qwen3.5 reasoning tokens)
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()

    finish_reason = "tool_calls" if tool_calls else "stop"
    return ParseResult(
        content=content if content else None,
        tool_calls=tool_calls if tool_calls else None,
        finish_reason=finish_reason,
    )


def _parse_param_value(value: str):
    """Try to parse a parameter value as a JSON type."""
    # Try JSON parse first (handles numbers, bools, arrays, objects)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        pass
    # Return as string
    return value


def has_tool_call_start(text: str) -> bool:
    """Check if text contains the start of a tool call (for streaming detection)."""
    return "<tool_call>" in text


def has_tool_call_end(text: str) -> bool:
    """Check if text contains a complete tool call."""
    return "</tool_call>" in text
