"""OpenAI-compatible request/response Pydantic models."""

import time
import uuid
from typing import Any, Optional, Union

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────


class FunctionDefinition(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}


class ToolDefinition(BaseModel):
    type: str = "function"
    function: FunctionDefinition


class ContentPart(BaseModel):
    """Multimodal content part (text or image_url)."""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[dict[str, str]] = None  # {"url": "..."}


class ChatMessage(BaseModel):
    role: str  # system, user, assistant, tool
    content: Optional[Union[str, list[ContentPart]]] = None
    name: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    tools: Optional[list[ToolDefinition]] = None
    tool_choice: Optional[Union[str, dict]] = None
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: Union[str, list[str]] = ""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Optional[Union[str, list[str]]] = None
    n: int = 1


# ── Response Models ─────────────────────────────────────────────────────────


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCall(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolCallFunction


class ChatResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatResponseMessage
    finish_reason: Optional[str] = None  # "stop", "length", "tool_calls"


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = []
    usage: UsageInfo = UsageInfo()

    def __init__(self, **kwargs):
        if not kwargs.get("id"):
            kwargs["id"] = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if not kwargs.get("created"):
            kwargs["created"] = int(time.time())
        super().__init__(**kwargs)


# ── Streaming (SSE) Models ──────────────────────────────────────────────────


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[StreamChoice] = []

    def __init__(self, **kwargs):
        if not kwargs.get("id"):
            kwargs["id"] = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if not kwargs.get("created"):
            kwargs["created"] = int(time.time())
        super().__init__(**kwargs)


# ── Completion Response ─────────────────────────────────────────────────────


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = ""
    object: str = "text_completion"
    created: int = 0
    model: str = ""
    choices: list[CompletionChoice] = []
    usage: UsageInfo = UsageInfo()

    def __init__(self, **kwargs):
        if not kwargs.get("id"):
            kwargs["id"] = f"cmpl-{uuid.uuid4().hex[:12]}"
        if not kwargs.get("created"):
            kwargs["created"] = int(time.time())
        super().__init__(**kwargs)


# ── Model Info ──────────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "openvino"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = []
