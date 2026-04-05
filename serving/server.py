"""Qwen3.5 OpenAI-compatible API server.

Endpoints:
  POST /v1/chat/completions  — Chat (multi-turn, tools, streaming)
  POST /v1/completions       — Text completion
  GET  /v1/models            — List models
  GET  /health               — Health check
"""

import asyncio
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import ServerConfig, parse_args
from engine import Engine
from vl_backend import VLBackend, has_image_content, decode_image_content, extract_text_from_content
from schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatResponseMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    ModelInfo,
    ModelList,
    StreamChoice,
    ToolCall,
    ToolCallFunction,
    UsageInfo,
)
from tool_parser import parse_tool_calls

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("server")

engine: Engine = None
config: ServerConfig = None
vl_backend: VLBackend = None


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, config, vl_backend
    config = parse_args()
    engine = Engine(config)
    vl_backend = VLBackend(
        model_path=config.model_path,
        device=config.device,
        exe_path=config.vl_exe if config.vl_exe else "",
    )
    logger.info(f"Starting engine: model={config.model_path}, device={config.device}, workers={config.num_workers}")
    logger.info(f"VL backend: {'available' if vl_backend.available else 'NOT available'} (exe: {vl_backend.exe_path})")
    await engine.start()
    logger.info("Server ready")
    yield
    if engine:
        await engine.shutdown()

app = FastAPI(title="Qwen3.5 OpenAI API", version="0.1.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": {"message": str(exc), "type": "server_error"}})


# ── Helpers ─────────────────────────────────────────────────────────────────


def _messages_to_dicts(messages) -> list[dict]:
    """Convert Pydantic ChatMessage list to plain dicts for apply_chat_template."""
    result = []
    for m in messages:
        d = {"role": m.role}
        if m.content is not None:
            if isinstance(m.content, str):
                d["content"] = m.content
            else:
                # Multimodal: extract text parts (VL support TBD)
                texts = [p.text for p in m.content if p.type == "text" and p.text]
                d["content"] = "\n".join(texts)
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.name:
            d["name"] = m.name
        result.append(d)
    return result


def _tools_to_dicts(tools) -> list[dict]:
    """Convert Pydantic ToolDefinition list to plain dicts."""
    if not tools:
        return None
    return [t.model_dump() for t in tools]


def _get_stop_list(stop) -> list[str]:
    """Normalize stop parameter to a list."""
    if stop is None:
        return None
    if isinstance(stop, str):
        return [stop]
    return list(stop)


# ── Chat Completions ────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required and must be non-empty")

    max_tokens = request.max_completion_tokens or request.max_tokens or config.max_tokens_default
    if max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be >= 1")

    model_name = request.model or config.model_name

    # Route VL requests (messages with images) to VL backend
    if has_image_content(request.messages):
        return await _handle_vl_request(request, model_name, max_tokens)

    messages = _messages_to_dicts(request.messages)
    tools = _tools_to_dicts(request.tools)
    stop = _get_stop_list(request.stop)

    # Format prompt via chat template
    try:
        prompt = engine.apply_chat_template(messages, tools=tools)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat template error: {e}")

    try:
        if request.stream:
            return _stream_chat(prompt, model_name, max_tokens, request.temperature,
                                request.top_p, request.top_k, stop, tools is not None)
        else:
            return await _complete_chat(prompt, model_name, max_tokens, request.temperature,
                                        request.top_p, request.top_k, stop, tools is not None)
    except RuntimeError as e:
        if "workers busy" in str(e).lower():
            raise HTTPException(status_code=503, detail="Server overloaded, try again later")
        raise HTTPException(status_code=500, detail=str(e))


async def _handle_vl_request(request: ChatCompletionRequest, model_name: str, max_tokens: int):
    """Handle vision-language requests via subprocess backend."""
    if not vl_backend or not vl_backend.available:
        raise HTTPException(status_code=501, detail="VL backend not available (modeling_qwen3_5.exe not found)")

    # Extract image from the last user message with image content
    image_data = None
    text_prompt = ""
    for msg in reversed(request.messages):
        if msg.role == "user" and isinstance(msg.content, list):
            image_data = await decode_image_content(msg.content)
            text_prompt = extract_text_from_content(msg.content)
            break

    if not image_data:
        raise HTTPException(status_code=400, detail="No valid image found in messages")
    if not text_prompt:
        text_prompt = "Describe this image."

    temperature = request.temperature if request.temperature is not None else 0.0

    try:
        if request.stream:
            return _stream_vl(text_prompt, image_data, model_name, max_tokens, temperature)
        else:
            result = await vl_backend.generate(
                prompt=text_prompt,
                image_data=image_data,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            message = ChatResponseMessage(role="assistant", content=result.text)
            return ChatCompletionResponse(
                model=model_name,
                choices=[ChatChoice(index=0, message=message, finish_reason=result.finish_reason)],
                usage=UsageInfo(
                    prompt_tokens=result.prompt_tokens or max(1, len(text_prompt) // 4),
                    completion_tokens=result.completion_tokens or max(1, len(result.text) // 4),
                    total_tokens=(result.prompt_tokens or max(1, len(text_prompt) // 4))
                               + (result.completion_tokens or max(1, len(result.text) // 4)),
                ),
            )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="VL generation timed out")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"VL generation failed: {e}")


def _stream_vl(text_prompt, image_data, model_name, max_tokens, temperature):
    """Streaming VL response via SSE."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_generator():
        # First chunk: role
        chunk = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        accumulated = ""
        try:
            async for token in vl_backend.generate_stream(
                prompt=text_prompt,
                image_data=image_data,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                accumulated += token

                # Buffer <think>...</think> blocks
                if "<think>" in accumulated and "</think>" not in accumulated:
                    continue
                if "</think>" in accumulated:
                    accumulated = ""
                    continue

                chunk = ChatCompletionStreamResponse(
                    id=request_id, created=created, model=model_name,
                    choices=[StreamChoice(delta=DeltaMessage(content=token))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                accumulated = ""

        except Exception as e:
            logger.error(f"VL streaming error: {e}")
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # Final chunk with finish_reason from subprocess metadata
        finish_reason = "stop"
        if vl_backend._last_stream_result:
            finish_reason = vl_backend._last_stream_result.finish_reason

        done_chunk = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _complete_chat(prompt, model_name, max_tokens, temperature, top_p, top_k, stop, has_tools):
    """Non-streaming chat completion."""
    result = await engine.generate(prompt, max_tokens, temperature, top_p, top_k, stop)

    # Strip <think>...</think> reasoning blocks
    import re
    text = re.sub(r"<think>.*?</think>\s*", "", result.text, flags=re.DOTALL).strip()

    # Parse tool calls if tools were provided
    if has_tools:
        parsed = parse_tool_calls(text)
        message = ChatResponseMessage(
            role="assistant",
            content=parsed.content,
            tool_calls=parsed.tool_calls,
        )
        finish_reason = parsed.finish_reason
    else:
        message = ChatResponseMessage(role="assistant", content=text)
        finish_reason = result.finish_reason

    return ChatCompletionResponse(
        model=model_name,
        choices=[ChatChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


def _stream_chat(prompt, model_name, max_tokens, temperature, top_p, top_k, stop, has_tools):
    """Streaming chat completion via SSE."""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_generator():
        # First chunk: role
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        accumulated = ""
        emitted_tool_calls = False
        tool_call_index = 0
        buffering_tool = False
        # Track if we might be in a tag prefix (for partial tag detection)
        _TAG_PREFIX = "<tool_call>"

        try:
            async for token in engine.generate_stream(
                prompt, max_tokens, temperature, top_p, top_k, stop,
            ):
                accumulated += token

                if has_tools:
                    # Check if we're starting to buffer a potential tool call
                    if not buffering_tool:
                        # Check if accumulated ends with a prefix of <tool_call>
                        for plen in range(1, len(_TAG_PREFIX) + 1):
                            if accumulated.endswith(_TAG_PREFIX[:plen]):
                                buffering_tool = True
                                break

                    if "<tool_call>" in accumulated and "</tool_call>" not in accumulated:
                        # Inside a tool call block — keep buffering
                        buffering_tool = True
                        continue

                    if "</tool_call>" in accumulated:
                        # Complete tool call — parse and emit
                        parsed = parse_tool_calls(accumulated)
                        # Emit any text content before the tool call
                        if parsed.content:
                            content_chunk = ChatCompletionStreamResponse(
                                id=request_id, created=created, model=model_name,
                                choices=[StreamChoice(delta=DeltaMessage(content=parsed.content))],
                            )
                            yield f"data: {content_chunk.model_dump_json()}\n\n"

                        if parsed.tool_calls:
                            for tc in parsed.tool_calls:
                                tc_chunk = ChatCompletionStreamResponse(
                                    id=request_id, created=created, model=model_name,
                                    choices=[StreamChoice(delta=DeltaMessage(
                                        tool_calls=[{
                                            "index": tool_call_index,
                                            "id": tc.id,
                                            "type": "function",
                                            "function": {
                                                "name": tc.function.name,
                                                "arguments": tc.function.arguments,
                                            },
                                        }],
                                    ))],
                                )
                                yield f"data: {tc_chunk.model_dump_json()}\n\n"
                                tool_call_index += 1
                            emitted_tool_calls = True

                        accumulated = ""
                        buffering_tool = False
                        continue

                    if buffering_tool:
                        # Still might be a partial tag — keep buffering
                        continue

                # Regular content token (or no tools)
                # Strip <think>...</think> blocks for streaming
                if "<think>" in accumulated and "</think>" not in accumulated:
                    continue  # buffer thinking
                if "</think>" in accumulated:
                    accumulated = ""
                    continue

                chunk = ChatCompletionStreamResponse(
                    id=request_id, created=created, model=model_name,
                    choices=[StreamChoice(delta=DeltaMessage(content=token))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                accumulated = ""

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Emit error as a final chunk
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # Final chunk with correct finish_reason
        finish_reason = "tool_calls" if emitted_tool_calls else "stop"
        done_chunk = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Text Completions ────────────────────────────────────────────────────────


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    max_tokens = request.max_tokens or config.max_tokens_default
    stop = _get_stop_list(request.stop)
    model_name = request.model or config.model_name

    if request.stream:
        return _stream_completion(prompt, model_name, max_tokens, request.temperature,
                                  request.top_p, request.top_k, stop)

    result = await engine.generate(prompt, max_tokens, request.temperature,
                                   request.top_p, request.top_k, stop)
    return CompletionResponse(
        model=model_name,
        choices=[CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


def _stream_completion(prompt, model_name, max_tokens, temperature, top_p, top_k, stop):
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    async def event_generator():
        async for token in engine.generate_stream(
            prompt, max_tokens, temperature, top_p, top_k, stop,
        ):
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": token, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        done = {
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Models & Health ─────────────────────────────────────────────────────────


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[
        ModelInfo(id=config.model_name, created=int(time.time())),
    ])


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": config.model_name,
        "workers": config.num_workers,
        "vl_available": vl_backend.available if vl_backend else False,
    }


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # parse_args() will be called inside lifespan, just need host/port here
    import sys
    _cfg = parse_args()
    uvicorn.run(
        "server:app",
        host=_cfg.host,
        port=_cfg.port,
        log_level="info",
    )
