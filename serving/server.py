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
import openvino_genai as og
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

import os as _os
_log_dir = _os.environ.get("OV_SERVER_LOG_DIR", "")
_file_handler = None
if _log_dir:
    _os.makedirs(_log_dir, exist_ok=True)
    _log_file = _os.path.join(_log_dir, f"server_{int(time.time())}.log")
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    _file_handler.setLevel(logging.DEBUG)

def _get_logger(name: str) -> logging.Logger:
    """Get a logger with console + optional file handler."""
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)
    if not lg.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        ch.setLevel(logging.INFO)
        lg.addHandler(ch)
    if _file_handler and _file_handler not in lg.handlers:
        lg.addHandler(_file_handler)
    lg.propagate = False
    return lg

logger = _get_logger("server")
_get_logger("vl_backend")
if _file_handler:
    logger.info(f"Logging to file: {_log_file}")

tokenizer: og.Tokenizer = None
config: ServerConfig = None
vl_backend: VLBackend = None
text_backend: VLBackend = None


def apply_chat_template(
    messages: list[dict],
    tools: list[dict] = None,
    enable_thinking: bool = None,
) -> str:
    """Format messages using the model's chat template."""
    kwargs = {"add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools

    # Detect /nothink prefix in last user message
    if enable_thinking is None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.lstrip().startswith("/nothink"):
                    enable_thinking = False
                break

    if enable_thinking is not None:
        kwargs["extra_context"] = {"enable_thinking": enable_thinking}

    return tokenizer.apply_chat_template(messages, **kwargs)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, config, vl_backend, text_backend
    config = parse_args()

    # Load tokenizer for chat template formatting
    tokenizer = og.Tokenizer(config.model_path)
    logger.info(f"Tokenizer loaded from {config.model_path}")

    # Single VL-mode exe handles both text and VL requests.
    vl_backend = VLBackend(
        model_path=config.model_path,
        device=config.device,
        exe_path=config.vl_exe if config.vl_exe else "",
        use_serve=config.serve_vl,
        mode="vl",
    )
    text_backend = vl_backend  # Same process handles text requests too
    logger.info(f"Backend exe: {'available' if vl_backend.available else 'NOT available'} (exe: {vl_backend.exe_path})")

    if not vl_backend.available:
        logger.error("Backend exe not available — server cannot serve requests")

    # Start single serve process for both text and VL
    if config.serve_vl and vl_backend.available:
        logger.info("Starting persistent subprocess (handles text+VL)...")
        try:
            await vl_backend.start_serve()
            logger.info("Serve process started successfully")
        except Exception as e:
            logger.error(f"Failed to start serve process: {e}")
            logger.info("Falling back to per-request subprocess mode")
            vl_backend.use_serve = False

    logger.info("Server ready")
    yield
    if vl_backend and vl_backend.use_serve:
        await vl_backend.stop_serve()

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
@app.post("/v3/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages is required and must be non-empty")

    max_tokens = request.max_completion_tokens or request.max_tokens or config.max_tokens_default
    if max_tokens < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be >= 1")

    model_name = request.model or config.model_name
    last_msg = request.messages[-1].content if request.messages else ""
    if isinstance(last_msg, list):
        last_msg = str(last_msg)[:80]
    else:
        last_msg = str(last_msg)[:80]
    n_tools = len(request.tools) if request.tools else 0
    logger.info(f"[chat] model={model_name}, msgs={len(request.messages)}, tools={n_tools}, "
                f"stream={request.stream}, temp={request.temperature}, max_tokens={max_tokens}, "
                f"last_msg={last_msg!r}")

    # Route VL requests (messages with images) to VL backend
    if has_image_content(request.messages):
        return await _handle_vl_request(request, model_name, max_tokens)

    messages = _messages_to_dicts(request.messages)
    tools = _tools_to_dicts(request.tools)

    # All text requests go through exe backend
    return await _handle_text_via_backend(request, messages, model_name, max_tokens, tools)


async def _handle_text_via_backend(request: ChatCompletionRequest, messages: list[dict],
                                    model_name: str, max_tokens: int,
                                    tools: list[dict] = None):
    """Handle text chat via exe backend subprocess (all text including tool calls)."""
    # Apply chat template in Python, send pre-formatted prompt to exe with raw_prompt=True
    try:
        prompt = apply_chat_template(messages, tools=tools, enable_thinking=config.think)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Chat template error: {e}")

    temperature = request.temperature if request.temperature is not None else 1.0
    has_tools = tools is not None
    # Force greedy decoding for tool calling — prevents degenerate repetition loops
    if has_tools:
        temperature = 0.0

    logger.info(f"[text_backend] has_tools={has_tools}, temperature={temperature}, "
                f"max_tokens={max_tokens}, stream={request.stream}, think={config.think}, "
                f"prompt_len={len(prompt)}")
    if has_tools:
        logger.debug(f"[text_backend] prompt_tail=...{prompt[-300:]}")

    try:
        if request.stream:
            return _stream_text_via_backend(prompt, model_name, max_tokens, temperature, has_tools)
        else:
            return await _complete_text_via_backend(prompt, model_name, max_tokens, temperature, has_tools)
    except RuntimeError as e:
        logger.error(f"[text_backend] RuntimeError: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _complete_text_via_backend(prompt: str, model_name: str, max_tokens: int,
                                     temperature: float, has_tools: bool = False):
    """Non-streaming text completion via exe backend."""
    result = await text_backend.generate(
        prompt=prompt, image_data=None, max_tokens=max_tokens,
        temperature=temperature, think=config.think, raw_prompt=True,
    )

    # Strip thinking from output.
    # When think=True: model outputs thinking, then </think>, then content → take AFTER.
    # When think=False: model may still output content then a stray </think> then garbage → take BEFORE.
    text = result.text
    logger.info(f"[complete] raw output ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")
    if "</think>" in text:
        if config.think:
            text = text.split("</think>", 1)[1].strip()
        else:
            text = text.split("</think>", 1)[0].strip()
        logger.info(f"[complete] after think-strip ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")

    # Parse tool calls if tools were provided
    if has_tools:
        parsed = parse_tool_calls(text)
        logger.info(f"[complete] tool_parse: content={parsed.content!r:.100}, "
                     f"tool_calls={len(parsed.tool_calls) if parsed.tool_calls else 0}, "
                     f"finish_reason={parsed.finish_reason}")
        message = ChatResponseMessage(
            role="assistant",
            content=parsed.content,
            tool_calls=parsed.tool_calls,
        )
        finish_reason = parsed.finish_reason
    else:
        message = ChatResponseMessage(role="assistant", content=text)
        finish_reason = result.finish_reason or "stop"

    request_id = f"chatcmpl-{int(time.time()*1000)}"
    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=model_name,
        choices=[ChatChoice(
            message=message,
            finish_reason=finish_reason,
        )],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


def _stream_text_via_backend(prompt: str, model_name: str, max_tokens: int,
                             temperature: float, has_tools: bool = False):
    """Streaming text completion via exe backend with think filtering and tool call detection."""
    request_id = f"chatcmpl-{int(time.time()*1000)}"
    created = int(time.time())

    async def generate():
        # Role chunk
        chunk = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        accumulated = ""
        think_done = not config.think  # Skip think filtering if think=OFF
        emitted_tool_calls = False
        tool_call_index = 0
        buffering_tool = False
        _TAG_PREFIX = "<tool_call>"

        try:
            async for token in text_backend.generate_stream(
                prompt=prompt, image_data=None, max_tokens=max_tokens,
                temperature=temperature, think=config.think, raw_prompt=True,
            ):
                accumulated += token

                # Phase 1: Think filtering
                if not think_done:
                    if "</think>" not in accumulated:
                        continue
                    remainder = accumulated.split("</think>", 1)[1]
                    think_done = True
                    accumulated = remainder
                    if not accumulated.strip():
                        accumulated = ""
                        continue
                    token = accumulated

                # Phase 1b: think=OFF stop at stray </think>
                if not config.think and "</think>" in accumulated:
                    before = accumulated.split("</think>", 1)[0]
                    already_len = len(accumulated) - len(token)
                    new_content = before[already_len:]
                    if new_content.strip():
                        c = ChatCompletionStreamResponse(
                            id=request_id, created=created, model=model_name,
                            choices=[StreamChoice(delta=DeltaMessage(content=new_content))],
                        )
                        yield f"data: {c.model_dump_json()}\n\n"
                    break

                # Phase 2: Tool call detection
                if has_tools:
                    if not buffering_tool:
                        for plen in range(1, len(_TAG_PREFIX) + 1):
                            if accumulated.endswith(_TAG_PREFIX[:plen]):
                                buffering_tool = True
                                break

                    if "<tool_call>" in accumulated and "</tool_call>" not in accumulated:
                        buffering_tool = True
                        continue

                    if "</tool_call>" in accumulated:
                        parsed = parse_tool_calls(accumulated)
                        if parsed.content:
                            c = ChatCompletionStreamResponse(
                                id=request_id, created=created, model=model_name,
                                choices=[StreamChoice(delta=DeltaMessage(content=parsed.content))],
                            )
                            yield f"data: {c.model_dump_json()}\n\n"
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
                        continue

                # Phase 3: Regular content
                chunk = ChatCompletionStreamResponse(
                    id=request_id, created=created, model=model_name,
                    choices=[StreamChoice(delta=DeltaMessage(content=token))],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                accumulated = ""

        except Exception as e:
            logger.error(f"Text backend streaming error: {e}")
            logger.error(f"[stream] accumulated at error ({len(accumulated)} chars): {accumulated[:300]}")
            error_chunk = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        # Final chunk
        finish_reason = "tool_calls" if emitted_tool_calls else "stop"
        logger.info(f"[stream] done: finish_reason={finish_reason}, accumulated_len={len(accumulated)}, "
                     f"tool_calls_emitted={tool_call_index}")
        final = ChatCompletionStreamResponse(
            id=request_id, created=created, model=model_name,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason=finish_reason)],
        )
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


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

    temperature = request.temperature if request.temperature is not None else 1.0

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

            # Strip thinking from VL output (VL mode always enables thinking).
            # If </think> present: take content after it.
            # If not present (ran out of tokens while thinking): strip <think> prefix,
            # return thinking content directly (it IS the description).
            text = result.text
            if "</think>" in text:
                text = text.split("</think>", 1)[1].strip()
            elif text.lstrip().startswith("<think>"):
                text = text.lstrip().removeprefix("<think>").strip()

            message = ChatResponseMessage(role="assistant", content=text)
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
        think_done = False
        think_prefix_stripped = False
        try:
            async for token in vl_backend.generate_stream(
                prompt=text_prompt,
                image_data=image_data,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                accumulated += token

                if not think_done:
                    # Buffer until </think> — thinking block ends
                    if "</think>" in accumulated:
                        # Strip everything up to and including </think>
                        remainder = accumulated.split("</think>", 1)[1]
                        think_done = True
                        if not remainder.strip():
                            accumulated = ""
                            continue
                        # Emit the remainder
                        chunk = ChatCompletionStreamResponse(
                            id=request_id, created=created, model=model_name,
                            choices=[StreamChoice(delta=DeltaMessage(content=remainder.lstrip()))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                        accumulated = ""
                        continue
                    # If model is still thinking but we haven't stripped <think> prefix yet,
                    # check if we have enough accumulated to strip and start emitting
                    if not think_prefix_stripped:
                        stripped = accumulated.lstrip()
                        if stripped.startswith("<think>"):
                            accumulated = stripped.removeprefix("<think>").lstrip()
                            think_prefix_stripped = True
                            if accumulated:
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id, created=created, model=model_name,
                                    choices=[StreamChoice(delta=DeltaMessage(content=accumulated))],
                                )
                                yield f"data: {chunk.model_dump_json()}\n\n"
                                accumulated = ""
                            continue
                        elif len(accumulated) < 10:
                            # Wait for more tokens to decide
                            continue
                        else:
                            # No <think> prefix, just emit directly
                            think_done = True
                    else:
                        # <think> already stripped, emit thinking content as-is
                        chunk = ChatCompletionStreamResponse(
                            id=request_id, created=created, model=model_name,
                            choices=[StreamChoice(delta=DeltaMessage(content=token))],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
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



# ── Text Completions ────────────────────────────────────────────────────────


@app.post("/v1/completions")
@app.post("/v3/completions")
async def completions(request: CompletionRequest):
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    max_tokens = request.max_tokens or config.max_tokens_default
    model_name = request.model or config.model_name
    temperature = request.temperature if request.temperature is not None else 1.0

    result = await text_backend.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        raw_prompt=True,
    )

    if request.stream:
        # For streaming completions, wrap result as SSE
        request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        async def event_generator():
            chunk = {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": result.text, "finish_reason": None}],
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

    return CompletionResponse(
        model=model_name,
        choices=[CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


# ── Models & Health ─────────────────────────────────────────────────────────


@app.get("/v1/models")
@app.get("/v3/models")
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
