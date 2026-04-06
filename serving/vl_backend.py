"""VL (Vision-Language) backend using modeling_qwen3_5 subprocess.

VLMPipeline does NOT support Qwen3.5 via Modeling API. The VL support
is only in the C++ modeling sample (modeling_qwen3_5 --mode vl).
This module wraps that exe as a subprocess for VL requests.

Supports both non-streaming and streaming modes. In streaming mode,
the exe emits JSON lines to stdout (one per token chunk) while logs
go to stderr.

Serve mode (--serve): Keeps exe running as a persistent subprocess.
Model is compiled once at startup; each request sends JSON to stdin
and reads JSON lines from stdout. Eliminates ~5s compile_model overhead.
"""

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("vl_backend")


@dataclass
class VLResult:
    text: str
    load_time_ms: float = 0
    generate_time_ms: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class VLBackend:
    """Subprocess-based VL backend using modeling_qwen3_5 binary.

    Supports two modes:
    1. Per-request subprocess (default): spawns exe for each request
    2. Serve mode (use_serve=True): persistent subprocess, model loaded once
    """

    def __init__(
        self,
        model_path: str,
        exe_path: str = "",
        device: str = "GPU",
        max_pixels: int = 602112,
        cache_model: bool = True,
        use_serve: bool = False,
        mode: str = "vl",  # "vl" or "text"
    ):
        self.model_path = model_path
        self.device = device
        self.max_pixels = max_pixels
        self.cache_model = cache_model
        self.use_serve = use_serve
        self.mode = mode
        self._lock = asyncio.Lock()

        # Serve mode state
        self._serve_proc: Optional[asyncio.subprocess.Process] = None
        self._serve_ready = False
        self._stderr_task: Optional[asyncio.Task] = None
        self._serve_needs_drain = False  # True after cancelled request
        self._serve_needs_restart = False  # True after degenerate output detected

        # Find exe
        if exe_path:
            self.exe_path = exe_path
        else:
            self.exe_path = self._find_exe()

        if not os.path.isfile(self.exe_path):
            logger.warning(f"VL exe not found: {self.exe_path} — VL requests will fail")

    def _find_exe(self) -> str:
        """Auto-detect modeling_qwen3_5 location (cross-platform).

        Search order:
        1. Deployment: runtime/openvino_genai/ (sibling to serving/)
        2. Dev build paths (for development use)
        """
        import sys
        is_win = sys.platform == "win32"
        exe_name = "modeling_qwen3_5.exe" if is_win else "modeling_qwen3_5"

        serving_dir = Path(__file__).parent
        deploy_root = serving_dir.parent
        deployment_candidates = [
            deploy_root / "runtime" / "openvino_genai" / exe_name,
        ]
        # Dev build paths
        dev_candidates = [
            Path(r"D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe"),
            Path(r"D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\modeling_qwen3_5.exe"),
            Path("/home/gta/chuansheng/src_code/openvino.genai/build/bin/Release/modeling_qwen3_5"),
        ]
        for p in deployment_candidates + dev_candidates:
            if p.is_file():
                return str(p)
        return str(deployment_candidates[0])

    @property
    def available(self) -> bool:
        return os.path.isfile(self.exe_path)

    # ----------------------------------------------------------------
    # Serve mode: persistent subprocess management
    # ----------------------------------------------------------------

    async def start_serve(self):
        """Start persistent serve subprocess. Blocks until ready signal."""
        if not self.use_serve:
            return
        if self._serve_proc and self._serve_proc.returncode is None:
            logger.info("Serve process already running")
            return

        cmd = [
            self.exe_path,
            "--model", self.model_path,
            "--mode", self.mode,
            "--device", self.device,
        ]
        if self.mode == "vl":
            cmd.extend(["--max-pixels", str(self.max_pixels)])
        cmd.append("--serve")
        if self.cache_model:
            cmd.append("--cache-model")

        env = self._make_env()
        logger.info(f"Starting {self.mode} serve subprocess: {' '.join(cmd[:6])}...")
        t0 = time.time()

        self._serve_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Drain stderr in background
        self._stderr_task = asyncio.create_task(self._drain_serve_stderr())

        # Wait for {"ready":true} signal on stdout
        try:
            ready = await asyncio.wait_for(self._wait_for_ready(), timeout=120)
            if not ready:
                raise RuntimeError("Serve process exited before ready signal")
        except asyncio.TimeoutError:
            logger.error("Serve process timed out waiting for ready signal")
            await self.stop_serve()
            raise RuntimeError("VL serve startup timed out (120s)")

        elapsed = time.time() - t0
        self._serve_ready = True
        logger.info(f"VL serve process ready in {elapsed:.1f}s")

    async def _wait_for_ready(self) -> bool:
        """Read stdout lines until we get the ready signal."""
        while True:
            line = await self._serve_proc.stdout.readline()
            if not line:
                return False  # process exited
            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str or not line_str.startswith("{"):
                continue
            try:
                msg = json.loads(line_str)
                if msg.get("ready"):
                    return True
            except json.JSONDecodeError:
                continue

    async def _drain_serve_stderr(self):
        """Continuously drain stderr from serve process to logger."""
        try:
            while self._serve_proc and self._serve_proc.returncode is None:
                line = await self._serve_proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug(f"[serve-exe] {text}")
        except Exception as e:
            logger.warning(f"Serve stderr drain error: {e}")

    async def stop_serve(self):
        """Stop persistent serve subprocess."""
        self._serve_ready = False
        if self._serve_proc and self._serve_proc.returncode is None:
            logger.info("Stopping VL serve subprocess")
            try:
                self._serve_proc.stdin.close()
                await asyncio.wait_for(self._serve_proc.wait(), timeout=10)
            except (asyncio.TimeoutError, Exception):
                try:
                    self._serve_proc.kill()
                    await self._serve_proc.wait()
                except Exception:
                    pass
        self._serve_proc = None
        if self._stderr_task:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def _ensure_serve(self):
        """Ensure serve process is running, restart if crashed."""
        if not self._serve_proc or self._serve_proc.returncode is not None:
            logger.warning("Serve process not running, restarting...")
            self._serve_ready = False
            self._serve_needs_drain = False
            await self.start_serve()

    async def _drain_cancelled_output(self):
        """Drain leftover output from a previously cancelled serve request."""
        logger.info("Draining leftover output from cancelled request...")
        try:
            while True:
                line = await asyncio.wait_for(
                    self._serve_proc.stdout.readline(),
                    timeout=30,
                )
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("{"):
                    continue
                try:
                    msg = json.loads(line_str)
                    if msg.get("done") or "error" in msg:
                        logger.info("Drain complete — subprocess ready for next request")
                        break
                except json.JSONDecodeError:
                    continue
        except asyncio.TimeoutError:
            logger.warning("Drain timed out — restarting serve process")
            await self.stop_serve()
            await self.start_serve()
        except Exception as e:
            logger.warning(f"Drain error: {e} — restarting serve process")
            await self.stop_serve()
            await self.start_serve()

    @staticmethod
    def _is_degenerate_output(text: str) -> bool:
        """Detect degenerate repetitive output that indicates state corruption.

        Common pattern: model outputs '<tool_call>!!!...' or just '!!!...'
        With C++ early stop (16 repeated tokens), degenerate output may be short.
        """
        if not text or len(text) < 10:
            return False
        from collections import Counter
        counts = Counter(text)
        if counts:
            most_common_char, most_common_count = counts.most_common(1)[0]
            # >50% single char dominance (e.g. '<tool_call>!!!...' = 16!/27 chars)
            if most_common_count > len(text) * 0.5 and most_common_count >= 8:
                return True
        return False

    async def _serve_request(
        self,
        prompt: str,
        image_data: Optional[bytes],
        max_tokens: int,
        temperature: float,
        think: bool,
        stream: bool = True,
        raw_prompt: bool = False,
    ):
        """Send request to serve subprocess and yield response chunks.

        Writes image + prompt to temp files, sends JSON request via stdin,
        reads JSON lines from stdout.
        """
        t0 = time.time()

        # Restart subprocess if previous request produced degenerate output (state corruption)
        if self._serve_needs_restart:
            logger.warning("Restarting serve subprocess due to prior degenerate output...")
            await self.stop_serve()
            self._serve_needs_restart = False

        await self._ensure_serve()

        # Drain any leftover output from a previously cancelled request
        if self._serve_needs_drain:
            await self._drain_cancelled_output()
            self._serve_needs_drain = False

        image_path, prompt_path = self._write_temp_files(prompt, image_data)
        t_files = time.time()
        try:
            request_json = json.dumps({
                "image_path": image_path,
                "prompt_file": prompt_path,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                "think": think,
                "raw_prompt": raw_prompt,
            })

            logger.info(f"Serve request: prompt='{prompt[:50]}...', max_tokens={max_tokens}, temperature={temperature}")
            self._serve_proc.stdin.write((request_json + "\n").encode("utf-8"))
            await self._serve_proc.stdin.drain()
            t_send = time.time()

            # Read response lines
            accumulated_text = ""
            t_first_line = None
            while True:
                line = await asyncio.wait_for(
                    self._serve_proc.stdout.readline(),
                    timeout=300,
                )
                if t_first_line is None:
                    t_first_line = time.time()
                if not line:
                    raise RuntimeError("Serve process died during request")

                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("{"):
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if "error" in msg:
                    raise RuntimeError(f"VL serve error: {msg['error']}")

                if "text" in msg:
                    chunk = msg["text"]
                    accumulated_text += chunk
                    yield chunk

                if msg.get("done"):
                    self._last_stream_result = VLResult(
                        text=accumulated_text,
                        generate_time_ms=msg.get("decode_ms", 0),
                        prompt_tokens=msg.get("prompt_tokens", 0),
                        completion_tokens=msg.get("completion_tokens", 0),
                        finish_reason=msg.get("finish_reason", "stop"),
                    )
                    t_done = time.time()
                    logger.info(
                        f"Serve timing: files={1000*(t_files-t0):.0f}ms, "
                        f"send={1000*(t_send-t_files):.0f}ms, "
                        f"first_line={1000*((t_first_line or t_done)-t_send):.0f}ms, "
                        f"total={1000*(t_done-t0):.0f}ms"
                    )
                    # Detect degenerate output — schedule subprocess restart
                    if self._is_degenerate_output(accumulated_text):
                        logger.warning(f"Degenerate output detected ({len(accumulated_text)} chars), "
                                       "scheduling subprocess restart")
                        self._serve_needs_restart = True
                    break

        except (asyncio.CancelledError, GeneratorExit):
            # Don't kill the serve process — mark for draining on next request
            self._serve_needs_drain = True
            logger.info("Serve request cancelled by client — will drain on next request")
            raise
        finally:
            self._cleanup_temp_files(image_path, prompt_path)

    # ----------------------------------------------------------------
    # Public API: generate (non-streaming) and generate_stream
    # ----------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        image_data: Optional[bytes] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: bool = False,
        raw_prompt: bool = False,
    ) -> VLResult:
        """Run generation. Uses serve mode if available, else subprocess."""
        if not self.available:
            raise RuntimeError(f"VL exe not found: {self.exe_path}")

        if self.use_serve:
            async with self._lock:
                accumulated = ""
                async for chunk in self._serve_request(
                    prompt, image_data, max_tokens, temperature, think,
                    stream=False, raw_prompt=raw_prompt,
                ):
                    accumulated += chunk
                return self._last_stream_result or VLResult(text=accumulated)

        async with self._lock:
            return await self._run_subprocess(prompt, image_data, max_tokens, temperature, think)

    async def _run_subprocess(
        self,
        prompt: str,
        image_data: bytes,
        max_tokens: int,
        temperature: float,
        think: bool,
    ) -> VLResult:
        """Execute the VL subprocess."""
        image_path, prompt_path = self._write_temp_files(prompt, image_data)
        try:
            cmd = self._build_cmd(image_path, prompt_path, max_tokens, temperature, think)
            env = self._make_env()

            logger.info(f"VL subprocess: prompt='{prompt[:50]}...', image={len(image_data)} bytes")
            t0 = time.time()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=300,  # 5 min max
            )

            elapsed = time.time() - t0
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                logger.error(f"VL subprocess failed (rc={proc.returncode}): {stderr_text[:500]}")
                raise RuntimeError(f"VL generation failed: {stderr_text[:200]}")

            # Parse output — extract generated text
            text = self._parse_output(stdout_text)
            logger.info(f"VL done in {elapsed:.1f}s, output={len(text)} chars")

            return VLResult(text=text, generate_time_ms=elapsed * 1000)

        finally:
            self._cleanup_temp_files(image_path, prompt_path)

    async def generate_stream(
        self,
        prompt: str,
        image_data: Optional[bytes] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: bool = False,
        raw_prompt: bool = False,
    ):
        """Streaming generation. Yields text chunks as they arrive.

        Uses serve mode if available (persistent subprocess),
        otherwise spawns a per-request subprocess with --stream.

        Protocol: exe emits JSON lines to stdout:
          {"text":"chunk"}          — incremental decoded text
          {"done":true,...}         — generation complete with metadata
        Logs/metrics go to stderr.

        Yields: str chunks of generated text
        Returns metadata via VLResult stored in self._last_stream_result
        """
        if not self.available:
            raise RuntimeError(f"VL exe not found: {self.exe_path}")

        if self.use_serve:
            async with self._lock:
                async for chunk in self._serve_request(
                    prompt, image_data, max_tokens, temperature, think,
                    stream=True, raw_prompt=raw_prompt,
                ):
                    yield chunk
            return

        async with self._lock:
            async for chunk in self._run_subprocess_stream(
                prompt, image_data, max_tokens, temperature, think
            ):
                yield chunk

    async def _run_subprocess_stream(
        self,
        prompt: str,
        image_data: bytes,
        max_tokens: int,
        temperature: float,
        think: bool,
    ):
        """Execute VL subprocess in streaming mode."""
        image_path, prompt_path = self._write_temp_files(prompt, image_data)
        proc = None
        try:
            cmd = self._build_cmd(
                image_path, prompt_path, max_tokens, temperature, think,
                stream=True,
            )
            env = self._make_env()

            logger.info(f"VL stream subprocess: prompt='{prompt[:50]}...', image={len(image_data)} bytes")
            t0 = time.time()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Drain stderr concurrently to prevent blocking
            stderr_lines = []
            async def _drain_stderr():
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    stderr_lines.append(line.decode("utf-8", errors="replace"))
            stderr_task = asyncio.create_task(_drain_stderr())

            # Read stdout line by line — parse JSON streaming protocol
            self._last_stream_result = None
            accumulated_text = ""

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                line_str = line.decode("utf-8", errors="replace").strip()
                if not line_str or not line_str.startswith("{"):
                    continue  # skip non-JSON lines (log output)

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if "text" in msg:
                    chunk = msg["text"]
                    accumulated_text += chunk
                    yield chunk
                elif msg.get("done"):
                    elapsed = time.time() - t0
                    self._last_stream_result = VLResult(
                        text=accumulated_text,
                        generate_time_ms=elapsed * 1000,
                        prompt_tokens=msg.get("prompt_tokens", 0),
                        completion_tokens=msg.get("completion_tokens", 0),
                        finish_reason=msg.get("finish_reason", "stop"),
                    )

            await stderr_task
            await proc.wait()

            if proc.returncode != 0:
                stderr_text = "".join(stderr_lines)
                logger.error(f"VL stream subprocess failed (rc={proc.returncode}): {stderr_text[:500]}")
                raise RuntimeError(f"VL generation failed: {stderr_text[:200]}")

            logger.info(f"VL stream done in {time.time()-t0:.1f}s, output={len(accumulated_text)} chars")

        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected — kill subprocess
            if proc and proc.returncode is None:
                logger.info("VL stream cancelled, killing subprocess")
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            raise
        finally:
            self._cleanup_temp_files(image_path, prompt_path)

    def _write_temp_files(self, prompt: str, image_data: Optional[bytes] = None) -> tuple[str, str]:
        """Write prompt and optionally image to temp files, return paths."""
        image_path = ""
        if image_data:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_data)
                image_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as f:
            f.write(prompt)
            prompt_path = f.name
        return image_path, prompt_path

    def _build_cmd(self, image_path, prompt_path, max_tokens, temperature, think, stream=False):
        """Build subprocess command line."""
        cmd = [
            self.exe_path,
            "--model", self.model_path,
            "--mode", self.mode,
        ]
        if image_path:
            cmd.extend(["--image", image_path])
        cmd.extend([
            "--prompt-file", prompt_path,
            "--device", self.device,
            "--output-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--think", "1" if think else "0",
        ])
        if self.mode == "vl":
            cmd.extend(["--max-pixels", str(self.max_pixels)])
        if self.cache_model:
            cmd.append("--cache-model")
        if stream:
            cmd.append("--stream")
        return cmd

    def _make_env(self):
        """Build environment for subprocess (cross-platform)."""
        import sys
        env = os.environ.copy()
        env["OV_GENAI_USE_MODELING_API"] = "1"
        env["OV_GENAI_INFLIGHT_QUANT_MODE"] = "int4_asym"
        env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = "128"
        env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = "int4_asym"
        exe_dir = str(Path(self.exe_path).parent)
        if sys.platform == "win32":
            # Windows: add DLL directory to PATH
            env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
        else:
            # Linux/macOS: add .so directory to LD_LIBRARY_PATH
            env["LD_LIBRARY_PATH"] = exe_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        return env

    @staticmethod
    def _cleanup_temp_files(*paths):
        """Remove temp files, ignoring errors."""
        for p in paths:
            if not p:
                continue
            try:
                os.unlink(p)
            except OSError:
                pass

    def _parse_output(self, stdout: str) -> str:
        """Extract generated text from exe stdout.

        The exe output format:
          [log lines...]
          Throughput: XX.XX tokens/s
          <actual model output here>

        The generated text always comes after the last "Throughput:" or "tokens/s" line.
        """
        lines = stdout.split("\n")

        # Find the last metric line — generated text is everything after it
        last_metric_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if any(marker in stripped for marker in [
                "tokens/s", "Throughput:", "TPOT:", "TTFT:",
                "Decode time:", "Output token size:", "Prompt token size:",
                "Mode: hf",
            ]):
                last_metric_idx = i

        if last_metric_idx >= 0:
            text_lines = lines[last_metric_idx + 1:]
        else:
            # Fallback: skip known log prefixes
            text_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("[") or stripped.startswith("  -> "):
                    continue
                if any(kw in stripped for kw in [
                    "Time:", "Statistics:", "Loading", "Compiling",
                    "Mapping", "Mapped", "Quantiz", "Total weights",
                    "Primary mode:", "Backup mode:", "Timing (ms):",
                    "Fetch:", "Quant:", "Graph:", "Total:",
                    "cache-model", "Zero-Copy", "coverage",
                ]):
                    continue
                text_lines.append(line)

        text = "\n".join(text_lines).strip()
        # Strip <think>...</think> (closed)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
        # Strip unclosed <think>... (truncated output)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
        return text


async def decode_image_content(content_parts: list) -> Optional[bytes]:
    """Extract image bytes from OpenAI content parts.

    Supports:
      - {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      - {"type": "image_url", "image_url": {"url": "https://..."}}
    """
    for part in content_parts:
        if not hasattr(part, "type") or part.type != "image_url":
            continue
        if not part.image_url or "url" not in part.image_url:
            continue

        url = part.image_url["url"]

        # Base64 data URL
        if url.startswith("data:"):
            # data:image/jpeg;base64,/9j/4AAQ...
            try:
                _, encoded = url.split(",", 1)
                return base64.b64decode(encoded)
            except Exception as e:
                logger.error(f"Failed to decode base64 image: {e}")
                continue

        # HTTP(S) URL
        if url.startswith("http://") or url.startswith("https://"):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    return r.content
            except Exception as e:
                logger.error(f"Failed to fetch image from {url}: {e}")
                continue

    return None


def has_image_content(messages: list) -> bool:
    """Check if any message contains image content."""
    for m in messages:
        if hasattr(m, "content") and isinstance(m.content, list):
            for part in m.content:
                if hasattr(part, "type") and part.type == "image_url":
                    return True
    return False


def extract_text_from_content(content) -> str:
    """Extract text from string or multimodal content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if hasattr(part, "type") and part.type == "text" and part.text:
                texts.append(part.text)
        return "\n".join(texts)
    return str(content) if content else ""
