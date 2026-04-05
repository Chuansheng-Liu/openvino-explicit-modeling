"""VL (Vision-Language) backend using modeling_qwen3_5.exe subprocess.

VLMPipeline does NOT support Qwen3.5 via Modeling API. The VL support
is only in the C++ modeling sample (modeling_qwen3_5.exe --mode vl).
This module wraps that exe as a subprocess for VL requests.

Supports both non-streaming and streaming modes. In streaming mode,
the exe emits JSON lines to stdout (one per token chunk) while logs
go to stderr.
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
    """Subprocess-based VL backend using modeling_qwen3_5.exe."""

    def __init__(
        self,
        model_path: str,
        exe_path: str = "",
        device: str = "GPU",
        max_pixels: int = 602112,
        cache_model: bool = True,
    ):
        self.model_path = model_path
        self.device = device
        self.max_pixels = max_pixels
        self.cache_model = cache_model
        self._lock = asyncio.Lock()

        # Find exe
        if exe_path:
            self.exe_path = exe_path
        else:
            self.exe_path = self._find_exe()

        if not os.path.isfile(self.exe_path):
            logger.warning(f"VL exe not found: {self.exe_path} — VL requests will fail")

    def _find_exe(self) -> str:
        """Auto-detect modeling_qwen3_5.exe location.

        Search order:
        1. Deployment: runtime/openvino_genai/ (sibling to serving/)
        2. Dev build paths (for development use)
        """
        # Deployment layout: serving/ is CWD, runtime/ is sibling
        serving_dir = Path(__file__).parent
        deploy_root = serving_dir.parent
        deployment_candidates = [
            deploy_root / "runtime" / "openvino_genai" / "modeling_qwen3_5.exe",
        ]
        # Dev build paths
        dev_candidates = [
            Path(r"D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe"),
            Path(r"D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\modeling_qwen3_5.exe"),
        ]
        for p in deployment_candidates + dev_candidates:
            if p.is_file():
                return str(p)
        return str(deployment_candidates[0])

    @property
    def available(self) -> bool:
        return os.path.isfile(self.exe_path)

    async def generate(
        self,
        prompt: str,
        image_data: bytes,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: bool = False,
    ) -> VLResult:
        """Run VL generation via subprocess. Only one at a time."""
        if not self.available:
            raise RuntimeError(f"VL exe not found: {self.exe_path}")

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
        image_data: bytes,
        max_tokens: int = 512,
        temperature: float = 0.0,
        think: bool = False,
    ):
        """Streaming VL generation. Yields text chunks as they arrive.

        Protocol: exe with --stream emits JSON lines to stdout:
          {"text":"chunk"}          — incremental decoded text
          {"done":true,...}         — generation complete with metadata
        Logs/metrics go to stderr.

        Yields: str chunks of generated text
        Returns metadata via VLResult stored in self._last_stream_result
        """
        if not self.available:
            raise RuntimeError(f"VL exe not found: {self.exe_path}")

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

    def _write_temp_files(self, prompt: str, image_data: bytes) -> tuple[str, str]:
        """Write prompt and image to temp files, return paths."""
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
            "--mode", "vl",
            "--image", image_path,
            "--prompt-file", prompt_path,
            "--device", self.device,
            "--output-tokens", str(max_tokens),
            "--temperature", str(temperature),
            "--think", "1" if think else "0",
            "--max-pixels", str(self.max_pixels),
        ]
        if self.cache_model:
            cmd.append("--cache-model")
        if stream:
            cmd.append("--stream")
        return cmd

    def _make_env(self):
        """Build environment for subprocess."""
        env = os.environ.copy()
        env["OV_GENAI_USE_MODELING_API"] = "1"
        env["OV_GENAI_INFLIGHT_QUANT_MODE"] = "int4_asym"
        env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = "128"
        env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = "int4_asym"
        exe_dir = str(Path(self.exe_path).parent)
        env["PATH"] = exe_dir + os.pathsep + env.get("PATH", "")
        return env

    @staticmethod
    def _cleanup_temp_files(*paths):
        """Remove temp files, ignoring errors."""
        for p in paths:
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
