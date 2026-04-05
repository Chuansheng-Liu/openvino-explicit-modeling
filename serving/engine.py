"""Worker pool engine wrapping openvino_genai LLMPipeline instances.

Each worker owns one LLMPipeline (B=1) with isolated InferRequest and
Variable state. Workers are acquired from a pool and released after use.
"""

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import openvino_genai as og

from config import ServerConfig

logger = logging.getLogger("engine")


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"


class Worker:
    """A single LLMPipeline instance."""

    def __init__(self, worker_id: int, config: ServerConfig):
        self.worker_id = worker_id
        self.config = config
        self.pipeline: Optional[og.LLMPipeline] = None
        self.tokenizer: Optional[og.Tokenizer] = None
        self._lock = threading.Lock()

    def initialize(self):
        """Load the model (called once at startup)."""
        logger.info(f"Worker {self.worker_id}: Loading model from {self.config.model_path}")
        t0 = time.time()
        self.pipeline = og.LLMPipeline(
            self.config.model_path,
            self.config.device,
        )
        self.tokenizer = self.pipeline.get_tokenizer()
        logger.info(f"Worker {self.worker_id}: Model loaded in {time.time()-t0:.1f}s")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
        streamer_callback: Optional[Callable[[str], bool]] = None,
    ) -> GenerateResult:
        """Run synchronous generation. Called from thread pool."""
        config = og.GenerationConfig()
        config.max_new_tokens = max_tokens

        if temperature is not None:
            if temperature < 1e-6:
                # Greedy
                config.do_sample = False
            else:
                config.do_sample = True
                config.temperature = temperature

        if top_p is not None:
            config.top_p = top_p
        if top_k is not None:
            config.top_k = top_k
        if stop:
            config.stop_strings = set(stop)

        # Estimate prompt tokens (approximate — tokenizer.encode could be used for precision)
        prompt_tokens = len(prompt) // 4  # rough estimate

        if streamer_callback:
            # Streaming: use TextStreamer with callback
            completion_tokens = 0

            def _callback(text: str) -> og.StreamingStatus:
                nonlocal completion_tokens
                completion_tokens += 1
                keep_going = streamer_callback(text)
                if keep_going:
                    return og.StreamingStatus.RUNNING
                return og.StreamingStatus.CANCEL

            streamer = og.TextStreamer(self.tokenizer, _callback)
            result = self.pipeline.generate(prompt, config, streamer)
            return GenerateResult(
                text=str(result),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason="stop",
            )
        else:
            # Non-streaming
            result = self.pipeline.generate(prompt, config)
            text = str(result)
            # Estimate completion tokens (encode().input_ids has pybind issues)
            completion_tokens = max(1, len(text) // 4)
            return GenerateResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason="stop" if len(text) > 0 else "length",
            )

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Format messages using the model's chat template."""
        kwargs = {"add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **kwargs)


class Engine:
    """Pool of LLMPipeline workers with async interface."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.workers: list[Worker] = []
        self._pool: asyncio.Queue[Worker] = asyncio.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=config.num_workers,
            thread_name_prefix="ov-worker",
        )
        self._started = False

    async def start(self):
        """Initialize all workers. Call once at startup."""
        if self._started:
            return

        # Set environment variables for Modeling API
        os.environ["OV_GENAI_USE_MODELING_API"] = "1"
        if self.config.quant_mode:
            os.environ["OV_GENAI_INFLIGHT_QUANT_MODE"] = self.config.quant_mode
            os.environ["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(self.config.quant_group_size)
            if self.config.quant_backup_mode:
                os.environ["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = self.config.quant_backup_mode
            logger.info(f"Quantization: mode={self.config.quant_mode}, gs={self.config.quant_group_size}, backup={self.config.quant_backup_mode}")

        loop = asyncio.get_event_loop()
        for i in range(self.config.num_workers):
            worker = Worker(i, self.config)
            # Initialize in thread to not block event loop
            await loop.run_in_executor(self._executor, worker.initialize)
            self.workers.append(worker)
            await self._pool.put(worker)

        self._started = True
        logger.info(f"Engine started with {self.config.num_workers} worker(s)")

    async def acquire_worker(self, timeout: float = 300.0) -> Worker:
        """Get a free worker from the pool."""
        try:
            worker = await asyncio.wait_for(self._pool.get(), timeout=timeout)
            logger.info(f"Worker {worker.worker_id} acquired (pool size: {self._pool.qsize()})")
            return worker
        except asyncio.TimeoutError:
            raise RuntimeError("All workers busy, request timed out")

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self._pool.put(worker)
        logger.info(f"Worker {worker.worker_id} released (pool size: {self._pool.qsize()})")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> GenerateResult:
        """Non-streaming generation."""
        worker = await self.acquire_worker()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: worker.generate(
                    prompt, max_tokens, temperature, top_p, top_k, stop,
                ),
            )
            return result
        finally:
            await self.release_worker(worker)

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ):
        """Streaming generation. Yields text chunks via async generator."""
        worker = await self.acquire_worker()
        queue: asyncio.Queue[Optional[str | Exception]] = asyncio.Queue()
        loop = asyncio.get_event_loop()
        cancelled = threading.Event()

        def _callback(text: str) -> bool:
            if cancelled.is_set():
                return False  # stop generation
            loop.call_soon_threadsafe(queue.put_nowait, text)
            return True

        def _run():
            try:
                worker.generate(
                    prompt, max_tokens, temperature, top_p, top_k, stop,
                    streamer_callback=_callback,
                )
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # signal done

        future = self._executor.submit(_run)

        try:
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        except (asyncio.CancelledError, GeneratorExit):
            cancelled.set()
            # Wait for generation thread to finish before releasing worker
            await asyncio.get_event_loop().run_in_executor(None, future.result, 5.0)
        finally:
            # Ensure generation thread is done before releasing worker
            if not future.done():
                cancelled.set()
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, future.result),
                        timeout=10.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    pass
            await self.release_worker(worker)

    def apply_chat_template(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Use the first worker's tokenizer for chat template formatting."""
        if not self.workers:
            raise RuntimeError("Engine not started")
        return self.workers[0].apply_chat_template(messages, tools)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    async def shutdown(self):
        """Shutdown all workers."""
        self._executor.shutdown(wait=False)
        logger.info("Engine shut down")
