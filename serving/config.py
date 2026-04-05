"""Configuration for the Qwen3.5 OpenAI-compatible serving server."""

import argparse
import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    model_path: str = ""
    device: str = "GPU"
    host: str = "0.0.0.0"
    port: int = 8000
    num_workers: int = 1
    max_tokens_default: int = 2048
    quant_mode: str = ""  # "", "int4_asym", "int8_sym"
    quant_group_size: int = 128
    model_name: str = ""  # Display name for /v1/models

    def __post_init__(self):
        if not self.model_name and self.model_path:
            self.model_name = os.path.basename(self.model_path.rstrip("/\\"))


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Qwen3.5 OpenAI-compatible API server")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--device", default="GPU", help="Device (GPU/CPU)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1, help="Number of LLMPipeline instances")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Default max tokens")
    parser.add_argument("--quant", default="", help="Quantization mode (int4_asym, int8_sym)")
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--model-name", default="", help="Model display name")

    args = parser.parse_args()
    return ServerConfig(
        model_path=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        max_tokens_default=args.max_tokens,
        quant_mode=args.quant,
        quant_group_size=args.quant_group_size,
        model_name=args.model_name,
    )
