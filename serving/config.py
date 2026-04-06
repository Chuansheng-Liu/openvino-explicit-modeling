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
    max_tokens_default: int = 4096  # Large default for thinking models (thinking + content)
    model_name: str = ""  # Display name for /v1/models
    vl_exe: str = ""  # Path to modeling_qwen3_5.exe for VL
    serve_vl: bool = True  # Use persistent VL subprocess (eliminates model load per request)
    think: bool = False  # Enable thinking mode (default: off for 4B models)

    def __post_init__(self):
        if not self.model_name and self.model_path:
            self.model_name = os.path.basename(self.model_path.rstrip("/\\"))


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(description="Qwen3.5 OpenAI-compatible API server")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--device", default="GPU", help="Device (GPU/CPU)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Default max tokens")
    parser.add_argument("--model-name", default="", help="Model display name")
    parser.add_argument("--vl-exe", default="", help="Path to modeling_qwen3_5.exe for VL (auto-detected if empty)")
    parser.add_argument("--serve-vl", action="store_true", default=True, help="Use persistent VL subprocess (default: enabled)")
    parser.add_argument("--no-serve-vl", action="store_true", help="Disable persistent VL subprocess, use per-request mode")
    parser.add_argument("--think", action="store_true", default=False, help="Enable thinking mode (default: off)")
    # Legacy args (ignored, kept for backward compat with existing start.bat)
    parser.add_argument("--lazy-engine", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--quant", default="none", help=argparse.SUPPRESS)
    parser.add_argument("--quant-group-size", type=int, default=128, help=argparse.SUPPRESS)
    parser.add_argument("--quant-backup", default="", help=argparse.SUPPRESS)

    args = parser.parse_args()
    return ServerConfig(
        model_path=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        max_tokens_default=args.max_tokens,
        model_name=args.model_name,
        vl_exe=args.vl_exe,
        serve_vl=not args.no_serve_vl,
        think=args.think,
    )
