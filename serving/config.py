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
    quant_mode: str = "int4_asym"
    quant_group_size: int = 128
    quant_backup_mode: str = "int4_asym"
    model_name: str = ""  # Display name for /v1/models
    vl_exe: str = ""  # Path to modeling_qwen3_5.exe for VL

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
    parser.add_argument("--quant", default="int4_asym", help="Quantization mode (int4_asym, int8_sym, none)")
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--quant-backup", default="int4_asym", help="Backup quantization mode")
    parser.add_argument("--model-name", default="", help="Model display name")
    parser.add_argument("--vl-exe", default="", help="Path to modeling_qwen3_5.exe for VL (auto-detected if empty)")

    args = parser.parse_args()
    quant = "" if args.quant == "none" else args.quant
    return ServerConfig(
        model_path=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        num_workers=args.workers,
        max_tokens_default=args.max_tokens,
        quant_mode=quant,
        quant_group_size=args.quant_group_size,
        quant_backup_mode="" if args.quant == "none" else args.quant_backup,
        model_name=args.model_name,
        vl_exe=args.vl_exe,
    )
