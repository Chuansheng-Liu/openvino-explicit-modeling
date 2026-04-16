#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
One-click MMMU benchmark (VLMPipeline, same setup as run_vlm_qwen3_5).
Load parquet from local MMMU_Pro_dataset, infer with VLMPipeline, output for mmmu-pro/evaluate.py.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# Import datasets before script modifies sys.path (bat no longer sets PYTHONPATH)
try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("Missing dependency: pip install datasets") from None

SCRIPT_DIR = Path(__file__).resolve().parent
MMMU_ROOT = SCRIPT_DIR
MMMU_PRO = MMMU_ROOT / "mmmu-pro"
DATASET_PATH = MMMU_ROOT / "MMMU_Pro_dataset"
PROMPTS_YAML = MMMU_PRO / "prompts.yaml"
OUTPUT_DIR = MMMU_PRO / "output"

# === OpenVINO setup (same as run_vlm_qwen3_5) ===
_openvino_root = os.environ.get("OPENVINO_ROOT")
if _openvino_root:
    _openvino_root = Path(_openvino_root).resolve()
else:
    _openvino_root = (Path(__file__).resolve().parent / ".." / "..").resolve()

_ov_python = _openvino_root / "openvino" / "bin" / "intel64" / "Release" / "python"
_ov_genai = _openvino_root / "openvino.genai" / "build"
for _p in (_ov_python, _ov_genai):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _dll_paths = [
        _openvino_root / "openvino" / "bin" / "intel64" / "Release",
        _openvino_root / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin",
        _openvino_root / "openvino" / "build" / "install" / "runtime" / "bin" / "intel64" / "Release",
        _openvino_root / "openvino.genai" / "build" / "openvino_genai",
        _openvino_root / "openvino.genai" / "build" / "bin" / "Release",
    ]
    for _p in _dll_paths:
        if _p.exists():
            os.add_dll_directory(str(_p))
    for _ep in os.environ.get("OPENVINO_LIB_PATHS", "").split(";"):
        _ep = _ep.strip()
        if not _ep:
            continue
        _p = Path(_ep).resolve()
        if _p.exists() and _p.is_dir() and str(_p) != ".":
            os.add_dll_directory(str(_p))

import numpy as np

if os.environ.get("OV_GENAI_USE_MODELING_API", "").lower() not in ("1", "true", "yes"):
    os.environ["OV_GENAI_USE_MODELING_API"] = "1"

import openvino_genai
from openvino import Tensor

try:
    import yaml
except ImportError:
    raise SystemExit("Missing dependency: pip install pyyaml") from None


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    return "\n".join(f"{l}. {o}" for l, o in zip(option_letters, options))


def construct_prompt(doc, prompt_config: dict) -> str:
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    return f"{question}\n{parsed_options}\n{prompt_config['standard']}"


def pil_to_tensor(pil_image) -> Tensor:
    """Convert PIL Image to OpenVINO Tensor (RGB, uint8)."""
    arr = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    return Tensor(arr)


def _streamer(subword: str) -> bool:
    """Streaming callback: print each subword to terminal."""
    print(subword, end="", flush=True)
    return False


def extract_response(result) -> str:
    if isinstance(result, dict) and "error" in result:
        return ""
    if hasattr(result, "texts") and result.texts:
        return result.texts[0].strip()
    return str(result).strip()


def main():
    default_group_size = _safe_int(os.environ.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", "128"), 128)

    parser = argparse.ArgumentParser(
        description="Run MMMU benchmark with OpenVINO VLMPipeline (Qwen3.5-VL)"
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=os.environ.get("QWEN3_VL_MODEL", r"D:\Data\models\Huggingface\Qwen3.5-35B-A3B"),
        help="Path to Qwen3.5-VL HF model directory",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "cot"],
        default="direct",
        help="Reasoning mode: direct or cot (default: direct)",
    )
    parser.add_argument(
        "--setting",
        default="vision",
        help="Dataset setting: vision | 'standard (10 options)' (default: vision)",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device: CPU or GPU (default: GPU)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit samples (0=all, for testing)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluate.py after inference",
    )
    parser.add_argument(
        "--inflight-quant-mode",
        default=os.environ.get("OV_GENAI_INFLIGHT_QUANT_MODE", ""),
        help="In-flight quantization mode",
    )
    parser.add_argument(
        "--inflight-quant-group-size",
        type=int,
        default=default_group_size,
        help="In-flight quant group size",
    )
    parser.add_argument(
        "--inflight-quant-backup-mode",
        default=os.environ.get("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", ""),
        help="Backup quant mode",
    )
    args = parser.parse_args()

    quant_mode = (args.inflight_quant_mode or "").strip().lower()
    if quant_mode in ("", "none", "off", "disable", "disabled", "0"):
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_MODE", None)
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", None)
        os.environ.pop("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", None)
    else:
        os.environ["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant_mode
        os.environ["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(args.inflight_quant_group_size)
        if args.inflight_quant_backup_mode:
            os.environ["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = args.inflight_quant_backup_mode

    with open(PROMPTS_YAML, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)[args.mode]

    parquet_dir = DATASET_PATH / args.setting
    parquet_files = sorted(glob.glob(str(parquet_dir / "test-*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files in {parquet_dir}. "
            "Download MMMU_Pro dataset (e.g. from HuggingFace) to MMMU_Pro_dataset/vision/ or standard (10 options)/."
        )

    dataset = load_dataset("parquet", data_files={"test": parquet_files}, split="test")
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    model_name = Path(args.model_dir).name
    output_path = OUTPUT_DIR / f"{model_name}_{args.setting}_{args.mode}.jsonl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_dir}")
    print(f"Device: {args.device}, Mode: {args.mode}, Setting: {args.setting}")
    print(f"Total samples: {len(dataset)}")
    print(f"Output: {output_path}")

    enable_cache = {}
    if args.device == "GPU":
        enable_cache["CACHE_DIR"] = "vlm_qwen3_5_cache"

    pipe = openvino_genai.VLMPipeline(args.model_dir, args.device, **enable_cache)
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 2048
    config.do_sample = False
    config.repetition_penalty = 1.1
    config.temperature = 0.0
    config.top_p = 1.0
    config.top_k = 0

    results = []
    for i, data in enumerate(dataset):
        print(f"[{i + 1}/{len(dataset)}] id={data.get('id', i)}...")

        if args.setting == "vision":
            prompt = prompt_config["vision"]
            img = data.get("image")
            if img is None:
                results.append(({"error": "no image"}, data))
                continue
            images = [pil_to_tensor(img)]
        elif "standard" in args.setting:
            prompt = construct_prompt(data, prompt_config)
            img = data.get("image_1")
            if img is None:
                results.append(({"error": "no image_1"}, data))
                continue
            images = [pil_to_tensor(img)]
        else:
            results.append(({"error": f"unsupported setting: {args.setting}"}, data))
            continue

        try:
            print("Prompt:", prompt[:300] + ("..." if len(prompt) > 300 else ""))
            print("Response: ", end="", flush=True)
            out = pipe.generate(
                prompt,
                images=images,
                generation_config=config,
                streamer=_streamer,
            )
            print()  # newline after streaming
            response = extract_response(out)
        except Exception as e:
            response = {"error": str(e)}
            print("Response ERROR:", str(e))
        print("-" * 60)
        results.append((response, data))

        if (i + 1) % 10 == 0 or i + 1 == len(dataset):
            with open(output_path, "w", encoding="utf-8") as f:
                for resp, d in results:
                    row = {k: v for k, v in d.items() if not (k.startswith("image_") or k == "image")}
                    if "subdomain" not in row and "subject" in row:
                        row["subdomain"] = row["subject"]
                    row["response"] = resp if isinstance(resp, str) else (resp.get("error", "") if isinstance(resp, dict) else str(resp))
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  -> Checkpoint: saved {len(results)}")

    print(f"Saved to {output_path}")

    if not args.skip_eval and MMMU_PRO.exists():
        print("Running MMMU evaluate.py...")
        subprocess.run(
            [sys.executable, str(MMMU_PRO / "evaluate.py")],
            cwd=str(MMMU_PRO),
            check=True,
        )
    else:
        if args.skip_eval:
            print("Skipped evaluation (--skip-eval)")
        else:
            print("Skipped evaluation (mmmu-pro not found)")


if __name__ == "__main__":
    main()