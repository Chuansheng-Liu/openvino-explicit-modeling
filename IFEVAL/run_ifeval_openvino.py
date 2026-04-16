
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IFEVAL_ROOT = SCRIPT_DIR
DEFAULT_INPUT = IFEVAL_ROOT / "data" / "input_data.jsonl"
DEFAULT_OUTPUT_BASE = IFEVAL_ROOT / "data"

SYSTEM_PREFIX = "<|im_start|>system\n"
SYSTEM_SUFFIX = "<|im_end|>\n"
USER_PREFIX = "<|im_start|>user\n"
USER_SUFFIX = "<|im_end|>\n"
ASSISTANT_PREFIX = "<|im_start|>assistant\n"


def _safe_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_prompt(user_message: str) -> str:
    """Build ChatML-format prompt (same as inst_eval manual template)."""
    return (
        f"{SYSTEM_PREFIX}{SYSTEM_SUFFIX}"
        f"{USER_PREFIX}{user_message}{USER_SUFFIX}"
        f"{ASSISTANT_PREFIX}"
    )


def _streamer(subword: str) -> bool:
    """Streaming callback: print each subword to terminal."""
    print(subword, end="", flush=True)
    return False


def extract_response(result) -> str:
    """Extract text from pipe.generate return value."""
    if hasattr(result, "texts") and result.texts:
        text = result.texts[0]
    else:
        text = str(result)
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0]
    return text.strip()


def main():
    default_group_size = _safe_int(os.environ.get("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", "128"), 128)

    parser = argparse.ArgumentParser(
        description="Run IFEval benchmark with OpenVINO LLMPipeline (Qwen3.5)"
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        default=os.environ.get("MODEL", r"D:\Data\models\Huggingface\Qwen3.5-35B-A3B"),
        help="Path to Qwen3.5 HF model directory",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device: CPU or GPU (default: GPU)",
    )
    parser.add_argument(
        "--input-data",
        default=str(DEFAULT_INPUT),
        help=f"Path to IFEval input_data.jsonl (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-base",
        default=str(DEFAULT_OUTPUT_BASE),
        help=f"Base output directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name for output subdir (default: basename of model_dir)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens (default: 2048)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of prompts to process (0=all, for testing)",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluation_main.py after generation",
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

    if os.environ.get("OV_GENAI_USE_MODELING_API", "").lower() not in ("1", "true", "yes"):
        os.environ["OV_GENAI_USE_MODELING_API"] = "1"

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

    model_name = args.model_name or Path(args.model_dir).name
    output_dir = Path(args.output_base) / model_name / "ifeval"
    output_jsonl = output_dir / "output.jsonl"

    if not Path(args.input_data).exists():
        raise FileNotFoundError(f"Input data not found: {args.input_data}")

    with open(args.input_data, "r", encoding="utf-8") as f:
        json_list = [json.loads(line) for line in f if line.strip()]

    prompts_raw = [j["prompt"] for j in json_list]
    if args.limit > 0:
        prompts_raw = prompts_raw[: args.limit]
        json_list = json_list[: args.limit]
    prompts = [build_prompt(p) for p in prompts_raw]

    print(f"Loading model: {args.model_dir}")
    print(f"Device: {args.device}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Output: {output_jsonl}")

    import openvino_genai

    pipe = openvino_genai.LLMPipeline(args.model_dir, args.device)
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 512
    config.do_sample = True
    config.temperature = 1.0
    config.top_p = 0.95
    config.top_k = 20
    config.presence_penalty = 1.5
    config.repetition_penalty = 1.0
    config.no_repeat_ngram_size = 3

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for i, (prompt, raw) in enumerate(zip(prompts, prompts_raw)):
        print(f"\n[{i + 1}/{len(prompts)}] Generating...")
        print("-" * 60)
        print("Prompt:", raw[:500] + ("..." if len(raw) > 500 else ""))
        print("Response: ", end="", flush=True)
        result = pipe.generate(prompt, config, streamer=_streamer)
        print()  # newline after streaming
        response_text = extract_response(result)
        print("-" * 60)
        results.append({"prompt": raw, "response": response_text})

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to {output_jsonl}")

    if not args.skip_eval and (IFEVAL_ROOT / "evaluation_main.py").exists():
        eval_cmd = [
            sys.executable,
            str(IFEVAL_ROOT / "evaluation_main.py"),
            f"--input_data={args.input_data}",
            f"--input_response_data={output_jsonl}",
            f"--output_dir={output_dir}",
        ]
        print("Running evaluation...")
        subprocess.run(eval_cmd, cwd=str(IFEVAL_ROOT), check=True)
    else:
        if args.skip_eval:
            print("Skipped evaluation (--skip-eval)")
        else:
            print("Skipped evaluation (evaluation_main.py not found)")


if __name__ == "__main__":
    main()
