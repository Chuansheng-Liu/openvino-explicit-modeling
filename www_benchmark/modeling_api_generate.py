import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


DEFAULT_MODEL_PATH = r"D:\Data\models\Huggingface\SmolLM3-3B"
DEFAULT_EXE_PATH = r"D:\openvino-modeling-api\openvino.genai\build\bin\Release\greedy_causal_lm.exe"
DEFAULT_QWEN35_MODEL_PATH = r"D:\Data\models\Huggingface\Qwen3.5-35B-A3B"
DEFAULT_QWEN35_EXE_PATH = r"D:\openvino-modeling-api\openvino.genai\build\bin\Release\modeling_qwen3_5.exe"
DEFAULT_OPENVINO_ROOT = r"D:\openvino-modeling-api"




def parse_args():
    parser = argparse.ArgumentParser(
        prog="Generate",
        description="Generate answers by calling an external exe once per prompt.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Model path passed to exe as the first positional model argument.",
    )
    parser.add_argument(
        "--modeltype",
        default="exe_causal",
        choices=["exe_causal", "modeling_qwen3_5"],
        help=(
            "Inference mode: exe_causal uses greedy_causal_lm-style positional args; "
            "modeling_qwen3_5 uses modeling_qwen3_5.exe named args"
        ),
    )
    parser.add_argument(
        "--csv",
        default="simple.csv",
        help="CSV file with prompts. Must contain column 'questions'.",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.csv",
        help="Path to save generated outputs CSV.",
    )

    parser.add_argument(
        "--exe_path",
        default=DEFAULT_EXE_PATH,
        help="Path to external exe, e.g. greedy_causal_lm.exe.",
    )
    parser.add_argument(
        "--exe_work_dir",
        default=None,
        help="Working directory for exe. Defaults to exe parent folder.",
    )
    parser.add_argument(
        "--exe_common_args",
        nargs="+",
        default=["GPU", "1", "1", "100"],
        help="Args appended after prompt in exe mode. Default: GPU 1 1 100",
    )
    parser.add_argument(
        "--exe_quant_args",
        nargs=3,
        metavar=("QUANT_MODE", "GROUP_SIZE", "BACKUP_MODE"),
        default=None,
        help="Optional 3 quantization args, e.g. int4_asym -1 int8_asym",
    )
    parser.add_argument(
        "--openvino_root",
        default=DEFAULT_OPENVINO_ROOT,
        help="OpenVINO source/build root used to auto-compose PATH entries.",
    )
    parser.add_argument(
        "--exe_path_prepend",
        nargs="*",
        default=None,
        help="Optional custom PATH prepend entries for exe runtime.",
    )
    parser.add_argument(
        "--print_exe_command",
        action="store_true",
        help="Print one full exe command per prompt.",
    )

    parser.add_argument(
        "--qwen35_exe_path",
        default=DEFAULT_QWEN35_EXE_PATH,
        help="Path to modeling_qwen3_5.exe for modeltype=modeling_qwen3_5.",
    )
    parser.add_argument(
        "--qwen35_work_dir",
        default=None,
        help="Working directory for modeling_qwen3_5.exe. Defaults to exe parent folder.",
    )
    parser.add_argument(
        "--qwen35_mode",
        choices=["text", "vl"],
        default="text",
        help="Mode for modeling_qwen3_5.exe.",
    )
    parser.add_argument(
        "--qwen35_image_path",
        default=None,
        help="Image path used when --qwen35_mode vl.",
    )
    parser.add_argument(
        "--qwen35_output_tokens",
        type=int,
        default=300,
        help="Value for --output-tokens passed to modeling_qwen3_5.exe.",
    )
    parser.add_argument(
        "--qwen35_device",
        default=None,
        help="Optional --device value for modeling_qwen3_5.exe (e.g. GPU/CPU).",
    )
    parser.add_argument(
        "--qwen35_no_cache_model",
        action="store_true",
        help="Do not pass --cache-model for modeling_qwen3_5.exe.",
    )
    parser.add_argument(
        "--qwen35_quant_env",
        nargs=3,
        metavar=("QUANT_MODE", "GROUP_SIZE", "BACKUP_MODE"),
        default=None,
        help=(
            "Optional in-flight quant env for modeling_qwen3_5 mode, "
            "e.g. int4_asym 128 int4_asym"
        ),
    )

    return parser.parse_args()


def command_to_string(args):
    quoted = []
    for arg in args:
        if " " in arg or "\t" in arg:
            quoted.append(f'"{arg}"')
        else:
            quoted.append(arg)
    return " ".join(quoted)


def build_default_path_entries(openvino_root):
    root = openvino_root
    candidates = []

    tbb_dir = os.path.join(root, "openvino", "temp", "Windows_AMD64", "tbb", "bin")
    if os.path.isdir(tbb_dir) and os.path.isfile(os.path.join(tbb_dir, "tbb12.dll")):
        candidates.append(tbb_dir)

    candidates.extend(
        [
            os.path.join(root, "openvino.genai", "build", "openvino_genai"),
            os.path.join(root, "openvino", "bin", "intel64", "Release"),
            os.path.join(root, "openvino.genai", "build", "bin", "Release"),
            os.path.join(root, "openvino", "build", "bin", "Release"),
        ]
    )

    return [entry for entry in candidates if os.path.isdir(entry)]


def build_env(args):
    env = os.environ.copy()

    if args.exe_path_prepend is not None:
        prepend_entries = [entry for entry in args.exe_path_prepend if os.path.isdir(entry)]
    else:
        prepend_entries = build_default_path_entries(args.openvino_root)

    if prepend_entries:
        env["PATH"] = ";".join(prepend_entries) + ";" + env.get("PATH", "")

    # Clear env-based quant settings first to avoid accidental interference.
    env.pop("OV_GENAI_INFLIGHT_QUANT_MODE", None)
    env.pop("OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE", None)
    env.pop("OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE", None)

    env["OV_GENAI_USE_MODELING_API"] = env.get("OV_GENAI_USE_MODELING_API", "1")

    if args.modeltype == "modeling_qwen3_5":
        # Align with auto_tests.py qwen3.5 runtime environment.
        env["OV_GPU_MOE_DISABLE_ONEDNN"] = "1"
        if args.qwen35_quant_env is not None:
            env["OV_GENAI_INFLIGHT_QUANT_MODE"] = args.qwen35_quant_env[0]
            env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = args.qwen35_quant_env[1]
            env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = args.qwen35_quant_env[2]

    return env


def resolve_model_path(args):
    if args.modeltype == "modeling_qwen3_5" and args.model == DEFAULT_MODEL_PATH:
        return DEFAULT_QWEN35_MODEL_PATH
    return args.model


def extract_generated_text(output):
    marker = "Generated text:"
    idx = output.find(marker)
    if idx != -1:
        text_block = output[idx + len(marker) :].lstrip("\r\n")
        if text_block.strip():
            return text_block.strip()

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def generate_with_exe(args):
    data = pd.read_csv(args.csv)
    if "questions" not in data.columns:
        raise ValueError("Input CSV must contain a 'questions' column")

    questions = data["questions"]
    model_path = resolve_model_path(args)

    if args.modeltype == "exe_causal":
        exe_path = args.exe_path
        work_dir = args.exe_work_dir if args.exe_work_dir else os.path.dirname(exe_path)
    elif args.modeltype == "modeling_qwen3_5":
        exe_path = args.qwen35_exe_path
        work_dir = args.qwen35_work_dir if args.qwen35_work_dir else os.path.dirname(exe_path)

        if args.qwen35_mode == "vl":
            if not args.qwen35_image_path:
                raise ValueError("--qwen35_image_path is required when --qwen35_mode vl")
            if not os.path.isfile(args.qwen35_image_path):
                raise FileNotFoundError(f"Qwen3.5 image not found: {args.qwen35_image_path}")
    else:
        raise ValueError("Unsupported modeltype")

    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"Exe not found: {exe_path}")
    if not os.path.isdir(work_dir):
        raise FileNotFoundError(f"Exe work dir not found: {work_dir}")

    env = build_env(args)

    answers = []
    for q in tqdm(questions.values):
        if args.modeltype == "exe_causal":
            cmd = [exe_path, model_path, str(q), *args.exe_common_args]
            if args.exe_quant_args is not None:
                cmd.extend(args.exe_quant_args)
        else:
            cmd = [exe_path, "--model", model_path]
            if not args.qwen35_no_cache_model:
                cmd.append("--cache-model")
            cmd.extend(["--mode", args.qwen35_mode])
            if args.qwen35_mode == "vl":
                cmd.extend(["--image", args.qwen35_image_path])
            cmd.extend(["--prompt", str(q), "--output-tokens", str(args.qwen35_output_tokens)])
            if args.qwen35_device is not None:
                cmd.extend(["--device", args.qwen35_device])

        if args.print_exe_command:
            print("Command:", command_to_string(cmd))

        result = subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        output = result.stdout or ""
        if result.returncode != 0:
            tail = output[-4000:] if len(output) > 4000 else output
            raise RuntimeError(
                "External exe inference failed with return code "
                f"{result.returncode}\n"
                f"Command: {command_to_string(cmd)}\n"
                f"Output tail:\n{tail}"
            )

        answers.append(extract_generated_text(output))

    res_data = {"questions": list(questions.values), "answers": answers}
    df = pd.DataFrame(res_data)
    df.to_csv(args.save_generations_path, index=False)


def main():
    args = parse_args()
    generate_with_exe(args)


if __name__ == "__main__":
    main()
