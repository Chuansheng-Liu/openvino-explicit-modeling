import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_PROMPT_FILE = ROOT_DIR / "wwb_prompt.txt"
DEFAULT_RESULTS_DIR = ROOT_DIR / "wwb_results"
OPENVINO_BIN = ROOT_DIR / "openvino" / "bin" / "intel64" / "Release"
TBB_BIN = ROOT_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin"
GENAI_DLL_DIR = ROOT_DIR / "openvino.genai" / "build" / "openvino_genai"
BIN_DIR = ROOT_DIR / "openvino.genai" / "build" / "bin"
EXE_PATH = BIN_DIR / "modeling_qwen3_5.exe"

MODELS = [
    Path(r"D:\data\models\Huggingface\Qwen3-0.6B"),
    Path(r"D:\data\models\Huggingface\Qwen3-2B"),
    Path(r"D:\data\models\Huggingface\Qwen3-4B"),
    Path(r"D:\data\models\Huggingface\Qwen3-8B"),
    Path(r"D:\data\models\Huggingface\Qwen3.5-0.8B"),
    Path(r"D:\data\models\Huggingface\Qwen3.5-2B"),
    Path(r"D:\data\models\Huggingface\Qwen3.5-4B"),
    Path(r"D:\data\models\Huggingface\Qwen3.5-9B"),
    Path(r"D:\data\models\Huggingface\Qwen3.5-35B-A3B"),
]

ENV_OVERRIDES = {
    "OV_GENAI_SAVE_OV_MODEL": "1",
    "OV_GENAI_USE_MODELING_API": "1",
}


@dataclass(frozen=True)
class QuantPreset:
    mode: str
    group_size: int
    backup_mode: str

    @property
    def disabled(self) -> bool:
        return self.mode.lower() == "none"

    @property
    def tag(self) -> str:
        if self.disabled:
            return "q0_none"
        return f"{self.mode}_g{self.group_size}_{self.backup_mode}"

    @property
    def display(self) -> str:
        if self.disabled:
            return "[none, none, none]"
        return f"[{self.mode}, {self.group_size}, {self.backup_mode}]"


QUANT_PRESETS: Dict[int, QuantPreset] = {
    0: QuantPreset("none", 0, "none"),
    1: QuantPreset("int4_asym", 32, "int4_asym"),
    2: QuantPreset("int4_sym", 64, "int4_sym"),
    3: QuantPreset("int4_asym", 128, "int4_asym"),
    4: QuantPreset("int4_asym", 32, "int8_asym"),
    5: QuantPreset("int8_asym", 64, "int8_asym"),
    6: QuantPreset("int8_sym", 128, "int8_sym"),
}


def build_runtime_env(quant_preset: QuantPreset) -> dict:
    env = os.environ.copy()
    env.update(ENV_OVERRIDES)

    for key in [
        "OV_GENAI_INFLIGHT_QUANT_MODE",
        "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE",
        "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE",
    ]:
        env.pop(key, None)

    if not quant_preset.disabled:
        env["OV_GENAI_INFLIGHT_QUANT_MODE"] = quant_preset.mode
        env["OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE"] = str(quant_preset.group_size)
        env["OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE"] = quant_preset.backup_mode

    prepend_dirs = [OPENVINO_BIN, TBB_BIN, GENAI_DLL_DIR, BIN_DIR]
    path_value = os.pathsep.join(str(p) for p in prepend_dirs) + os.pathsep + env.get("PATH", "")
    env["PATH"] = path_value
    return env


def validate_runtime_layout() -> None:
    required_dirs = [
        OPENVINO_BIN,
        TBB_BIN,
        GENAI_DLL_DIR,
        BIN_DIR,
    ]
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required runtime directory not found: {dir_path}")
    if not EXE_PATH.exists():
        raise FileNotFoundError(f"Executable not found: {EXE_PATH}")


def parse_index_selection(spec: str, min_index: int, max_index: int, arg_name: str, allow_all: bool) -> List[int]:
    if not spec:
        return list(range(min_index, max_index + 1))

    if allow_all and spec.strip().lower() == "all":
        return list(range(min_index, max_index + 1))

    tokens = [token.strip() for token in spec.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"`{arg_name}` is empty.")

    chosen: List[int] = []
    seen = set()
    for token in tokens:
        range_match = re.fullmatch(r"(\d+)\s*[~-]\s*(\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            if start <= end:
                expanded = range(start, end + 1)
            else:
                expanded = range(start, end - 1, -1)
            for idx in expanded:
                if idx < min_index or idx > max_index:
                    raise ValueError(f"Index out of range in {arg_name}: {idx}. Valid range is {min_index}~{max_index}.")
                if idx not in seen:
                    chosen.append(idx)
                    seen.add(idx)
            continue

        if not token.isdigit():
            raise ValueError(f"Invalid selector in {arg_name}: `{token}`.")
        idx = int(token)
        if idx < min_index or idx > max_index:
            raise ValueError(f"Index out of range in {arg_name}: {idx}. Valid range is {min_index}~{max_index}.")
        if idx not in seen:
            chosen.append(idx)
            seen.add(idx)

    return chosen


def parse_model_selection(spec: str, max_index: int) -> List[int]:
    return parse_index_selection(spec, 1, max_index, "--models", allow_all=False)


def parse_quant_selection(spec: str) -> List[int]:
    selected = parse_index_selection(spec, 0, max(QUANT_PRESETS.keys()), "--quant-list", allow_all=True)
    invalid = [idx for idx in selected if idx not in QUANT_PRESETS]
    if invalid:
        raise ValueError(f"Unsupported quant preset index in --quant-list: {invalid}")
    return selected


def load_prompts(prompt_file: Path) -> List[str]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    prompts: List[str] = []
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)

    if not prompts:
        raise ValueError(f"No valid prompts found in: {prompt_file}")
    return prompts


def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", name)


def run_for_model(
    model_path: Path,
    quant_index: int,
    quant_preset: QuantPreset,
    prompts: List[str],
    output_tokens: int,
    results_dir: Path,
    env: dict,
) -> int:
    model_name = model_path.name
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    quant_tag = sanitize_filename(f"q{quant_index}_{quant_preset.tag}")
    result_file = results_dir / f"{sanitize_filename(model_name)}_{quant_tag}_{timestamp}.txt"

    fail_count = 0
    with result_file.open("w", encoding="utf-8", errors="replace") as out:
        out.write(f"Model: {model_path}\n")
        out.write(f"Quant preset: {quant_index} {quant_preset.display}\n")
        out.write(f"Prompt count: {len(prompts)}\n")
        out.write(f"Timestamp: {timestamp}\n")
        out.write(f"OV_GENAI_INFLIGHT_QUANT_MODE={env.get('OV_GENAI_INFLIGHT_QUANT_MODE', '<unset>')}\n")
        out.write(f"OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE={env.get('OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE', '<unset>')}\n")
        out.write(f"OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE={env.get('OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE', '<unset>')}\n")
        out.write("=" * 80 + "\n")

        for i, prompt in enumerate(prompts, start=1):
            cmd = [
                str(EXE_PATH),
                "--model",
                str(model_path),
                "--cache-model",
                "--mode",
                "text",
                "--prompt",
                prompt,
                "--output-tokens",
                str(output_tokens),
            ]

            header = (
                "\n"
                + "=" * 80
                + f"\nQuestion {i}/{len(prompts)}\nPrompt: {prompt}\n"
                + "Command: "
                + " ".join(f"\"{c}\"" if " " in c else c for c in cmd)
                + "\n"
                + "=" * 80
                + "\n"
            )
            out.write(header)
            out.flush()

            print(f"[{model_name}][Q{quant_index}] Running question {i}/{len(prompts)} ...")
            completed = subprocess.run(
                cmd,
                cwd=str(BIN_DIR),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            out.write(f"\n[Return code] {completed.returncode}\n")
            out.flush()

            if completed.returncode != 0:
                fail_count += 1
                print(f"[{model_name}][Q{quant_index}] Question {i} failed with return code {completed.returncode}.")

    print(f"[{model_name}][Q{quant_index}] Finished. Result: {result_file}")
    return fail_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all prompts in wwb_prompt.txt against selected Qwen models and save raw outputs."
    )
    parser.add_argument(
        "--models",
        "--models-list",
        dest="models",
        default="1~9",
        help="Model index selectors. Examples: 1,3,4  |  1~5  |  2,4~5,6,8~9",
    )
    parser.add_argument(
        "--quant-list",
        default="3",
        help="Quant preset selectors. Examples: 1 | 2,3,4 | all | 1~6 | 1~3,4,5~6",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help=f"Prompt file path (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=3000,
        help="Value passed to --output-tokens (default: 3000).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model index mapping and exit.",
    )
    parser.add_argument(
        "--list-quants",
        action="store_true",
        help="Print quant preset mapping and exit.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("Available models:")
    for idx, model in enumerate(MODELS, start=1):
        print(f"  {idx}. {model}")
    print("Available quant presets:")
    for idx, preset in QUANT_PRESETS.items():
        print(f"  {idx}. {preset.display}")

    if args.list_models:
        return 0
    if args.list_quants:
        return 0

    try:
        validate_runtime_layout()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        selected_indices = parse_model_selection(args.models, len(MODELS))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        selected_quant_indices = parse_quant_selection(args.quant_list)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        prompts = load_prompts(Path(args.prompt_file))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected models: {selected_indices}")
    print(f"Selected quant presets: {selected_quant_indices}")
    print(f"Prompt file: {args.prompt_file} (loaded {len(prompts)} prompts)")
    print(f"Results dir: {results_dir}")
    print(f"BIN dir: {BIN_DIR}")

    total_failures = 0
    for idx in selected_indices:
        for quant_idx in selected_quant_indices:
            quant_preset = QUANT_PRESETS[quant_idx]
            env = build_runtime_env(quant_preset)
            total_failures += run_for_model(
                model_path=MODELS[idx - 1],
                quant_index=quant_idx,
                quant_preset=quant_preset,
                prompts=prompts,
                output_tokens=args.output_tokens,
                results_dir=results_dir,
                env=env,
            )

    if total_failures > 0:
        print(f"Completed with failures. Failed runs: {total_failures}")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
