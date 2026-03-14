import argparse
import datetime as dt
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent
DEFAULT_RESULTS_DIR = ROOT_DIR / "wwb_results"
OPENVINO_BIN = ROOT_DIR / "openvino" / "bin" / "intel64" / "Release"
TBB_BIN = ROOT_DIR / "openvino" / "temp" / "Windows_AMD64" / "tbb" / "bin"
GENAI_DLL_DIR = ROOT_DIR / "openvino.genai" / "build" / "openvino_genai"
BIN_DIR = ROOT_DIR / "openvino.genai" / "build" / "bin"
EXE_PATH = BIN_DIR / "modeling_qwen3_5.exe"
DEFAULT_MODEL_ROOT = Path(r"C:\data\models\Huggingface")
SUMMARY_FILE_NAME = "summary.md"

MODEL_NAMES = [
    "Qwen3.5-0.8B",
    "Qwen3.5-2B",
    "Qwen3.5-4B",
    "Qwen3.5-9B",
    "Qwen3.5-35B-A3B",
]

BUILTIN_PROMPTS = [
    "Who is Mark Twain?",
    "Who is William Shakespeare?",
    "Who is Agatha Christie?",
    "Who is Barbara Cartland?",
    "Who is Danielle Steel?",
    "Who is Harold Robbins?",
    "Who is Georges Simenon?",
    "Who is Enid Blyton?",
    "Who is Sidney Sheldon?",
    "Who is Akira Toriyama?",
    "Who is Leo Tolstoy?",
    "Who is Alexander Pushkin?",
    "Who is Stephen King?",
    "What is C++?",
    "What is Python?",
    "What is Java?",
    "What is JavaScript?",
    "What is Perl?",
    "What is OpenCV?",
    "Who is the most famous writer?",
    "Who is the most famous inventor?",
    "Who is the most famous mathematician?",
    "Who is the most famous composer?",
    "Who is the most famous programmer?",
    "Who is the most famous athlete?",
    "Who is the most famous ancient Greek scientist?",
    "What color will you get when you mix blue and yellow?",
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
    1: QuantPreset("int4_asym", 128, "int4_asym"),
    2: QuantPreset("int4_asym", 128, "int8_asym"),
    3: QuantPreset("none", 0, "none"),
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


def build_model_paths(model_root: Path) -> List[Path]:
    return [model_root / model_name for model_name in MODEL_NAMES]


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
    selected = parse_index_selection(spec, 1, max(QUANT_PRESETS.keys()), "--quant-list", allow_all=True)
    invalid = [idx for idx in selected if idx not in QUANT_PRESETS]
    if invalid:
        raise ValueError(f"Unsupported quant preset index in --quant-list: {invalid}")
    return selected


def parse_prompt_selection(spec: str) -> List[int]:
    return parse_index_selection(spec, 1, len(BUILTIN_PROMPTS), "--prompt-list", allow_all=True)


def parse_think_selection(spec: str) -> List[int]:
    return parse_index_selection(spec, 0, 1, "--think", allow_all=True)


def summarize_selection(indices: List[int], min_index: int, max_index: int) -> str:
    normalized = sorted(set(indices))
    if normalized == list(range(min_index, max_index + 1)):
        return "all"
    if not normalized:
        return "none"

    parts: List[str] = []
    start = normalized[0]
    end = normalized[0]
    for idx in normalized[1:]:
        if idx == end + 1:
            end = idx
        else:
            parts.append(f"{start}" if start == end else f"{start}~{end}")
            start = idx
            end = idx
    parts.append(f"{start}" if start == end else f"{start}~{end}")
    return ",".join(parts)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", name)


def extract_first_match(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def parse_single_log_for_summary(log_path: Path) -> List[Dict[str, str]]:
    content = log_path.read_text(encoding="utf-8", errors="replace")

    model_full_path = extract_first_match(r"^Model:\s*(.+)$", content) or ""
    model_name = Path(model_full_path.strip("\"")).name if model_full_path else "N/A"
    if model_name in ("", ".", "\\", "/"):
        name_match = re.search(r"^m\d+_(.+?)__q\d+_", log_path.name)
        model_name = name_match.group(1) if name_match else "N/A"

    quant_tuple = extract_first_match(r"^Quant preset:\s*\d+\s*(\[[^\]]+\])\s*$", content) or "N/A"
    think_value = extract_first_match(r"^Think:\s*([01])\s*$", content) or "N/A"

    question_matches = list(re.finditer(r"^Question\s+(\d+)\s*/\s*(\d+)\s*$", content, re.MULTILINE))
    if not question_matches:
        return [
            {
                "model_name": model_name,
                "quant_tuple": quant_tuple,
                "think": think_value,
                "input_tokens": "N/A",
                "output_tokens": "N/A",
                "ttft_ms": "N/A",
                "throughput_tps": "N/A",
            }
        ]

    rows: List[Dict[str, str]] = []
    for idx, match in enumerate(question_matches):
        block_start = match.start()
        block_end = question_matches[idx + 1].start() if idx + 1 < len(question_matches) else len(content)
        block = content[block_start:block_end]

        return_code = extract_first_match(r"^\[Return code\]\s*(-?\d+)\s*$", block)
        failed = return_code is None or return_code != "0"

        input_tokens = extract_first_match(r"^Prompt token size:\s*(\d+)\s*$", block) or "N/A"
        output_tokens = extract_first_match(r"^Output token size:\s*(\d+)\s*$", block) or "N/A"
        ttft_ms = extract_first_match(r"^TTFT:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\s*$", block) or "N/A"
        throughput_tps = (
            extract_first_match(r"^Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*tokens/s\s*$", block) or "N/A"
        )

        if failed:
            input_tokens = "N/A"
            output_tokens = "N/A"
            ttft_ms = "N/A"
            throughput_tps = "N/A"

        rows.append(
            {
                "model_name": model_name,
                "quant_tuple": quant_tuple,
                "think": think_value,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ttft_ms": ttft_ms,
                "throughput_tps": throughput_tps,
            }
        )

    return rows


def to_markdown_cell(value: str) -> str:
    return value.replace("|", r"\|")


def build_summary_markdown(rows: List[Dict[str, str]]) -> str:
    lines = [
        "# WWB Test Summary",
        "",
        f"- Total tests: {len(rows)}",
        "",
        "| Model | Quant (3-tuple) | Think | Input Tokens | Output Tokens | TTFT (ms) | Throught (tokens/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    if not rows:
        lines.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
    else:
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        to_markdown_cell(row["model_name"]),
                        to_markdown_cell(row["quant_tuple"]),
                        to_markdown_cell(row["think"]),
                        to_markdown_cell(row["input_tokens"]),
                        to_markdown_cell(row["output_tokens"]),
                        to_markdown_cell(row["ttft_ms"]),
                        to_markdown_cell(row["throughput_tps"]),
                    ]
                )
                + " |"
            )

    return "\n".join(lines) + "\n"


def write_summary_markdown(run_dir: Path) -> Path:
    rows: List[Dict[str, str]] = []
    for log_path in sorted(run_dir.glob("*.txt")):
        rows.extend(parse_single_log_for_summary(log_path))

    summary_path = run_dir / SUMMARY_FILE_NAME
    summary_path.write_text(build_summary_markdown(rows), encoding="utf-8")
    return summary_path


def run_for_model(
    model_index: int,
    model_path: Path,
    quant_index: int,
    quant_preset: QuantPreset,
    prompts: List[str],
    prompt_selection_tag: str,
    output_tokens: int,
    run_dir: Path,
    env: dict,
    repetition_penalty: Optional[float] = None,
    think: int = 1,
) -> int:
    model_name = model_path.name
    model_tag = sanitize_filename(f"m{model_index}_{model_name}")
    quant_desc = f"{quant_preset.mode}_g{quant_preset.group_size}_{quant_preset.backup_mode}"
    quant_tag = sanitize_filename(f"q{quant_index}_{quant_desc}")
    think_tag = sanitize_filename(f"t{think}")
    prompt_tag = sanitize_filename(f"p{prompt_selection_tag}")
    token_tag = sanitize_filename(f"ot{output_tokens}")
    result_file = run_dir / f"{model_tag}__{quant_tag}__{think_tag}__{prompt_tag}__{token_tag}.txt"

    fail_count = 0
    with result_file.open("w", encoding="utf-8", errors="replace") as out:
        out.write(f"Model: {model_path}\n")
        out.write(f"Model index: {model_index}\n")
        out.write(f"Quant preset: {quant_index} {quant_preset.display}\n")
        out.write(f"Prompt selection: {prompt_selection_tag}\n")
        out.write(f"Prompt count: {len(prompts)}\n")
        out.write(f"Think: {think}\n")
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
            if repetition_penalty is not None:
                cmd.extend(["--repetition-penalty", str(repetition_penalty)])
            cmd.extend(["--think", str(think)])

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

            print(f"[{model_name}][Q{quant_index}][T{think}] Running question {i}/{len(prompts)} ...")
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
                print(
                    f"[{model_name}][Q{quant_index}][T{think}] "
                    f"Question {i} failed with return code {completed.returncode}."
                )

    print(f"[{model_name}][Q{quant_index}][T{think}] Finished. Result: {result_file}")
    return fail_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run built-in prompts against selected Qwen models and save raw outputs.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    model_mapping = "\n".join(f"  {idx}. {model_name}" for idx, model_name in enumerate(MODEL_NAMES, start=1))
    quant_mapping = "\n".join(f"  {idx}. {preset.display}" for idx, preset in QUANT_PRESETS.items())
    prompt_mapping = "\n".join(f"  {idx}. {prompt}" for idx, prompt in enumerate(BUILTIN_PROMPTS, start=1))
    parser.add_argument(
        "--models",
        "--models-list",
        dest="models",
        default="1",
        help=(
            "Model index selectors (default: 1). Examples: 1,3,4 | 1~5 | 2,4~5,6,8~9\n"
            "Model index mapping:\n"
            f"{model_mapping}"
        ),
    )
    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
        help=f"Model root folder (default: {DEFAULT_MODEL_ROOT}).",
    )
    parser.add_argument(
        "--quant-list",
        default="1",
        help=(
            "Quant preset selectors (default: 1). Examples: 1 | 2,3,4 | all | 1~7 | 1~3,4,5~7\n"
            "Quant preset mapping:\n"
            f"{quant_mapping}"
        ),
    )
    parser.add_argument(
        "--prompt-list",
        default="1",
        help=(
            "Prompt selectors (default: 1). Examples: 1 | 2,3,4 | all | 1~27 | 1~3,4,5~7\n"
            "Prompt index mapping:\n"
            f"{prompt_mapping}"
        ),
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=2000,
        help="Value passed to --output-tokens (default: 2000).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Repetition penalty passed to the exe (default: use exe built-in default 1.1).",
    )
    parser.add_argument(
        "--think",
        metavar="THINK_LIST",
        default="1",
        help="Think selectors (default: 1). Examples: 1 | 0,1 | all | 0~1",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        validate_runtime_layout()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    model_root = Path(args.model_root)
    models = build_model_paths(model_root)

    print(f"Model root: {model_root}")
    print("Available models:")
    for idx, model in enumerate(models, start=1):
        print(f"  {idx}. {model}")
    print("Available quant presets:")
    for idx, preset in QUANT_PRESETS.items():
        print(f"  {idx}. {preset.display}")

    try:
        selected_indices = parse_model_selection(args.models, len(models))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        selected_quant_indices = parse_quant_selection(args.quant_list)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        selected_prompt_indices = parse_prompt_selection(args.prompt_list)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    try:
        selected_think_values = parse_think_selection(args.think)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    prompts = [BUILTIN_PROMPTS[idx - 1] for idx in selected_prompt_indices]
    missing_models = [models[idx - 1] for idx in selected_indices if not models[idx - 1].exists()]
    if missing_models:
        print("ERROR: The following model paths do not exist:", file=sys.stderr)
        for model_path in missing_models:
            print(f"  {model_path}", file=sys.stderr)
        return 2

    results_dir = DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_selection_tag = summarize_selection(selected_prompt_indices, 1, len(BUILTIN_PROMPTS))

    print(f"Selected models: {selected_indices}")
    print(f"Selected quant presets: {selected_quant_indices}")
    print(f"Selected prompt indices: {selected_prompt_indices}")
    print(f"Selected think values: {selected_think_values}")
    print(f"Built-in prompts selected: {len(prompts)}/{len(BUILTIN_PROMPTS)}")
    print(f"Run id: {run_id}")
    print(f"Results root dir: {results_dir}")
    print(f"Current run dir: {run_dir}")
    print(f"BIN dir: {BIN_DIR}")

    total_failures = 0
    for idx in selected_indices:
        for quant_idx in selected_quant_indices:
            for think in selected_think_values:
                quant_preset = QUANT_PRESETS[quant_idx]
                env = build_runtime_env(quant_preset)
                total_failures += run_for_model(
                    model_index=idx,
                    model_path=models[idx - 1],
                    quant_index=quant_idx,
                    quant_preset=quant_preset,
                    prompts=prompts,
                    prompt_selection_tag=prompt_selection_tag,
                    output_tokens=args.output_tokens,
                    run_dir=run_dir,
                    env=env,
                    repetition_penalty=args.repetition_penalty,
                    think=think,
                )

    summary_path = write_summary_markdown(run_dir)
    print(f"Summary markdown: {summary_path}")
    try:
        summary_text = summary_path.read_text(encoding="utf-8", errors="replace")
        print("=" * 80)
        print("Summary Markdown Content")
        print("=" * 80)
        print(summary_text, end="" if summary_text.endswith("\n") else "\n")
        print("=" * 80)
    except OSError as e:
        print(f"WARNING: Failed to read summary markdown for console output: {e}", file=sys.stderr)

    if total_failures > 0:
        print(f"Completed with failures. Failed runs: {total_failures}")
        return 1

    print("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
