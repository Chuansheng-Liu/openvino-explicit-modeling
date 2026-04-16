# MTP Test Script Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/mtp_test.py` — a standalone CLI script that runs one `modeling_qwen3_5.exe` MTP test with fully configurable parameters and saves a markdown report.

**Architecture:** Single self-contained Python file. Pure functions for path resolution, command building, output parsing, and report formatting, wired together by `main()`. No dependency on `auto_tests.py`.

**Tech Stack:** Python 3.8+, stdlib only (`argparse`, `subprocess`, `pathlib`, `os`, `datetime`)

**Spec:** `docs/superpowers/specs/2026-03-24-mtp-test-script-design.md`

---

## Chunk 1: Scaffold, arg parsing, and prompt resolution

### Task 1: Create script scaffold with arg parsing

**Files:**
- Create: `scripts/mtp_test.py`

- [ ] **Step 1: Create `scripts/mtp_test.py` with imports, constants, and `parse_args()`**

```python
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "prompt_1k.txt"
DEFAULT_PROMPT = "introduce ffmpeg in details"
DEFAULT_BUILD_TYPE = "Release"
FALLBACK_BUILD_TYPE = "RelWithDebInfo"

EXE_REL = (
    Path("openvino.genai") / "build" / "bin" / "{build_type}" / "modeling_qwen3_5.exe"
)
WORK_DIR_REL = Path("openvino") / "bin" / "intel64" / "{build_type}"
GENAI_RT_REL = Path("openvino.genai") / "build" / "openvino_genai"
GENAI_BIN_REL = Path("openvino.genai") / "build" / "bin"
TBB_FIXED_REL = Path("openvino") / "temp" / "Windows_AMD64" / "tbb" / "bin"
TBB_GLOB_ROOT_REL = Path("openvino") / "temp"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run one modeling_qwen3_5 MTP test with configurable parameters.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/mtp_test.py --model C:/data/models/Huggingface/Qwen3.5-4B\n"
            "  python scripts/mtp_test.py --root .. --model C:/data/models/Huggingface/Qwen3.5-35B-A3B"
            " --mtp-draft-n 1\n"
            "  python scripts/mtp_test.py --model C:/data/models/Huggingface/Qwen3.5-4B"
            " --mtp-draft-n 0  # baseline\n"
        ),
    )
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="Workspace root containing openvino/ and openvino.genai/")
    p.add_argument("--model", required=True, help="Path to model directory")
    p.add_argument("--mtp-draft-n", type=int, default=3,
                   help="Draft tokens N (1-16); 0 = baseline run without MTP")
    p.add_argument("--quant-mode", default="int4_asym",
                   help="Weight quantization mode (default: int4_asym)")
    p.add_argument("--quant-group-size", type=int, default=128,
                   help="Quantization group size; -1 = channel-wise (default: 128)")
    p.add_argument("--quant-backup-mode", default="int4_asym",
                   help="Backup quantization mode (default: int4_asym)")
    p.add_argument("--output-tokens", type=int, default=300,
                   help="Number of output tokens to generate (default: 300)")
    p.add_argument("--prompt", default=None,
                   help="Prompt string (mutually exclusive with --prompt-file)")
    p.add_argument("--prompt-file", default=None,
                   help="Path to prompt text file (mutually exclusive with --prompt)")
    p.add_argument("--think", type=int, choices=[0, 1], default=None,
                   help="Think mode: 0 or 1 (omit = model default)")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature; 0 = greedy (default: 0)")
    p.add_argument("--device", default="GPU", help="Inference device (default: GPU)")
    p.add_argument("--report-dir", default=None,
                   help="Directory for markdown report (default: <root>/reports/)")
    p.add_argument("--build-type", default=None,
                   help="Build type: Release or RelWithDebInfo (default: auto-detect)")
    return p.parse_args()
```

- [ ] **Step 2: Add `resolve_prompt()` below `parse_args()`**

```python
def resolve_prompt(args: argparse.Namespace) -> str:
    """Resolve prompt string from --prompt, --prompt-file, or defaults."""
    if args.prompt is not None and args.prompt_file is not None:
        print("Error: --prompt and --prompt-file are mutually exclusive.", file=sys.stderr)
        sys.exit(2)
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        path = Path(args.prompt_file)
        if not path.is_file():
            print(f"Error: --prompt-file not found: {path}", file=sys.stderr)
            sys.exit(2)
        return path.read_text(encoding="utf-8").strip()
    # Default: try scripts/prompt_1k.txt, else hardcoded fallback
    if DEFAULT_PROMPT_FILE.is_file():
        content = DEFAULT_PROMPT_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content
    return DEFAULT_PROMPT
```

- [ ] **Step 3: Verify parse_args runs without error**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling/scripts
python mtp_test.py --help
```

Expected: prints usage and exits 0.

- [ ] **Step 4: Commit**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling
git add scripts/mtp_test.py
git commit --author "Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(mtp_test): scaffold with arg parsing and prompt resolution"
```

---

## Chunk 2: Path resolution and environment building

### Task 2: Path resolution and build_env

**Files:**
- Modify: `scripts/mtp_test.py`

- [ ] **Step 1: Add `find_tbb_bin_dir()` and `resolve_paths()`**

```python
def find_tbb_bin_dir(root: Path) -> Optional[str]:
    """Resolve TBB bin dir: fixed candidate first, then sorted glob."""
    fixed = root / TBB_FIXED_REL
    if fixed.is_dir() and (fixed / "tbb12.dll").is_file():
        return str(fixed)
    glob_root = root / TBB_GLOB_ROOT_REL
    if glob_root.is_dir():
        for candidate in sorted(glob_root.glob("*/tbb/bin")):
            if candidate.is_dir() and (candidate / "tbb12.dll").is_file():
                return str(candidate)
    return None


def resolve_paths(root: Path, build_type: str) -> Tuple[Path, Path]:
    """Return (exe_path, work_dir). Exits with code 2 if not found."""
    exe = root / str(EXE_REL).format(build_type=build_type)
    work_dir = root / str(WORK_DIR_REL).format(build_type=build_type)
    if not exe.is_file():
        print(f"Error: exe not found: {exe}", file=sys.stderr)
        sys.exit(2)
    if not work_dir.is_dir():
        print(f"Error: work_dir not found: {work_dir}", file=sys.stderr)
        sys.exit(2)
    return exe, work_dir


def detect_build_type(root: Path) -> str:
    """Try Release first, fall back to RelWithDebInfo."""
    for bt in (DEFAULT_BUILD_TYPE, FALLBACK_BUILD_TYPE):
        exe = root / str(EXE_REL).format(build_type=bt)
        if exe.is_file():
            return bt
    return DEFAULT_BUILD_TYPE  # will fail later with a clear error
```

- [ ] **Step 2: Add `build_env()`**

```python
def build_env(
    root: Path,
    work_dir: Path,
    build_type: str,
    quant_mode: str,
    quant_group_size: int,
    quant_backup_mode: str,
) -> Tuple[Dict[str, str], List[str]]:
    """Build subprocess env and a list of 'set KEY=VALUE' strings for the report."""
    tbb = find_tbb_bin_dir(root)
    if tbb is None:
        print("Warning: TBB bin dir not found; omitting from PATH.", file=sys.stderr)

    genai_rt = str(root / GENAI_RT_REL)
    genai_bin = str(root / GENAI_BIN_REL)
    ov_bin = str(work_dir)

    path_parts = [p for p in [tbb, genai_rt, genai_bin, ov_bin] if p]
    original_path = os.environ.get("PATH", "")
    new_path = ";".join(path_parts)
    if original_path:
        new_path = f"{new_path};{original_path}"

    extra = {
        "OV_GENAI_USE_MODELING_API": "1",
        "OV_GPU_MOE_DISABLE_ONEDNN": "1",
        "OV_GENAI_INFLIGHT_QUANT_MODE": quant_mode,
        "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE": str(quant_group_size),
        "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE": quant_backup_mode,
    }

    env = os.environ.copy()
    env["PATH"] = new_path
    env.update(extra)

    report_lines = [f"set {k}={v}" for k, v in extra.items()]
    report_lines.append(f"set PATH={new_path}")
    report_lines.append(f"cd {work_dir}")
    return env, report_lines
```

- [ ] **Step 3: Commit**

```bash
git add scripts/mtp_test.py
git commit --author "Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(mtp_test): path resolution and environment building"
```

---

## Chunk 3: Command construction, execution, and output parsing

### Task 3: Build command and run subprocess

**Files:**
- Modify: `scripts/mtp_test.py`

- [ ] **Step 1: Add `build_command()`**

```python
def build_command(
    exe: Path,
    model: str,
    mtp_draft_n: int,
    think: Optional[int],
    temperature: float,
    device: str,
    prompt: str,
    output_tokens: int,
) -> List[str]:
    """Build the args list for subprocess.run (no shell)."""
    args: List[str] = [
        str(exe),
        "--model", model,
        "--cache-model",
        "--mode", "text",
    ]
    if mtp_draft_n > 0:
        args += ["--mtp", "--mtp-draft-n", str(mtp_draft_n)]
    if think is not None:
        args += ["--think", str(think)]
    args += [
        "--temperature", str(temperature),
        "--device", device,
        "--prompt", prompt,
        "--output-tokens", str(output_tokens),
    ]
    return args


def command_to_string(args: List[str]) -> str:
    """Format args list as a readable command string for the report."""
    def quote(s: str) -> str:
        return f'"{s}"' if (" " in s or "\t" in s) else s
    return " ".join(quote(a) for a in args)
```

- [ ] **Step 2: Add `run_exe()`**

```python
def run_exe(
    args_list: List[str],
    work_dir: Path,
    env: Dict[str, str],
) -> Tuple[int, str]:
    """Run the exe, merging stderr into stdout. Returns (returncode, output)."""
    result = subprocess.run(
        args_list,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode, result.stdout or ""
```

- [ ] **Step 3: Add `parse_output()`**

```python
PERF_LABELS = [
    "Prompt token size:",
    "Output token size:",
    "TTFT:",
    "Decode time:",
    "TPOT:",
    "Throughput:",
]
MTP_LABEL = "[MTP] Accept rate:"


def parse_output(output: str, mtp_enabled: bool) -> str:
    """Extract performance lines from stdout. Returns a formatted block string."""
    lines_out: List[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        for label in PERF_LABELS:
            if stripped.startswith(label):
                lines_out.append(stripped)
                break
        if mtp_enabled and stripped.startswith(MTP_LABEL):
            lines_out.append(stripped)
    return "\n".join(lines_out) if lines_out else "Not found in output."
```

- [ ] **Step 4: Commit**

```bash
git add scripts/mtp_test.py
git commit --author "Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(mtp_test): command construction, subprocess execution, output parsing"
```

---

## Chunk 4: Report generation and main()

### Task 4: Report writer and main()

**Files:**
- Modify: `scripts/mtp_test.py`

- [ ] **Step 1: Add `format_duration()` and `write_report()`**

```python
def format_duration(delta: _dt.timedelta) -> str:
    total = delta.total_seconds()
    if total < 60:
        return f"{total:.2f}s"
    minutes = int(total // 60)
    seconds = total % 60
    return f"{minutes}m{seconds:05.2f}s"


def write_report(
    report_path: Path,
    timestamp: str,
    args: argparse.Namespace,
    build_type: str,
    env_lines: List[str],
    cmd_string: str,
    perf_block: str,
    returncode: int,
    duration: _dt.timedelta,
    raw_output: str,
) -> None:
    mtp_label = str(args.mtp_draft_n) if args.mtp_draft_n > 0 else "disabled"
    think_label = str(args.think) if args.think is not None else "(default)"

    lines = [
        f"# MTP Test Report {timestamp}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| Model | {args.model} |",
        f"| MTP draft N | {mtp_label} |",
        f"| Quant mode | {args.quant_mode} |",
        f"| Quant group size | {args.quant_group_size} |",
        f"| Quant backup mode | {args.quant_backup_mode} |",
        f"| Output tokens | {args.output_tokens} |",
        f"| Temperature | {args.temperature} |",
        f"| Device | {args.device} |",
        f"| Think | {think_label} |",
        f"| Build type | {build_type} |",
        "",
        "## Environment",
        "",
        "```text",
        *env_lines,
        "```",
        "",
        "## Command",
        "",
        "```text",
        cmd_string,
        "```",
        "",
        "## Performance",
        "",
        "```text",
        perf_block,
        "```",
        "",
        f"Return code: {returncode}",
        f"Duration: {format_duration(duration)}",
        "",
        "## Full Output",
        "",
        "```text",
        raw_output.strip(),
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```

- [ ] **Step 2: Add `main()`**

```python
def main() -> int:
    args = parse_args()

    # Resolve workspace root
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    if not root.is_dir():
        print(f"Error: root not found: {root}", file=sys.stderr)
        return 2

    # Build type
    build_type = args.build_type or detect_build_type(root)

    # Paths
    exe, work_dir = resolve_paths(root, build_type)

    # Report dir
    report_dir = Path(args.report_dir) if args.report_dir else root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Prompt
    prompt = resolve_prompt(args)

    # Environment
    env, env_lines = build_env(
        root, work_dir, build_type,
        args.quant_mode, args.quant_group_size, args.quant_backup_mode,
    )

    # Command
    args_list = build_command(
        exe, args.model,
        args.mtp_draft_n, args.think, args.temperature,
        args.device, prompt, args.output_tokens,
    )
    cmd_string = command_to_string(args_list)

    # Run
    timestamp = _dt.datetime.now()
    stamp_name = timestamp.strftime("%Y%m%d_%H%M%S")
    stamp_title = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    mtp_label = f"N={args.mtp_draft_n}" if args.mtp_draft_n > 0 else "baseline (no MTP)"
    print(f"MTP Test: {mtp_label}  |  Model: {Path(args.model).name}")
    print(f"Command: {cmd_string}")

    run_start = _dt.datetime.now()
    returncode, output = run_exe(args_list, work_dir, env)
    duration = _dt.datetime.now() - run_start

    # Parse and display
    mtp_enabled = args.mtp_draft_n > 0
    perf_block = parse_output(output, mtp_enabled)
    print(f"Return code: {returncode}")
    print("Performance:")
    print(perf_block)

    # Report
    report_path = report_dir / f"mtp_test_{stamp_name}.md"
    write_report(
        report_path, stamp_title, args, build_type, env_lines,
        cmd_string, perf_block, returncode, duration, output,
    )
    print("=" * 80)
    print(f"Report saved to: {report_path}")

    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Commit**

```bash
git add scripts/mtp_test.py
git commit --author "Chuansheng Liu <chuansheng.liu@intel.com>" -m "feat(mtp_test): report writer and main() — script complete"
```

---

## Chunk 5: Manual verification

### Task 5: Smoke-test the script

**Files:** none (verification only)

- [ ] **Step 1: Verify `--help` works**

```bash
cd D:/chuansheng/src_code/explicit_modeling/openvino-explicit-modeling/scripts
python mtp_test.py --help
```

Expected: clean usage output, exit 0.

- [ ] **Step 2: Verify mutual-exclusion guard**

```bash
python mtp_test.py --model C:/data/models/Huggingface/Qwen3.5-4B \
  --prompt "hello" --prompt-file prompt_1k.txt
```

Expected: `Error: --prompt and --prompt-file are mutually exclusive.`, exit 2.

- [ ] **Step 3: Run baseline (no MTP) with 4B, 10 output tokens (fast smoke test)**

```bash
python mtp_test.py \
  --root .. \
  --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --mtp-draft-n 0 \
  --output-tokens 10
```

Expected: runs to completion, prints `Throughput:` line, saves report to `../reports/mtp_test_*.md`, exit 0.

- [ ] **Step 4: Run MTP N=3 with 4B, 10 output tokens**

```bash
python mtp_test.py \
  --root .. \
  --model "C:/data/models/Huggingface/Qwen3.5-4B" \
  --mtp-draft-n 3 \
  --output-tokens 10
```

Expected: runs to completion, prints `[MTP] Accept rate:` line, saves report, exit 0.

- [ ] **Step 5: Spot-check the generated report**

Open `../reports/mtp_test_*.md` — verify all sections present: Configuration, Environment, Command, Performance, Full Output.

- [ ] **Step 6: Final commit if any fixups needed**

```bash
git add scripts/mtp_test.py
git commit --author "Chuansheng Liu <chuansheng.liu@intel.com>" -m "fix(mtp_test): smoke-test fixups"
```
