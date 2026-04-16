# MTP Test Script Design

**Date:** 2026-03-24
**Script:** `scripts/mtp_test.py`
**Status:** Approved

## Purpose

A standalone CLI script to run a single `modeling_qwen3_5.exe` MTP test with
fully configurable model, MTP settings, quantization, prompt, and generation
parameters. Produces console output and a markdown report.

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | grandparent of script dir | Workspace root containing `openvino/` and `openvino.genai/` |
| `--model` | *(required)* | Path to model directory |
| `--mtp-draft-n` | `3` | Draft tokens N (1–16); use `0` to run baseline without MTP |
| `--quant-mode` | `int4_asym` | Weight quantization mode (`int4_asym`, `int8_asym`, etc.) |
| `--quant-group-size` | `128` | Quantization group size; `-1` = channel-wise |
| `--quant-backup-mode` | `int4_asym` | Backup quantization mode |
| `--output-tokens` | `300` | Number of output tokens to generate |
| `--prompt` | *(see below)* | Prompt string (mutually exclusive with `--prompt-file`) |
| `--prompt-file` | *(see below)* | Path to prompt text file |
| `--think` | *(omitted)* | Think mode: `0` or `1`; omit to use model default |
| `--temperature` | `0` | Sampling temperature; `0` = greedy |
| `--device` | `GPU` | Inference device (`GPU` or `CPU`) |
| `--report-dir` | `<root>/reports/` | Directory for markdown report; created if absent |
| `--build-type` | `Release` | Build type (`Release` or `RelWithDebInfo`) |

**Prompt resolution order:** `--prompt` → `--prompt-file` → `scripts/prompt_1k.txt`
(relative to script dir, if present) → hardcoded fallback string.
`--prompt` and `--prompt-file` together is an error (exit 2).

## Path Resolution

Paths derived from `--root`:

```
exe:         {root}/openvino.genai/build/bin/{build_type}/modeling_qwen3_5.exe
work_dir:    {root}/openvino/bin/intel64/{build_type}/        (subprocess cwd)
TBB bin:     resolved by scanning (see below)
genai_rt:    {root}/openvino.genai/build/openvino_genai/      (no build-type segment)
genai_bin:   {root}/openvino.genai/build/bin/                 (no build-type segment)
```

**TBB resolution (matching `auto_tests.py` priority):**
1. Try fixed candidate: `{root}/openvino/temp/Windows_AMD64/tbb/bin/`
2. If not found, sorted-glob: `{root}/openvino/temp/*/tbb/bin/`, take first hit that contains `tbb12.dll`
3. If still not found, omit TBB from PATH (warn to stderr)

## PATH Prepend

Prepend order (highest priority first), then append existing `%PATH%`:

```
TBB bin  →  genai_rt  →  genai_bin  →  work_dir
```

This matches `auto_tests.py`'s `build_path_entries()` exactly.

## Environment Variables

Set for every run:

```
OV_GENAI_USE_MODELING_API=1
OV_GPU_MOE_DISABLE_ONEDNN=1
OV_GENAI_INFLIGHT_QUANT_MODE=<quant-mode>
OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=<quant-group-size>
OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=<quant-backup-mode>
```

## Command Construction

```
{exe}
  --model {model}
  --cache-model
  --mode text
  [--mtp --mtp-draft-n N]     # omitted when --mtp-draft-n 0
  [--think 0|1]               # omitted when --think not specified
  --temperature {T}           # always included
  --device {device}           # always included
  --prompt <prompt>           # or --prompt-file {path}; passed as list element, no shell quoting
  --output-tokens {N}
```

The subprocess is invoked via `subprocess.run(args_list, ...)` — no shell.
The displayed command in the report is formatted for readability only (no real shell quoting).

## Output Parsing

Exact stdout prefixes emitted by `modeling_qwen3_5.exe`:

| Label | Present | Example |
|-------|---------|---------|
| `Prompt token size:` | always | `Prompt token size: 1024` |
| `Output token size:` | always | `Output token size: 300` |
| `TTFT:` | always | `TTFT: 1523.40 ms` |
| `Decode time:` | always | `Decode time: 14231.10 ms` |
| `TPOT:` | always | `TPOT: 47.44 ms/token` |
| `Throughput:` | always | `Throughput: 21.08 tokens/s` |
| `[MTP] Accept rate:` | MTP runs only | `[MTP] Accept rate: 98/300 = 32.67%` |

No `Load time:` or `Accept length:` labels exist in the exe output.
For baseline runs (`--mtp-draft-n 0`), the `[MTP] Accept rate:` line is absent from the
Performance block entirely — it is not shown as `N/A`.

Parsing rule: for each label, find the first line starting with that prefix (strip leading
whitespace), take everything after the colon as the value string.

## Report Format

Saved to `{report_dir}/mtp_test_YYYYMMDD_HHMMSS.md`. Directory created if absent.

```markdown
# MTP Test Report YYYY-MM-DD HH:MM:SS

## Configuration
| Parameter | Value |
|-----------|-------|
| Model | ... |
| MTP draft N | 3 (or "disabled") |
| Quant mode | int4_asym |
| Quant group size | 128 |
| Quant backup mode | int4_asym |
| Output tokens | 300 |
| Temperature | 0 |
| Device | GPU |
| Think | (default) |
| Build type | Release |

## Environment
```text
set OV_GENAI_USE_MODELING_API=1
set OV_GPU_MOE_DISABLE_ONEDNN=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym
set PATH=...
cd {work_dir}
```

## Command
```text
modeling_qwen3_5.exe --model ... --cache-model ...
```

## Performance
```text
Prompt token size: 1024
Output token size: 300
TTFT: 1523.40 ms
Decode time: 14231.10 ms
TPOT: 47.44 ms/token
Throughput: 21.08 tokens/s
[MTP] Accept rate: 98/300 = 32.67%   ← MTP runs only
```

Return code: 0
Duration: 16.42s

## Full Output
```text
{raw stdout}
```
```

**Duration format:** matches `auto_tests.py`'s `format_duration()`:
`{S:.2f}s` for under 60s, `{M}m{S:05.2f}s` for longer.

## Error Handling

| Condition | Behaviour |
|-----------|-----------|
| `--prompt` and `--prompt-file` both given | Print error, exit 2 |
| exe or work_dir not found | Print clear error with path, exit 2 |
| TBB dir not found | Warn to stderr, continue without it |
| `--report-dir` does not exist | Create it (`mkdir -p`), continue |
| exe returns non-zero | Record in report, exit with same code |
| stderr handling | `stderr=subprocess.STDOUT` — merge stderr into stdout so both are captured together and appear in the Full Output section |
