# CLIntercept GPU Kernel Analysis: Baseline INT4 vs DFlash INT4/INT4

## 1. Goal

Capture CLIntercept kernel timing logs for two configs of Qwen3.5-4B:

1. **Baseline INT4** — `modeling_qwen3_5.exe` with `OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym`
2. **DFlash INT4/INT4** — `modeling_qwen3_5_dflash.exe` with `target=INT4_ASYM draft=INT4_ASYM`

From each log, extract **one representative decode step** and break down GPU time by
sub-phase (MLP, full attention, RMS norm, RoPE, lm_head, copy ops, etc.) arranged in
execution time order. Produce a side-by-side comparison table.

**Success criteria:** Both logs captured without crash; decode breakdown table shows
per-kernel-group time in ms sorted chronologically for one decode step.

---

## 2. Role

You are a GPU performance engineer. Your job is to instrument the two executables with
CLIntercept, run them with a short output (32 tokens) to limit log size, extract one
clean decode step from each log, classify kernels into sub-phases, and produce a
readable breakdown. Do not modify any source code.

**IMPORTANT — known incompatibility:** CLIntercept's `opencl.dll` is known to cause
heap corruption (0xC0000374) during `compile_model()` in OpenVINO when `--cache-model`
is used with a fresh IR compile. **Always use a pre-cached model** (run once without
CLIntercept first to populate the cache, then run with CLIntercept). The `--cache-model`
flag reuses the cached XML/blob; CLIntercept is safe after compile_model completes.

---

## 3. Process

1. **Warm up model cache** (no CLIntercept) — ensure cached IR exists for both configs.
2. **Capture Baseline INT4** — inject CLIntercept DLL, run 32-token decode, remove DLL.
3. **Capture DFlash INT4/INT4** — same, but with dflash exe.
4. **Parse logs** — locate the Chrome JSON (`*.json`) in `CLI_DumpDir`. Identify one
   decode step by finding a repeating kernel sequence after the first large TTFT block.
5. **Classify kernels** into sub-phases by name pattern:
   - `rms` / `layer_norm` → RMS Norm
   - `rope` / `rotary` → RoPE
   - `gemm` / `fc` / `mm` / `mlp` / `gate` / `up` / `down` → MLP / Linear
   - `sdpa` / `attention` / `kv` → Attention
   - `lm_head` → LM Head
   - `copy` / `reorder` / `concat` → Data movement
6. **Produce breakdown table** per config: sub-phase, kernel count, total ms, % of step.
7. **Compare** the two configs side-by-side.

---

## 4. Tools

- CLIntercept DLL: `D:\tools\clintercept-3.0.6-win64\Release\opencl.dll`
- Baseline exe: `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe`
- DFlash exe:   `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5_dflash.exe`
- Exe dir (inject target): `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release`
- Model (target): `C:\data\models\Huggingface\Qwen3.5-4B`
- Model (draft):  `C:\data\models\Huggingface\Qwen3.5-4B-DFlash`
- Prompt file: `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling\scripts\prompt_1k.txt`
- OV bin DLLs: `D:\chuansheng\src_code\explicit_modeling\openvino\bin\intel64\Release`
- TBB DLLs:    `D:\chuansheng\src_code\explicit_modeling\openvino\temp\Windows_AMD64\tbb\bin`
- GenAI DLLs:  `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\openvino_genai`
- Log output base: `D:\chuansheng\src_code\explicit_modeling\clintercept_logs\dflash_analysis`

---

## 5. Build

No build needed — executables are already built from the `dflash` branch.
If a rebuild is needed:
```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
.\build.bat
```

---

## 6. Test

> **RE-READ this section before executing any command.**

### Step 0 — Set PATH for DLLs (run once per PowerShell session)

```powershell
$root = "D:\chuansheng\src_code\explicit_modeling"
$env:PATH = "$root\openvino\bin\intel64\Release;" +
            "$root\openvino\temp\Windows_AMD64\tbb\bin;" +
            "$root\openvino.genai\build\openvino_genai;" +
            "$root\openvino.genai\build\bin\Release;" +
            "$root\openvino.genai\build\bin;" +
            $env:PATH
$env:OV_GENAI_USE_MODELING_API = "1"
$exeDir = "$root\openvino.genai\build\bin\Release"
$baselineExe = "$exeDir\modeling_qwen3_5.exe"
$dflashExe   = "$exeDir\modeling_qwen3_5_dflash.exe"
$modelDir    = "C:\data\models\Huggingface\Qwen3.5-4B"
$draftDir    = "C:\data\models\Huggingface\Qwen3.5-4B-DFlash"
$promptFile  = "D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling\scripts\prompt_1k.txt"
$prompt      = (Get-Content -Raw $promptFile).Trim() -replace '\r?\n', ' '
$clinterceptDll = "D:\tools\clintercept-3.0.6-win64\Release\opencl.dll"
$logBase     = "D:\chuansheng\src_code\explicit_modeling\clintercept_logs\dflash_analysis"
```

### Step 1 — Warm up model cache (no CLIntercept)

Run each config once WITHOUT CLIntercept to ensure cached IR exists.
Skip if `qwen3_5_text_q4a_b4a_g128.xml` already exists under the model dir.

```powershell
# Baseline INT4 warm-up
$env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
$env:OV_GENAI_SAVE_OV_MODEL = "1"
& $baselineExe --model $modelDir --cache-model --mode text `
  --prompt "Hello" --output-tokens 4 --think 0 --temperature 0
Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
Remove-Item Env:\OV_GENAI_SAVE_OV_MODEL -ErrorAction SilentlyContinue

# DFlash INT4/INT4 warm-up (dflash exe uses positional args, no --cache-model flag)
# Just run it once; it will compile and cache internally if needed.
& $dflashExe $modelDir $draftDir $prompt "GPU" 4 16 "INT4_ASYM" "INT4_ASYM"
```

### Step 2 — Capture Baseline INT4 with CLIntercept

```powershell
New-Item -ItemType Directory -Force -Path "$logBase\baseline_int4" | Out-Null

# Inject CLIntercept
Copy-Item $clinterceptDll "$exeDir\opencl.dll"

# Set CLIntercept env vars
$env:CLI_DevicePerformanceTiming = "1"
$env:CLI_ChromePerformanceTiming = "1"
$env:CLI_DumpDir = "$logBase\baseline_int4"
$env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
$env:OV_GENAI_SAVE_OV_MODEL = "1"

# Run — 32 output tokens to limit log size; use --cache-model (IR already cached)
& $baselineExe --model $modelDir --cache-model --mode text `
  --prompt-file $promptFile --output-tokens 32 --think 0 --temperature 0

# Clean up
Remove-Item "$exeDir\opencl.dll"
Remove-Item Env:\CLI_DevicePerformanceTiming, Env:\CLI_ChromePerformanceTiming,
            Env:\CLI_DumpDir, Env:\OV_GENAI_INFLIGHT_QUANT_MODE,
            Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE, Env:\OV_GENAI_SAVE_OV_MODEL `
            -ErrorAction SilentlyContinue

Write-Host "Log saved to: $logBase\baseline_int4"
```

### Step 3 — Capture DFlash INT4/INT4 with CLIntercept

```powershell
New-Item -ItemType Directory -Force -Path "$logBase\dflash_int4_int4" | Out-Null

# Inject CLIntercept
Copy-Item $clinterceptDll "$exeDir\opencl.dll"

$env:CLI_DevicePerformanceTiming = "1"
$env:CLI_ChromePerformanceTiming = "1"
$env:CLI_DumpDir = "$logBase\dflash_int4_int4"

# DFlash exe uses positional args: model draft prompt device max_tokens block_size target_quant draft_quant
& $dflashExe $modelDir $draftDir $prompt "GPU" 32 16 "INT4_ASYM" "INT4_ASYM"

# Clean up
Remove-Item "$exeDir\opencl.dll"
Remove-Item Env:\CLI_DevicePerformanceTiming, Env:\CLI_ChromePerformanceTiming,
            Env:\CLI_DumpDir -ErrorAction SilentlyContinue

Write-Host "Log saved to: $logBase\dflash_int4_int4"
```

### Step 4 — Analyze logs

Open the Chrome JSON files in `chrome://tracing` or parse programmatically:

```powershell
# Find the JSON files
Get-ChildItem "$logBase\baseline_int4\*.json"
Get-ChildItem "$logBase\dflash_int4_int4\*.json"
```

For automated analysis, use `scripts/analyze_clintercept.py` if available, or parse
the `DevicePerformanceTiming` report files (`*.txt`) directly:

```powershell
Get-ChildItem "$logBase\baseline_int4\*.txt"
Get-ChildItem "$logBase\dflash_int4_int4\*.txt"
```

**Identifying one decode step:** In the timing report, look for a repeating group of
kernels after the first large TTFT cluster. The TTFT block typically has many more
unique kernels (compile + first prefill). A decode step is a shorter repeating unit
~every N kernels. Isolate step 3–5 (skip early steps that may still have warm-up
variance).

**Kernel classification patterns** (case-insensitive match on kernel name):

| Sub-phase | Kernel name pattern |
|---|---|
| RMS Norm | `rms`, `layer_norm`, `rmsnorm` |
| RoPE | `rope`, `rotary` |
| Attention (SDPA) | `sdpa`, `attention`, `kv_cache`, `paged_attn` |
| MLP / Linear | `gemm`, `fc_`, `mlp`, `gate`, `silu`, `act_`, `mm_` |
| LM Head | `lm_head`, `head_` |
| Data movement | `copy`, `reorder`, `concat`, `gather` |

---

## 7. Commit & Summary

No code changes — this is a profiling-only task.

After analysis, summarize:

```
## CLIntercept Decode Step Breakdown: Baseline INT4 vs DFlash INT4/INT4
### Qwen3.5-4B, 1K prompt, 32 output tokens, Arc 140T iGPU

| Sub-phase       | Baseline INT4 (ms) | DFlash INT4/INT4 (ms) | Delta |
|---|---|---|---|
| RMS Norm        | ... | ... | ... |
| RoPE            | ... | ... | ... |
| Attention (SDPA)| ... | ... | ... |
| MLP / Linear    | ... | ... | ... |
| LM Head         | ... | ... | ... |
| Data movement   | ... | ... | ... |
| **Total decode**| ... | ... | ... |
```

Include:
- Which decode step was used for each config
- Any notable kernel differences between baseline and DFlash
- Whether DFlash adds extra kernels (draft model kernels visible in log)
- Any gaps / idle time between kernels
```
