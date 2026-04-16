# Checkout dflash Branches and Build

## 1. Goal

Check out the `dflash` branch across all three project repositories and produce a
working Release build via `build.bat`. The build is successful when
`modeling_qwen3_5.exe` is present in
`openvino.genai\build\bin\Release\`.

## 2. Role

You are a build engineer responsible for syncing the three repositories to their
correct feature branches and producing a clean, runnable build. Do not modify any
source files. If a build step fails, diagnose the error and fix the environment
(missing cache, generator mismatch, etc.) before retrying.

## 3. Process

1. Fetch and checkout `dflash` in each repo (see Build section for exact commands).
2. For `openvino-explicit-modeling`, the branch is on a different fork
   (`liqianhao111`); add it as a remote if not already present.
3. Run `build.bat` from `openvino-explicit-modeling\`.
4. If CMake fails with a generator mismatch, run `build.bat --clean` to wipe stale
   caches and retry.
5. Verify the output executable exists.

**Known caveats:**
- The openvino build tree may contain stale VS 2022 generator caches from a
  previous configure. `build.bat` auto-detects this and cleans; if it still fails,
  use `--clean`.
- SSH (port 22) is blocked on the corporate network. All `git fetch/push` must use
  HTTPS remotes. Credentials may need to be entered interactively; if they fail in
  bash, switch to a PowerShell or Git GUI window.

## 4. Tools

- Build script: `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling\build.bat`
- openvino repo: `D:\chuansheng\src_code\explicit_modeling\openvino`
- openvino.genai repo: `D:\chuansheng\src_code\explicit_modeling\openvino.genai`
- openvino-explicit-modeling repo: `D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling`
- Output exe: `D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe`

## 5. Build

> **RE-READ this section before executing any command.**

### Step 1 — openvino: fetch and checkout `dflash` from liangali's fork

```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino
git fetch origin dflash
git checkout dflash
```

### Step 2 — openvino.genai: fetch and checkout `dflash` from liangali's fork

```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino.genai
git fetch origin dflash
git checkout dflash
```

### Step 3 — openvino-explicit-modeling: fetch and checkout from liqianhao111's fork

```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling

# Add liqianhao111's fork as a remote if not already present
git remote get-url liqianhao111 2>$null
if ($LASTEXITCODE -ne 0) {
    git remote add liqianhao111 https://github.com/liqianhao111/openvino-explicit-modeling.git
}

git fetch liqianhao111
git checkout -b dflash liqianhao111/dflash
# If branch already exists locally: git checkout dflash && git reset --hard liqianhao111/dflash
```

### Step 4 — Build

```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
.\build.bat
```

If it fails with a CMake generator mismatch or stale cache error:

```powershell
.\build.bat --clean
```

## 6. Test

After a successful build, verify the executable exists:

```powershell
$exe = "D:\chuansheng\src_code\explicit_modeling\openvino.genai\build\bin\Release\modeling_qwen3_5.exe"
Test-Path $exe   # must be True
```

Run the dflash benchmark (LLM-only, skip VLM since no test image required):

```powershell
cd D:\chuansheng\src_code\explicit_modeling\openvino-explicit-modeling
.\run_dflash.bat --skip-vlm
```

> **Note:** `run_dflash_benchmark.ps1` hardcodes model paths to `D:\Data\models\Huggingface`.
> The actual models live at `C:\data\models\Huggingface`. Before running, patch the
> path variables in `run_dflash_benchmark.ps1` lines 53-54:
>   `$modelDir = "C:\data\models\Huggingface\Qwen3.5-4B"`
>   `$draftDir = "C:\data\models\Huggingface\Qwen3.5-4B-DFlash"`
>
> The benchmark requires two executables in `openvino.genai\build\bin\Release\`:
>   - `modeling_qwen3_5.exe`       (baseline)
>   - `modeling_qwen3_5_dflash.exe` (dflash — built from the dflash branch)

Expected: exits 0, prints summary table with TTFT/TPOT/Speedup for all
Baseline FP16 / DFlash FP16+FP16 / DFlash FP16+INT4 / Baseline INT4 /
DFlash INT4+FP16 / DFlash INT4+INT4 configs.

## 7. Commit & Summary

No source changes are expected from this task (branch checkout + build only).
After the build succeeds, summarize:

- Which commit SHAs are checked out in each repo (`git rev-parse HEAD`)
- Build duration
- Whether the smoke test passed and the reported throughput
- Any issues encountered (generator mismatch, auth errors, etc.) and how they
  were resolved
