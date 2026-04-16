param(
    [string]$LogName = "dflash_snapshot_dtype_f16",
    [int]$MaxTokens = 512
)

$root       = "D:\chuansheng\src_code\explicit_modeling"
$exeDir     = "$root\openvino.genai\build\bin\Release"
$dflashExe  = "$exeDir\modeling_qwen3_5_dflash.exe"
$modelDir   = "C:\data\models\Huggingface\Qwen3.5-4B"
$draftDir   = "C:\data\models\Huggingface\Qwen3.5-4B-DFlash"
$promptFile = "$root\openvino-explicit-modeling\scripts\prompt_1k.txt"
$clDll      = "D:\tools\clintercept-3.0.6-win64\Release\opencl.dll"
$logDir     = "$root\clintercept_logs\$LogName"

$prompt = (Get-Content -Raw $promptFile -Encoding UTF8).Trim() -replace '\r?\n', ' '

$env:PATH = "$exeDir;$root\openvino.genai\build\bin;$root\openvino.genai\build\openvino_genai;" +
            "$root\openvino\bin\intel64\Release;$root\openvino\temp\Windows_AMD64\tbb\bin;" + $env:PATH
$env:OV_GENAI_USE_MODELING_API   = "1"
$env:OV_GENAI_DISABLE_THINKING   = "1"

# ── Warmup (no CLIntercept) ──────────────────────────────────
Write-Host "=== Warmup (model cache build) ===" -ForegroundColor Cyan
& $dflashExe $modelDir $draftDir $prompt "GPU" 4 16 "INT4_ASYM" "INT4_ASYM"
if ($LASTEXITCODE -ne 0) { Write-Host "[WARN] Warmup exited $LASTEXITCODE" -ForegroundColor Yellow }

# ── CLIntercept capture ──────────────────────────────────────
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
Copy-Item $clDll "$exeDir\opencl.dll" -Force

$env:CLI_DevicePerformanceTiming = "1"
$env:CLI_ChromePerformanceTiming = "1"
$env:CLI_DumpDir                 = $logDir

Write-Host ""
Write-Host "=== Capturing DFlash INT4/INT4 ($MaxTokens tokens) ===" -ForegroundColor Cyan
Write-Host "    Log dir: $logDir"
& $dflashExe $modelDir $draftDir $prompt "GPU" $MaxTokens 16 "INT4_ASYM" "INT4_ASYM"
$rc = $LASTEXITCODE

Remove-Item "$exeDir\opencl.dll" -ErrorAction SilentlyContinue
"CLI_DevicePerformanceTiming","CLI_ChromePerformanceTiming","CLI_DumpDir" |
    ForEach-Object { Remove-Item "Env:\$_" -ErrorAction SilentlyContinue }

Write-Host ""
if ($rc -eq 0) {
    Write-Host "[OK] Capture complete. Log: $logDir" -ForegroundColor Green
} else {
    Write-Host "[ERROR] DFlash exited $rc" -ForegroundColor Red
}
exit $rc
