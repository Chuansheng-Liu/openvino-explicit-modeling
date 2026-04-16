# Capture CLIntercept log for Fix4: no RemoteTensor pre-allocation for snapshot outputs
# Fix1+2+3+4 applied: bfzyx format + conv f16 + recurrent f16 + no RemoteTensor
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\tasks\capture_fix4_no_remote_tensor.ps1

param(
    [string]$LogName = "fix4_no_remote_tensor_final",
    [int]$MaxTokens = 512
)

$root       = "D:\chuansheng\src_code\explicit_modeling"
# Fix4 exe lives in build\bin (not build\bin\Release)
$exeDir     = "$root\openvino.genai\build\bin"
$dflashExe  = "$exeDir\modeling_qwen3_5_dflash.exe"
$modelDir   = "C:\data\models\Huggingface\Qwen3.5-4B"
$draftDir   = "C:\data\models\Huggingface\Qwen3.5-4B-DFlash"
$promptFile = "$root\openvino-explicit-modeling\scripts\prompt_1k.txt"
$clDll      = "D:\tools\clintercept-3.0.6-win64\Release\opencl.dll"
$logBase    = "$root\clintercept_logs"
$logDir     = "$logBase\$LogName"

$prompt = (Get-Content -Raw $promptFile -Encoding UTF8).Trim() -replace '\r?\n', ' '

$env:PATH = "$exeDir;$root\openvino.genai\build\openvino_genai;$root\openvino\bin\intel64\Release;$root\openvino\temp\Windows_AMD64\tbb\bin;" + $env:PATH
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_DISABLE_THINKING = "1"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $clDll)) {
    Write-Error "CLIntercept DLL not found at: $clDll"
    exit 1
}
Write-Host "Copying CLIntercept DLL to exe dir..."
Copy-Item $clDll "$exeDir\opencl.dll" -Force

$env:CLI_DevicePerformanceTiming = "1"
$env:CLI_ChromePerformanceTiming = "1"
$env:CLI_DumpDir = $logDir

Write-Host "=== Capturing DFlash INT4/INT4 (Fix4: no RemoteTensor) ==="
Write-Host "    EXE:     $dflashExe"
Write-Host "    Log dir: $logDir"
Write-Host "    Tokens:  $MaxTokens"
Write-Host ""

& $dflashExe $modelDir $draftDir $prompt "GPU" $MaxTokens 16 "INT4_ASYM" "INT4_ASYM"
$rc = $LASTEXITCODE

Remove-Item "$exeDir\opencl.dll" -ErrorAction SilentlyContinue
"CLI_DevicePerformanceTiming","CLI_ChromePerformanceTiming","CLI_DumpDir" |
    ForEach-Object { Remove-Item "Env:\$_" -ErrorAction SilentlyContinue }

Write-Host ""
Write-Host "Done (exit $rc). Log: $logDir"
if (Test-Path "$logDir\clintercept_trace.json") {
    $sz = (Get-Item "$logDir\clintercept_trace.json").Length
    Write-Host "Trace file: $([math]::Round($sz/1MB, 1)) MB"
}
