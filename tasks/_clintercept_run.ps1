param([string]$Step)

$root       = "D:\chuansheng\src_code\explicit_modeling"
$exeDir     = "$root\openvino.genai\build\bin\Release"
$baselineExe = "$exeDir\modeling_qwen3_5.exe"
$dflashExe   = "$exeDir\modeling_qwen3_5_dflash.exe"
$modelDir    = "C:\data\models\Huggingface\Qwen3.5-4B"
$draftDir    = "C:\data\models\Huggingface\Qwen3.5-4B-DFlash"
$promptFile  = "$root\openvino-explicit-modeling\scripts\prompt_1k.txt"
$prompt      = (Get-Content -Raw $promptFile -Encoding UTF8).Trim() -replace '\r?\n', ' '
$clDll       = "D:\tools\clintercept-3.0.6-win64\Release\opencl.dll"
$logBase     = "$root\clintercept_logs\dflash_analysis"

$env:PATH = "$exeDir;$root\openvino.genai\build\bin;$root\openvino.genai\build\openvino_genai;$root\openvino\bin\intel64\Release;$root\openvino\temp\Windows_AMD64\tbb\bin;" + $env:PATH
$env:OV_GENAI_USE_MODELING_API = "1"

switch ($Step) {

    "warmup_dflash" {
        Write-Host "=== Warming up DFlash INT4/INT4 cache ==="
        & $dflashExe $modelDir $draftDir $prompt "GPU" 4 16 "INT4_ASYM" "INT4_ASYM"
        Write-Host "Warmup done. Exit: $LASTEXITCODE"
    }

    "capture_baseline" {
        New-Item -ItemType Directory -Force -Path "$logBase\baseline_int4" | Out-Null
        Copy-Item $clDll "$exeDir\opencl.dll" -Force
        $env:CLI_DevicePerformanceTiming = "1"
        $env:CLI_ChromePerformanceTiming = "1"
        $env:CLI_DumpDir = "$logBase\baseline_int4"
        $env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
        $env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
        $env:OV_GENAI_SAVE_OV_MODEL = "1"

        Write-Host "=== Capturing Baseline INT4 ==="
        & $baselineExe --model $modelDir --cache-model --mode text `
            --prompt-file $promptFile --output-tokens 32 --think 0 --temperature 0

        Remove-Item "$exeDir\opencl.dll" -ErrorAction SilentlyContinue
        "CLI_DevicePerformanceTiming","CLI_ChromePerformanceTiming","CLI_DumpDir",
        "OV_GENAI_INFLIGHT_QUANT_MODE","OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE","OV_GENAI_SAVE_OV_MODEL" |
            ForEach-Object { Remove-Item "Env:\$_" -ErrorAction SilentlyContinue }
        Write-Host "Log saved to: $logBase\baseline_int4"
    }

    "capture_dflash" {
        New-Item -ItemType Directory -Force -Path "$logBase\dflash_int4_int4" | Out-Null
        Copy-Item $clDll "$exeDir\opencl.dll" -Force
        $env:CLI_DevicePerformanceTiming = "1"
        $env:CLI_ChromePerformanceTiming = "1"
        $env:CLI_DumpDir = "$logBase\dflash_int4_int4"

        Write-Host "=== Capturing DFlash INT4/INT4 ==="
        & $dflashExe $modelDir $draftDir $prompt "GPU" 32 16 "INT4_ASYM" "INT4_ASYM"

        Remove-Item "$exeDir\opencl.dll" -ErrorAction SilentlyContinue
        "CLI_DevicePerformanceTiming","CLI_ChromePerformanceTiming","CLI_DumpDir" |
            ForEach-Object { Remove-Item "Env:\$_" -ErrorAction SilentlyContinue }
        Write-Host "Log saved to: $logBase\dflash_int4_int4"
    }
}
