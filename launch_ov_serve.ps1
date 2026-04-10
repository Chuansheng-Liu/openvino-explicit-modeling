<#
.SYNOPSIS
    Launch ov_serve.exe with optimal configuration for Qwen3.5 on Intel Arc GPU.
.DESCRIPTION
    Sets up environment variables and starts the HTTP server with best-practice
    defaults for INT4 quantized Qwen3.5 models.
    Automatically detects standalone package layout vs. build tree.
.PARAMETER Model
    Path to HF model directory (default: Qwen3.5-4B)
.PARAMETER NoVL
    Disable vision-language mode (VL is on by default)
.PARAMETER Port
    HTTP port (default: 8080)
.PARAMETER WarmupTokens
    GPU warmup sequence length (default: 4096, 0=disable)
.PARAMETER Thinking
    Enable thinking mode (default: off)
.EXAMPLE
    # VL 4B (default)
    .\launch_ov_serve.ps1
    # Text-only 4B
    .\launch_ov_serve.ps1 -NoVL
    # 35B text
    .\launch_ov_serve.ps1 -Model C:\data\models\Huggingface\Qwen3.5-35B-A3B
    # From standalone package
    cd package_serve\Release
    .\launch_ov_serve.ps1 -Model C:\data\models\Huggingface\Qwen3.5-4B -VL
#>
param(
    [string]$Model = "C:\data\models\Huggingface\Qwen3.5-4B",
    [switch]$NoVL,
    [int]$Port = 8080,
    [int]$WarmupTokens = 4096,
    [switch]$Thinking,
    [string]$Device = "GPU",
    [int]$Workers = 1,
    [float]$RepPenalty = 1.1,
    [float]$PresPenalty = 0.0,
    [float]$MinTemp = 0.0,
    [int]$MaxTokens = 2048,
    [switch]$NoLog
)

$ErrorActionPreference = "Stop"

# ── Detect standalone package vs build tree ──
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Standalone mode: ov_serve.exe is in the same directory as this script
$STANDALONE_EXE = Join-Path $SCRIPT_DIR "ov_serve.exe"

if (Test-Path $STANDALONE_EXE) {
    # ── Standalone package mode ──
    $EXE = $STANDALONE_EXE
    $env:PATH = "$SCRIPT_DIR;$env:PATH"
    $LaunchMode = "standalone"
} else {
    # ── Build tree mode ──
    $REPO_ROOT = "D:\chuansheng\src_code\explicit_modeling"
    $OV_ROOT   = "$REPO_ROOT\openvino"
    $GENAI_ROOT = "$REPO_ROOT\openvino.genai"
    $EXE = "$GENAI_ROOT\build\bin\Release\ov_serve.exe"

    if (-not (Test-Path $EXE)) {
        Write-Error "ov_serve.exe not found at $EXE. Run build.bat first."
        exit 1
    }

    $env:PATH = @(
        "$OV_ROOT\bin\intel64\Release",
        "$OV_ROOT\temp\Windows_AMD64\tbb\bin",
        "$OV_ROOT\build\bin\Release",
        "$GENAI_ROOT\build\openvino_genai",
        "$GENAI_ROOT\build\bin\Release",
        $env:PATH
    ) -join ";"
    $LaunchMode = "build-tree"
}

if (-not (Test-Path $Model)) {
    Write-Error "Model directory not found: $Model"
    exit 1
}

# Modeling API + INT4 asymmetric quantization with group_size=128
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_INFLIGHT_QUANT_MODE = "int4_asym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"

# ── Build command line ──
$args_list = @(
    "--model", $Model,
    "--port", $Port,
    "--device", $Device,
    "--workers", $Workers,
    "--rep-penalty", $RepPenalty,
    "--pres-penalty", $PresPenalty,
    "--max-tokens", $MaxTokens,
    "--warmup-tokens", $WarmupTokens
)

if (-not $NoVL) { $args_list += "--vl" }
if (-not $Thinking) { $args_list += "--no-thinking" }
if ($MinTemp -gt 0) { $args_list += @("--min-temp", $MinTemp) }
if ($NoLog) { $args_list += "--no-log" }

# ── Launch ──
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ov_serve — Qwen3.5 OpenVINO Inference Server"     -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Launch mode:    $LaunchMode"
Write-Host "  Model:          $Model"
Write-Host "  Device:         $Device"
Write-Host "  Port:           $Port"
Write-Host "  Workers:        $Workers"
Write-Host "  Vision (VL):    $(-not $NoVL.IsPresent)"
Write-Host "  Thinking:       $($Thinking.IsPresent)"
Write-Host "  Rep.Penalty:    $RepPenalty"
Write-Host "  Pres.Penalty:   $PresPenalty"
Write-Host "  Max Tokens:     $MaxTokens"
Write-Host "  Warmup Tokens:  $WarmupTokens"
Write-Host "  Logging:        $(-not $NoLog.IsPresent)"
Write-Host "  Quant:          int4_asym / group_size=128"
Write-Host ""
Write-Host "  Endpoint:       http://localhost:${Port}/v1/chat/completions"
Write-Host ""

# ── DLL verification ──
$keyDlls = @("openvino.dll", "openvino_genai.dll", "openvino_intel_gpu_plugin.dll", "tbb12.dll")
Write-Host "  DLL resolution:" -ForegroundColor Yellow
foreach ($dll in $keyDlls) {
    $resolved = (where.exe $dll 2>$null) | Select-Object -First 1
    if ($resolved) {
        $color = if ($LaunchMode -eq "standalone" -and $resolved.StartsWith($SCRIPT_DIR)) { "Green" }
                 elseif ($LaunchMode -eq "build-tree") { "Green" }
                 else { "Red" }
        Write-Host "    $dll -> $resolved" -ForegroundColor $color
    } else {
        Write-Host "    $dll -> NOT FOUND" -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# ── Log file: stderr saved to file, stdout to terminal ──
$LogFile = Join-Path $SCRIPT_DIR "ov_serve.log"
Write-Host "  Log file:       $LogFile" -ForegroundColor Yellow
Write-Host ""

& $EXE @args_list 2>$LogFile
