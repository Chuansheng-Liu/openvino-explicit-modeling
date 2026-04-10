<#
.SYNOPSIS
    Launch ov_serve.exe with optimal configuration for Qwen3.5 on Intel Arc GPU.
.DESCRIPTION
    Sets up environment variables and starts the HTTP server with best-practice
    defaults for INT4 quantized Qwen3.5 models.
.PARAMETER Model
    Path to HF model directory (default: Qwen3.5-4B)
.PARAMETER VL
    Enable vision-language mode (loads vision encoder)
.PARAMETER Port
    HTTP port (default: 8080)
.PARAMETER WarmupTokens
    GPU warmup sequence length (default: 4096, 0=disable)
.PARAMETER NoThinking
    Disable thinking mode
.EXAMPLE
    # Text-only 4B
    .\launch_ov_serve.ps1
    # VL 4B
    .\launch_ov_serve.ps1 -VL
    # 35B text
    .\launch_ov_serve.ps1 -Model C:\data\models\Huggingface\Qwen3.5-35B-A3B
#>
param(
    [string]$Model = "C:\data\models\Huggingface\Qwen3.5-4B",
    [switch]$VL,
    [int]$Port = 8080,
    [int]$WarmupTokens = 4096,
    [switch]$NoThinking,
    [string]$Device = "GPU",
    [int]$Workers = 1,
    [float]$RepPenalty = 1.1,
    [float]$PresPenalty = 1.5,
    [float]$MinTemp = 0.0,
    [int]$MaxTokens = 2048
)

$ErrorActionPreference = "Stop"

# ── Paths ──
$REPO_ROOT = "D:\chuansheng\src_code\explicit_modeling"
$OV_ROOT   = "$REPO_ROOT\openvino"
$GENAI_ROOT = "$REPO_ROOT\openvino.genai"
$EXE = "$GENAI_ROOT\build\bin\Release\ov_serve.exe"

if (-not (Test-Path $EXE)) {
    Write-Error "ov_serve.exe not found at $EXE. Run build.bat first."
    exit 1
}
if (-not (Test-Path $Model)) {
    Write-Error "Model directory not found: $Model"
    exit 1
}

# ── Environment ──
$env:PATH = @(
    "$OV_ROOT\bin\intel64\Release",
    "$OV_ROOT\temp\Windows_AMD64\tbb\bin",
    "$OV_ROOT\build\bin\Release",
    "$GENAI_ROOT\build\openvino_genai",
    "$GENAI_ROOT\build\bin\Release",
    $env:PATH
) -join ";"

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

if ($VL) { $args_list += "--vl" }
if ($NoThinking) { $args_list += "--no-thinking" }
if ($MinTemp -gt 0) { $args_list += @("--min-temp", $MinTemp) }

# ── Launch ──
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ov_serve — Qwen3.5 OpenVINO Inference Server"     -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Model:          $Model"
Write-Host "  Device:         $Device"
Write-Host "  Port:           $Port"
Write-Host "  Workers:        $Workers"
Write-Host "  Vision (VL):    $($VL.IsPresent)"
Write-Host "  Thinking:       $(-not $NoThinking.IsPresent)"
Write-Host "  Rep.Penalty:    $RepPenalty"
Write-Host "  Pres.Penalty:   $PresPenalty"
Write-Host "  Max Tokens:     $MaxTokens"
Write-Host "  Warmup Tokens:  $WarmupTokens"
Write-Host "  Quant:          int4_asym / group_size=128"
Write-Host ""
Write-Host "  Endpoint:       http://localhost:${Port}/v1/chat/completions"
Write-Host "═══════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

& $EXE @args_list
