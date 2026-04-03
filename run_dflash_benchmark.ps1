#Requires -Version 5.1
<#
.SYNOPSIS
    DFlash EXE-only benchmark (no Python pipeline dependency).
.DESCRIPTION
    Runs Baseline (modeling_qwen3_5.exe) and DFlash (modeling_qwen3_5_dflash.exe)
    configs for LLM and VLM, prints summary tables, saves report to dflash_exe_reports/.
.EXAMPLE
    .\run_dflash_benchmark.ps1
    .\run_dflash_benchmark.ps1 --skip-vlm
    .\run_dflash_benchmark.ps1 --skip-baseline --skip-vlm
#>
param()

$ErrorActionPreference = 'Continue'
$startTime = Get-Date

# ── Parse flags ──────────────────────────────────────────────────
$skipVlm              = $args -contains "--skip-vlm"
$skipBaseline         = $args -contains "--skip-baseline"
$skipBaselineInt4     = $args -contains "--skip-baseline-int4"
$skipDflashFp16Fp16   = $args -contains "--skip-dflash-fp16-fp16"
$skipDflashFp16Int4   = $args -contains "--skip-dflash-fp16-int4"
$skipDflashInt4Fp16   = $args -contains "--skip-dflash-int4-fp16"
$skipDflashInt4Int4   = $args -contains "--skip-dflash-int4-int4"
$skipVlmBaseline      = $args -contains "--skip-vlm-baseline"
$skipVlmDflashFp16Fp16 = $args -contains "--skip-vlm-dflash-fp16-fp16"
$skipVlmDflashInt4Fp16 = $args -contains "--skip-vlm-dflash-int4-fp16"

# ── Paths ────────────────────────────────────────────────────────
$scriptDir  = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$rootDir    = Split-Path -Parent $scriptDir

# ── DLL environment (same as run_dflash.bat) ─────────────────────
$dllDirs = @(
    (Join-Path $rootDir "openvino\bin\intel64\Release"),
    (Join-Path $rootDir "openvino\temp\Windows_AMD64\tbb\bin"),
    (Join-Path $rootDir "openvino.genai\build\openvino_genai"),
    (Join-Path $rootDir "openvino.genai\build\bin"),
    (Join-Path $rootDir "openvino.genai\build\bin\Release")
)
$extraPath = ($dllDirs | Where-Object { Test-Path $_ }) -join ";"
$env:PATH = "$extraPath;$($env:PATH)"
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_DISABLE_THINKING = "1"  # Disable thinking mode for benchmarks (matches --no-think)

# Prefer build\bin\ (current CMake output) over build\bin\Release\ (stale)
$binDir = Join-Path $rootDir "openvino.genai\build\bin"
$binDirRelease = Join-Path $rootDir "openvino.genai\build\bin\Release"
$baselineExe = Join-Path $binDir "modeling_qwen3_5.exe"
if (-not (Test-Path $baselineExe)) { $baselineExe = Join-Path $binDirRelease "modeling_qwen3_5.exe" }
$dflashExe = Join-Path $binDir "modeling_qwen3_5_dflash.exe"
if (-not (Test-Path $dflashExe)) { $dflashExe = Join-Path $binDirRelease "modeling_qwen3_5_dflash.exe" }

if (-not (Test-Path $baselineExe)) { Write-Error "Baseline exe not found: $baselineExe"; exit 1 }
if (-not (Test-Path $dflashExe))   { Write-Error "DFlash exe not found: $dflashExe"; exit 1 }

$modelDir    = "E:\data\models\Huggingface\Qwen3.5-4B"
$draftDir    = "E:\data\models\Huggingface\Qwen3.5-4B-DFlash"
$device      = "GPU"
$maxTokens   = 512
$blockSize   = 16
$vlmMaxTokens = 256
$vlmImage    = Join-Path $scriptDir "scripts\test.jpg"
$vlmPrompt   = "Describe this image in detail."
$reportDir   = Join-Path $rootDir "dflash_exe_reports"

# ── Read LLM prompt ─────────────────────────────────────────────
$promptFile = Join-Path $scriptDir "scripts\prompt_1k.txt"
if (Test-Path $promptFile) {
    $prompt = (Get-Content -Raw $promptFile -Encoding UTF8).Trim() -replace '\r?\n', ' '
} else {
    $prompt = "Tell me a short story about a robot."
    Write-Warning "Prompt file not found: $promptFile (using default)"
}

# ═════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════

function Parse-ExeOutput([string]$stdout) {
    $m = @{
        Label = ""; IsDFlash = $false; IsVlm = $false
        TargetQuant = "FP16"; DraftQuant = "-"
        Tokens = 0; TTFT = 0.0; TPOT = 0.0; E2E = 0.0
        DraftSteps = 0; AcceptedDraft = 0; AcceptRate = 0.0; AvgAccepted = 0.0
        OutputText = ""
    }
    if ($stdout -match 'Output token size:\s*(\d+)')        { $m.Tokens = [int]$Matches[1] }
    if ($stdout -match 'TTFT:\s*([\d.]+)\s*ms')             { $m.TTFT = [double]$Matches[1] }
    $decodeMs = 0.0
    if ($stdout -match 'Decode time:\s*([\d.]+)\s*ms')      { $decodeMs = [double]$Matches[1] }
    if ($stdout -match 'TPOT:\s*([\d.]+)\s*ms/token')       { $m.TPOT = [double]$Matches[1] }
    $m.E2E = $m.TTFT + $decodeMs

    # DFlash-specific
    if ($stdout -match 'Draft steps:\s*(\d+)')               { $m.DraftSteps = [int]$Matches[1] }
    if ($stdout -match 'Accepted draft tokens:\s*(\d+)')     { $m.AcceptedDraft = [int]$Matches[1] }
    if ($stdout -match 'Acceptance rate:\s*([\d.]+)')         { $m.AcceptRate = [double]$Matches[1] }
    if ($stdout -match 'Avg accepted per step:\s*([\d.]+)')   { $m.AvgAccepted = [double]$Matches[1] }

    # Output text
    if ($stdout -match '(?s)\[Output\]\r?\n(.*?)\r?\n\r?\n\[Generation Complete\]') {
        $m.OutputText = $Matches[1].Trim()
    } else {
        # Baseline exe prints output after "Throughput:" line
        $lines = $stdout -split '\r?\n'
        for ($i = 0; $i -lt $lines.Count; $i++) {
            if ($lines[$i] -match '^Throughput:') {
                $remaining = ($lines[($i+1)..($lines.Count-1)] -join "`n").Trim()
                if ($remaining) { $m.OutputText = $remaining }
                break
            }
        }
    }
    return $m
}

function Invoke-BaselineExe {
    param(
        [string]$Label, [string]$Prompt, [string]$Mode = "text",
        [int]$MaxTok, [string]$QuantMode = "", [string]$ImagePath = ""
    )

    Write-Host "`n$('=' * 70)"
    Write-Host "  $Label"
    Write-Host ('=' * 70)

    # Quantization via env vars
    if ($QuantMode) {
        $env:OV_GENAI_INFLIGHT_QUANT_MODE = $QuantMode.ToLower()
        $env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE = "128"
    } else {
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
        Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue
    }

    $exeArgs = @("--model", $modelDir, "--prompt", $Prompt,
                 "--device", $device, "--output-tokens", "$MaxTok",
                 "--temperature", "0", "--think", "0", "--mode", $Mode)
    if ($ImagePath) { $exeArgs += @("--image", $ImagePath) }

    Write-Host "[exe] modeling_qwen3_5.exe ($Mode)"
    Write-Host "--- output ---"

    try {
        $output = & $baselineExe @exeArgs 2>&1 | Out-String
    } catch {
        Write-Host "[exe] FAILED: $_" -ForegroundColor Red
        $output = ""
    }
    Write-Host $output
    Write-Host "--- end ---"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[exe] WARNING: exit code $LASTEXITCODE" -ForegroundColor Yellow
    }

    # Cleanup env vars
    Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_MODE -ErrorAction SilentlyContinue
    Remove-Item Env:\OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE -ErrorAction SilentlyContinue

    $m = Parse-ExeOutput $output
    $m.Label = $Label
    $m.IsDFlash = $false
    $m.IsVlm = ($Mode -eq "vl")
    $m.TargetQuant = $(if ($QuantMode) { "INT4" } else { "FP16" })
    $m.DraftQuant = "-"

    Write-Host ("ttft={0:F1}ms  tpot={1:F2}ms  e2e={2:F1}ms  tokens={3}" -f $m.TTFT, $m.TPOT, $m.E2E, $m.Tokens)
    return $m
}

function Invoke-DFlashExe {
    param(
        [string]$Label, [string]$Prompt, [int]$MaxTok,
        [string]$TargetQuant = "FP16", [string]$DraftQuant = "FP16",
        [string]$ImagePath = ""
    )

    Write-Host "`n$('=' * 70)"
    Write-Host "  $Label"
    Write-Host ('=' * 70)

    $tqLabel = $(if ($TargetQuant -match "INT4") { "INT4" } else { "FP16" })
    $dqLabel = $(if ($DraftQuant -match "INT4") { "INT4" } else { "FP16" })

    $exeArgs = @($modelDir, $draftDir, $Prompt, $device,
                 "$MaxTok", "$blockSize", $TargetQuant, $DraftQuant)
    if ($ImagePath) { $exeArgs += $ImagePath }

    Write-Host "[exe] modeling_qwen3_5_dflash.exe"
    Write-Host "[exe] target_quant=$TargetQuant  draft_quant=$DraftQuant"
    if ($ImagePath) { Write-Host "[exe] image=$ImagePath" }
    Write-Host "--- output ---"

    try {
        $output = & $dflashExe @exeArgs 2>&1 | Out-String
    } catch {
        Write-Host "[exe] FAILED: $_" -ForegroundColor Red
        $output = ""
    }
    Write-Host $output
    Write-Host "--- end ---"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[exe] WARNING: exit code $LASTEXITCODE" -ForegroundColor Yellow
    }

    $m = Parse-ExeOutput $output
    $m.Label = $Label
    $m.IsDFlash = $true
    $m.IsVlm = [bool]$ImagePath
    $m.TargetQuant = $tqLabel
    $m.DraftQuant = $dqLabel

    Write-Host ("ttft={0:F1}ms  tpot={1:F2}ms  e2e={2:F1}ms  tokens={3}" -f $m.TTFT, $m.TPOT, $m.E2E, $m.Tokens)
    if ($m.DraftSteps -gt 0) {
        Write-Host ("[DFlash] draft_steps={0}  accepted_draft={1}  avg_per_step={2:F2}  acceptance_rate={3:F1}%" -f `
            $m.DraftSteps, $m.AcceptedDraft, $m.AvgAccepted, ($m.AcceptRate * 100))
    }
    return $m
}

function Build-SummaryLines {
    param([array]$Results, $BaselineFp16, $BaselineInt4 = $null, $VlmBaseline = $null)

    $sep = "=" * 105
    $lines = @()
    $lines += ""
    $lines += $sep
    $lines += "  BENCHMARK SUMMARY"
    $lines += $sep
    $lines += "  Note: Speedup = Baseline_TPOT / DFlash_TPOT (decode speed improvement):"
    $lines += "        - target=FP16 configs use Baseline FP16 (or VLM Baseline for VLM configs)"
    $lines += "        - target=INT4 configs use Baseline INT4"
    $lines += $sep
    $lines += "{0,-30} {1,10} {2,10} {3,10} {4,7} {5,9} {6,9} {7,8}" -f `
              "Config","TTFT(ms)","TPOT(ms)","E2E(ms)","Tokens","Speedup","Accept%","AvgAcc"
    $lines += "-" * 105

    foreach ($m in $Results) {
        $speedup = ""
        if ($m.IsDFlash) {
            $baseRef = $null
            if ($m.IsVlm) { $baseRef = $VlmBaseline }
            elseif ($m.TargetQuant -eq "INT4") { $baseRef = $BaselineInt4 }
            else { $baseRef = $BaselineFp16 }
            if ($baseRef -and $baseRef.TPOT -gt 0 -and $m.TPOT -gt 0) {
                $speedup = "{0:F2}x" -f ($baseRef.TPOT / $m.TPOT)
            }
        }
        $acceptPct = $(if ($m.IsDFlash) { "{0:F1}%" -f ($m.AcceptRate * 100) } else { "-" })
        $avgAcc    = $(if ($m.IsDFlash) { "{0:F2}" -f $m.AvgAccepted } else { "-" })

        $lines += "{0,-30} {1,10:F1} {2,10:F2} {3,10:F1} {4,7} {5,9} {6,9} {7,8}" -f `
                  $m.Label, $m.TTFT, $m.TPOT, $m.E2E, $m.Tokens, $speedup, $acceptPct, $avgAcc
    }
    $lines += $sep
    return $lines
}

function Build-TextDiffLines([array]$Results) {
    $lines = @()
    $blFp16 = $Results | Where-Object { $_.Label -eq "Baseline FP16" } | Select-Object -First 1
    $blInt4 = $Results | Where-Object { $_.Label -eq "Baseline INT4" } | Select-Object -First 1
    $dfFp16 = @($Results | Where-Object { $_.IsDFlash -and $_.TargetQuant -eq "FP16" -and -not $_.IsVlm })
    $dfInt4 = @($Results | Where-Object { $_.IsDFlash -and $_.TargetQuant -eq "INT4" -and -not $_.IsVlm })

    if ($blFp16 -and $dfFp16.Count -gt 0) {
        $lines += ""
        $lines += "=" * 105
        $lines += "  OUTPUT TEXT DIFF: target=FP16 DFlash vs Baseline FP16"
        $lines += "=" * 105
        foreach ($d in $dfFp16) {
            if ($blFp16.OutputText -eq $d.OutputText) {
                $lines += "  Baseline FP16  vs  $($d.Label)  =>  IDENTICAL"
            } else {
                # Simple char-level similarity
                $a = $blFp16.OutputText; $b = $d.OutputText
                $maxLen = [Math]::Max($a.Length, $b.Length)
                if ($maxLen -eq 0) { $sim = 1.0 }
                else {
                    $common = 0
                    $minLen = [Math]::Min($a.Length, $b.Length)
                    for ($i = 0; $i -lt $minLen; $i++) { if ($a[$i] -eq $b[$i]) { $common++ } }
                    $sim = [double]$common / $maxLen
                }
                $lines += "  Baseline FP16  vs  $($d.Label)  =>  DIFFERENT  (similarity $("{0:F1}%" -f ($sim * 100)))"
            }
            $lines += ""
        }
    }

    if ($blInt4 -and $dfInt4.Count -gt 0) {
        $lines += "=" * 105
        $lines += "  OUTPUT TEXT DIFF: target=INT4 DFlash vs Baseline INT4"
        $lines += "=" * 105
        foreach ($d in $dfInt4) {
            if ($blInt4.OutputText -eq $d.OutputText) {
                $lines += "  Baseline INT4  vs  $($d.Label)  =>  IDENTICAL"
            } else {
                $a = $blInt4.OutputText; $b = $d.OutputText
                $maxLen = [Math]::Max($a.Length, $b.Length)
                if ($maxLen -eq 0) { $sim = 1.0 }
                else {
                    $common = 0
                    $minLen = [Math]::Min($a.Length, $b.Length)
                    for ($i = 0; $i -lt $minLen; $i++) { if ($a[$i] -eq $b[$i]) { $common++ } }
                    $sim = [double]$common / $maxLen
                }
                $lines += "  Baseline INT4  vs  $($d.Label)  =>  DIFFERENT  (similarity $("{0:F1}%" -f ($sim * 100)))"
            }
            $lines += ""
        }
    }
    return $lines
}

function Save-Report {
    param([array]$Results, $BaselineFp16, $BaselineInt4 = $null, $VlmBaseline = $null)

    if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir -Force | Out-Null }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $reportPath = Join-Path $reportDir "dflash_benchmark_${ts}.txt"

    $lines = @()
    $lines += "DFlash EXE Benchmark Report"
    $lines += "=" * 80
    $lines += "timestamp=$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ss')"
    $lines += "model_dir=$modelDir"
    $lines += "draft_dir=$draftDir"
    $lines += "device=$device"
    $lines += "max_tokens=$maxTokens"
    $lines += "block_size=$blockSize"
    $lines += "vlm_max_tokens=$vlmMaxTokens"
    $lines += "vlm_image=$vlmImage"
    $lines += ""

    # Summary table
    $lines += Build-SummaryLines -Results $Results -BaselineFp16 $BaselineFp16 `
                                 -BaselineInt4 $BaselineInt4 -VlmBaseline $VlmBaseline
    $lines += ""

    # Text diffs
    $lines += Build-TextDiffLines $Results
    $lines += ""

    # Per-config details
    $lines += "=" * 105
    $lines += "PER-CONFIG DETAILS"
    $lines += "=" * 105
    foreach ($m in $Results) {
        $lines += "[$($m.Label)]"
        $lines += "ttft_ms=$("{0:F1}" -f $m.TTFT)"
        $lines += "tpot_ms=$("{0:F2}" -f $m.TPOT)"
        $lines += "e2e_ms=$("{0:F1}" -f $m.E2E)"
        $lines += "generated_tokens=$($m.Tokens)"
        $lines += "is_dflash=$($m.IsDFlash)"
        $lines += "target_quant=$($m.TargetQuant)"
        $lines += "draft_quant=$($m.DraftQuant)"
        if ($m.IsDFlash) {
            $lines += "draft_steps=$($m.DraftSteps)"
            $lines += "accepted_draft_tokens=$($m.AcceptedDraft)"
            $lines += "acceptance_rate=$("{0:F6}" -f $m.AcceptRate)"
            $lines += "avg_accepted_per_step=$("{0:F4}" -f $m.AvgAccepted)"
        }
        if ($m.OutputText) {
            $lines += "[Output Text]"
            $lines += $m.OutputText
        }
        $lines += ""
    }

    $lines -join "`n" | Set-Content -Path $reportPath -Encoding UTF8
    return $reportPath
}

# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

Write-Host "============================================================"
Write-Host "  DFlash EXE Benchmark"
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "============================================================"

$llmResults = [System.Collections.ArrayList]::new()
$vlmResults = [System.Collections.ArrayList]::new()
$baselineFp16 = $null
$baselineInt4 = $null
$vlmBaselineFp16 = $null

# ── 1. Baseline FP16 ────────────────────────────────────────────
if (-not $skipBaseline) {
    $m = Invoke-BaselineExe -Label "Baseline FP16" -Prompt $prompt -Mode "text" -MaxTok $maxTokens
    [void]$llmResults.Add($m)
    $baselineFp16 = $m
}

# ── 2. DFlash FP16 / FP16 ───────────────────────────────────────
if (-not $skipDflashFp16Fp16) {
    $m = Invoke-DFlashExe -Label "DFlash  target=FP16  draft=FP16" `
         -Prompt $prompt -MaxTok $maxTokens -TargetQuant "FP16" -DraftQuant "FP16"
    [void]$llmResults.Add($m)
}

# ── 3. DFlash FP16 / INT4 ───────────────────────────────────────
if (-not $skipDflashFp16Int4) {
    $m = Invoke-DFlashExe -Label "DFlash  target=FP16  draft=INT4" `
         -Prompt $prompt -MaxTok $maxTokens -TargetQuant "FP16" -DraftQuant "INT4_ASYM"
    [void]$llmResults.Add($m)
}

# ── 4. Baseline INT4 ────────────────────────────────────────────
if ((-not $skipBaseline) -and (-not $skipBaselineInt4)) {
    $m = Invoke-BaselineExe -Label "Baseline INT4" -Prompt $prompt -Mode "text" `
         -MaxTok $maxTokens -QuantMode "int4_asym"
    [void]$llmResults.Add($m)
    $baselineInt4 = $m
}

# ── 5. DFlash INT4 / FP16 ───────────────────────────────────────
if (-not $skipDflashInt4Fp16) {
    $m = Invoke-DFlashExe -Label "DFlash  target=INT4  draft=FP16" `
         -Prompt $prompt -MaxTok $maxTokens -TargetQuant "INT4_ASYM" -DraftQuant "FP16"
    [void]$llmResults.Add($m)
}

# ── 6. DFlash INT4 / INT4 ───────────────────────────────────────
if (-not $skipDflashInt4Int4) {
    $m = Invoke-DFlashExe -Label "DFlash  target=INT4  draft=INT4" `
         -Prompt $prompt -MaxTok $maxTokens -TargetQuant "INT4_ASYM" -DraftQuant "INT4_ASYM"
    [void]$llmResults.Add($m)
}

# ── Print LLM table ─────────────────────────────────────────────
foreach ($line in (Build-SummaryLines -Results $llmResults -BaselineFp16 $baselineFp16 -BaselineInt4 $baselineInt4)) {
    Write-Host $line
}
foreach ($line in (Build-TextDiffLines $llmResults)) {
    Write-Host $line
}

# ── VLM configs ─────────────────────────────────────────────────
if ((-not $skipVlm) -and (Test-Path $vlmImage)) {
    Write-Host "`n$('#' * 70)"
    Write-Host "  VLM MODE - image: $vlmImage"
    Write-Host ('#' * 70)

    # ── 7. VLM Baseline FP16 ────────────────────────────────────
    if (-not $skipVlmBaseline) {
        $m = Invoke-BaselineExe -Label "VLM Baseline FP16" -Prompt $vlmPrompt `
             -Mode "vl" -MaxTok $vlmMaxTokens -ImagePath $vlmImage
        [void]$vlmResults.Add($m)
        $vlmBaselineFp16 = $m
    }

    # ── 8. VLM DFlash FP16 / FP16 ──────────────────────────────
    if (-not $skipVlmDflashFp16Fp16) {
        $m = Invoke-DFlashExe -Label "VLM DFlash  target=FP16  draft=FP16" `
             -Prompt $vlmPrompt -MaxTok $vlmMaxTokens `
             -TargetQuant "FP16" -DraftQuant "FP16" -ImagePath $vlmImage
        [void]$vlmResults.Add($m)
    }

    # ── 9. VLM DFlash INT4 / FP16 ──────────────────────────────
    if (-not $skipVlmDflashInt4Fp16) {
        $m = Invoke-DFlashExe -Label "VLM DFlash  target=INT4  draft=FP16" `
             -Prompt $vlmPrompt -MaxTok $vlmMaxTokens `
             -TargetQuant "INT4_ASYM" -DraftQuant "FP16" -ImagePath $vlmImage
        [void]$vlmResults.Add($m)
    }

    # ── Print VLM table ─────────────────────────────────────────
    if ($vlmResults.Count -gt 0) {
        Write-Host "`n$('#' * 105)"
        Write-Host "  VLM BENCHMARK (image: $(Split-Path -Leaf $vlmImage))"
        Write-Host ('#' * 105)
        foreach ($line in (Build-SummaryLines -Results $vlmResults -BaselineFp16 $vlmBaselineFp16 -VlmBaseline $vlmBaselineFp16)) {
            Write-Host $line
        }
    }
}

# ── Save report ─────────────────────────────────────────────────
$allResults = @($llmResults) + @($vlmResults)
if ($allResults.Count -gt 0) {
    try {
        $rp = Save-Report -Results $allResults -BaselineFp16 $baselineFp16 `
                          -BaselineInt4 $baselineInt4 -VlmBaseline $vlmBaselineFp16
        Write-Host "[Report] Saved: $rp"
    } catch {
        Write-Error "[Report] Failed: $_"
    }
}

$elapsed = (Get-Date) - $startTime
Write-Host "`n[Total time: $("{0:F0}" -f $elapsed.TotalSeconds)s]"
