# Dump runtime model XML for graph analysis
# Runs with a short generation to get past compilation, saves runtime XML

$ROOT = "D:\chuansheng\src_code\explicit_modeling"
$env:PATH = "$ROOT\openvino.genai\build\bin;$ROOT\openvino.genai\build\openvino_genai;$ROOT\openvino\bin\intel64\Release;$ROOT\openvino\temp\Windows_AMD64\tbb\bin;" + $env:PATH
$env:OV_GENAI_USE_MODELING_API = "1"
$env:OV_GENAI_DISABLE_THINKING = "1"
$env:OV_GENAI_DUMP_RUNTIME_MODEL = "$ROOT\openvino-explicit-modeling\runtime_graphs"

Write-Host "Dumping runtime model to: $($env:OV_GENAI_DUMP_RUNTIME_MODEL)"

& "$ROOT\openvino.genai\build\bin\modeling_qwen3_5_dflash.exe" `
  "C:\data\models\Huggingface\Qwen3.5-4B" `
  "C:\data\models\Huggingface\Qwen3.5-4B-DFlash" `
  "Hello world" `
  GPU 4 16 INT4_ASYM INT4_ASYM *>&1 | Select-Object -Last 5

Write-Host "Done. Files:"
Get-ChildItem "$ROOT\openvino-explicit-modeling\runtime_graphs" | Format-Table Name, Length
