param(
    [string]$Python = (Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"),
    [string]$Device = "cuda",
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$OcrDevice = "auto",
    [switch]$GpuOcr,
    [int]$Epochs = 5,
    [int]$BatchSize = 64
)

$ErrorActionPreference = "Stop"

$IndexScript = Join-Path $PSScriptRoot "build_ui_index_from_pdfs.py"
$TripletScript = Join-Path $PSScriptRoot "build_triplet_dataset.py"
$TrainScript = Join-Path $PSScriptRoot "train_3b_projection_adapter.py"
$EvalScript = Join-Path $PSScriptRoot "evaluate_projection_adapter.py"

$PdfDir = Join-Path $PSScriptRoot "pdf"
$IndexDir = Join-Path $PSScriptRoot "generated\ui_index"
$Triplets = Join-Path $PSScriptRoot "generated\triplets.jsonl"

$ocrArgs = @(
    "--pdf-dir", $PdfDir,
    "--out-dir", $IndexDir,
    "--resume",
    "--ocr-device", $OcrDevice
)
if ($GpuOcr) {
    $ocrArgs += @("--ocr-device", "cuda")
}

Write-Host "[1/4] Building UI index from PDFs..."
& $Python $IndexScript @ocrArgs

Write-Host "[2/4] Building triplet dataset..."
& $Python $TripletScript --ui-items (Join-Path $IndexDir "ui_items.jsonl") --out $Triplets

Write-Host "[3/4] Training 3B projection adapter..."
& $Python $TrainScript --triplets $Triplets --model-size "3B" --device $Device --epochs $Epochs --batch-size $BatchSize

Write-Host "[4/4] Evaluating projection adapter..."
& $Python $EvalScript --model-size "3B" --device $Device
