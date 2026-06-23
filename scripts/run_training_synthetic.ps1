# run_training_synthetic.ps1
# -----------------------------------------------
# Полный пайплайн обучения Dual Encoder на синтетическом датасете
# Запускать из корня проекта:
#   .\scripts\run_training_synthetic.ps1
# -----------------------------------------------

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$JsonPath   = "training\synthetic_dataset\triplet_dataset_clean.json"
$SplitPath  = "training\synthetic_dataset\dual_encoder_splits.json"
$ConfigPath = "training\synthetic_dataset\train_config.json"
$OutputDir  = "artifacts\synthetic_dual_encoder"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   Visual Language UI Embedder - Training Pipeline" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# ---- Step 1: Junction -------------------------------------------------------
Write-Host "[1/4] Checking image directory junction..." -ForegroundColor Yellow
$JunctionPath = "training\synthetic_dataset\dataset_images"
if (-not (Test-Path $JunctionPath)) {
    cmd /c mklink /J $JunctionPath "training\synthetic_dataset\data"
    Write-Host "      Junction created: $JunctionPath -> training\synthetic_dataset\data" -ForegroundColor Green
} else {
    Write-Host "      Junction already exists: $JunctionPath" -ForegroundColor Green
}

# ---- Step 2: Build splits ---------------------------------------------------
Write-Host ""
Write-Host "[2/4] Building train/val/test splits..." -ForegroundColor Yellow
python scripts/build_splits.py `
    --json-path $JsonPath `
    --output-path $SplitPath `
    --train-ratio 0.8 `
    --val-ratio 0.1 `
    --test-ratio 0.1 `
    --seed 42
Write-Host "      Splits saved to: $SplitPath" -ForegroundColor Green

# ---- Step 3: Train ----------------------------------------------------------
Write-Host ""
Write-Host "[3/4] Starting training..." -ForegroundColor Yellow
Write-Host "      Config:  $ConfigPath"
Write-Host "      Output:  $OutputDir"
Write-Host ""

python scripts/train_dual_encoder.py `
    --json-path  $JsonPath `
    --split-path $SplitPath `
    --output-dir $OutputDir `
    --config-path $ConfigPath

Write-Host ""
Write-Host "[3/4] Training complete." -ForegroundColor Green

# ---- Step 4: Evaluate on test split -----------------------------------------
Write-Host ""
Write-Host "[4/4] Evaluating best checkpoint on test split..." -ForegroundColor Yellow

$CheckpointPath = "$OutputDir\best_recall_at_1.ckpt"
$SavedConfigPath = "$OutputDir\train_config.json"

if (Test-Path $CheckpointPath) {
    python scripts/eval_dual_encoder.py `
        --json-path       $JsonPath `
        --split-path      $SplitPath `
        --checkpoint-path $CheckpointPath `
        --config-path     $SavedConfigPath `
        --split           test

    Write-Host ""
    Write-Host "[4/4] Evaluation complete." -ForegroundColor Green
    Write-Host "      Qualitative report: $OutputDir\test_qualitative.json" -ForegroundColor Green
} else {
    Write-Host "[4/4] WARNING: Checkpoint not found at $CheckpointPath" -ForegroundColor Red
    Write-Host "      Training may have failed or produced no improvement." -ForegroundColor Red
}

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "   ALL DONE" -ForegroundColor Cyan
Write-Host "   Metrics history: $OutputDir\metrics_history.jsonl" -ForegroundColor Cyan
Write-Host "   Best checkpoint: $OutputDir\best_recall_at_1.ckpt" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
