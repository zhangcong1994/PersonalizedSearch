# =============================================================================
# Exp-009 检索管道 PowerShell 脚本 (Windows)
# 用法: .\scripts\exp009\run_retrieval_pipeline.ps1
#       .\scripts\exp009\run_retrieval_pipeline.ps1 -SkipAugment   # 跳过查询增强
#       .\scripts\exp009\run_retrieval_pipeline.ps1 -Device cpu    # CPU 模式
# =============================================================================

param(
    [string]$Device = "cuda",
    [string]$EmbeddingModel = "models/m3e-base-t2ranking-phase3-2/ep1/merged",
    [string]$RerankerModel = "BAAI/bge-reranker-v2-m3",
    [switch]$SkipAugment,
    [switch]$SkipIndex,
    [switch]$RebuildIndex,
    [string]$Backend = "vllm",
    [string]$VllmUrl = "http://localhost:8000/v1",
    [string]$DatasetPrefix = "exp009",
    [int]$PerRouteK = 50,
    [int]$RrfK = 60,
    [int]$OutputTopK = 10,
    [string]$SampledQueries = "data/processed/exp009_sampled_queries.jsonl"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path "$ScriptDir\..\.."

# ── 路径 ──
$Python = ".\.venv\Scripts\python.exe"
$DataDir = "data\processed"
$RwFile = "$DataDir\${DatasetPrefix}_rewritten_queries.jsonl"
$HyFile = "$DataDir\${DatasetPrefix}_hyde_answers.jsonl"
$DenseB0 = "$DataDir\${DatasetPrefix}_dense_B0.jsonl"
$DenseP2 = "$DataDir\${DatasetPrefix}_dense_P2.jsonl"
$DenseH2 = "$DataDir\${DatasetPrefix}_dense_H2.jsonl"
$Bm25File = "$DataDir\${DatasetPrefix}_bm25_B0.jsonl"
$RrfFile = "$DataDir\${DatasetPrefix}_rrf_fused.jsonl"
$RerankFile = "$DataDir\${DatasetPrefix}_reranked_top10.jsonl"

Write-Host "=" * 60
Write-Host "  Exp-009 Retrieval Pipeline (Windows)"
Write-Host "=" * 60
Write-Host "  Device:        $Device"
Write-Host "  Backend:       $Backend"
Write-Host "  Embedding:     $EmbeddingModel"
Write-Host "  Reranker:      $RerankerModel"
Write-Host "  Sampled:       $SampledQueries"
Write-Host "  Skip augment:  $SkipAugment"
Write-Host "  Skip index:    $SkipIndex"
Write-Host "-" * 60

Push-Location $ProjectRoot

try {

    # =========================================================================
    # Step 1: Query Augmentation (改写 + HyDE)
    # =========================================================================
    if (-not $SkipAugment) {
        if (Test-Path $RwFile -and (Test-Path $HyFile)) {
            $rwCount = (Get-Content $RwFile | Measure-Object -Line).Lines
            $hyCount = (Get-Content $HyFile | Measure-Object -Line).Lines
            Write-Host "[Step 1] SKIP: $rwCount rewritten + $hyCount hyde already cached" -ForegroundColor Yellow
        } else {
            Write-Host "[Step 1] Query augmentation (rewrite + HyDE) backend=$Backend..." -ForegroundColor Cyan
            $augArgs = @(
                "scripts/exp009/run_query_augment.py",
                "--backend", $Backend,
                "--input", $SampledQueries,
                "--output-rw", $RwFile,
                "--output-hy", $HyFile
            )
            if ($Backend -eq "vllm") {
                $augArgs += "--llm-url"
                $augArgs += $VllmUrl
            }
            & $Python $augArgs
            if ($LASTEXITCODE -ne 0) { throw "Step 1 failed" }
            Write-Host "[Step 1] DONE" -ForegroundColor Green
        }
    }

    # =========================================================================
    # Step 2: Build FAISS Index
    # =========================================================================
    if (-not $SkipIndex) {
        $ModelShort = ($EmbeddingModel -split "[/\\]")[-1]
        $IndexDir = "data\vector_db\t2ranking\$ModelShort"
        if ((Test-Path "$IndexDir\index.faiss") -and (-not $RebuildIndex)) {
            Write-Host "[Step 2] SKIP: FAISS index exists at $IndexDir" -ForegroundColor Yellow
        } else {
            Write-Host "[Step 2] Building FAISS index ($EmbeddingModel)..." -ForegroundColor Cyan
            $idxArgs = @(
                "scripts/exp007/build_faiss_index.py",
                "--model", $EmbeddingModel,
                "--device", $Device,
                "--offline"
            )
            if ($RebuildIndex) { $idxArgs += "--rebuild" }
            & $Python $idxArgs
            if ($LASTEXITCODE -ne 0) { throw "Step 2 failed" }
            Write-Host "[Step 2] DONE" -ForegroundColor Green
        }
    }

    # =========================================================================
    # Step 3: Dense Retrieval (3 routes: original + rewrite + HyDE)
    # =========================================================================
    $denseMissing = @()
    if (-not (Test-Path $DenseB0)) { $denseMissing += "B0" }
    if (-not (Test-Path $DenseP2)) { $denseMissing += "P2" }
    if (-not (Test-Path $DenseH2)) { $denseMissing += "H2" }

    if ($denseMissing.Count -eq 0) {
        Write-Host "[Step 3] SKIP: all 3 dense routes cached" -ForegroundColor Yellow
    } else {
        Write-Host "[Step 3] Dense retrieval (routes: $($denseMissing -join ', '))..." -ForegroundColor Cyan
        & $Python scripts/exp009/dense_retrieve.py `
            --model $EmbeddingModel `
            --device $Device `
            --input-queries $SampledQueries `
            --input-rewritten $RwFile `
            --input-hyde $HyFile `
            --output-b0 $DenseB0 `
            --output-p2 $DenseP2 `
            --output-h2 $DenseH2 `
            --top-k $PerRouteK
        if ($LASTEXITCODE -ne 0) { throw "Step 3 failed" }
        Write-Host "[Step 3] DONE" -ForegroundColor Green
    }

    # =========================================================================
    # Step 4: BM25 Retrieval (1 route)
    # =========================================================================
    if (Test-Path $Bm25File) {
        $bmCount = (Get-Content $Bm25File | Measure-Object -Line).Lines
        Write-Host "[Step 4] SKIP: $bmCount BM25 results cached" -ForegroundColor Yellow
    } else {
        Write-Host "[Step 4] BM25 retrieval..." -ForegroundColor Cyan
        & $Python scripts/exp009/bm25_retrieve.py `
            --input-queries $SampledQueries `
            --output $Bm25File `
            --top-k $PerRouteK
        if ($LASTEXITCODE -ne 0) { throw "Step 4 failed" }
        Write-Host "[Step 4] DONE" -ForegroundColor Green
    }

    # =========================================================================
    # Step 5: RRF Fusion
    # =========================================================================
    if (Test-Path $RrfFile) {
        $rrfCount = (Get-Content $RrfFile | Measure-Object -Line).Lines
        Write-Host "[Step 5] SKIP: $rrfCount RRF results cached" -ForegroundColor Yellow
    } else {
        Write-Host "[Step 5] RRF fusion (4 routes, per-route K=$PerRouteK, RRF k=$RrfK)..." -ForegroundColor Cyan
        & $Python scripts/exp009/rrf_fuse.py `
            --route-files $DenseB0 $DenseP2 $DenseH2 $Bm25File `
            --per-route-k $PerRouteK `
            --rrf-k $RrfK `
            --output-top-k $PerRouteK `
            --output $RrfFile
        if ($LASTEXITCODE -ne 0) { throw "Step 5 failed" }
        Write-Host "[Step 5] DONE" -ForegroundColor Green
    }

    # =========================================================================
    # Step 6: Reranker
    # =========================================================================
    if (Test-Path $RerankFile) {
        $reCount = (Get-Content $RerankFile | Measure-Object -Line).Lines
        Write-Host "[Step 6] SKIP: $reCount reranked results cached" -ForegroundColor Yellow
    } else {
        Write-Host "[Step 6] Reranker ($RerankerModel)..." -ForegroundColor Cyan
        & $Python scripts/exp009/rerank.py `
            --input $RrfFile `
            --model $RerankerModel `
            --device $Device `
            --top-k $OutputTopK `
            --output $RerankFile
        if ($LASTEXITCODE -ne 0) { throw "Step 6 failed" }
        Write-Host "[Step 6] DONE" -ForegroundColor Green
    }

    # =========================================================================
    # Summary
    # =========================================================================
    Write-Host ""
    Write-Host "=" * 60
    Write-Host "  Pipeline Complete!"
    Write-Host "=" * 60
    $finalCount = if (Test-Path $RerankFile) { (Get-Content $RerankFile | Measure-Object -Line).Lines } else { 0 }
    Write-Host "  Output: $RerankFile ($finalCount queries, top-$OutputTopK each)"
    Write-Host ""
    Write-Host "  Next: python scripts/exp009/generate_teacher_answers.py"
    Write-Host "=" * 60

} finally {
    Pop-Location
}
