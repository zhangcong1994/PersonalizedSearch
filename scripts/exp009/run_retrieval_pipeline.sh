#!/usr/bin/env bash
# =============================================================================
# Exp-009 检索管道 Bash 脚本 (Linux / AutoDL)
# 用法: bash scripts/exp009/run_retrieval_pipeline.sh
#       bash scripts/exp009/run_retrieval_pipeline.sh --skip-augment
#       bash scripts/exp009/run_retrieval_pipeline.sh --device cpu
# =============================================================================
set -euo pipefail

# ── 默认参数 ──
DEVICE="${DEVICE:-cuda}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-models/m3e-base-t2ranking-phase3-2/ep1/merged}"
RERANKER_MODEL="${RERANKER_MODEL:-BAAI/bge-reranker-v2-m3}"
SKIP_AUGMENT=false
SKIP_INDEX=false
REBUILD_INDEX=false
BACKEND="${BACKEND:-vllm}"
VLLM_URL="${VLLM_URL:-http://localhost:8000/v1}"
DATASET_PREFIX="exp009"
PER_ROUTE_K=50
RRF_K=60
OUTPUT_TOP_K=10
SAMPLED_QUERIES="data/processed/exp009_sampled_queries.jsonl"

# ── 解析命令行参数 ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --skip-augment) SKIP_AUGMENT=true; shift ;;
        --skip-index) SKIP_INDEX=true; shift ;;
        --rebuild-index) REBUILD_INDEX=true; shift ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --vllm-url) VLLM_URL="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="python"
DATA_DIR="data/processed"
RW_FILE="$DATA_DIR/${DATASET_PREFIX}_rewritten_queries.jsonl"
HY_FILE="$DATA_DIR/${DATASET_PREFIX}_hyde_answers.jsonl"
DENSE_B0="$DATA_DIR/${DATASET_PREFIX}_dense_B0.jsonl"
DENSE_P2="$DATA_DIR/${DATASET_PREFIX}_dense_P2.jsonl"
DENSE_H2="$DATA_DIR/${DATASET_PREFIX}_dense_H2.jsonl"
BM25_FILE="$DATA_DIR/${DATASET_PREFIX}_bm25_B0.jsonl"
RRF_FILE="$DATA_DIR/${DATASET_PREFIX}_rrf_fused.jsonl"
RERANK_FILE="$DATA_DIR/${DATASET_PREFIX}_reranked_top10.jsonl"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Exp-009 Retrieval Pipeline (Linux)"
echo "============================================================"
echo "  Device:        $DEVICE"
echo "  Backend:       $BACKEND"
echo "  Embedding:     $EMBEDDING_MODEL"
echo "  Reranker:      $RERANKER_MODEL"
echo "  Sampled:       $SAMPLED_QUERIES"
echo "  Skip augment:  $SKIP_AUGMENT"
echo "  Skip index:    $SKIP_INDEX"
echo "------------------------------------------------------------"

# ============================================================================
# Step 1: Query Augmentation (改写 + HyDE)
# ============================================================================
if [ "$SKIP_AUGMENT" = false ]; then
    if [ -f "$RW_FILE" ] && [ -f "$HY_FILE" ]; then
        RW_COUNT=$(wc -l < "$RW_FILE")
        HY_COUNT=$(wc -l < "$HY_FILE")
        echo "[Step 1] SKIP: $RW_COUNT rewritten + $HY_COUNT hyde already cached"
    else
        echo "[Step 1] Query augmentation (rewrite + HyDE) backend=$BACKEND..."
        AUG_ARGS=(scripts/exp009/run_query_augment.py
            --backend "$BACKEND"
            --input "$SAMPLED_QUERIES"
            --output-rw "$RW_FILE"
            --output-hy "$HY_FILE")
        if [ "$BACKEND" = "vllm" ]; then
            AUG_ARGS+=(--llm-url "$VLLM_URL")
        fi
        $PYTHON "${AUG_ARGS[@]}"
        echo "[Step 1] DONE"
    fi
fi

# ============================================================================
# Step 2: Build FAISS Index
# ============================================================================
if [ "$SKIP_INDEX" = false ]; then
    MODEL_SHORT=$(basename "$EMBEDDING_MODEL")
    INDEX_DIR="data/vector_db/t2ranking/$MODEL_SHORT"
    if [ -f "$INDEX_DIR/index.faiss" ] && [ "$REBUILD_INDEX" = false ]; then
        echo "[Step 2] SKIP: FAISS index exists at $INDEX_DIR"
    else
        echo "[Step 2] Building FAISS index ($EMBEDDING_MODEL)..."
        IDX_ARGS=(scripts/exp007/build_faiss_index.py --model "$EMBEDDING_MODEL" --device "$DEVICE" --offline)
        if [ "$REBUILD_INDEX" = true ]; then
            IDX_ARGS+=(--rebuild)
        fi
        $PYTHON "${IDX_ARGS[@]}"
        echo "[Step 2] DONE"
    fi
fi

# ============================================================================
# Step 3: Dense Retrieval (3 routes: original + rewrite + HyDE)
# ============================================================================
DENSE_MISSING=""
[ -f "$DENSE_B0" ] || DENSE_MISSING="$DENSE_MISSING B0"
[ -f "$DENSE_P2" ] || DENSE_MISSING="$DENSE_MISSING P2"
[ -f "$DENSE_H2" ] || DENSE_MISSING="$DENSE_MISSING H2"

if [ -z "$DENSE_MISSING" ]; then
    echo "[Step 3] SKIP: all 3 dense routes cached"
else
    echo "[Step 3] Dense retrieval (routes:$DENSE_MISSING)..."
    $PYTHON scripts/exp009/dense_retrieve.py \
        --model "$EMBEDDING_MODEL" \
        --device "$DEVICE" \
        --input-queries "$SAMPLED_QUERIES" \
        --input-rewritten "$RW_FILE" \
        --input-hyde "$HY_FILE" \
        --output-b0 "$DENSE_B0" \
        --output-p2 "$DENSE_P2" \
        --output-h2 "$DENSE_H2" \
        --top-k "$PER_ROUTE_K"
    echo "[Step 3] DONE"
fi

# ============================================================================
# Step 4: BM25 Retrieval (1 route)
# ============================================================================
if [ -f "$BM25_FILE" ]; then
    BM_COUNT=$(wc -l < "$BM25_FILE")
    echo "[Step 4] SKIP: $BM_COUNT BM25 results cached"
else
    echo "[Step 4] BM25 retrieval..."
    $PYTHON scripts/exp009/bm25_retrieve.py \
        --input-queries "$SAMPLED_QUERIES" \
        --output "$BM25_FILE" \
        --top-k "$PER_ROUTE_K"
    echo "[Step 4] DONE"
fi

# ============================================================================
# Step 5: RRF Fusion
# ============================================================================
if [ -f "$RRF_FILE" ]; then
    RRF_COUNT=$(wc -l < "$RRF_FILE")
    echo "[Step 5] SKIP: $RRF_COUNT RRF results cached"
else
    echo "[Step 5] RRF fusion (4 routes, per-route K=$PER_ROUTE_K, RRF k=$RRF_K)..."
    $PYTHON scripts/exp009/rrf_fuse.py \
        --route-files "$DENSE_B0" "$DENSE_P2" "$DENSE_H2" "$BM25_FILE" \
        --per-route-k "$PER_ROUTE_K" \
        --rrf-k "$RRF_K" \
        --output-top-k "$PER_ROUTE_K" \
        --output "$RRF_FILE"
    echo "[Step 5] DONE"
fi

# ============================================================================
# Step 6: Reranker
# ============================================================================
if [ -f "$RERANK_FILE" ]; then
    RE_COUNT=$(wc -l < "$RERANK_FILE")
    echo "[Step 6] SKIP: $RE_COUNT reranked results cached"
else
    echo "[Step 6] Reranker ($RERANKER_MODEL)..."
    $PYTHON scripts/exp009/rerank.py \
        --input "$RRF_FILE" \
        --model "$RERANKER_MODEL" \
        --device "$DEVICE" \
        --top-k "$OUTPUT_TOP_K" \
        --output "$RERANK_FILE"
    echo "[Step 6] DONE"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "  Pipeline Complete!"
echo "============================================================"
FINAL_COUNT=$(wc -l < "$RERANK_FILE" 2>/dev/null || echo 0)
echo "  Output: $RERANK_FILE ($FINAL_COUNT queries, top-$OUTPUT_TOP_K each)"
echo ""
echo "  Next: python scripts/exp009/generate_teacher_answers.py"
echo "============================================================"
