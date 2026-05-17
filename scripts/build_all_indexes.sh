#!/bin/bash
# ================================================================
#  串行构建 4 个嵌入模型的全量向量索引（GPU 显存不够用）
#  每个模型自动断点续跑（无 --rebuild），互不干扰。
#
#  用法:
#    bash scripts/build_all_indexes.sh
#    nohup bash scripts/build_all_indexes.sh > logs/build_all.log 2>&1 &
# ================================================================
set -u

MODELS=(
    "BAAI/bge-small-zh-v1.5"
    "moka-ai/m3e-base"
    "BAAI/bge-large-zh-v1.5"
    "BAAI/bge-m3"
)

LOGDIR="logs/build_indexes"
mkdir -p "$LOGDIR"

START_TIME=$(date +%s)
FAILED=0

echo "[$(date '+%H:%M:%S')] GPU memory:"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    short=$(echo "$model" | cut -d/ -f2)
    logfile="$LOGDIR/${short}.log"

    echo "[$(date '+%H:%M:%S')] [$((i+1))/${#MODELS[@]}] Launching: $model  →  $logfile"

    python scripts/build_t2ranking_index.py \
        --model "$model" \
        --device cuda \
        --prefetch \
        > "$logfile" 2>&1

    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] FAILED: $model (exit $rc) → $logfile"
        FAILED=$((FAILED + 1))
    else
        echo "[$(date '+%H:%M:%S')] DONE:   $model"
    fi
done

ELAPSED=$(($(date +%s) - START_TIME))
echo ""
echo "================================================"
echo "  $FAILED failed / ${#MODELS[@]} total  |  $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "================================================"
