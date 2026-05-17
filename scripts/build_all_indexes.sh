#!/bin/bash
# ================================================================
#  一次性构建 4 个嵌入模型的全量向量索引
#  用法:
#    bash scripts/build_all_indexes.sh           # 前台运行
#    nohup bash scripts/build_all_indexes.sh > build_all.log 2>&1 &  # 后台运行
# ================================================================
set -euo pipefail

MODELS=(
    "BAAI/bge-small-zh-v1.5"
    "BAAI/bge-large-zh-v1.5"
    "moka-ai/m3e-base"
    "BAAI/bge-m3"
)

TOTAL=${#MODELS[@]}
START_TIME=$(date +%s)

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    idx=$((i + 1))

    echo ""
    echo "========================================================"
    echo "  [$idx/$TOTAL] Building: $model"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================================"

    python scripts/build_t2ranking_index.py \
        --model "$model" \
        --device cuda \
        --prefetch \
        --rebuild

    echo ""
    echo "  [$idx/$TOTAL] Done: $model  ($(date '+%Y-%m-%d %H:%M:%S'))"
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo ""
echo "========================================================"
echo "  All $TOTAL indexes built in $((ELAPSED / 60)) min $((ELAPSED % 60)) sec"
echo "========================================================"

# Print summary
echo ""
echo "  Index directories:"
for model in "${MODELS[@]}"; do
    short=$(echo "$model" | cut -d/ -f2)
    echo "    data/vector_db/t2ranking/$short/"
done
