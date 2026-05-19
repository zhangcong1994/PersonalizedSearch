#!/bin/bash
# 串行运行 exp-002 全部分组（m3e-base Dense 模式）
# 用法: bash scripts/run_all_exp002_m3e.sh
set -e

SAMPLE=2000
TOP_K=20
EMBEDDING="moka-ai/m3e-base"
VECTOR_DB="/root/autodl-tmp/data/vector_db/t2ranking/m3e-base"

# E2d-P0 与 E2a-B0 重复（均为 none strategy），跳过
EXPERIMENTS=(
    # E2a: Prompt 迭代 (7 组)
    E2a-B0
    E2a-B1
    E2a-P1
    E2a-P2
    E2a-P3
    E2a-P4
    E2a-P5
    # E2b: Multi-Query 融合 (3 组)
    E2b-M1
    E2b-M2
    E2b-M3
    # E2c: HyDE 预回答扩展 (3 组)
    E2c-H1
    E2c-H2
    E2c-H3
    # E2d: PRF (2 组，P0 已跳过)
    E2d-P1
    E2d-P2
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

for EXP in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=============================================="
    echo "  [$CURRENT/$TOTAL] Running: $EXP"
    echo "=============================================="
    python scripts/evaluate_exp002.py \
        --experiment "$EXP" \
        --sample "$SAMPLE" \
        --top-k "$TOP_K" \
        --device cuda \
        --embedding-model "$EMBEDDING" \
        --vector-db "$VECTOR_DB"
    echo ""
    echo "  Completed: $EXP"
done

echo ""
echo "=============================================="
echo "  All $TOTAL experiments completed!"
echo "=============================================="
