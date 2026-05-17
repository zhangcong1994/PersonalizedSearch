#!/bin/bash
# ================================================================
#  并行构建 4 个嵌入模型的全量向量索引
#  每个模型自动断点续跑（无 --rebuild），互不干扰。
#
#  用法:
#    bash scripts/build_all_indexes.sh
#    nohup bash scripts/build_all_indexes.sh > logs/build_all.log 2>&1 &
#
#  GPU 显存需求: 4模型同时约 9-10GB。如果 < 12GB 显存会 OOM。
#  降级方案: 把最后一行 bge-m3 后面的 & 去掉，串行跑大模型。
# ================================================================
set -u

MODELS=(
    "BAAI/bge-small-zh-v1.5"      # ~0.2 GB VRAM — almost done, finishes quickly
    "moka-ai/m3e-base"           # ~0.4 GB VRAM
    "BAAI/bge-large-zh-v1.5"     # ~2.4 GB VRAM
    "BAAI/bge-m3"                # ~4.3 GB VRAM — heaviest, runs last
)

LOGDIR="logs/build_indexes"
mkdir -p "$LOGDIR"

START_TIME=$(date +%s)
PIDS=()

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
        > "$logfile" 2>&1 &

    PIDS+=($!)

    # Stagger: 15s gap so small model finishes before large ones start encoding
    sleep 15
done

echo ""
echo "[$(date '+%H:%M:%S')] All ${#MODELS[@]} launched. PIDs: ${PIDS[*]}"
echo "  Monitor: tail -f $LOGDIR/*.log"
echo "  GPU:     watch -n 5 nvidia-smi"
echo ""

FAILED=0
for i in "${!MODELS[@]}"; do
    pid="${PIDS[$i]}"
    model="${MODELS[$i]}"
    short=$(echo "$model" | cut -d/ -f2)
    wait "$pid" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%H:%M:%S')] FAILED: $model (exit $rc) → $LOGDIR/${short}.log"
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
