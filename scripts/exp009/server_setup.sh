#!/bin/bash
# Exp-009 QLoRA SFT 训练服务器一键部署脚本
# 适用于 AutoDL / 恒源云 / 矩池云等 GPU 云平台

set -e

echo "=== Exp-009 SFT Training Server Setup ==="

# ── 0. 数据盘路径（按 GPU 云平台修改） ──
# AutoDL 默认 /root/autodl-tmp，恒源云 /hy-tmp，矩池云 /mnt
DATA_ROOT="${PERSONALIZEDSEARCH_DATA_ROOT:-/root/autodl-tmp}"
echo ">>> DATA_ROOT = $DATA_ROOT"
mkdir -p "$DATA_ROOT/models" "$DATA_ROOT/data/processed"

# ── 1. 克隆项目（如果还没克隆） ──
if [ ! -d "PersonalizedSearch" ]; then
    echo ">>> Cloning project..."
    git clone <your_repo_url> PersonalizedSearch
fi

cd PersonalizedSearch

# ── 2. 创建 Python 3.11 环境 ──
echo ">>> Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# ── 3. 安装依赖 ──
echo ">>> Installing dependencies..."
pip install --upgrade pip

# 基础依赖
pip install -r requirements.txt

# SFT 训练依赖（flash-attn 单独处理）
pip install datasets trl peft accelerate bitsandbytes

echo "=== Core dependencies installed ==="

# ── 3.5 flash-attn（可选，安装失败不影响训练） ──
echo ""
echo ">>> Attempting to install flash-attn (optional, skip if fails)..."
echo "    PyTorch SDPA will be used as fallback."

# 获取当前 CUDA 版本
CUDA_VER=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:4])" 2>/dev/null || echo "")
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')" 2>/dev/null || echo "")

if [ -n "$CUDA_VER" ] && [ -n "$PY_VER" ]; then
    # 尝试用预编译 wheel（快，几秒搞定）
    echo "    Trying pre-built wheel for CUDA ${CUDA_VER}..."
    pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu${CUDA_VER}torch-cp${PY_VER}-cp${PY_VER}-linux_x86_64.whl" 2>/dev/null && echo "    flash-attn installed via wheel!" || {
        # 回退：从源码编译
        echo "    Wheel not found, trying source build (this may take 10-30min)..."
        pip install flash-attn --no-build-isolation 2>/dev/null && echo "    flash-attn installed from source!" || {
            echo "    flash-attn installation failed — training will use PyTorch SDPA instead."
            echo "    This is fine for QLoRA SFT, no action needed."
        }
    }
else
    echo "    Cannot detect CUDA/Python version, skipping flash-attn."
fi

echo ""

# ── 4. 下载模型 ──
echo ">>> Downloading Qwen3-4B to $DATA_ROOT/models/Qwen3-4B..."
export HF_ENDPOINT=https://hf-mirror.com
export HF_XET_HIGH_PERFORMANCE=1
hf download Qwen/Qwen3-4B --local-dir "$DATA_ROOT/models/Qwen3-4B"

echo "=== Model downloaded ==="

# ── 5. 上传训练数据 ──
echo ""
echo "=== IMPORTANT: Upload training data ==="
echo "Run on your local machine:"
echo "  scp data/processed/exp009_sft_train.jsonl <server>:$DATA_ROOT/data/processed/"
echo "  scp data/processed/exp009_sft_val.jsonl   <server>:$DATA_ROOT/data/processed/"
echo ""

# ── 6. 配置环境变量 ──
echo ">>> Creating .env file..."
cat > .env << EOF
PERSONALIZEDSEARCH_DATA_ROOT=$DATA_ROOT
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
EOF

echo ""
echo "=== Server setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Upload training data: scp data/processed/exp009_sft_*.jsonl <server>:$DATA_ROOT/data/processed/"
echo "  2. Activate: source .venv/bin/activate"
echo "  3. Train:    python scripts/exp009/train_sft.py --base-model $DATA_ROOT/models/Qwen3-4B"
