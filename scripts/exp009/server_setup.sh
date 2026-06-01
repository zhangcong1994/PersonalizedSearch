#!/bin/bash
# Exp-009 QLoRA SFT 训练服务器一键部署脚本
# 适用于 AutoDL / 恒源云 / 矩池云等 GPU 云平台

set -e

echo "=== Exp-009 SFT Training Server Setup ==="

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

# SFT 训练依赖
pip install \
    datasets \
    trl \
    peft \
    accelerate \
    bitsandbytes \
    flash-attn \
    --no-build-isolation

echo "=== All dependencies installed ==="

# ── 4. 下载模型 ──
echo ">>> Downloading Qwen3-4B..."
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B

echo "=== Model downloaded ==="

# ── 5. 上传训练数据 ──
echo ""
echo "=== IMPORTANT: Upload training data ==="
echo "Run on your local machine:"
echo "  scp data/processed/exp009_sft_train.jsonl <server>:/path/to/PersonalizedSearch/data/processed/"
echo "  scp data/processed/exp009_sft_val.jsonl   <server>:/path/to/PersonalizedSearch/data/processed/"
echo ""

# ── 6. 配置环境变量 ──
echo ">>> Creating .env file..."
cat > .env << 'EOF'
PERSONALIZEDSEARCH_DATA_ROOT=./data
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
EOF

echo ""
echo "=== Server setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Upload training data: scp data/processed/exp009_sft_*.jsonl ..."
echo "  2. Activate: source .venv/bin/activate"
echo "  3. Train:    python scripts/exp009/train_sft.py --base-model models/Qwen3-4B"
