"""
Exp-012: 自对比式 QLoRA DPO 训练 Qwen3-8B。

从自对比式 DPO 数据中微调基座模型，让模型倾向输出更高质量的搜索答案。

数据格式要求（与 construct_dpo_pairs.py 产出一致）：
  {"prompt": "...", "chosen": "...", "rejected": "...", "chosen_score": ..., "rejected_score": ..., "gap": ...}

用法:
  # 默认：best_vs_first_below_gap20
  python scripts/exp012/train_dpo.py

  # 指定数据文件
  python scripts/exp012/train_dpo.py \
      --data data/processed/exp012/exp012_dpo_best_vs_first_below_gap20.jsonl

  # 调整超参
  python scripts/exp012/train_dpo.py --beta 0.5 --lr 1e-5 --epochs 2
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import DATA_ROOT, MODEL_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 默认路径 ────────────────────────────────────────────────

DEFAULT_DATA = (
    DATA_ROOT / "data" / "processed" / "exp012"
    / "exp012_dpo_best_vs_first_below_gap20.jsonl"
)
DEFAULT_OUTPUT = DATA_ROOT / "models" / "exp012-dpo-pilot"

# ── 工具 ────────────────────────────────────────────────────

def load_dpo_dataset(filepath: Path) -> Dataset:
    """加载 DPO JSONL 并转为 HuggingFace Dataset。

    DPOTrainer 可以直接接受含 prompt/chosen/rejected 字符串字段的数据集，
    内部会自动 tokenize。
    """
    records = []
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            prompt = entry.get("prompt", "")
            chosen = entry.get("chosen", "")
            rejected = entry.get("rejected", "")

            if not prompt or not chosen or not rejected:
                skipped += 1
                continue

            records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "query_id": entry.get("query_id", ""),
                "chosen_score": entry.get("chosen_score", 0),
                "rejected_score": entry.get("rejected_score", 0),
                "gap": entry.get("gap", 0),
            })

    ds = Dataset.from_list(records)

    # 统计
    if len(records) > 0:
        gaps = [r["gap"] for r in records]
        chosen_lens = [len(r["chosen"]) for r in records]
        rejected_lens = [len(r["rejected"]) for r in records]
        prompt_lens = [len(r["prompt"]) for r in records]

        logger.info(f"  Loaded {len(records)} pairs (skipped {skipped})")
        logger.info(f"  Gap:       mean={sum(gaps)/len(gaps):.1f}  min={min(gaps):.1f}  max={max(gaps):.1f}")
        logger.info(f"  Chosen len:   mean={sum(chosen_lens)/len(chosen_lens):.0f}  min={min(chosen_lens)}  max={max(chosen_lens)}")
        logger.info(f"  Rejected len: mean={sum(rejected_lens)/len(rejected_lens):.0f}  min={min(rejected_lens)}  max={max(rejected_lens)}")
        logger.info(f"  Prompt len:   mean={sum(prompt_lens)/len(prompt_lens):.0f}  min={min(prompt_lens)}  max={max(prompt_lens)}")

    return ds


def resolve_model_path(model_id: str, cache_dir: Path) -> str:
    """解析模型路径：优先找本地已下载的，否则走 HF 下载。

    HuggingFace 缓存结构：
      - `snapshots` 模式: cache_dir/models--Qwen--Qwen3-8B/snapshots/<hash>/
      - 手动 save_pretrained: cache_dir/Qwen3-8B/  （含 config.json）

    检测顺序：
      1. model_id 本身就是本地路径且存在 → 直接用
      2. cache_dir/models--<org>--<model>/  HF 缓存 → 直接用
      3. cache_dir/<model_name>/  手动保存的 → 直接用
      4. 都不存在 → 返回原始 HF ID，让 transformers 去下载
    """
    model_path = Path(model_id)
    if model_path.exists():
        logger.info(f"直接使用本地模型路径: {model_path}")
        return str(model_path)

    # HuggingFace 缓存目录名
    hf_cache_name = f"models--{model_id.replace('/', '--')}"
    hf_cache_path = cache_dir / hf_cache_name
    if hf_cache_path.exists():
        logger.info(f"找到 HF 缓存: {hf_cache_path}")
        return model_id  # 返回 HF ID，配合 cache_dir 参数自动命中

    # 手动 save_pretrained 的可能路径
    local_name = model_id.split("/")[-1]
    local_path = cache_dir / local_name
    if local_path.exists() and (local_path / "config.json").exists():
        logger.info(f"找到本地模型目录: {local_path}")
        return str(local_path)

    logger.info(f"本地未找到 {model_id}，将从 HuggingFace 下载到 {cache_dir}")
    return model_id


# ── 主逻辑 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-012: QLoRA DPO Training")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="DPO 训练数据 JSONL")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B",
                        help="基座模型 HuggingFace ID")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="输出目录（LoRA adapter）")
    parser.add_argument("--epochs", type=int, default=1,
                        help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="每 GPU 的 batch size")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="DPO beta（KL 散度约束系数，越大越远离 ref model）")
    parser.add_argument("--max-length", type=int, default=7168,
                        help="最大序列长度（chosen/rejected 总 token 数）")
    parser.add_argument("--max-prompt-length", type=int, default=6144,
                        help="prompt 最大 token 数")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="每 N 步打印日志")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="每 N 步保存 checkpoint")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup 比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    # ── 路径验证 ──
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        return 1

    if os.path.exists(args.output_dir):
        logger.info(f"Removing previous output: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # ── 日志 ──
    logger.info("=" * 60)
    logger.info("  Exp-012 QLoRA DPO Training")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {args.base_model}")
    logger.info(f"  Data:          {args.data}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Batch size:    {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"  LR:            {args.lr}")
    logger.info(f"  DPO beta:      {args.beta}")
    logger.info(f"  Max length:    {args.max_length}")
    logger.info(f"  Max prompt:    {args.max_prompt_length}")
    logger.info(f"  LoRA:          r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    logger.info(f"  Seed:          {args.seed}")
    logger.info("=" * 60)

    # ── GPU 检查 ──
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info("=" * 60)

    # ── 解析模型路径 ──
    base_model_path = resolve_model_path(args.base_model, MODEL_CACHE_DIR)
    logger.info(f"  Resolved model path: {base_model_path}")
    logger.info("=" * 60)

    # ── 加载数据 ──
    logger.info("Loading DPO dataset...")
    train_dataset = load_dpo_dataset(Path(args.data))

    import math
    total_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_dataset) / total_batch)
    total_steps = steps_per_epoch * args.epochs
    logger.info(f"  Steps per epoch: ~{steps_per_epoch}, total: ~{total_steps}")
    logger.info("=" * 60)

    # ── 加载 tokenizer ──
    logger.info("Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, cache_dir=str(MODEL_CACHE_DIR))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Qwen3 的 tokenizer 默认 left-padding 可能导致 DPO trainer 问题，用 right
    tokenizer.padding_side = "right"

    # ── Flash Attention 检测 ──
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("flash-attn detected, using FlashAttention2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash-attn not found, falling back to PyTorch SDPA")

    # ── 加载模型 ──
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model (4-bit) for training...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 参考模型（冻结，用于 KL 散度计算）──
    # DPO 需要 ref model 提供基线概率分布。用相同的 4-bit base model，
    # 不加 LoRA，冻结，以节省显存。
    # DPOTrainer 内部会自动把 ref model 设为 eval mode。
    logger.info("Loading reference model (4-bit, frozen)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        cache_dir=str(MODEL_CACHE_DIR),
    )
    # ref model 不加 LoRA，直接冻结
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # ── Training Arguments ──
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # ── DPO Trainer ──
    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        loss_type="sigmoid",
    )

    # ── 训练 ──
    logger.info("Starting DPO training...")
    logger.info("=" * 60)
    dpo_trainer.train()

    # ── 保存 ──
    logger.info("Saving adapter...")
    dpo_trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # ── 打印最终指标 ──
    try:
        final_logs = dpo_trainer.state.log_history[-2:]  # 最后两个 log entry（含 eval）
        logger.info("Final training metrics:")
        for entry in final_logs:
            relevant = {k: v for k, v in entry.items() if "loss" in k.lower() or "reward" in k.lower() or "margin" in k.lower()}
            if relevant:
                logger.info(f"  {relevant}")
    except Exception:
        pass

    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  Adapter saved to: {args.output_dir}")
    logger.info("")
    logger.info("  评估方法:")
    logger.info(f"    用 vLLM 加载 LoRA adapter:")
    logger.info(f"    vllm serve Qwen/Qwen3-8B \\")
    logger.info(f"      --enable-lora \\")
    logger.info(f"      --lora-modules dpo-pilot={args.output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
