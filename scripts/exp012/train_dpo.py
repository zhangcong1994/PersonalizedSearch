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
import math
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import TrainerCallback

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

def load_dpo_dataset(
    filepath: Path, min_chosen_score: float = 0.0
) -> tuple[list[dict], dict]:
    """加载 DPO JSONL 并返回 records 列表和统计信息。"""
    records = []
    skipped = 0
    filtered = 0

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

            chosen_score = entry.get("chosen_score", 0)
            if min_chosen_score > 0 and chosen_score < min_chosen_score:
                filtered += 1
                continue

            records.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "query_id": entry.get("query_id", ""),
                "chosen_score": chosen_score,
                "rejected_score": entry.get("rejected_score", 0),
                "gap": entry.get("gap", 0),
            })

    stats = {"total": len(records), "skipped": skipped, "filtered": filtered}
    if filtered > 0:
        logger.info(f"  Filtered out {filtered} pairs (chosen_score < {min_chosen_score})")
    if skipped > 0:
        logger.info(f"  Skipped {skipped} empty entries")
    if len(records) > 0:
        gaps = [r["gap"] for r in records]
        chosen_lens = [len(r["chosen"]) for r in records]
        rejected_lens = [len(r["rejected"]) for r in records]
        prompt_lens = [len(r["prompt"]) for r in records]
        stats.update({
            "gap_mean": sum(gaps) / len(gaps),
            "gap_min": min(gaps),
            "gap_max": max(gaps),
            "chosen_len_mean": sum(chosen_lens) / len(chosen_lens),
            "rejected_len_mean": sum(rejected_lens) / len(rejected_lens),
            "prompt_len_mean": sum(prompt_lens) / len(prompt_lens),
            "len_ratio": (sum(chosen_lens) / len(chosen_lens)) / max(1, sum(rejected_lens) / len(rejected_lens)),
        })
        logger.info(f"  Loaded {len(records)} pairs (skipped {skipped})")
        logger.info(f"  Gap:       mean={stats['gap_mean']:.1f}  min={stats['gap_min']:.1f}  max={stats['gap_max']:.1f}")
        logger.info(f"  Chosen len:   mean={stats['chosen_len_mean']:.0f}  "
                     f"min={min(chosen_lens)}  max={max(chosen_lens)}")
        logger.info(f"  Rejected len: mean={stats['rejected_len_mean']:.0f}  "
                     f"min={min(rejected_lens)}  max={max(rejected_lens)}")
        logger.info(f"  Prompt len:   mean={stats['prompt_len_mean']:.0f}  "
                     f"min={min(prompt_lens)}  max={max(prompt_lens)}")
        logger.info(f"  Length ratio (chosen/rejected): {stats['len_ratio']:.2f}")
        if stats['len_ratio'] > 1.5:
            logger.warning(
                f"  ⚠️  Chosen responses are {stats['len_ratio']:.1f}x longer than rejected! "
                f"DPO may learn to exploit length. Consider length normalization."
            )

    return records, stats


def build_dataset_from_records(records: list[dict]) -> Dataset:
    return Dataset.from_list(records)


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


class DPOMonitorCallback(TrainerCallback):
    """DPO 训练专属监控回调。

    追踪 DPO 关键指标并在异常时发出 WARNING：
      - rewards/chosen: 对 chosen 回答的隐式 reward，应小幅上升或稳定
      - rewards/rejected: 对 rejected 回答的隐式 reward，应下降
      - rewards/margins: chosen - rejected 的差距，持续暴涨 >10 是过拟合信号
      - rewards/accuracies: chosen > rejected 的比例，接近 1.0 是过拟合信号
      - kl: 与 ref model 的 KL 散度，应缓慢增长，暴涨是 diverging 信号
      - logps/chosen: 若持续下降，说明模型在"遗忘"好的行为
    """

    WARNING_THRESHOLDS = {
        "rewards/margins": {"high": 10.0, "desc": "reward margin > 10 → 过拟合风险"},
        "rewards/accuracies": {"high": 0.98, "desc": "accuracy > 0.98 → 可能死记"},
    }

    def __init__(self, log_file: Optional[Path] = None):
        self.metrics_history: list[dict] = []
        self.log_file = log_file
        self._margin_rising_count = 0
        self._chosen_logp_dropping_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step_metrics = {"step": state.global_step, "epoch": round(state.epoch, 2) if state.epoch else 0}
        dpo_keys = [
            "loss",
            "rewards/chosen",
            "rewards/rejected",
            "rewards/margins",
            "rewards/accuracies",
            "kl",
            "logps/chosen",
            "logps/rejected",
        ]
        for key in dpo_keys:
            if key in logs:
                val = float(logs[key])
                step_metrics[key] = val

        if len(step_metrics) > 2:
            self.metrics_history.append(step_metrics)

        # ── 实时告警 ──
        margin = step_metrics.get("rewards/margins")
        accuracy = step_metrics.get("rewards/accuracies")
        kl = step_metrics.get("kl")
        chosen_logp = step_metrics.get("logps/chosen")

        if margin is not None and margin > self.WARNING_THRESHOLDS["rewards/margins"]["high"]:
            self._margin_rising_count += 1
            if self._margin_rising_count >= 2:
                logger.warning(
                    f"  ⚠️  Step {state.global_step}: DPO reward margin={margin:.2f} "
                    f"持续偏高 — 模型可能过拟合到偏好信号！考虑增大 beta 或减少训练"
                )
        else:
            self._margin_rising_count = 0

        if accuracy is not None and accuracy > self.WARNING_THRESHOLDS["rewards/accuracies"]["high"]:
            logger.warning(
                f"  ⚠️  Step {state.global_step}: DPO accuracy={accuracy:.3f} — "
                f"几乎所有 pair 的 chosen > rejected，可能死记硬背"
            )

        if kl is not None and kl > 5.0:
            logger.warning(
                f"  ⚠️  Step {state.global_step}: KL={kl:.2f} 偏高 — "
                f"模型正在偏离 ref model，建议增大 beta"
            )

        # 追踪 chosen logp 趋势（需要至少 3 个数据点）
        if chosen_logp is not None and len(self.metrics_history) >= 4:
            recent = [m.get("logps/chosen") for m in self.metrics_history[-4:]
                      if m.get("logps/chosen") is not None]
            if len(recent) >= 3 and all(
                recent[i] > recent[i + 1] for i in range(len(recent) - 1)
            ):
                self._chosen_logp_dropping_count += 1
                if self._chosen_logp_dropping_count >= 2:
                    logger.warning(
                        f"  ⚠️  Step {state.global_step}: chosen logp 持续下降 "
                        f"({recent[0]:.1f} → {recent[-1]:.1f}) — 模型可能遗忘好的行为"
                    )
            else:
                self._chosen_logp_dropping_count = 0

    def on_train_end(self, args, state, control, **kwargs):
        if self.log_file and self.metrics_history:
            os.makedirs(self.log_file.parent, exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
            logger.info(f"DPO metrics history saved to {self.log_file}")
            self._print_summary()

    def _print_summary(self):
        if not self.metrics_history:
            return
        logger.info("\n" + "=" * 60)
        logger.info("  DPO 训练指标终态")
        logger.info("=" * 60)
        first = self.metrics_history[0]
        last = self.metrics_history[-1]
        for key in ["rewards/margins", "rewards/accuracies", "kl",
                     "logps/chosen", "logps/rejected", "loss"]:
            if key in first and key in last:
                delta = last[key] - first[key]
                direction = "↑" if delta > 0 else "↓"
                logger.info(f"  {key:25s}: {first[key]:8.3f} → {last[key]:8.3f}  ({direction}{abs(delta):.3f})")

        margins = [m["rewards/margins"] for m in self.metrics_history if "rewards/margins" in m]
        if len(margins) >= 2:
            trend = margins[-1] - margins[0]
            if trend > 5:
                logger.warning(f"  ⚠️  Margin trend: +{trend:.1f} (从 {margins[0]:.2f} 到 {margins[-1]:.2f}) — 过拟合风险")
            elif trend < -5:
                logger.warning(f"  ⚠️  Margin trend: {trend:.1f} (从 {margins[0]:.2f} 到 {margins[-1]:.2f}) — 训练崩溃")
            else:
                logger.info(f"  ✓  Margin trend: {trend:+.1f} — 合理范围")
        logger.info("=" * 60 + "\n")


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
    parser.add_argument("--max-length", type=int, default=5120,
                        help="最大序列长度（chosen/rejected 总 token 数，47G GPU 建议 5120）")
    parser.add_argument("--max-prompt-length", type=int, default=4300,
                        help="prompt 最大 token 数（仅日志参考，训练不截断）")
    parser.add_argument("--min-chosen-score", type=float, default=0.0,
                        help="最低 chosen 分数阈值，低于此分数的训练对将被过滤（数据已预过滤时默认 0）")
    parser.add_argument("--lora-r", type=int, default=8,
                        help="LoRA rank（30+对数据时建议 4-8）")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha（建议 alpha=2*r）")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--logging-steps", type=int, default=5,
                        help="每 N 步打印日志")
    parser.add_argument("--save-steps", type=int, default=50,
                        help="每 N 步保存 checkpoint")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup 比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--no-merge", action="store_true",
                        help="跳过 LoRA 合并（只保存 adapter）")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="禁用 gradient checkpointing（95G GPU 可关掉加速）")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="验证集比例（0=不划分验证集，建议 0.1~0.15）")
    parser.add_argument("--eval-steps", type=int, default=0,
                        help="每 N 步在验证集上评估（0=不评估，需要 --val-split > 0）")
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
    logger.info(f"  Min chosen:    {args.min_chosen_score}")
    logger.info(f"  Max length:    {args.max_length}")
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
    records, data_stats = load_dpo_dataset(Path(args.data), min_chosen_score=args.min_chosen_score)
    full_dataset = build_dataset_from_records(records)

    eval_dataset = None
    if args.val_split > 0 and args.val_split < 1:
        split = full_dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"  Train: {len(train_dataset)}, Val: {len(eval_dataset)} "
                     f"(split={args.val_split})")
        if args.eval_steps > 0:
            logger.info(f"  Eval every {args.eval_steps} steps")
    else:
        train_dataset = full_dataset
        if args.eval_steps > 0:
            logger.warning("  --eval-steps set but --val-split=0, disabling evaluation")
            args.eval_steps = 0

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
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer, DPOConfig

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

    # gradient checkpointing: 不缓存中间激活，显著节省显存（47G GPU 必需）
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ── DPO Config（含 precompute_ref_log_probs）──
    # ref model 的 log prob 预计算后释放，训练时只保留 policy model，节省显存
    eval_enabled = eval_dataset is not None and args.eval_steps > 0
    dpo_config = DPOConfig(
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
        gradient_checkpointing=not args.no_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        # DPO 专属参数
        beta=args.beta,
        max_length=args.max_length,
        loss_type="sigmoid",
        precompute_ref_log_probs=True,
        # 验证集评估（仅在 val_split > 0 且 eval_steps > 0 时启用）
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=args.eval_steps if eval_enabled else None,
        per_device_eval_batch_size=args.batch_size if eval_enabled else None,
    )

    # ── DPO Trainer ──
    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── 注册监控回调 ──
    metrics_log_path = Path(args.output_dir) / "dpo_metrics.json"
    monitor_callback = DPOMonitorCallback(log_file=metrics_log_path)
    dpo_trainer.add_callback(monitor_callback)

    # ── 训练 ──
    logger.info("Starting DPO training...")
    logger.info("=" * 60)
    dpo_trainer.train()

    # ── 保存 Adapter ──
    logger.info("Saving adapter...")
    dpo_trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # ── 合并 LoRA（默认开启）──
    if not args.no_merge:
        logger.info("Merging adapter into base model...")
        merged_path = os.path.join(str(args.output_dir), "merged")

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

        # Qwen3 的 save_pretrained 可能产生重复 chat_template，vLLM 会报错
        tcfg = Path(merged_path) / "tokenizer_config.json"
        if tcfg.exists():
            tcfg.unlink()
            logger.info("  Removed tokenizer_config.json (avoiding vLLM duplicate template error)")

        logger.info(f"  Merged model -> {merged_path}")

    # ── 保存数据统计 ──
    stats_path = Path(args.output_dir) / "data_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(data_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"Data stats saved to {stats_path}")

    # ── 验证集最终评估 ──
    if eval_dataset is not None:
        logger.info("Running final evaluation on validation set...")
        eval_results = dpo_trainer.evaluate()
        eval_path = Path(args.output_dir) / "eval_results.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Eval results saved to {eval_path}")
        logger.info(f"  Val metrics: {json.dumps(eval_results, indent=2)}")

    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  Adapter:    {args.output_dir}")
    logger.info(f"  Metrics:    {args.output_dir}/dpo_metrics.json")
    if not args.no_merge:
        logger.info(f"  Merged:     {args.output_dir}/merged")
    if eval_dataset is not None:
        logger.info(f"  Eval:       {args.output_dir}/eval_results.json")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
