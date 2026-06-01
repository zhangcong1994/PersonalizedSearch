"""
Exp-009 阶段七：QLoRA SFT 微调 Qwen3-4B。

用法:
  python scripts/exp009/train_sft.py --train data/processed/exp009_sft_train.jsonl --val data/processed/exp009_sft_val.jsonl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.generation.prompts import SYSTEM_PROMPT, FEW_SHOT, CONTEXT_TEMPLATE, QUESTION_TEMPLATE
from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SYSTEM_TEXT = SYSTEM_PROMPT + FEW_SHOT

DEFAULT_TRAIN = DATA_ROOT / "data" / "processed" / "exp009_sft_train.jsonl"
DEFAULT_VAL = DATA_ROOT / "data" / "processed" / "exp009_sft_val.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "models" / "qwen3-4b-t2ranking-sft"


MONITOR_CATEGORIES = {"standard", "citation_emphasis", "refusal", "noise", "contradiction"}


def format_passage(idx: int, pid: str, text: str) -> str:
    return f"[{idx}] 来源: {pid}\n{text}"


def build_user_message(query: str, passages: list[dict]) -> str:
    lines = [format_passage(i + 1, p["pid"], p["text"]) for i, p in enumerate(passages)]
    ctx = CONTEXT_TEMPLATE.format(context="\n\n".join(lines))
    q = QUESTION_TEMPLATE.format(question=query)
    return ctx + "\n\n" + q


def load_and_format_data(path: Path, tokenizer) -> Dataset:
    conversations = []
    stats = {cat: 0 for cat in MONITOR_CATEGORIES}
    skipped = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            query = entry.get("query", "")
            passages = entry.get("passages", [])
            answer = entry.get("answer", "")
            category = entry.get("category", "unknown")

            if not query or not answer:
                skipped += 1
                continue

            user_msg = build_user_message(query, passages)

            messages = [
                {"role": "system", "content": SYSTEM_TEXT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            conversations.append({"text": text, "category": category, "answer_len": len(answer)})

            if category in stats:
                stats[category] += 1

    logger.info(f"  Loaded {len(conversations):,} examples (skipped {skipped})")
    for cat, count in stats.items():
        if count > 0:
            logger.info(f"    {cat}: {count:,}")

    return Dataset.from_list(conversations)


def main():
    parser = argparse.ArgumentParser(description="Exp-009 QLoRA SFT Training")
    parser.add_argument("--train", type=str, default=str(DEFAULT_TRAIN),
                        help="Training data path")
    parser.add_argument("--val", type=str, default=str(DEFAULT_VAL),
                        help="Validation data path")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B",
                        help="Base model HF ID")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory for adapter + merged model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=6144,
                        help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--save-steps", type=int, default=200,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging adapter into base model")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Exp-009 QLoRA SFT Training")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {args.base_model}")
    logger.info(f"  Train data:    {args.train}")
    logger.info(f"  Val data:      {args.val}")
    logger.info(f"  Output dir:    {args.output_dir}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Batch size:    {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    logger.info(f"  LR:            {args.lr}")
    logger.info(f"  Max seq len:   {args.max_seq_length}")
    logger.info(f"  LoRA r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available! This script requires a GPU.")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info("=" * 60)

    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        logger.info("flash-attn detected, using FlashAttention2")
    except ImportError:
        attn_impl = "sdpa"
        logger.info("flash-attn not found, falling back to PyTorch SDPA")

    logger.info("Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
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

    logger.info("Loading and formatting data...")
    train_dataset = load_and_format_data(Path(args.train), tokenizer)
    val_dataset = load_and_format_data(Path(args.val), tokenizer)

    import math
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    logger.info(f"  Steps per epoch: ~{steps_per_epoch}, total: ~{total_steps}")

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving final adapter...")
    trainer.save_model(str(args.output_dir))

    if not args.no_merge:
        logger.info("Merging adapter into base model...")
        merged_path = os.path.join(str(args.output_dir), "merged")

        merged_model = model.merge_and_unload()

        merged_model.save_pretrained(merged_path, safe_serialization=True)
        tokenizer.save_pretrained(merged_path)

        logger.info(f"  Merged model → {merged_path}")

    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info(f"  Adapter: {args.output_dir}")
    logger.info(f"  Merged:  {args.output_dir}/merged")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
