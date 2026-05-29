"""
exp-007 Phase 2: Multi-Input Distribution Alignment (M3E-base + MNRL).

Fine-tunes moka-ai/m3e-base from pretrained weights on a mixture of
three query types (original 55% / rewritten 25% / HyDE 20%), using
MultipleNegativesRankingLoss with in-batch negatives.

Key differences from Phase 1:
  - Training data: mixed input types (not just original queries)
  - Starts from pretrained M3E-base (not Phase 1 checkpoint)
  - lr=1e-5 (half of Phase 1, compensating for LLM-generated noise)
  - epochs=3 (compensating for lower lr)
  - No instruction prefix (deferred to ablation — isolate one variable)
  - No LoRA (full fine-tuning, same as Phase 1)

Prerequisite: Run merge_training_data_phase2.py to build the training JSONL,
which requires generate_training_augmentations.py to have populated the
rewrite and HyDE cache files.

Usage:
  # Quick local test (CPU, 500 training pairs)
  python scripts/exp007/train_embedding_phase2.py --sample 500 --device cpu --batch-size 8

  # Full training (GPU, recommended)
  python scripts/exp007/train_embedding_phase2.py --device cuda --offline

  # Full training with custom output path
  python scripts/exp007/train_embedding_phase2.py --output models/m3e-phase2-v2
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, resolve_model_local_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_ROOT / "data" / "processed"
DEFAULT_TRAIN_FILE = PROCESSED_DIR / "embedding_train_phase2.jsonl"
DEFAULT_SAMPLE_FILE = PROCESSED_DIR / "embedding_train_phase2_sample.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "models" / "m3e-base-t2ranking-phase2"

MODEL_ID = "moka-ai/m3e-base"

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def _resolve_model_path(model_id: str, offline: bool = False) -> str:
    if offline:
        local_path = resolve_model_local_path(model_id)
        if local_path is not None:
            logger.info(f"Offline mode: resolved {model_id} -> {local_path}")
            return str(local_path)
        logger.warning(f"Offline mode: no local cache found for {model_id}, will attempt download")
    return model_id


def load_training_data(path: Path, max_pairs: int = 0) -> list[dict[str, str]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_pairs > 0 and len(records) >= max_pairs:
                break
    logger.info(f"Loaded {len(records)} training pairs from {path.name}")
    return records


def main():
    parser = argparse.ArgumentParser(
        description="exp-007 Phase 2: Multi-input distribution alignment with MNRL"
    )
    parser.add_argument(
        "--train-file", type=Path, default=None,
        help="Training data JSONL (default: auto-select based on --sample)"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Use only first N training pairs (0 = full dataset)"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--model", default=MODEL_ID,
        help="Base model HF ID"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use offline mode (resolve model from local cache, HF_HUB_OFFLINE=1)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (e.g., 'cpu', 'cuda:0')"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=WARMUP_RATIO,
        help=f"Warmup ratio (default: {WARMUP_RATIO})"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use mixed precision training (default: True)"
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable mixed precision"
    )
    args = parser.parse_args()

    use_fp16 = args.fp16 and not args.no_fp16

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    train_file = args.train_file
    if train_file is None:
        train_file = DEFAULT_SAMPLE_FILE if args.sample > 0 else DEFAULT_TRAIN_FILE

    if not train_file.exists():
        logger.error(f"Training data not found: {train_file}")
        logger.error("Run scripts/exp007/merge_training_data_phase2.py first")
        return 1

    logger.info("=" * 60)
    logger.info("  EXP-007 PHASE 2: Multi-Input Distribution Alignment")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Start from: pretrained (not Phase 1 checkpoint)")
    logger.info(f"  Train file: {train_file}")
    logger.info(f"  Output:     {args.output}")
    logger.info(f"  Device:     {args.device or 'auto'}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  LR:         {args.lr}")
    logger.info(f"  FP16:       {use_fp16}")
    logger.info(f"  Offline:    {args.offline}")
    logger.info("-" * 60)

    records = load_training_data(train_file, max_pairs=args.sample)
    if not records:
        logger.error("No training data loaded")
        return 1

    logger.info("Loading model from pretrained weights...")
    from sentence_transformers import SentenceTransformer, InputExample
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses
    from torch.utils.data import DataLoader

    model_path = _resolve_model_path(args.model, offline=args.offline)

    model = SentenceTransformer(
        model_path,
        device=args.device,
    )
    dim = model.get_sentence_embedding_dimension() if hasattr(model, "get_sentence_embedding_dimension") else model.get_embedding_dimension()
    logger.info(f"Model loaded: {args.model} (dim={dim})")

    train_examples = [
        InputExample(texts=[r["query"], r["positive"]])
        for r in records
    ]

    steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Training pairs:     {len(train_examples)}")
    logger.info(f"Steps per epoch:    {steps_per_epoch}")
    logger.info(f"Total steps:        {total_steps}")
    logger.info(f"Warmup steps:       {warmup_steps}")
    logger.info("-" * 60)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    checkpoint_dir = args.output / "checkpoints"

    logger.info("Starting training...")
    logger.info(f"Per-epoch checkpoints: {checkpoint_dir}/")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        scheduler="warmupcosine",
        use_amp=use_fp16,
        output_path=str(args.output),
        save_best_model=False,
        show_progress_bar=True,
        checkpoint_path=str(checkpoint_dir),
        checkpoint_save_steps=steps_per_epoch,
        checkpoint_save_total_limit=args.epochs,
    )
    logger.info("Training complete.")

    logger.info(f"Model saved to: {args.output}")

    print()
    print("=" * 60)
    print("  PHASE 2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:       {args.output}")
    print(f"  Start:       pretrained m3e-base")
    print(f"  Pairs:       {len(train_examples)}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch:       {args.batch_size}")
    print(f"  Checkpoints: {checkpoint_dir}/")
    print()
    print("  Evaluate each epoch checkpoint:")
    for ep in range(1, args.epochs + 1):
        step = ep * steps_per_epoch
        cp_path = checkpoint_dir / str(step)
        print(f"    # Epoch {ep}")
        print(f"    python scripts/exp007/build_index.py --model {cp_path} --device cuda")
        print(f"    python scripts/exp007/evaluate_embedding.py --model {cp_path} \\")
        print(f"        --device cuda --offline --baseline-values")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
