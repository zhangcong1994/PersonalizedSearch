"""
exp-007 Phase 3.1: CachedMNRL training (Loss upgrade from MNRL).

Loads the Phase 1 fine-tuned model and continues training with
CachedMultipleNegativesRankingLoss, which caches passage embeddings
across mini-batches to provide 5-10x more negative samples per step
at the same GPU memory budget.

Usage:
  # Full training (GPU, recommended)
  python scripts/exp007/train_embedding_phase3_1.py

  # From a different starting model
  python scripts/exp007/train_embedding_phase3_1.py \
      --model models/m3e-base-t2ranking-phase1 \
      --device cuda --offline

  # Quick local test (CPU)
  python scripts/exp007/train_embedding_phase3_1.py \
      --sample 500 --batch-size 8 --epochs 1 --device cpu --no-fp16 --offline
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TRAIN_FILE = DATA_ROOT / "data" / "processed" / "embedding_train_phase1.jsonl"
DEFAULT_SAMPLE_FILE = DATA_ROOT / "data" / "processed" / "embedding_train_phase1_sample.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "m3e-base-t2ranking-phase3-1"
DEFAULT_MODEL = "models/m3e-base-t2ranking-phase1"

BATCH_SIZE = 64
MINI_BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def _resolve_model_path(model_id: str, offline: bool = False) -> str:
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)

    if offline:
        local_path = resolve_model_local_path(model_id)
        if local_path is not None:
            logger.info(f"Offline: resolved {model_id} → {local_path}")
            return str(local_path)
        logger.warning(f"Offline: no local cache for {model_id}, will attempt download")

    relative_to_data_root = (DATA_ROOT / model_id).resolve()
    if relative_to_data_root.is_dir():
        logger.info(f"Resolved via DATA_ROOT: {model_id} → {relative_to_data_root}")
        return str(relative_to_data_root)

    if not os.path.isabs(model_id):
        candidate = (Path.cwd() / model_id).resolve()
        if candidate.is_dir():
            logger.info(f"Resolved via CWD: {model_id} → {candidate}")
            return str(candidate)

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
        description="exp-007 Phase 3.1: CachedMNRL training"
    )
    parser.add_argument(
        "--train-file", type=Path, default=None,
        help="Training data JSONL"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Use only first N training pairs"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Output directory for the fine-tuned model"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Starting model path (Phase 1 output) or HF ID"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Offline mode (HF_HUB_OFFLINE=1)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--mini-batch-size", type=int, default=MINI_BATCH_SIZE,
        help=f"Mini-batch size for CachedMNRL (default: {MINI_BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of epochs (default: {EPOCHS})"
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
        help="Use mixed precision"
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
        logger.error("Run scripts/exp007/prepare_training_data.py first")
        return 1

    logger.info("=" * 60)
    logger.info("  EXP-007 PHASE 3.1: CachedMNRL training")
    logger.info("=" * 60)
    logger.info(f"  Start model: {args.model}")
    logger.info(f"  Train file:  {train_file}")
    logger.info(f"  Output:      {args.output}")
    logger.info(f"  Device:      {args.device or 'auto'}")
    logger.info(f"  Batch size:  {args.batch_size}")
    logger.info(f"  Mini-batch:  {args.mini_batch_size}")
    logger.info(f"  Epochs:      {args.epochs}")
    logger.info(f"  LR:          {args.lr}")
    logger.info(f"  FP16:        {use_fp16}")
    logger.info(f"  Offline:     {args.offline}")
    logger.info("-" * 60)

    records = load_training_data(train_file, max_pairs=args.sample)
    if not records:
        logger.error("No training data loaded")
        return 1

    logger.info("Loading model...")
    from sentence_transformers import SentenceTransformer, InputExample
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses
    from torch.utils.data import DataLoader

    model_path = _resolve_model_path(args.model, offline=args.offline)

    model = SentenceTransformer(model_path, device=args.device)

    try:
        dim = model.get_embedding_dimension()
    except AttributeError:
        dim = model.get_sentence_embedding_dimension()
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
    logger.info(f"Negatives per step: ~{args.mini_batch_size} (in-batch) "
                f"+ cached from {args.batch_size // args.mini_batch_size - 1} prev mini-batches")
    logger.info("-" * 60)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    train_loss = losses.CachedMultipleNegativesRankingLoss(
        model,
        mini_batch_size=args.mini_batch_size,
        show_progress_bar=False,
    )

    logger.info("Starting training...")
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
        checkpoint_path=None,
        checkpoint_save_steps=0,
    )
    logger.info("Training complete.")
    logger.info(f"Model saved to: {args.output}")

    print()
    print("=" * 60)
    print("  PHASE 3.1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Start:  {args.model}")
    print(f"  Model:  {args.output}")
    print(f"  Pairs:  {len(train_examples)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {args.batch_size}")
    print(f"  Loss:   CachedMultipleNegativesRankingLoss")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
