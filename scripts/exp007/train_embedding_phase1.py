"""
exp-007 Phase 1: Dual Encoder fine-tuning baseline (M3E-base + MNRL).

Fine-tunes moka-ai/m3e-base on T2Ranking training data using
MultipleNegativesRankingLoss. Only uses (query, positive) pairs —
in-batch negatives are automatically generated during training.

Usage:
  # Quick local test (CPU, 500 training pairs)
  python scripts/exp007/train_embedding_phase1.py --sample 500 --device cpu --batch-size 8

  # Full training (GPU, recommended)
  python scripts/exp007/train_embedding_phase1.py

  # Full training with custom output path
  python scripts/exp007/train_embedding_phase1.py --output models/m3e-phase1-v2
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
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "m3e-base-t2ranking-phase1"

MODEL_ID = "moka-ai/m3e-base"

BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0


def _resolve_model_path(model_id: str, offline: bool = False) -> str:
    if offline:
        local_path = resolve_model_local_path(model_id)
        if local_path is not None:
            logger.info(f"Offline mode: resolved {model_id} → {local_path}")
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
        description="exp-007 Phase 1: Fine-tune M3E-base with MNRL"
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
    parser.add_argument(
        "--eval-after-train", action="store_true",
        help="Run quick Recall@10 evaluation after training on a 500-query sample"
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
    logger.info("  EXP-007 PHASE 1: M3E-base fine-tuning")
    logger.info("=" * 60)
    logger.info(f"  Model:      {args.model}")
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

    logger.info("Loading model...")
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

    total_steps = math.ceil(len(train_examples) / args.batch_size) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info(f"Training pairs:     {len(train_examples)}")
    logger.info(f"Steps per epoch:    {math.ceil(len(train_examples) / args.batch_size)}")
    logger.info(f"Total steps:        {total_steps}")
    logger.info(f"Warmup steps:       {warmup_steps}")
    logger.info("-" * 60)

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

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

    if args.eval_after_train:
        print()
        logger.info("=" * 60)
        logger.info("  Post-training evaluation (500 query sample)")
        logger.info("=" * 60)
        _run_quick_eval(str(args.output), args.device or "cpu")

    print()
    print("=" * 60)
    print("  PHASE 1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Model:  {args.output}")
    print(f"  Pairs:  {len(train_examples)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {args.batch_size}")
    print("=" * 60)

    return 0


def _run_quick_eval(model_path: str, device: str):
    """
    Quick post-training sanity check: compute Recall@10 on a small
    subset of the dev set (500 queries sampled from the first 2000).
    Loads all passages from collection.tsv as the corpus.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import random

    from src.utils.config import RAW_DATA_DIR
    from src.evaluation.data_loader import load_queries, load_qrels, load_passages

    random.seed(42)
    np.random.seed(42)

    t2r = RAW_DATA_DIR / "t2ranking"
    queries_file = t2r / "queries.dev.tsv"
    qrels_file = t2r / "qrels.retrieval.dev.tsv"
    collection_file = t2r / "collection.tsv"

    logger.info("Loading evaluation data ...")
    all_queries = load_queries(queries_file)
    qrels = load_qrels(qrels_file)

    eval_queries = [(qid, text) for qid, text in all_queries if qid in qrels]
    if len(eval_queries) > 500:
        eval_queries = random.sample(eval_queries, 500)
    eval_qids = {qid for qid, _ in eval_queries}

    logger.info(f"Evaluation queries: {len(eval_queries)}")
    logger.info("Loading passages (this may take a minute)...")

    passages = load_passages(collection_file)
    if not passages or not passages[0]:
        logger.error("No passages loaded")
        return
    all_pids, all_texts = passages

    logger.info(f"Loaded {len(all_pids)} passages, encoding...")
    model = SentenceTransformer(model_path, device=device)

    pid_to_idx = {pid: i for i, pid in enumerate(all_pids)}

    query_texts = [text for _, text in eval_queries]
    q_embs = model.encode(query_texts, show_progress_bar=True, normalize_embeddings=True)
    p_embs = model.encode(all_texts, show_progress_bar=True, normalize_embeddings=True)

    scores_matrix = q_embs @ p_embs.T

    recall_sum = 0.0
    valid = 0
    for i, (qid, _) in enumerate(eval_queries):
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        top10_idx = np.argsort(scores_matrix[i])[::-1][:10]
        top10_pids = {all_pids[j] for j in top10_idx}
        hits = len(top10_pids & relevant)
        recall_sum += hits / len(relevant)
        valid += 1

    if valid > 0:
        recall_at_10 = recall_sum / valid
        logger.info(f"Quick eval Recall@10: {recall_at_10:.4f} ({valid} queries)")
    else:
        logger.warning("No valid queries for evaluation")


if __name__ == "__main__":
    sys.exit(main())
