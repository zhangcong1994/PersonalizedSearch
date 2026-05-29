"""
exp-007 Phase 3.2: Dynamic Dense Hard Negative Mining + TripletLoss + LoRA.

Per-epoch workflow (matching the experiment plan):
  epoch 0 (mine only): Encode 2.3M passages → FAISS IndexFlatIP →
      retrieve top-50 per query → filter out positives → top-5 hard negatives
  epoch 1: Load Phase 1 → add LoRA adapter → TripletLoss(margin=0.05) + MNRL
      → merge_and_unload → save merged → re-mine hard negatives for epoch 2
  epoch 2: Load ep1 merged → add fresh LoRA → TripletLoss(margin=0.03) + MNRL
      → merge_and_unload → save → re-mine for epoch 3
  epoch 3: Load ep2 merged → add fresh LoRA → TripletLoss(margin=0.03) + MNRL
      → merge_and_unload → save final

LoRA (default ON): trains only ~1.2% of parameters (attention projections),
prevents catastrophic forgetting on the already-converged Phase 1 model.
Each epoch creates a fresh adapter, then merges into full weights for
evaluation and next-epoch mining.

Usage:
  # Quick local test (100 queries, 5K passages, 1 epoch)
  python scripts/exp007/train_embedding_phase3_2.py \
      --sample-queries 100 --sample-passages 5000 --device cpu --batch-size 8

  # Full training on GPU (LoRA enabled by default)
  python scripts/exp007/train_embedding_phase3_2.py --device cuda --offline

  # Full fine-tuning without LoRA
  python scripts/exp007/train_embedding_phase3_2.py --device cuda --offline --no-lora

  # Start from a specific epoch checkpoint (resume)
  python scripts/exp007/train_embedding_phase3_2.py \
      --start-from models/m3e-base-t2ranking-phase3-2/ep1/merged --device cuda --offline
"""

import os
import sys
import re
import html
import json
import math
import time
import argparse
import logging
import random
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    DATA_ROOT, RAW_DATA_DIR,
    resolve_model_local_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = DATA_ROOT / "models" / "m3e-base-t2ranking-phase1"
DEFAULT_OUTPUT_BASE = DATA_ROOT / "models" / "m3e-base-t2ranking-phase3-2"

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_TRAIN_FILE = T2RANKING_DIR / "queries.train.tsv"
QRELS_TRAIN_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
QUERIES_DEV_FILE = T2RANKING_DIR / "queries.dev.tsv"
QRELS_DEV_FILE = T2RANKING_DIR / "qrels.retrieval.dev.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"

TRAIN_DATA_DIR = DATA_ROOT / "data" / "processed"
HARD_NEG_FILE_TEMPLATE = "exp007_hard_neg_epoch{epoch}.jsonl"

DEV_MONITOR_SAMPLE_SIZE = 500

BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
MARGIN_EPOCH1 = 0.05
MARGIN_LATER = 0.03
# Margin rationale: hard negatives are passages the model already finds very similar
# to the query. For TripletLoss with COSINE_DISTANCE:
#   loss = max(0, cos_dist(q,p) - cos_dist(q,n) + margin)
# If cos(q,n)=0.7 (typical for mined HN), cos_dist(q,n)=0.3.
# margin=0.5 requires cos_dist(q,p)<-0.2 → IMPOSSIBLE → model never converges.
# margin=0.05 only requires cos(q,p)>cos(q,n), i.e. positive marginally better
# than hard negative. This is achievable and stable.
DENSE_SEARCH_TOP_K = 50
NUM_HARD_NEGS_PER_QUERY = 5
HARD_NEG_SKIP_TOP = 10
ENCODE_BATCH_SIZE = 256
ENCODE_BATCH_SIZE_FP16 = 1024  # FP16 halves memory, can 4x batch size

# LoRA: 只训练 ~1.2% 参数，防止灾难性遗忘
# target_modules: BERT-base 的 attention 投影矩阵
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]

HTML_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUA_RE = re.compile(r"[\uE000-\uF8FF\u200E\u200F\u202A-\u202E\uFEFF]+")


def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = CONTROL_RE.sub("", text)
    text = PUA_RE.sub("", text)
    return text.strip()


def _resolve_model_path(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)
    local_path = resolve_model_local_path(model_id_or_path)
    if local_path is not None:
        return str(local_path)
    relative = (DATA_ROOT / model_id_or_path).resolve()
    if relative.is_dir():
        return str(relative)
    if not os.path.isabs(model_id_or_path):
        candidate = (Path.cwd() / model_id_or_path).resolve()
        if candidate.is_dir():
            return str(candidate)
    return model_id_or_path


def load_train_queries(max_queries: int = 0) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(QUERIES_TRAIN_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
            if max_queries > 0 and len(queries) >= max_queries:
                break
    logger.info(f"Loaded {len(queries):,} training queries")
    return queries


def load_train_qrels(target_qids: set[str] | None = None) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(QRELS_TRAIN_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qid, pid = parts[0], parts[1]
            if target_qids is not None and qid not in target_qids:
                continue
            qrels.setdefault(qid, set()).add(pid)
    total = sum(len(v) for v in qrels.values())
    logger.info(f"Loaded qrels: {len(qrels):,} queries, {total:,} pairs")
    return qrels


def load_all_passages(max_passages: int = 0) -> tuple[list[str], list[str]]:
    pids: list[str] = []
    texts: list[str] = []
    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            pid, raw_text = parts[0], parts[1]
            text = clean_text(raw_text)
            if len(text) < 10:
                continue
            if len(text) > 2000:
                text = text[:2000]
            pids.append(pid)
            texts.append(text)
            if max_passages > 0 and len(pids) >= max_passages:
                break
    logger.info(f"Loaded {len(pids):,} passages")
    return pids, texts


def load_training_pairs(qrels: dict[str, set[str]], queries: dict[str, str]) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for qid, pid_set in qrels.items():
        query_text = queries.get(qid)
        if query_text is None:
            continue
        for pid in pid_set:
            pairs.append({"qid": qid, "query": query_text, "pid": pid})
    logger.info(f"Built {len(pairs):,} training pairs from qrels")
    return pairs


def encode_batched(model, texts: list[str], batch_size: int = ENCODE_BATCH_SIZE) -> "np.ndarray":
    import numpy as np
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vectors.append(vecs)
    return np.concatenate(all_vectors, axis=0)


def _load_model_for_mining(model_path: str, device: str, fp16: bool = False):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_path, device=device)
    if fp16:
        model.half()
        dtype = str(next(model.parameters()).dtype)
        logger.info(f"Mining model FP16: {model_path} (dtype={dtype})")
    else:
        logger.info(f"Mining model FP32: {model_path}")
    return model


def mine_hard_negatives(
    model,
    passage_pids: list[str],
    passage_texts: list[str],
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    top_k: int = DENSE_SEARCH_TOP_K,
    num_hard_negs: int = NUM_HARD_NEGS_PER_QUERY,
    skip_top: int = HARD_NEG_SKIP_TOP,
    encode_batch_size: int = ENCODE_BATCH_SIZE,
) -> dict[str, list[str]]:
    import numpy as np
    import faiss

    logger.info("Encoding passages...")
    t0 = time.time()
    passage_vecs = encode_batched(model, passage_texts, batch_size=encode_batch_size)
    dim = passage_vecs.shape[1]
    logger.info(f"  {len(passage_texts):,} passages → {passage_vecs.shape} ({time.time() - t0:.0f}s)")

    logger.info("Building FAISS index...")
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(passage_vecs.astype(np.float32))
    logger.info(f"  index built ({time.time() - t0:.0f}s)")

    query_list = [(qid, text) for qid, text in queries.items()]
    query_texts = [text for _, text in query_list]
    query_ids = [qid for qid, _ in query_list]

    logger.info(f"Encoding {len(query_texts):,} queries...")
    t0 = time.time()
    query_vecs = encode_batched(model, query_texts, batch_size=encode_batch_size)
    logger.info(f"  {query_vecs.shape} ({time.time() - t0:.0f}s)")

    search_k = top_k + max(150, skip_top + num_hard_negs * 30)
    logger.info(
        f"Searching FAISS (top_k={search_k}, skip_top={skip_top}, "
        f"num_hard_negs={num_hard_negs})..."
    )
    t0 = time.time()
    scores, indices = index.search(query_vecs.astype(np.float32), search_k)
    logger.info(f"  search complete ({time.time() - t0:.0f}s)")

    hard_negatives: dict[str, list[str]] = {}
    skipped_no_pos = 0
    skipped_insufficient = 0

    for i, qid in enumerate(query_ids):
        positive_pids = qrels.get(qid, set())
        if not positive_pids:
            skipped_no_pos += 1
            continue

        neg_candidates = []
        for j in range(skip_top, len(indices[i])):
            pid = passage_pids[indices[i][j]]
            if pid not in positive_pids:
                neg_candidates.append(pid)

        if len(neg_candidates) < num_hard_negs:
            skipped_insufficient += 1
            if neg_candidates:
                hard_negatives[qid] = neg_candidates[:num_hard_negs]
            continue

        hn_pids = []
        segment_size = len(neg_candidates) // num_hard_negs
        for s in range(num_hard_negs):
            start = s * segment_size
            end = start + segment_size if s < num_hard_negs - 1 else len(neg_candidates)
            hn_pids.append(neg_candidates[start])

        hard_negatives[qid] = hn_pids

    logger.info(
        f"  Hard negatives mined: {len(hard_negatives):,} queries, "
        f"skipped: {skipped_no_pos} no-pos, {skipped_insufficient} insufficient"
    )
    return hard_negatives


def load_passage_texts(pids: list[str], passage_map: dict[str, str] | None = None) -> dict[str, str]:
    if passage_map is not None:
        return passage_map

    target = set(pids)
    result: dict[str, str] = {}
    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid in target:
                result[pid] = clean_text(parts[1])[:2000]
                if len(result) >= len(target):
                    break
    return result


def load_dev_pairs(
    sample_size: int = DEV_MONITOR_SAMPLE_SIZE,
) -> list[dict[str, str]]:
    dev_queries: dict[str, str] = {}
    with open(QUERIES_DEV_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                dev_queries[parts[0]] = parts[1]

    dev_qrels: dict[str, set[str]] = {}
    with open(QRELS_DEV_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            qid, pid = parts[0], parts[1]
            if qid in dev_queries:
                dev_qrels.setdefault(qid, set()).add(pid)

    pair_list: list[tuple[str, str]] = []
    for qid, pid_set in dev_qrels.items():
        query_text = dev_queries[qid]
        for pid in pid_set:
            pair_list.append((query_text, pid))

    random.seed(42)
    random.shuffle(pair_list)
    pair_list = pair_list[:sample_size]

    all_needed_pids = {pid for _, pid in pair_list}
    pid_to_text: dict[str, str] = {}
    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid in all_needed_pids:
                pid_to_text[pid] = clean_text(parts[1])[:2000]
                if len(pid_to_text) >= len(all_needed_pids):
                    break

    pairs: list[dict[str, str]] = []
    for query_text, pid in pair_list:
        pos_text = pid_to_text.get(pid)
        if pos_text:
            pairs.append({"query": query_text, "positive": pos_text})

    logger.info(
        f"Dev monitor pairs loaded: {len(pairs):,} (requested {sample_size})"
    )
    return pairs


def evaluate_dev_cosine_similarity(
    model,
    dev_pairs: list[dict[str, str]],
) -> tuple[float, float]:
    import numpy as np

    queries = [p["query"] for p in dev_pairs]
    positives = [p["positive"] for p in dev_pairs]

    q_vecs = model.encode(
        queries,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    p_vecs = model.encode(
        positives,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    sims = np.sum(q_vecs * p_vecs, axis=1)
    mean_sim = float(np.mean(sims))
    std_sim = float(np.std(sims))

    return mean_sim, std_sim


def build_triplets(
    training_pairs: list[dict[str, str]],
    hard_negatives: dict[str, list[str]],
    passage_map: dict[str, str],
) -> list[dict[str, str]]:
    triplets: list[dict[str, str]] = []
    missing_passage = 0
    missing_hn = 0

    for pair in training_pairs:
        qid = pair["qid"]
        pid = pair["pid"]
        positive_text = passage_map.get(pid)
        if positive_text is None:
            missing_passage += 1
            continue

        hard_negs = hard_negatives.get(qid, [])
        if not hard_negs:
            missing_hn += 1
            continue

        for hn_pid in hard_negs:
            hn_text = passage_map.get(hn_pid)
            if hn_text is None:
                continue
            triplets.append({
                "qid": qid,
                "query": pair["query"],
                "positive": positive_text,
                "positive_pid": pid,
                "hard_negative": hn_text,
                "hard_negative_pid": hn_pid,
            })

    logger.info(f"Built {len(triplets):,} triplets")
    if missing_passage:
        logger.warning(f"  {missing_passage} pairs skipped (passage not found)")
    if missing_hn:
        logger.warning(f"  {missing_hn} pairs skipped (no hard negatives)")
    return triplets


def save_triplets_jsonl(triplets: list[dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(triplets):,} triplets to {path}")


def load_triplets_jsonl(path: Path) -> list[dict[str, str]]:
    triplets: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    logger.info(f"Loaded {len(triplets):,} triplets from {path.name}")
    return triplets


def get_margin_for_epoch(epoch: int) -> float:
    return MARGIN_EPOCH1 if epoch == 1 else MARGIN_LATER


def main():
    parser = argparse.ArgumentParser(
        description="exp-007 Phase 3.2: Dynamic Dense Hard Negative Mining + TripletLoss"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="Phase 1 fine-tuned model path (default: models/m3e-base-t2ranking-phase1)"
    )
    parser.add_argument(
        "--output-base", type=Path, default=DEFAULT_OUTPUT_BASE,
        help="Base output directory (ep1/ep2/ep3 will be created under it)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (cpu, cuda, cuda:0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Training batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Number of training epochs after mining (default: {EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--sample-queries", type=int, default=0,
        help="Use first N training queries (0 = all 258K)"
    )
    parser.add_argument(
        "--sample-passages", type=int, default=0,
        help="Use first N passages for mining (0 = all 2.3M)"
    )
    parser.add_argument(
        "--start-from", type=Path, default=None,
        help="Resume from a specific epoch checkpoint (e.g. models/m3e-base-t2ranking-phase3-2-ep2)"
    )
    parser.add_argument(
        "--skip-mining", action="store_true",
        help="Skip hard negative mining, use existing JSONL files"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True,
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable mixed precision"
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Disable LoRA, do full fine-tuning instead (default: LoRA enabled)"
    )
    parser.add_argument(
        "--encode-fp16", action="store_true",
        help="Use FP16 for passage/query encoding during mining (faster, larger batch)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="HF offline mode"
    )
    parser.add_argument(
        "--dev-monitor-size", type=int, default=DEV_MONITOR_SAMPLE_SIZE,
        help=f"Number of dev (query, positive) pairs for training stability monitoring (default: {DEV_MONITOR_SAMPLE_SIZE}, 0 to disable)"
    )
    args = parser.parse_args()

    use_fp16 = args.fp16 and not args.no_fp16
    use_lora = not args.no_lora
    encode_fp16 = args.encode_fp16
    mining_bs = ENCODE_BATCH_SIZE_FP16 if encode_fp16 else ENCODE_BATCH_SIZE

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    is_sample = args.sample_queries > 0 or args.sample_passages > 0
    max_q = args.sample_queries if args.sample_queries > 0 else 0
    max_p = args.sample_passages if args.sample_passages > 0 else 0

    if is_sample:
        logger.warning("Sample mode: results are for verification only, not for evaluation")

    logger.info("=" * 60)
    logger.info("  EXP-007 PHASE 3.2: Dynamic Hard Negative Mining + TripletLoss")
    logger.info("=" * 60)
    logger.info(f"  Phase 1 model:  {args.model}")
    logger.info(f"  Output base:    {args.output_base}")
    logger.info(f"  Device:         {args.device or 'auto'}")
    logger.info(f"  Batch size:     {args.batch_size}")
    logger.info(f"  Epochs:         {args.epochs}")
    logger.info(f"  LR:             {args.lr}")
    logger.info(f"  FP16:           {use_fp16}")
    logger.info(f"  LoRA:           {use_lora} (r={LORA_R}, alpha={LORA_ALPHA})")
    logger.info(f"  Encode FP16:    {encode_fp16} (mining bs={mining_bs})")
    logger.info(f"  Offline:        {args.offline}")
    if args.start_from:
        logger.info(f"  Resume from:    {args.start_from}")
    logger.info(f"  Sample queries: {max_q if max_q > 0 else 'all (~258K)'}")
    logger.info(f"  Sample passages:{max_p if max_p > 0 else 'all (~2.3M)'}")
    logger.info("-" * 60)

    # ── Load data ──
    logger.info("Loading training data...")
    queries = load_train_queries(max_queries=max_q)
    qrels = load_train_qrels(target_qids=set(queries.keys()))
    passage_pids, passage_texts = load_all_passages(max_passages=max_p)

    # Build passage lookup map (pid → text) for all passages in collection
    passage_map: dict[str, str] = {pid: text for pid, text in zip(passage_pids, passage_texts)}

    # Also need to look up positive passages that may be outside the sampled set
    training_pairs = load_training_pairs(qrels, queries)
    all_needed_pids = set(passage_pids)
    for pair in training_pairs:
        all_needed_pids.add(pair["pid"])
    missing = all_needed_pids - set(passage_map.keys())
    if missing:
        logger.info(f"Loading {len(missing)} positive passages outside sampled set...")
        extra = load_passage_texts(list(missing))
        passage_map.update(extra)

    logger.info(f"Passage map: {len(passage_map):,} entries")

    # ── Load model ──
    from sentence_transformers import SentenceTransformer, InputExample
    try:
        from sentence_transformers.sentence_transformer import losses
    except ImportError:
        from sentence_transformers import losses
    from torch.utils.data import DataLoader

    # ── Build MNRL pairs (in-batch easy negatives) ──
    mnrl_examples: list[InputExample] = []
    for pair in training_pairs:
        pos_text = passage_map.get(pair["pid"])
        if pos_text is not None:
            mnrl_examples.append(InputExample(texts=[pair["query"], pos_text]))
    logger.info(f"MNRL training pairs: {len(mnrl_examples):,}")

    dev_pairs: list[dict[str, str]] = []
    if args.dev_monitor_size > 0:
        dev_pairs = load_dev_pairs(sample_size=args.dev_monitor_size)
        dev_sim_history: list[tuple[float, float]] = []

    model_path_str = _resolve_model_path(str(args.model))
    logger.info(f"Loading Phase 1 model: {model_path_str}")

    # ── Determine start epoch ──
    start_epoch = 0
    current_model_path = model_path_str
    if args.start_from:
        resolved = _resolve_model_path(str(args.start_from))
        current_model_path = resolved
        import re as _re
        m = _re.search(r'ep(\d+)', str(args.start_from))
        if m:
            start_epoch = int(m.group(1))
            logger.info(f"Resuming from epoch {start_epoch} checkpoint: {current_model_path}")

    # ── Epoch loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        print()
        logger.info("=" * 60)

        hard_neg_file = TRAIN_DATA_DIR / HARD_NEG_FILE_TEMPLATE.format(epoch=epoch)

        if epoch == 0:
            # ── Epoch 0: Mine only, no training ──
            logger.info(f"  EPOCH 0: Mining hard negatives (no training)")
            logger.info("=" * 60)

            model = _load_model_for_mining(current_model_path, args.device, fp16=encode_fp16)

            if args.skip_mining and hard_neg_file.exists():
                logger.info(f"  Skip mining: using existing {hard_neg_file}")
            else:
                hard_negatives = mine_hard_negatives(
                    model,
                    passage_pids,
                    passage_texts,
                    queries,
                    qrels,
                    top_k=DENSE_SEARCH_TOP_K,
                    num_hard_negs=NUM_HARD_NEGS_PER_QUERY,
                    encode_batch_size=mining_bs,
                )
                triplets = build_triplets(
                    training_pairs, hard_negatives, passage_map,
                )
                save_triplets_jsonl(triplets, hard_neg_file)

            del model
            torch.cuda.empty_cache()
            continue

        # ── Epochs 1-3: Train with previous epoch's hard negatives → Save → Mine for next ──
        margin = get_margin_for_epoch(epoch)
        output_dir = args.output_base / f"ep{epoch}"

        # Training data comes from the PREVIOUS epoch's mining
        prev_hard_neg_file = TRAIN_DATA_DIR / HARD_NEG_FILE_TEMPLATE.format(epoch=epoch - 1)
        if not prev_hard_neg_file.exists():
            logger.error(f"Hard negative file not found: {prev_hard_neg_file}")
            return 1

        logger.info(f"  EPOCH {epoch}: margin={margin}")
        logger.info(f"  Training data: {prev_hard_neg_file.name}")
        logger.info("=" * 60)

        # Load model. For epoch 1: Phase 1 full model.
        # For epochs 2+: previous epoch's merged model.
        model = SentenceTransformer(current_model_path, device=args.device)
        dim = model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension") else model.get_sentence_embedding_dimension()
        logger.info(f"Model: {current_model_path} (dim={dim})")

        # ── LoRA: add fresh adapter to prevent catastrophic forgetting ──
        if use_lora:
            from peft import LoraConfig, TaskType

            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET_MODULES,
            )
            model.add_adapter(peft_config)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            logger.info(f"LoRA adapter added: {trainable:,}/{total:,} params trainable ({trainable/total*100:.1f}%)")

        # Load triplets from previous epoch's mining result
        triplets = load_triplets_jsonl(prev_hard_neg_file)
        if not triplets:
            logger.error(f"No triplets found at {prev_hard_neg_file}")
            return 1

        # Shuffle
        random.seed(42 + epoch)
        random.shuffle(triplets)

        # Sub-sample triplets to balance with MNRL pairs (avoid hard-neg domination)
        if len(triplets) > len(mnrl_examples):
            triplets = triplets[:len(mnrl_examples)]

        # Build InputExamples
        triplet_examples = [
            InputExample(texts=[t["query"], t["positive"], t["hard_negative"]])
            for t in triplets
        ]

        triplet_steps = math.ceil(len(triplet_examples) / args.batch_size)
        mnrl_steps = math.ceil(len(mnrl_examples) / args.batch_size)
        warmup_steps = int(max(triplet_steps, mnrl_steps) * WARMUP_RATIO)

        hard_count = len(triplet_examples)
        easy_per_step = args.batch_size - 1

        logger.info(f"Triplets (hard negs):{hard_count:,} ({args.batch_size} hard/step)")
        logger.info(f"MNRL pairs (easy):   {len(mnrl_examples):,} ({easy_per_step} in-batch/query)")
        logger.info(f"Total negs/step:      hard={args.batch_size} + easy={args.batch_size * easy_per_step} ≈ {args.batch_size + args.batch_size * easy_per_step}")
        logger.info(f"Hard ratio:           {args.batch_size}/{args.batch_size + args.batch_size * easy_per_step} = {args.batch_size / (args.batch_size + args.batch_size * easy_per_step) * 100:.0f}%")
        logger.info(f"Triplet steps:        {triplet_steps:,}")
        logger.info(f"MNRL steps:           {mnrl_steps:,}")
        logger.info(f"Warmup steps:         {warmup_steps}")
        logger.info(f"Batch size:           {args.batch_size}")
        logger.info(f"Margin:               {margin}")
        logger.info(f"Output:               {output_dir}")
        logger.info("-" * 60)

        triplet_dataloader = DataLoader(
            triplet_examples, shuffle=True, batch_size=args.batch_size,
        )
        mnrl_dataloader = DataLoader(
            mnrl_examples, shuffle=True, batch_size=args.batch_size,
        )

        try:
            from sentence_transformers.sentence_transformer.losses import (
                TripletLoss, SiameseDistanceMetric, MultipleNegativesRankingLoss,
            )
        except ImportError:
            from sentence_transformers.losses import (
                TripletLoss, SiameseDistanceMetric, MultipleNegativesRankingLoss,
            )

        triplet_loss = TripletLoss(
            model,
            distance_metric=SiameseDistanceMetric.COSINE_DISTANCE,
            triplet_margin=margin,
        )
        mnrl_loss = MultipleNegativesRankingLoss(model)

        output_dir.mkdir(parents=True, exist_ok=True)

        if dev_pairs and not dev_sim_history:
            dev_mean, dev_std = evaluate_dev_cosine_similarity(model, dev_pairs)
            dev_sim_history.append((dev_mean, dev_std))
            logger.info(
                f"[DEV BASELINE] {len(dev_pairs)} pairs: "
                f"cos_sim={dev_mean:.4f} ± {dev_std:.4f}  (before epoch {epoch} training)"
            )

        logger.info("Training...")
        t_start = time.time()
        model.fit(
            train_objectives=[
                (triplet_dataloader, triplet_loss),
                (mnrl_dataloader, mnrl_loss),
            ],
            epochs=1,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": args.lr},
            weight_decay=WEIGHT_DECAY,
            max_grad_norm=MAX_GRAD_NORM,
            scheduler="warmupcosine",
            use_amp=use_fp16,
            output_path=str(output_dir),
            save_best_model=False,
            show_progress_bar=True,
            checkpoint_path=None,
            checkpoint_save_steps=0,
        )
        elapsed = time.time() - t_start
        logger.info(f"Epoch {epoch} training complete ({elapsed/60:.1f} min)")

        if dev_pairs:
            dev_mean, dev_std = evaluate_dev_cosine_similarity(
                model, dev_pairs,
            )
            prev_str = ""
            if dev_sim_history:
                prev_mean, _ = dev_sim_history[-1]
                delta = dev_mean - prev_mean
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                prev_str = f"  (prev={prev_mean:.4f} {arrow}{abs(delta):.4f})"
            dev_sim_history.append((dev_mean, dev_std))
            logger.info(
                f"[DEV MONITOR] {len(dev_pairs)} pairs: "
                f"cos_sim={dev_mean:.4f} ± {dev_std:.4f}{prev_str}"
            )

        # ── Merge LoRA into full weights & save for evaluation/mining ──
        if use_lora:
            merged_model = model.merge_and_unload()
            merged_dir = output_dir / "merged"
            merged_model.save(str(merged_dir))
            adapter_size = sum(
                f.stat().st_size for f in output_dir.rglob("adapter_model*")
            )
            logger.info(f"LoRA adapter: {adapter_size/1024:.0f} KB")
            logger.info(f"Merged model saved to: {merged_dir}")
        else:
            merged_dir = output_dir
            merged_model = model

        logger.info(f"Full model: {output_dir}")

        # Update current model path for next iteration
        current_model_path = str(merged_dir)

        # Free training model GPU memory before loading mining model
        del model
        del merged_model
        torch.cuda.empty_cache()

        # Mine hard negatives for the NEXT epoch (if not the final epoch)
        if epoch < args.epochs:
            next_hn_file = TRAIN_DATA_DIR / HARD_NEG_FILE_TEMPLATE.format(epoch=epoch)
            if args.skip_mining and next_hn_file.exists():
                logger.info(f"  Skip mining for next epoch: using existing {next_hn_file}")
            else:
                logger.info(f"  Mining hard negatives for epoch {epoch + 1} with ep{epoch} merged model...")
                model_for_mining = _load_model_for_mining(current_model_path, args.device, fp16=encode_fp16)
                hard_negatives = mine_hard_negatives(
                    model_for_mining,
                    passage_pids,
                    passage_texts,
                    queries,
                    qrels,
                    top_k=DENSE_SEARCH_TOP_K,
                    num_hard_negs=NUM_HARD_NEGS_PER_QUERY,
                    encode_batch_size=mining_bs,
                )
                triplets_next = build_triplets(
                    training_pairs, hard_negatives, passage_map,
                )
                save_triplets_jsonl(triplets_next, next_hn_file)
                del model_for_mining

    print()
    print("=" * 60)
    print("  PHASE 3.2 TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Output base: {args.output_base}")
    for ep in range(max(1, start_epoch + 1), args.epochs + 1):
        ep_dir = args.output_base / f"ep{ep}"
        eval_model = ep_dir / "merged" if use_lora else ep_dir
        print(f"    ep{ep}: {ep_dir}")
        if use_lora:
            print(f"      adapter: {ep_dir} ({ep_dir}/adapter_model.safetensors)")
            print(f"      merged:  {eval_model}  (full model)")
    print()
    print("  Evaluate each checkpoint:")
    for ep in range(max(1, start_epoch + 1), args.epochs + 1):
        ep_dir = args.output_base / f"ep{ep}"
        eval_model = ep_dir / "merged" if use_lora else ep_dir
        print(f"    # Epoch {ep}")
        print(f"    python scripts/exp007/build_index.py --model {eval_model} --device cuda")
        print(f"    python scripts/exp007/evaluate_embedding.py --model {eval_model} \\")
        print(f"        --device cuda --offline --baseline-values")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
