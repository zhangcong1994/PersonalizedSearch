"""
Exp-009 Step 2.6: Reranker.

Loads a Cross-Encoder reranker (FlagEmbedding or HuggingFace Transformers),
reads the RRF-fused top-50 JSONL, re-scores every (query, passage) pair,
re-sorts by score descending, and outputs top-K.

Backend auto-detection:
  - FlagEmbedding (FlagReranker): used for bge-reranker-v2-m3, bge-reranker-base
  - Transformers (AutoModelForSequenceClassification): fallback for other models

Reuses exp004's reranker loading pattern (load_flagembedding_reranker).

Usage:
  python scripts/exp009/rerank.py \
      --input data/processed/exp009_rrf_fused.jsonl \
      --model BAAI/bge-reranker-v2-m3 \
      --device cuda \
      --top-k 10 \
      --output data/processed/exp009_reranked_top10.jsonl
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, MODEL_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_model_path(model_id: str) -> str:
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)

    relative = (DATA_ROOT / model_id).resolve()
    if relative.is_dir():
        return str(relative)

    local_dir = MODEL_CACHE_DIR / model_id.replace("/", "--")
    if local_dir.is_dir():
        return str(local_dir)

    return model_id


def load_flagembedding_reranker(model_path: str, device: str):
    from FlagEmbedding import FlagReranker

    logger.info(f"Loading FlagReranker: {model_path}")
    reranker = FlagReranker(model_path, use_fp16=True, devices=[device])

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores.extend(reranker.compute_score(batch, normalize=True))
        return scores

    return score_pairs


def load_transformers_reranker(model_path: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading Transformers reranker: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)
    model.eval()

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        with torch.no_grad():
            scores = []
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                queries, docs = zip(*batch)
                inputs = tokenizer(
                    list(queries), list(docs),
                    padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1)
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                batch_scores = torch.sigmoid(logits).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
                scores.extend(batch_scores)
            return scores

    return score_pairs


def load_reranker(model_id: str, device: str):
    model_path = resolve_model_path(model_id)

    try:
        return load_flagembedding_reranker(model_path, device)
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"FlagEmbedding failed ({e}), falling back to Transformers")

    return load_transformers_reranker(model_path, device)


def load_rrf_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = DATA_ROOT / p
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 Step 2.6: Reranker"
    )
    parser.add_argument("--input", required=True,
                        help="RRF fused JSONL (top-50 per query)")
    parser.add_argument("--model", required=True,
                        help="Reranker model (HF ID or local path)")
    parser.add_argument("--device", default="cuda",
                        help="Device for reranker inference (default: cuda)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages to keep after reranking (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for reranker inference (default: 32)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL for reranked results")
    args = parser.parse_args()

    inp = _resolve(args.input)
    out = _resolve(args.output)

    logger.info("=" * 60)
    logger.info("  Exp-009 Step 2.6: Reranker")
    logger.info(f"  Input:     {inp.name}")
    logger.info(f"  Model:     {args.model}")
    logger.info(f"  Device:    {args.device}")
    logger.info(f"  Top-K:     {args.top_k}")
    logger.info(f"  Batch:     {args.batch_size}")
    logger.info("=" * 60)

    data = load_rrf_jsonl(inp)
    logger.info(f"Loaded {len(data):,} queries from {inp.name}")

    score_fn = load_reranker(args.model, args.device)

    results: list[dict] = []
    total_pairs = sum(len(d["results"]) for d in data)
    t_start = time.time()

    logger.info(f"Re-ranking {total_pairs:,} pairs ({len(data):,} queries × ~50 docs)...")

    for q_idx, entry in enumerate(data):
        qid = entry["qid"]
        query = entry["query"]
        passages = entry["results"]

        pairs = [(query, p["text"]) for p in passages]

        t_q0 = time.time()
        scores = score_fn(pairs, args.batch_size)
        t_q = time.time() - t_q0

        for p, s in zip(passages, scores):
            p["rerank_score"] = round(float(s), 6)

        passages.sort(key=lambda x: x["rerank_score"], reverse=True)
        topk = passages[:args.top_k]

        for rank, p in enumerate(topk, 1):
            p["rank"] = rank
            p["score"] = p.pop("rerank_score")

        results.append({
            "qid": qid,
            "query": query,
            "results": topk,
        })

        n_done = q_idx + 1
        if n_done % 500 == 0 or n_done == len(data):
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            eta = (len(data) - n_done) / max(rate, 0.01) / 60
            logger.info(f"  {n_done}/{len(data)} | {rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"Reranking done: {len(data):,} queries in {elapsed/60:.1f}min "
                f"({len(data)/max(elapsed,0.1):.1f}q/s)")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(results):,} entries → {out}")
    logger.info("=" * 60)
    logger.info("  Step 2.6 complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
