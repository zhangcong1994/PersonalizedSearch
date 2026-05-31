"""
Exp-009 Step 2.4: BM25 Retrieval (1 route).

Loads the pre-built BM25s sharded index, tokenizes each query with jieba,
retrieves top-K by BM25 scoring, then loads passage texts for all unique
retrieved pids from collection.tsv once. Outputs one JSONL file.

Reuses:
  - src/retrieval/bm25s_store.ShardedBM25S   (sharded index loading)
  - src/retrieval/bm25_store.tokenize_query  (jieba + stopwords)
  - src/retrieval/bm25_store._topk_indices   (argpartition top-K)

Index path: {DATA_ROOT}/data/bm25s_index/t2ranking/
  Built by: python scripts/exp003/build_bm25s_index.py

Usage:
  python scripts/exp009/bm25_retrieve.py \
      --input-queries data/processed/exp009_sampled_queries.jsonl \
      --output data/processed/exp009_bm25_B0.jsonl \
      --top-k 50
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR
from src.evaluation.data_loader import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"
DEFAULT_BM25S_DIR = DATA_ROOT / "data" / "bm25s_index" / "t2ranking"


def load_queries_jsonl(path: Path) -> list[tuple[str, str]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append((obj["qid"], obj["query"]))
    logger.info(f"Loaded {len(queries):,} queries from {path.name}")
    return queries


def load_pids_only(path: Path) -> list[str]:
    pids: list[str] = []
    logger.info(f"Loading pids from collection.tsv...")
    t0 = time.time()
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) >= 1:
                pids.append(parts[0])
    elapsed = time.time() - t0
    logger.info(f"  {len(pids):,} pids in {elapsed:.1f}s")
    return pids


def topk_indices(scores, k: int) -> list[int]:
    import numpy as np

    n = len(scores)
    if n <= k:
        return np.argsort(scores)[::-1].tolist()
    partition_idx = np.argpartition(scores, n - k)[n - k:]
    topk = partition_idx[np.argsort(scores[partition_idx])[::-1]]
    return topk.tolist()


def load_passage_texts(pids_to_load: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    missing = set(pids_to_load)
    logger.info(f"Loading texts for {len(pids_to_load):,} unique pids from collection.tsv...")
    t_start = time.time()

    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not missing:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid in missing:
                raw = clean_text(parts[1])
                texts[pid] = raw[:2000] if len(raw) > 2000 else raw
                missing.discard(pid)

    elapsed = time.time() - t_start
    logger.info(f"Loaded {len(texts):,}/{len(pids_to_load):,} texts in {elapsed:.1f}s "
                f"({len(missing):,} not found)")
    return texts


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = DATA_ROOT / p
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 Step 2.4: BM25 Retrieval (1 route)"
    )
    parser.add_argument("--input-queries", required=True,
                        help="Sampled queries JSONL (qid, query, ...)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL for BM25 retrieval results")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of passages to retrieve per query (default: 50)")
    parser.add_argument("--bm25s-dir", default=None,
                        help=f"BM25s sharded index dir (default: {DEFAULT_BM25S_DIR})")
    args = parser.parse_args()

    inp = _resolve(args.input_queries)
    out = _resolve(args.output)
    bm25s_dir = Path(args.bm25s_dir) if args.bm25s_dir else DEFAULT_BM25S_DIR
    if not bm25s_dir.is_absolute():
        bm25s_dir = DATA_ROOT / bm25s_dir

    logger.info("=" * 60)
    logger.info("  Exp-009 Step 2.4: BM25 Retrieval")
    logger.info(f"  Input:     {inp}")
    logger.info(f"  Output:    {out}")
    logger.info(f"  BM25s dir: {bm25s_dir}")
    logger.info(f"  Top-K:     {args.top_k}")
    logger.info("=" * 60)

    if not (bm25s_dir / "shards.json").exists():
        logger.error(f"BM25s sharded index not found: {bm25s_dir / 'shards.json'}")
        logger.error(f"  Build: python scripts/exp003/build_bm25s_index.py")
        return 1

    queries = load_queries_jsonl(inp)
    if not queries:
        logger.error("No queries loaded")
        return 1

    pids = load_pids_only(COLLECTION_FILE)
    logger.info(f"Collection: {len(pids):,} documents")

    logger.info(f"Loading BM25s sharded index from {bm25s_dir}...")
    t0 = time.time()
    from src.retrieval.bm25s_store import ShardedBM25S
    bm25 = ShardedBM25S.load(str(bm25s_dir))
    logger.info(f"BM25s index loaded: {bm25.n_docs:,} docs in {time.time() - t0:.1f}s")

    from src.retrieval.bm25_store import tokenize_query

    results: list[dict] = []
    all_retrieved_pids: set[str] = set()
    total_time = 0.0
    t_start = time.time()

    logger.info(f"Retrieving top-{args.top_k} for {len(queries)} queries...")

    for i, (qid, query_text) in enumerate(queries):
        t_q = time.time()
        tokenized = tokenize_query(query_text)
        scores = bm25.get_scores(tokenized)
        top_idx = topk_indices(scores, args.top_k)
        elapsed = time.time() - t_q
        total_time += elapsed

        route_results = []
        for rank, idx in enumerate(top_idx, 1):
            pid = pids[idx]
            route_results.append({
                "pid": pid,
                "score": round(float(scores[idx]), 6),
                "rank": rank,
            })
            all_retrieved_pids.add(pid)

        results.append({
            "qid": qid,
            "query": query_text,
            "results": route_results,
        })

        n_done = i + 1
        if n_done % 500 == 0 or n_done >= len(queries):
            elapsed_total = time.time() - t_start
            rate = n_done / max(elapsed_total, 0.1)
            eta = (len(queries) - n_done) / max(rate, 0.01) / 60
            avg_ms = total_time / n_done * 1000
            logger.info(f"  {n_done}/{len(queries)} | {rate:.1f}q/s | "
                        f"{avg_ms:.0f}ms/q | ETA {eta:.0f}min")

    elapsed_total = time.time() - t_start
    logger.info(f"BM25 retrieval done: {len(queries):,} queries in {elapsed_total/60:.1f}min "
                f"({len(queries)/max(elapsed_total,0.1):.1f}q/s, "
                f"{total_time/len(queries)*1000:.0f}ms/q avg)")

    pid_to_text = load_passage_texts(all_retrieved_pids)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for entry in results:
            for r in entry["results"]:
                r["text"] = pid_to_text.get(r["pid"], "")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(results):,} entries → {out}")
    total_passages = sum(len(entry["results"]) for entry in results)
    logger.info(f"Total passages: {total_passages:,}, unique pids: {len(all_retrieved_pids):,}")
    logger.info("=" * 60)
    logger.info("  Step 2.4 complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
