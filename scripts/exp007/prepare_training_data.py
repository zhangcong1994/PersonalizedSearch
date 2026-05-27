"""
Prepare training data for exp-007 Phase 1: Dual Encoder fine-tuning.

Reads T2Ranking train split (queries.train.tsv + qrels.retrieval.train.tsv +
collection.tsv) and outputs (query, positive_passage) pairs as JSONL
for sentence-transformers MultipleNegativesRankingLoss.

Steps:
  1. Load training queries       -> {qid: query_text}
  2. Load retrieval qrels (train)-> {qid: set(pid)}
  3. Collect needed pids         -> load only those passages from collection.tsv
  4. Join and output             -> JSONL with {"query": ..., "positive": ...}

Sample mode (--sample N): use only the first N queries for local verification.

Output: data/processed/embedding_train_phase1.jsonl
        data/processed/embedding_train_phase1_sample.jsonl  (--sample)

Usage:
  python scripts/exp007/prepare_training_data.py                          # full
  python scripts/exp007/prepare_training_data.py --sample 500             # sample
  python scripts/exp007/prepare_training_data.py --sample 500 --output data/processed/sample.jsonl
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.evaluation.data_loader import load_qrels, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_TRAIN_FILE = T2RANKING_DIR / "queries.train.tsv"
QRELS_RETRIEVAL_TRAIN_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"

DEFAULT_OUTPUT = DATA_ROOT / "data" / "processed" / "embedding_train_phase1.jsonl"
DEFAULT_SAMPLE_OUTPUT = DATA_ROOT / "data" / "processed" / "embedding_train_phase1_sample.jsonl"

TRUNCATE_LEN = 2000
MIN_TEXT_LEN = 10


def load_queries_dict(path: Path, max_queries: int = 0) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
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
    logger.info(f"Loaded {len(queries)} queries from {path.name}")
    return queries


def load_qrels_filtered(path: Path, target_qids: set[str]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                qid, pid = parts[0], parts[1]
                if qid in target_qids:
                    qrels.setdefault(qid, set()).add(pid)
    total_pairs = sum(len(v) for v in qrels.values())
    logger.info(f"Loaded qrels: {len(qrels)} queries, {total_pairs} pairs")
    return qrels


def load_passages_by_pids(path: Path, target_pids: set[str]) -> dict[str, str]:
    passages: dict[str, str] = {}
    found = 0
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            if pid not in target_pids:
                continue
            text = clean_text(text)
            if len(text) < MIN_TEXT_LEN:
                continue
            if len(text) > TRUNCATE_LEN:
                text = text[:TRUNCATE_LEN]
            passages[pid] = text
            found += 1
            if found >= len(target_pids):
                break
    logger.info(f"Loaded {len(passages)} passages from {path.name} "
                f"(target: {len(target_pids)}, missing: {len(target_pids) - len(passages)})")
    return passages


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for exp-007 Phase 1 (Dual Encoder fine-tuning)"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Sample first N queries for local verification (default: 0 = full dataset)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: auto based on --sample)"
    )
    args = parser.parse_args()

    is_sample = args.sample > 0
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_SAMPLE_OUTPUT if is_sample else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"{'SAMPLE' if is_sample else 'FULL'} MODE: "
                f"{f'{args.sample} queries' if is_sample else '258,042 queries'}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    queries = load_queries_dict(QUERIES_TRAIN_FILE, max_queries=args.sample)
    if not queries:
        logger.error("No queries loaded")
        return 1

    qrels = load_qrels_filtered(QRELS_RETRIEVAL_TRAIN_FILE, set(queries.keys()))
    if not qrels:
        logger.error("No qrels loaded")
        return 1

    needed_pids: set[str] = set()
    for pids in qrels.values():
        needed_pids.update(pids)
    logger.info(f"Unique pids needed: {len(needed_pids)}")

    passages = load_passages_by_pids(COLLECTION_FILE, needed_pids)

    written = 0
    skipped_missing = 0
    skipped_query = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for qid, query_text in queries.items():
            pos_pids = qrels.get(qid, set())
            if not pos_pids:
                skipped_query += 1
                continue
            for pid in pos_pids:
                passage_text = passages.get(pid)
                if not passage_text:
                    skipped_missing += 1
                    continue
                line = json.dumps(
                    {"query": query_text, "positive": passage_text},
                    ensure_ascii=False,
                )
                f.write(line + "\n")
                written += 1

    logger.info("-" * 60)
    logger.info(f"Output pairs:     {written}")
    logger.info(f"Skipped (missing): {skipped_missing} (pid not in collection)")
    logger.info(f"Skipped (no pos):  {skipped_query} (query has no qrels)")
    logger.info(f"Output file:       {output_path}")
    logger.info("=" * 60)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

    print()
    print("=" * 60)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Mode:      {'sample' if is_sample else 'full'}")
    if is_sample:
        print(f"  Queries:   {args.sample}")
    print(f"  Pairs:     {written}")
    print(f"  Output:    {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
