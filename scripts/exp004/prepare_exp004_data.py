"""
Prepare data for exp-004: reranker candidate pool + passage texts + qrels.

Steps:
  1. Load exp-002 route data (5 JSONL files)
  2. RRF fuse with S4 config (D-B0+D-P2+D-HyDE+B-B0, per-route K=50, RRF k=60)
  3. Split qids (seed=42, 50/50) — same as exp-003
  4. Load passage texts from collection.tsv for all candidate pids
  5. Load graded qrels (pid -> label 0-3)
  6. Save as JSONL (one line per query)

Output: results/exp004/exp004_prepared_data.jsonl
  Each line: {qid, query, split, relevant_pids, graded_qrels, candidates}

Usage:
  python scripts/prepare_exp004_data.py
  python scripts/prepare_exp004_data.py --output-dir results/exp004
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.evaluation.data_loader import load_qrels, load_qrels_graded, clean_text
from src.evaluation.result_cache import load_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results"
T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QRELS_FILE = T2RANKING_DIR / "qrels.retrieval.dev.tsv"
QRELS_GRADED_FILE = T2RANKING_DIR / "qrels.dev.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"

RANDOM_SEED = 42
SPLIT_RATIO = 0.5

ROUTE_FILES = {
    "D-B0": "results/exp002_dense_details/exp002_E2a-B0_s2000_similarity.jsonl",
    "D-P2": "results/exp002_dense_details/exp002_E2a-P2_s2000_similarity.jsonl",
    "D-HyDE": "results/exp002_dense_details/exp002_E2c-H2_s2000_similarity.jsonl",
    "B-B0": "results/exp002_bm25_details/exp002_E2a-B0_s2000_bm25.jsonl",
}

ROUTE_KEYS = {
    "D-B0": "similarity",
    "D-P2": "similarity",
    "D-HyDE": "hyde_sub",
    "B-B0": "bm25",
}

S4_ROUTES = ["D-B0", "D-P2", "D-HyDE", "B-B0"]
PER_ROUTE_K = 50
RRF_K = 60
OUTPUT_TOP_K = 50


def load_route_data(route_id: str, file_relpath: str) -> dict[str, dict]:
    filepath = DATA_ROOT / file_relpath
    results, meta = load_results(str(filepath))
    route_data = {}
    for r in results:
        route_data[r["qid"]] = r
    logger.info(f"Loaded route {route_id}: {len(route_data)} queries")
    return route_data


def rrf_fuse(route_results: list[list[dict]], per_route_k: int, rrf_k: int, top_k: int):
    scores = defaultdict(float)
    for retrievals in route_results:
        for item in retrievals[:per_route_k]:
            scores[item["pid"]] += 1.0 / (rrf_k + item["rank"])
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(pid, round(score, 6)) for pid, score in merged]


def load_passage_map(pid_set: set[str]) -> dict[str, str]:
    logger.info(f"Loading passages for {len(pid_set)} pids from collection.tsv ...")
    pid_to_text = {}
    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            if pid in pid_set:
                text = clean_text(text)
                if len(text) > 2000:
                    text = text[:2000]
                pid_to_text[pid] = text
                if len(pid_to_text) >= len(pid_set):
                    break
    logger.info(f"Loaded {len(pid_to_text)} passages")
    return pid_to_text


def main():
    parser = argparse.ArgumentParser(description="Prepare exp-004 data")
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "exp004"),
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "exp004_prepared_data.jsonl"
    split_file = output_dir / "exp004_split.json"

    print("=" * 70)
    print("  EXP-004 DATA PREPARATION")
    print("=" * 70)
    print(f"  Config: S4 (D-B0+D-P2+D-HyDE+B-B0), perK={PER_ROUTE_K}, RRF_k={RRF_K}")
    print(f"  Seed: {RANDOM_SEED}, split: 50/50")
    print(f"  Output: {output_file}")

    # ── Load route data ──
    print()
    logger.info("Loading route data from exp-002 JSONL files ...")
    route_data: dict[str, dict[str, dict]] = {}
    for route_id in S4_ROUTES:
        route_data[route_id] = load_route_data(route_id, ROUTE_FILES[route_id])

    # ── Align qids ──
    qid_sets = [set(d.keys()) for d in route_data.values()]
    common_qids = sorted(set.intersection(*qid_sets))
    qrels = load_qrels(QRELS_FILE)
    qids_with_qrels = sorted(set(common_qids) & set(qrels.keys()))
    logger.info(f"Common qids with qrels: {len(qids_with_qrels)}")

    # ── RRF fusion ──
    logger.info("Running RRF fusion (S4 config) ...")
    qid_to_query = {}
    qid_to_relevant = {}
    qid_to_candidates = {}
    all_pids = set()

    for qid in qids_with_qrels:
        ref = route_data[S4_ROUTES[0]][qid]
        qid_to_query[qid] = ref["query"]
        qid_to_relevant[qid] = ref["relevant_pids"]

        route_retrievals = []
        for rid in S4_ROUTES:
            key = ROUTE_KEYS[rid]
            items = route_data[rid][qid]["retrievals"].get(key, [])
            route_retrievals.append(items)

        fused = rrf_fuse(route_retrievals, PER_ROUTE_K, RRF_K, OUTPUT_TOP_K)
        candidates = []
        for rank, (pid, score) in enumerate(fused, 1):
            candidates.append({"pid": pid, "rrf_score": score, "rrf_rank": rank})
            all_pids.add(pid)
        qid_to_candidates[qid] = candidates

    logger.info(f"Fusion complete. Unique pids in candidates: {len(all_pids)}")

    # ── Split (same as exp-003, seed=42, 50/50) ──
    random.seed(RANDOM_SEED)
    random.shuffle(qids_with_qrels)
    mid = len(qids_with_qrels) // 2
    val_qids = set(qids_with_qrels[:mid])
    test_qids = set(qids_with_qrels[mid:])
    logger.info(f"Validation: {len(val_qids)} queries")
    logger.info(f"Test:       {len(test_qids)} queries")

    # ── Load passage texts ──
    pid_to_text = load_passage_map(all_pids)
    missing = all_pids - set(pid_to_text.keys())
    if missing:
        logger.warning(f"{len(missing)} pids not found in collection.tsv")

    # ── Load graded qrels ──
    graded_qrels = load_qrels_graded(QRELS_GRADED_FILE)

    # ── Build output ──
    logger.info("Building output ...")
    count_val = 0
    count_test = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for qid in qids_with_qrels:
            split = "val" if qid in val_qids else "test"
            if split == "val":
                count_val += 1
            else:
                count_test += 1

            candidates_with_text = []
            for cand in qid_to_candidates[qid]:
                text = pid_to_text.get(cand["pid"], "")
                candidates_with_text.append({
                    "pid": cand["pid"],
                    "text": text,
                    "rrf_score": cand["rrf_score"],
                    "rrf_rank": cand["rrf_rank"],
                })

            entry = {
                "qid": qid,
                "query": qid_to_query[qid],
                "split": split,
                "relevant_pids": sorted(qid_to_relevant[qid]),
                "graded_qrels": graded_qrels.get(qid, {}),
                "candidates": candidates_with_text,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {count_val} val + {count_test} test = {count_val + count_test} entries to {output_file}")

    # ── Save split assignment for reproducibility ──
    split_info = {
        "experiment": "exp-004",
        "random_seed": RANDOM_SEED,
        "split_ratio": SPLIT_RATIO,
        "total_queries": len(qids_with_qrels),
        "validation": {"count": count_val, "qids": sorted(val_qids)},
        "test": {"count": count_test, "qids": sorted(test_qids)},
        "rrf_config": {
            "routes": S4_ROUTES,
            "per_route_k": PER_ROUTE_K,
            "rrf_k": RRF_K,
            "output_top_k": OUTPUT_TOP_K,
        },
    }
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    logger.info(f"Split info saved to: {split_file}")

    # ── Summary ──
    print()
    print("=" * 70)
    print("  DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"  Validation queries:  {count_val}")
    print(f"  Test queries:        {count_test}")
    print(f"  Candidates per query: up to {OUTPUT_TOP_K}")
    print(f"  Unique passage pids: {len(all_pids)}")
    print(f"  Prepared data:       {output_file}")
    print(f"  Split info:          {split_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
