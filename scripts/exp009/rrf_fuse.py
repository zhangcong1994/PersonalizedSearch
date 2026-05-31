"""
Exp-009 Step 2.5: RRF Fusion (4 routes → top-K).

Reads per-route retrieval results from Step 2.3/2.4 JSONL files, applies
Reciprocal Rank Fusion, and outputs a single merged top-K JSONL.

Reuses exp003's rrf_fuse() core logic — it's a pure mathematical function,
no model, no API.

Input files (each JSONL with {"qid", "query", "results": [{"pid","score","rank"},...]}):
  - Dense B0  (original query)
  - Dense P2  (rewritten query)
  - Dense H2  (HyDE)
  - BM25  B0  (original query)

Usage:
  python scripts/exp009/rrf_fuse.py \
      --route-files \
          data/processed/exp009_dense_B0.jsonl \
          data/processed/exp009_dense_P2.jsonl \
          data/processed/exp009_dense_H2.jsonl \
          data/processed/exp009_bm25_B0.jsonl \
      --per-route-k 50 \
      --rrf-k 60 \
      --output-top-k 50 \
      --output data/processed/exp009_rrf_fused.jsonl
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def rrf_fuse(
    route_results: list[list[dict]],
    per_route_k: int,
    rrf_k: int,
    output_top_k: int,
) -> list[dict]:
    rrf_scores: dict[str, float] = defaultdict(float)
    for retrievals in route_results:
        for item in retrievals[:per_route_k]:
            rrf_scores[item["pid"]] += 1.0 / (rrf_k + item["rank"])
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:output_top_k]
    return [
        {"pid": pid, "score": round(score, 6), "rank": rank}
        for rank, (pid, score) in enumerate(merged, 1)
    ]


def load_route_file(path: Path) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data[obj["qid"]] = obj
    logger.info(f"Loaded {len(data):,} entries from {path.name}")
    return data


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = DATA_ROOT / p
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 Step 2.5: RRF Fusion"
    )
    parser.add_argument("--route-files", nargs="+", required=True,
                        help="2-4 route result JSONL files (order: D-B0 D-P2 D-H2 B-B0)")
    parser.add_argument("--per-route-k", type=int, default=50,
                        help="Number of passages to take from each route (default: 50)")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="RRF smoothing constant k (default: 60)")
    parser.add_argument("--output-top-k", type=int, default=50,
                        help="Number of passages in fused output (default: 50)")
    parser.add_argument("--output", required=True,
                        help="Output JSONL for fused results")
    args = parser.parse_args()

    route_paths = [_resolve(p) for p in args.route_files]
    out = _resolve(args.output)

    logger.info("=" * 60)
    logger.info("  Exp-009 Step 2.5: RRF Fusion")
    logger.info(f"  Routes:        {len(route_paths)} files")
    for i, p in enumerate(route_paths, 1):
        logger.info(f"    [{i}] {p.name}")
    logger.info(f"  Per-route K:   {args.per_route_k}")
    logger.info(f"  RRF k:         {args.rrf_k}")
    logger.info(f"  Output top-K:  {args.output_top_k}")
    logger.info("=" * 60)

    route_data = [load_route_file(p) for p in route_paths]

    all_qids = set(route_data[0].keys())
    for rd in route_data[1:]:
        all_qids &= set(rd.keys())

    logger.info(f"Common qids across all routes: {len(all_qids):,}")

    qids = sorted(all_qids)
    results: list[dict] = []

    for qid in qids:
        query = route_data[0][qid]["query"]
        route_results = [rd[qid]["results"] for rd in route_data]

        fused = rrf_fuse(route_results, args.per_route_k, args.rrf_k, args.output_top_k)

        text_map = {}
        for rd in route_data:
            for r in rd[qid]["results"]:
                pid = r["pid"]
                if "text" in r and r["text"] and pid not in text_map:
                    text_map[pid] = r["text"]

        for item in fused:
            item["text"] = text_map.get(item["pid"], "")

        results.append({
            "qid": qid,
            "query": query,
            "results": fused,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Wrote {len(results):,} entries → {out}")

    total_passages = sum(len(entry["results"]) for entry in results)
    unique_pids = len(set(
        r["pid"] for entry in results for r in entry["results"]
    ))
    logger.info(f"Total passages: {total_passages:,}, unique pids: {unique_pids:,}")
    logger.info("=" * 60)
    logger.info("  Step 2.5 complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
