"""
Phase 2: Merge training data from three query sources.

Reads original queries, rewritten queries, and HyDE pseudo-answers,
then merges them into a single JSONL training file.

Two modes:
  random (default): Randomly assign each qid to exactly one input type
    (55% original / 25% rewrite / 20% HyDE). If the assignment file from
    generate_training_augmentations.py exists, it is reused directly.
    Otherwise a local random assignment is performed.
  read: Use the best available augmentation per qid — no ratio assignment.
    For each qid: rewrite > HyDE > original. This is for use with a
    partially-generated augmentation set (e.g. --sample in the generation
    step). The resulting distribution reflects whatever was generated,
    not a target ratio.

Output format matches Phase 1: {"query": "...", "positive": "..."}
for direct compatibility with train_embedding_phase2.py.

Usage:
  # Quick test with 500 queries (random mode)
  python scripts/exp007/merge_training_data_phase2.py --sample 500

  # Full merge (random mode, 55/25/20)
  python scripts/exp007/merge_training_data_phase2.py

  # Read mode: use existing augmentations as-is, no ratio assignment
  python scripts/exp007/merge_training_data_phase2.py --mode read

  # Read mode with sampling
  python scripts/exp007/merge_training_data_phase2.py --mode read --sample 5000

  # Custom seed for random mode
  python scripts/exp007/merge_training_data_phase2.py --seed 123

Output:
  {DATA_ROOT}/data/processed/embedding_train_phase2.jsonl
"""

import os
import sys
import json
import random
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
QUERIES_TRAIN_FILE = T2RANKING_DIR / "queries.train.tsv"
QRELS_RETRIEVAL_TRAIN_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"

PROCESSED_DIR = DATA_ROOT / "data" / "processed"
REWRITE_FILE = PROCESSED_DIR / "exp007_rewritten_queries.jsonl"
HYDE_FILE = PROCESSED_DIR / "exp007_hyde_answers.jsonl"
ASSIGNMENT_FILE = PROCESSED_DIR / "exp007_phase2_assignment.jsonl"

DEFAULT_OUTPUT = PROCESSED_DIR / "embedding_train_phase2.jsonl"
DEFAULT_SAMPLE_OUTPUT = PROCESSED_DIR / "embedding_train_phase2_sample.jsonl"

ORIGINAL_RATIO = 0.55
REWRITE_RATIO = 0.25
HYDE_RATIO = 0.20

TRUNCATE_LEN = 2000
MIN_TEXT_LEN = 10


def _load_jsonl_dict(path: Path, key_field: str, value_field: str) -> dict[str, str]:
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("qid", "")
            val = obj.get(value_field, "")
            if qid and val and val.strip():
                result[qid] = val.strip()
    logger.info(f"Loaded {len(result)} entries from {path.name}")
    return result


def _load_assignment(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    result: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            result[obj["qid"]] = obj["type"]
    logger.info(f"Loaded {len(result)} query assignments from {path.name}")
    return result


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


def load_qrels_for_queries(path: Path, target_qids: set[str]) -> dict[str, set[str]]:
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


def assign_query_types(
    qids: list[str],
    seed: int = 42,
) -> dict[str, str]:
    rng = random.Random(seed)
    shuffled = list(qids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_rewrite = int(n * REWRITE_RATIO)
    n_hyde = int(n * HYDE_RATIO)

    assignments: dict[str, str] = {}
    for i, qid in enumerate(shuffled):
        if i < n_rewrite:
            assignments[qid] = "rewrite"
        elif i < n_rewrite + n_hyde:
            assignments[qid] = "hyde"
        else:
            assignments[qid] = "original"

    counts = {
        "original": sum(1 for v in assignments.values() if v == "original"),
        "rewrite": sum(1 for v in assignments.values() if v == "rewrite"),
        "hyde": sum(1 for v in assignments.values() if v == "hyde"),
    }
    logger.info(
        f"Query type assignment: original={counts['original']} "
        f"({100*counts['original']/n:.1f}%), "
        f"rewrite={counts['rewrite']} "
        f"({100*counts['rewrite']/n:.1f}%), "
        f"hyde={counts['hyde']} "
        f"({100*counts['hyde']/n:.1f}%)"
    )
    return assignments


def resolve_query_text(
    qid: str,
    original_text: str,
    assignment: str,
    rewritten_map: dict[str, str],
    hyde_map: dict[str, str],
) -> tuple[str, str]:
    if assignment == "rewrite":
        text = rewritten_map.get(qid)
        if text:
            return text, "rewrite"
        return original_text, "original(fallback)"

    if assignment == "hyde":
        text = hyde_map.get(qid)
        if text:
            return text, "hyde"
        return original_text, "original(fallback)"

    return original_text, "original"


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Merge original, rewritten, and HyDE training data"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Sample first N queries for testing (0 = full). In random mode, "
             "ignored if assignment file from generate_training_augmentations exists."
    )
    parser.add_argument(
        "--mode", choices=["random", "read"], default="random",
        help="Merge mode: random=55/25/20 ratio assignment (default), "
             "read=use best available augmentation per qid (rewrite > hyde > original)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for query type assignment (default: 42)"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSONL path (default: auto based on --sample)"
    )
    parser.add_argument(
        "--rewrite-file", type=Path, default=None,
        help="Path to rewritten queries JSONL (default: auto)"
    )
    parser.add_argument(
        "--hyde-file", type=Path, default=None,
        help="Path to HyDE answers JSONL (default: auto)"
    )
    args = parser.parse_args()

    is_sample = args.sample > 0
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_SAMPLE_OUTPUT if is_sample else DEFAULT_OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rewrite_file = args.rewrite_file or REWRITE_FILE
    hyde_file = args.hyde_file or HYDE_FILE

    logger.info("=" * 60)
    logger.info(f"  PHASE 2: Merge Training Data")
    logger.info(f"  Mode:       {args.mode}")
    logger.info(f"  Rewrite:    {rewrite_file}")
    logger.info(f"  HyDE:       {hyde_file}")
    logger.info(f"  Output:     {output_path}")

    if args.mode == "random":
        pre_assigned = _load_assignment(ASSIGNMENT_FILE)
    else:
        pre_assigned = None

    if pre_assigned is not None:
        logger.info("  Assignment: from generate_training_augmentations.py")
    elif args.mode == "random":
        logger.info(f"  Assignment: local 55/25/20 (seed={args.seed})")
    if is_sample:
        logger.info(f"  Sample:     {args.sample}")
    logger.info("=" * 60)

    all_queries = load_queries_dict(QUERIES_TRAIN_FILE, max_queries=0)

    if pre_assigned is not None:
        queries = {qid: text for qid, text in all_queries.items() if qid in pre_assigned}
        assignments = pre_assigned
    elif is_sample:
        queries = load_queries_dict(QUERIES_TRAIN_FILE, max_queries=args.sample)
        if args.mode == "random":
            assignments = assign_query_types(list(queries.keys()), seed=args.seed)
        else:
            assignments = {}
    else:
        queries = all_queries
        if args.mode == "random":
            assignments = assign_query_types(list(queries.keys()), seed=args.seed)
        else:
            assignments = {}

    if not queries:
        logger.error("No queries loaded")
        return 1

    qids = list(queries.keys())

    rewritten_map = _load_jsonl_dict(rewrite_file, "qid", "rewritten")
    hyde_map = _load_jsonl_dict(hyde_file, "qid", "hyde")

    rewrite_available = sum(1 for q in qids if q in rewritten_map)
    hyde_available = sum(1 for q in qids if q in hyde_map)
    logger.info(
        f"Augmentation coverage: rewrite={rewrite_available}/{len(qids)} "
        f"({100*rewrite_available/max(1,len(qids)):.1f}%), "
        f"hyde={hyde_available}/{len(qids)} "
        f"({100*hyde_available/max(1,len(qids)):.1f}%)"
    )

    qrels = load_qrels_for_queries(QRELS_RETRIEVAL_TRAIN_FILE, set(qids))
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
    fallback_count = 0
    type_counts = {"original": 0, "rewrite": 0, "hyde": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for qid in qids:
            original_text = queries[qid]

            if args.mode == "read":
                rewrite_text = rewritten_map.get(qid)
                hyde_text = hyde_map.get(qid)
                if rewrite_text:
                    query_text, actual_type = rewrite_text, "rewrite"
                elif hyde_text:
                    query_text, actual_type = hyde_text, "hyde"
                else:
                    query_text, actual_type = original_text, "original"
            else:
                assignment = assignments.get(qid, "original")
                query_text, actual_type = resolve_query_text(
                    qid, original_text, assignment, rewritten_map, hyde_map
                )
                if actual_type.endswith("(fallback)"):
                    fallback_count += 1

            base_type = actual_type.split("(")[0]
            type_counts[base_type] = type_counts.get(base_type, 0) + 1

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

    total_pairs_estimated = sum(len(qrels.get(q, set())) for q in qids)
    logger.info("-" * 60)
    logger.info(f"Output pairs:       {written}")
    logger.info(f"Skipped (missing):  {skipped_missing} (pid not in collection)")
    logger.info(f"Skipped (no pos):   {skipped_query} (query has no qrels)")
    if args.mode != "read":
        logger.info(f"Fallback to orig:   {fallback_count} (assigned augmentation not available)")
    logger.info(f"Final distribution:")
    logger.info(f"  original:          {type_counts['original']} queries")
    logger.info(f"  rewrite:           {type_counts['rewrite']} queries")
    logger.info(f"  hyde:              {type_counts['hyde']} queries")
    logger.info(f"Output file:        {output_path}")
    logger.info("=" * 60)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.1f} MB")

    print()
    print("=" * 60)
    print("  PHASE 2 DATA MERGE COMPLETE")
    print("=" * 60)
    print(f"  Mode:      {args.mode}")
    if args.mode == "random":
        ratio_orig = 100 * type_counts["original"] / max(1, len(qids))
        ratio_rewrite = 100 * type_counts["rewrite"] / max(1, len(qids))
        ratio_hyde = 100 * type_counts["hyde"] / max(1, len(qids))
        print(f"  Original:  {type_counts['original']} ({ratio_orig:.1f}%)")
        print(f"  Rewrite:   {type_counts['rewrite']} ({ratio_rewrite:.1f}%)")
        print(f"  HyDE:      {type_counts['hyde']} ({ratio_hyde:.1f}%)")
    else:
        print(f"  Rewrite avail:  {rewrite_available}/{len(qids)} queries")
        print(f"  HyDE avail:     {hyde_available}/{len(qids)} queries")
        print(f"  Final: original={type_counts['original']} rewrite={type_counts['rewrite']} hyde={type_counts['hyde']}")
    print(f"  Pairs:     {written}")
    print(f"  Output:    {output_path}")
    print(f"  Size:      {file_size_mb:.1f} MB")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
