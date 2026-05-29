"""
Phase 0: Zero-Hit Query Analysis for exp-006

Loads exp-003 test set results, finds queries where the RRF top-50
contains zero relevant passages (Hit@50=0), and outputs a formatted
report with passage snippets for manual failure analysis.

Usage:
    python scripts/analyze_zero_hit.py --results results/exp003/exp003_test_S4_K50_RRFk60.jsonl
    python scripts/analyze_zero_hit.py --results results/exp003/exp003_test_S4_K50_RRFk60.jsonl --collection data/raw/t2ranking/collection.tsv
    python scripts/analyze_zero_hit.py --results results/exp003/exp003_test_S4_K50_RRFk60.jsonl --n 30
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional


def _extract_pids(retrieved_items: list) -> list[str]:
    if not retrieved_items:
        return []
    if isinstance(retrieved_items[0], dict):
        return [item["pid"] for item in retrieved_items]
    return retrieved_items

PASSAGE_TRUNCATE = 120
RRF_KEY = "rrf@k60_perK50"
OUTPUT_TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-Hit Query Analysis for exp-006",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to exp-003 test set JSONL results file",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Path to collection.tsv (optional, for passage text lookup)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Number of zero-hit queries to sample for detailed output (0 = all)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save zero-hit queries JSON (default: results/exp006/zero_hit_queries.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to write full textual report (avoids terminal truncation)",
    )
    return parser.parse_args()


def load_results_jsonl(path: str) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "__meta__" in obj:
                continue
            relevant_pids = set(obj.pop("relevant_pids", []))
            retrievals = {}
            qid = obj.pop("qid")
            query = obj.pop("query", "")
            for key in list(obj.keys()):
                retrievals[key] = obj.pop(key)
            results.append({
                "qid": qid,
                "query": query,
                "relevant_pids": relevant_pids,
                "retrievals": retrievals,
            })
    print(f"Loaded {len(results)} results from {path}")
    return results


def load_collection(path: str) -> dict[str, str]:
    pid_to_text = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) >= 2:
                pid, text = parts[0], parts[1]
                pid_to_text[pid] = text
    print(f"Loaded {len(pid_to_text)} passages from collection")
    return pid_to_text


def find_zero_hit_queries(results: list[dict]) -> list[dict]:
    zero_hit = []
    for r in results:
        rrf_items = r["retrievals"].get(RRF_KEY, [])
        rrf_pids = set(_extract_pids(rrf_items)[:OUTPUT_TOP_K])
        if not (rrf_pids & r["relevant_pids"]):
            zero_hit.append(r)
    return zero_hit


def classify_failure(r: dict, pid_to_text: dict[str, str]) -> str:
    query = r["query"]
    query_len = len(query)
    rrf_items = r["retrievals"].get(RRF_KEY, [])[:10]
    num_qrels = len(r["relevant_pids"])

    hints = []
    if query_len <= 10:
        hints.append("candidate: too_short")
    if num_qrels <= 2:
        hints.append("candidate: sparse_qrels")

    preview_texts = []
    for item in rrf_items:
        pid = item["pid"] if isinstance(item, dict) else item
        text = pid_to_text.get(pid, "")
        if text:
            preview_texts.append(text[:80])

    if not hints:
        hints.append("unknown")

    return " | ".join(hints)


def print_zero_hit_report(
    zero_hit: list[dict],
    pid_to_text: dict[str, str],
    n_sample: int,
    seed: int,
    report_path: Optional[str] = None,
) -> None:
    total = len(zero_hit)

    if n_sample > 0 and n_sample < total:
        random.seed(seed)
        sampled = random.sample(zero_hit, n_sample)
    else:
        sampled = zero_hit
        n_sample = len(sampled)

    lines: list[str] = []
    to_file_only = report_path is not None

    def emit(s: str = "") -> None:
        if not to_file_only:
            print(s)
        lines.append(s)

    emit()
    emit("=" * 75)
    emit(f"  Zero-Hit Query Analysis (exp-003 S4 test set)")
    emit("=" * 75)
    emit(f"  Total zero-hit queries (Hit@50=0):     {total}")
    emit(f"  Sampled for detailed output:          {n_sample}")
    emit("=" * 75)

    if total == 0:
        emit()
        emit("  No zero-hit queries found.")
        _write_report(lines, report_path)
        return

    emit()
    emit("=" * 75)
    emit("  DETAILED ZERO-HIT QUERIES")
    emit("=" * 75)

    for i, r in enumerate(sampled, 1):
        query = r["query"]
        relevant_pids = r["relevant_pids"]
        rrf_items = r["retrievals"].get(RRF_KEY, [])[:10]
        rrf_pids = _extract_pids(rrf_items)

        emit(f"\n{'─' * 75}")
        emit(f"  [{i}/{n_sample}]  QID: {r['qid']}")
        emit(f"{'─' * 75}")
        emit(f"  Query:         {query}")
        emit(f"  Query length:  {len(query)} chars")
        emit(f"  Qrels count:   {len(relevant_pids)} relevant passages")

        if pid_to_text:
            emit(f"\n  Relevant passage previews:")
            for j, pid in enumerate(sorted(relevant_pids)[:5], 1):
                text = pid_to_text.get(pid, "(not in collection)")
                if len(text) > PASSAGE_TRUNCATE:
                    text = text[:PASSAGE_TRUNCATE] + "..."
                emit(f"    [{j}] {pid}: {text}")

        emit(f"\n  RRF Top-10 retrieved passages:")
        for j, (pid, item) in enumerate(zip(rrf_pids[:10], rrf_items[:10]), 1):
            score = item.get("score", "?") if isinstance(item, dict) else "?"
            if pid_to_text:
                text = pid_to_text.get(pid, "(not in collection)")
                if len(text) > PASSAGE_TRUNCATE:
                    text = text[:PASSAGE_TRUNCATE] + "..."
            else:
                text = "(collection.tsv not provided)"
            emit(f"    Rank {j:>2}: {pid}  score={score}")
            if pid_to_text:
                emit(f"             {text}")

        rrf_top10_pids = set(_extract_pids(rrf_items))
        emit(f"\n  RRF Top-10 hits in qrels: {len(rrf_top10_pids & relevant_pids)}")
        emit(f"  Failure hint: {classify_failure(r, pid_to_text)}")

    emit(f"\n{'=' * 75}")
    emit(f"  End of zero-hit report ({n_sample} queries shown)")
    emit(f"{'=' * 75}")

    _write_report(lines, report_path)


def _write_report(lines: list[str], report_path: Optional[str]) -> None:
    if report_path:
        out = Path(report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nFull report saved to: {report_path}")


def save_zero_hit_results(
    zero_hit: list[dict],
    output_path: str,
) -> None:
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for r in zero_hit:
        rrf_items = r["retrievals"].get(RRF_KEY, [])[:10]
        records.append({
            "qid": r["qid"],
            "query": r["query"],
            "relevant_pids": sorted(r["relevant_pids"]),
            "num_qrels": len(r["relevant_pids"]),
            "rrf_top10_pids": [
                item["pid"] if isinstance(item, dict) else item
                for item in rrf_items
            ],
            "query_length": len(r["query"]),
        })

    summary = {
        "total_zero_hit": len(zero_hit),
        "source": "exp-003 S4 test set, RRF k=60 per-route K=50",
        "rrf_key": RRF_KEY,
        "queries": records,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nZero-hit queries saved to: {output_path}")


def print_summary_stats(zero_hit: list[dict]) -> None:
    if not zero_hit:
        return

    qlen_dist = defaultdict(int)
    qrels_dist = defaultdict(int)
    for r in zero_hit:
        qlen = len(r["query"])
        if qlen <= 10:
            qlen_dist["1-10 chars"] += 1
        elif qlen <= 20:
            qlen_dist["11-20 chars"] += 1
        elif qlen <= 40:
            qlen_dist["21-40 chars"] += 1
        else:
            qlen_dist["40+ chars"] += 1

        nq = len(r["relevant_pids"])
        if nq == 1:
            qrels_dist["1"] += 1
        elif nq <= 3:
            qrels_dist["2-3"] += 1
        elif nq <= 10:
            qrels_dist["4-10"] += 1
        else:
            qrels_dist["10+"] += 1

    print(f"\n{'─' * 75}")
    print(f"  SUMMARY STATISTICS")
    print(f"{'─' * 75}")
    print(f"  Query length distribution:")
    for bucket in ["1-10 chars", "11-20 chars", "21-40 chars", "40+ chars"]:
        print(f"    {bucket:<15}: {qlen_dist.get(bucket, 0)}")
    print(f"  Qrels count distribution:")
    for bucket in ["1", "2-3", "4-10", "10+"]:
        print(f"    {bucket:<15}: {qrels_dist.get(bucket, 0)}")


def main() -> None:
    args = parse_args()

    results = load_results_jsonl(args.results)

    pid_to_text = {}
    if args.collection:
        pid_to_text = load_collection(args.collection)

    zero_hit = find_zero_hit_queries(results)
    print(f"Found {len(zero_hit)} zero-hit queries (Hit@50=0)")

    print_summary_stats(zero_hit)

    n_sample = args.n if args.n > 0 else len(zero_hit)
    print_zero_hit_report(zero_hit, pid_to_text, n_sample, args.seed, report_path=args.report)

    output_path = args.output
    if output_path is None:
        output_path = str(Path(__file__).resolve().parent.parent / "results" / "exp006" / "zero_hit_queries.json")
    save_zero_hit_results(zero_hit, output_path)


if __name__ == "__main__":
    main()
