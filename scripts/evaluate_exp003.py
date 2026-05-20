"""
Exp-003: Multi-Route Retrieval + RRF Fusion (Coarse-Ranking)

Pure post-processing experiment: reads per-route retrieval results from exp-002
JSONL files, applies RRF fusion with different route combinations / per-route K /
RRF k values. No re-retrieval or LLM API calls needed.

Experiment design:
  Phase 1 (validation): 6 route schemes × 3 per-route K = 18 configs (RRF k=60)
  Phase 2 (validation): Top-3 configs × 3 RRF k = 9 configs
  Phase 3 (test):        Best config on held-out test set

Usage:
  python scripts/evaluate_exp003.py
  python scripts/evaluate_exp003.py --output-dir results/exp003
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from src.evaluation.metrics import compute_metrics, get_metric_params
from src.evaluation.result_cache import load_results
from src.evaluation.data_loader import load_qrels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "t2ranking"
QRELS_FILE = RAW_DATA_DIR / "qrels.retrieval.dev.tsv"

RANDOM_SEED = 42
SPLIT_RATIO = 0.5
OUTPUT_TOP_K = 50

PER_ROUTE_K_VALUES = [20, 30, 50]
RRF_K_PHASE1 = 60
RRF_K_VALUES = [30, 60, 90]

# Route definitions: (route_id, jsonl_file, retrievals_key)
ROUTE_FILES = {
    "D-B0": "results/exp002_dense_details/exp002_E2a-B0_s2000_similarity.jsonl",
    "D-P2": "results/exp002_dense_details/exp002_E2a-P2_s2000_similarity.jsonl",
    "D-HyDE": "results/exp002_dense_details/exp002_E2c-H2_s2000_similarity.jsonl",
    "B-B0": "results/exp002_bm25_details/exp002_E2a-B0_s2000_bm25.jsonl",
    "B-P2": "results/exp002_bm25_details/exp002_E2a-P2_s2000_bm25.jsonl",
}

ROUTE_KEYS = {
    "D-B0": "similarity",
    "D-P2": "similarity",
    "D-HyDE": "hyde_sub",
    "B-B0": "bm25",
    "B-P2": "bm25",
}

EXPERIMENTS = {
    "S0": {
        "name": "baseline: Dense original query",
        "routes": ["D-B0"],
        "route_count": 1,
        "api_calls": 0,
    },
    "S1": {
        "name": "Dense original + rewrite (2 routes)",
        "routes": ["D-B0", "D-P2"],
        "route_count": 2,
        "api_calls": 1,
    },
    "S2": {
        "name": "Dense rewrite + HyDE (2 routes)",
        "routes": ["D-P2", "D-HyDE"],
        "route_count": 2,
        "api_calls": 2,
    },
    "S3": {
        "name": "Dense all 3 routes",
        "routes": ["D-B0", "D-P2", "D-HyDE"],
        "route_count": 3,
        "api_calls": 2,
    },
    "S4": {
        "name": "Dense 3 + BM25 original (4 routes)",
        "routes": ["D-B0", "D-P2", "D-HyDE", "B-B0"],
        "route_count": 4,
        "api_calls": 2,
    },
    "S5": {
        "name": "Dense 3 + BM25 dual (5 routes)",
        "routes": ["D-B0", "D-P2", "D-HyDE", "B-B0", "B-P2"],
        "route_count": 5,
        "api_calls": 2,
    },
}


def load_route_data(route_id: str, file_relpath: str) -> dict[str, dict]:
    filepath = PROJECT_ROOT / file_relpath
    results, meta = load_results(str(filepath))
    route_data = {}
    for r in results:
        route_data[r["qid"]] = r
    logger.info(f"Loaded route {route_id}: {len(route_data)} queries from {file_relpath}")
    return route_data


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


def build_result_entry(
    qid: str,
    query: str,
    relevant_pids: set,
    retrievals_map: dict[str, list[dict]],
) -> dict:
    return {
        "qid": qid,
        "query": query,
        "relevant_pids": relevant_pids,
        "retrievals": retrievals_map,
    }


def run_experiment(
    exp_id: str,
    exp_config: dict,
    route_data: dict[str, dict[str, dict]],
    qids: list[str],
    per_route_k: int,
    rrf_k: int,
) -> tuple[list[dict], dict]:
    route_ids = exp_config["routes"]
    results = []

    for qid in qids:
        ref = route_data[route_ids[0]][qid]
        query = ref["query"]
        relevant = ref["relevant_pids"]

        retrievals_map = {}

        route_retrievals = []
        for rid in route_ids:
            key = ROUTE_KEYS[rid]
            items = route_data[rid][qid]["retrievals"].get(key, [])
            route_retrievals.append(items)
            retrievals_map[f"{rid}@{per_route_k}"] = items[:per_route_k]

        fused = rrf_fuse(route_retrievals, per_route_k, rrf_k, OUTPUT_TOP_K)
        retrievals_map[f"rrf@k{rrf_k}_perK{per_route_k}"] = fused

        results.append(build_result_entry(qid, query, relevant, retrievals_map))

    return results, retrievals_map


def _find_rrf_key(metrics_map: dict[str, dict]) -> str:
    for k in metrics_map:
        if k.startswith("rrf@"):
            return k
    return list(metrics_map.keys())[0]


def compute_all_metrics(
    results: list[dict],
    top_k: int,
) -> dict[str, dict]:
    k_values, _ = get_metric_params(top_k)
    metrics = {}
    retrievals_keys = set()
    for r in results:
        retrievals_keys.update(r["retrievals"].keys())
    for key in sorted(retrievals_keys):
        metrics[key] = compute_metrics(results, key, k_values=k_values)
    return metrics


def print_phase_header(title: str, width: int = 80):
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_matrix_table(
    experiments: list[str],
    per_route_k_values: list[int],
    results_cache: dict[str, dict[str, dict]],
    metric_name: str = "Recall@50",
):
    col_width = 12
    header = f"  {'Experiment':<10}"
    for k in per_route_k_values:
        header += f" {'K=' + str(k):>{col_width}}"
    print(header)
    print("  " + "-" * (10 + len(per_route_k_values) * (col_width + 1)))

    best_configs = []
    for exp_id in experiments:
        row = f"  {exp_id:<10}"
        for perk in per_route_k_values:
            config_key = f"{exp_id}_K{perk}"
            if config_key in results_cache:
                mm = results_cache[config_key]
                primary = _find_rrf_key(mm)
                val = mm[primary].get(metric_name, 0.0)
                row += f" {val:>{col_width}.4f}"
                best_configs.append((val, exp_id, perk, config_key))
            else:
                row += f" {'--':>{col_width}}"
        print(row)

    best_configs.sort(key=lambda x: x[0], reverse=True)
    return best_configs


def main():
    parser = argparse.ArgumentParser(
        description="Exp-003: Multi-Route RRF Fusion Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "exp003"),
        help="Output directory for results",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_phase_header("Exp-003: Multi-Route Retrieval + RRF Fusion", 80)
    print("  Design: Pure post-processing (no re-retrieval, no LLM calls)")
    print(f"  Data:   T2Ranking dev, {OUTPUT_TOP_K} output top-K")
    print(f"  Split:  {int(SPLIT_RATIO*100)}/{int((1-SPLIT_RATIO)*100)} random (seed={RANDOM_SEED})")
    print(f"  Route sources: {len(ROUTE_FILES)} JSONL files from exp-002")

    # ── Load all route data ──
    print()
    logger.info("Loading route data from exp-002 JSONL files...")
    route_data: dict[str, dict[str, dict]] = {}
    for route_id, file_relpath in ROUTE_FILES.items():
        route_data[route_id] = load_route_data(route_id, file_relpath)

    # ── Align qids across all routes ──
    qid_sets = [set(d.keys()) for d in route_data.values()]
    common_qids = sorted(set.intersection(*qid_sets))
    logger.info(f"Common qids across all routes: {len(common_qids)}")

    # ── Load qrels from TSV for completeness ──
    qrels = load_qrels(QRELS_FILE)
    qids_with_qrels = sorted(set(common_qids) & set(qrels.keys()))
    logger.info(f"Qids with qrels in common: {len(qids_with_qrels)}")

    # ── Split into validation / test ──
    random.seed(RANDOM_SEED)
    random.shuffle(qids_with_qrels)
    mid = len(qids_with_qrels) // 2
    val_qids = qids_with_qrels[:mid]
    test_qids = qids_with_qrels[mid:]
    logger.info(f"Validation: {len(val_qids)} queries")
    logger.info(f"Test:       {len(test_qids)} queries")

    experiment_ids = list(EXPERIMENTS.keys())

    # =========================================================================
    # PHASE 1: Route Combination + Per-Route K Ablation
    # =========================================================================
    print_phase_header("PHASE 1: Route Combination + Per-Route K Ablation (Validation Set)")
    print(f"  Fixed RRF k = {RRF_K_PHASE1}")
    print(f"  {len(experiment_ids)} schemes × {len(PER_ROUTE_K_VALUES)} per-route K = "
          f"{len(experiment_ids) * len(PER_ROUTE_K_VALUES)} configs")
    print(f"  Primary metric: Recall@{OUTPUT_TOP_K}")

    phase1_cache: dict[str, dict[str, dict]] = {}

    for exp_id in experiment_ids:
        exp_config = EXPERIMENTS[exp_id]
        for perk in PER_ROUTE_K_VALUES:
            config_key = f"{exp_id}_K{perk}"
            logger.info(f"  [{config_key}] {exp_config['name']}, per-route K={perk}")

            results, _ = run_experiment(
                exp_id, exp_config, route_data, val_qids,
                per_route_k=perk, rrf_k=RRF_K_PHASE1,
            )
            metrics = compute_all_metrics(results, OUTPUT_TOP_K)
            phase1_cache[config_key] = metrics

    # ── Phase 1 summary table ──
    print()
    print(f"  Phase 1 Results — Recall@{OUTPUT_TOP_K} (Validation Set, RRF k={RRF_K_PHASE1})")
    best_configs = print_matrix_table(
        experiment_ids, PER_ROUTE_K_VALUES, phase1_cache, f"Recall@{OUTPUT_TOP_K}"
    )

    # ── Show top-3 to advance to Phase 2 ──
    print()
    print(f"  Top-3 configs advancing to Phase 2:")
    top3 = best_configs[:3]
    for rank, (recall, exp_id, perk, config_key) in enumerate(top3, 1):
        exp_name = EXPERIMENTS[exp_id]["name"]
        print(f"    #{rank}: {config_key}  Recall@50={recall:.4f}  ({exp_name})")

    # =========================================================================
    # PHASE 2: RRF k Ablation
    # =========================================================================
    print_phase_header("PHASE 2: RRF k Ablation (Validation Set)")
    print(f"  Top-3 configs × {len(RRF_K_VALUES)} RRF k values = "
          f"{len(top3) * len(RRF_K_VALUES)} configs")

    phase2_cache: dict[str, dict[str, dict]] = {}

    for _, exp_id, perk, _ in top3:
        exp_config = EXPERIMENTS[exp_id]
        for rrfk in RRF_K_VALUES:
            config_key = f"{exp_id}_K{perk}_RRFk{rrfk}"
            logger.info(f"  [{config_key}] {exp_config['name']}, K={perk}, RRF k={rrfk}")

            results, _ = run_experiment(
                exp_id, exp_config, route_data, val_qids,
                per_route_k=perk, rrf_k=rrfk,
            )
            metrics = compute_all_metrics(results, OUTPUT_TOP_K)
            phase2_cache[config_key] = metrics

    # ── Phase 2 summary table ──
    print()
    print(f"  Phase 2 Results — Recall@{OUTPUT_TOP_K} (Validation Set)")
    col_width = 14
    header = f"  {'Config':<22}"
    for rrfk in RRF_K_VALUES:
        header += f" {'RRF k=' + str(rrfk):>{col_width}}"
    print(header)
    print("  " + "-" * (22 + len(RRF_K_VALUES) * (col_width + 1)))

    best_overall = None
    best_recall = 0.0
    for _, exp_id, perk, _ in top3:
        row = f"  {exp_id}_K{perk:<15}"
        for rrfk in RRF_K_VALUES:
            config_key = f"{exp_id}_K{perk}_RRFk{rrfk}"
            if config_key in phase2_cache:
                mm = phase2_cache[config_key]
                primary = _find_rrf_key(mm)
                val = mm[primary].get(f"Recall@{OUTPUT_TOP_K}", 0.0)
                row += f" {val:>{col_width}.4f}"
                if val > best_recall:
                    best_recall = val
                    best_overall = (exp_id, perk, rrfk, config_key)
            else:
                row += f" {'--':>{col_width}}"
        print(row)

    if best_overall is None:
        logger.error("No best config found in Phase 2. Exiting.")
        return 1

    best_exp_id, best_perk, best_rrfk, best_config_key = best_overall
    print()
    print(f"  Best config: {best_config_key}")
    print(f"    Experiment: {best_exp_id} ({EXPERIMENTS[best_exp_id]['name']})")
    print(f"    Per-route K: {best_perk}")
    print(f"    RRF k: {best_rrfk}")
    print(f"    Validation Recall@{OUTPUT_TOP_K}: {best_recall:.4f}")

    # =========================================================================
    # PHASE 3: Test Set Final Evaluation
    # =========================================================================
    print_phase_header("PHASE 3: Test Set Final Evaluation")
    print(f"  Best config: {best_config_key}")
    print(f"  Test set: {len(test_qids)} queries")

    exp_config = EXPERIMENTS[best_exp_id]
    test_results, _ = run_experiment(
        best_exp_id, exp_config, route_data, test_qids,
        per_route_k=best_perk, rrf_k=best_rrfk,
    )
    test_metrics = compute_all_metrics(test_results, OUTPUT_TOP_K)

    # ── Test set detailed results ──
    _, metric_names = get_metric_params(OUTPUT_TOP_K)
    all_metric_names = metric_names + [f"Hit@{k}" for k in [10, 20, OUTPUT_TOP_K]] + \
                       [f"Precision@{k}" for k in [10, 20, OUTPUT_TOP_K]]

    print()
    print(f"  Test Set Results — {best_config_key}")
    print(f"  {'Metric':<18}", end="")
    for key in sorted(test_metrics.keys()):
        print(f" {key:>16}", end="")
    print()
    print("  " + "-" * (18 + len(test_metrics) * 17))

    for mname in all_metric_names:
        print(f"  {mname:<18}", end="")
        for key in sorted(test_metrics.keys()):
            val = test_metrics[key].get(mname, float("nan"))
            print(f" {val:>16.4f}", end="")
        print()

    # ── Save results ──
    test_save_path = output_dir / f"exp003_test_{best_config_key}.jsonl"
    meta = {
        "experiment_id": "exp-003",
        "best_config": best_config_key,
        "experiment_scheme": best_exp_id,
        "per_route_k": best_perk,
        "rrf_k": best_rrfk,
        "output_top_k": OUTPUT_TOP_K,
        "split": "test",
        "num_queries": len(test_qids),
        "random_seed": RANDOM_SEED,
        "dataset": "T2Ranking dev",
    }
    lines = [json.dumps({"__meta__": meta}, ensure_ascii=False)]
    for r in test_results:
        line = {
            "qid": r["qid"],
            "query": r["query"],
            "relevant_pids": sorted(r["relevant_pids"]),
        }
        for key, pids in r.get("retrievals", {}).items():
            line[key] = pids
        lines.append(json.dumps(line, ensure_ascii=False))
    with open(test_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Test results saved to: {test_save_path}")

    test_metrics_path = output_dir / f"exp003_test_{best_config_key}_metrics.json"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "metrics": test_metrics}, f, ensure_ascii=False, indent=2)
    logger.info(f"Test metrics saved to: {test_metrics_path}")

    # ── Also save Phase 1 & 2 validation results for reference ──
    val_summary = {
        "phase1": {
            config_key: {k: v for k, v in metrics.items()}
            for config_key, metrics in phase1_cache.items()
        },
        "phase2": {
            config_key: {k: v for k, v in metrics.items()}
            for config_key, metrics in phase2_cache.items()
        },
        "top3_configs": [
            {"rank": i + 1, "config": ck, f"Recall@{OUTPUT_TOP_K}": r}
            for i, (r, _, _, ck) in enumerate(top3)
        ],
    }
    val_summary_path = output_dir / "exp003_validation_summary.json"
    with open(val_summary_path, "w", encoding="utf-8") as f:
        json.dump(val_summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Validation summary saved to: {val_summary_path}")

    # ── Final summary ──
    print()
    print("=" * 80)
    print("  EXP-003 COMPLETE")
    print("=" * 80)
    print(f"  Best config:        {best_config_key}")
    print(f"  Per-route K:        {best_perk}")
    print(f"  RRF k:              {best_rrfk}")
    print(f"  Val Recall@{OUTPUT_TOP_K}:    {best_recall:.4f}")
    test_r50 = test_metrics[_find_rrf_key(test_metrics)].get(
        f"Recall@{OUTPUT_TOP_K}", 0.0
    )
    print(f"  Test Recall@{OUTPUT_TOP_K}:   {test_r50:.4f}")
    print(f"  Results saved to:    {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
