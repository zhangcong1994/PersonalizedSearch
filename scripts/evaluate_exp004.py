"""
Exp-004: Open-Source Cross-Encoder Reranker Comparison.

Pure inference experiment: loads exp-004 prepared data (from prepare_exp004_data.py),
runs each reranker model, computes ranking metrics with NDCG.

Experiment design:
  Phase 1 (validation): 5 models fixed output K=50 -> top-3 by NDCG@10
  Phase 2 (validation): Top-3 models x 3 output depths (10/20/50) -> best config
  Phase 3 (test):        Best model + best depth on held-out test set

Usage:
  python scripts/evaluate_exp004.py
  python scripts/evaluate_exp004.py --data results/exp004/exp004_prepared_data.jsonl
  python scripts/evaluate_exp004.py --model bge-v2-m3 --split val  # single model quick test
  python scripts/evaluate_exp004.py --batch-size 64
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import MODEL_CACHE_DIR
from src.evaluation.metrics import compute_reranker_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_DATA_FILE = RESULTS_DIR / "exp004" / "exp004_prepared_data.jsonl"

MODELS = {
    "bge-base": {
        "name": "BAAI/bge-reranker-base",
        "hf_id": "BAAI/bge-reranker-base",
        "backend": "flagembedding",
        "params": "278M",
        "contamination": True,
    },
    "gte-mul": {
        "name": "Alibaba-NLP/gte-multilingual-reranker-base",
        "hf_id": "Alibaba-NLP/gte-multilingual-reranker-base",
        "backend": "transformers",
        "params": "306M",
        "contamination": False,
    },
    "bge-v2-m3": {
        "name": "BAAI/bge-reranker-v2-m3",
        "hf_id": "BAAI/bge-reranker-v2-m3",
        "backend": "flagembedding",
        "params": "568M",
        "contamination": True,
    },
    "qwen3-rerank": {
        "name": "Qwen/Qwen3-Reranker-0.6B",
        "hf_id": "Qwen/Qwen3-Reranker-0.6B",
        "backend": "transformers",
        "params": "0.6B",
        "contamination": False,
    },
    "mxbai-v2": {
        "name": "mixedbread-ai/mxbai-rerank-base-v2",
        "hf_id": "mixedbread-ai/mxbai-rerank-base-v2",
        "backend": "transformers",
        "params": "0.5B",
        "contamination": False,
    },
}

EVAL_K_VALUES = [5, 10, 20, 50]
OUTPUT_DEPTHS = [10, 20, 50]
DEFAULT_BATCH_SIZE = 32


def resolve_model_dir(hf_id: str) -> str:
    cache_dir = str(MODEL_CACHE_DIR)
    local_dir = os.path.join(cache_dir, hf_id.replace("/", "--"))
    if os.path.isdir(local_dir):
        return local_dir
    return hf_id


def load_flagembedding_reranker(model_info: dict, device: str):
    from FlagEmbedding import FlagReranker

    model_path = resolve_model_dir(model_info["hf_id"])
    logger.info(f"Loading FlagReranker: {model_info['hf_id']} from {model_path}")
    reranker = FlagReranker(
        model_path,
        use_fp16=True,
        devices=[device],
    )

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores.extend(reranker.compute_score(batch, normalize=True))
        return scores

    def cleanup():
        del reranker
        torch.cuda.empty_cache()

    return score_pairs, cleanup


def load_transformers_reranker(model_info: dict, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = resolve_model_dir(model_info["hf_id"])
    logger.info(f"Loading Transformers reranker: {model_info['hf_id']} from {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    def score_pairs(pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            with torch.no_grad():
                inputs = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)
                outputs = model(**inputs)
                logits = outputs.logits.view(-1).float()
                batch_scores = torch.sigmoid(logits).cpu().tolist()
                scores.extend(batch_scores)
        return scores

    def cleanup():
        del model
        del tokenizer
        torch.cuda.empty_cache()

    return score_pairs, cleanup


def load_reranker(model_id: str, device: str = "cuda:0"):
    model_info = MODELS[model_id]
    if model_info["backend"] == "flagembedding":
        return load_flagembedding_reranker(model_info, device)
    else:
        return load_transformers_reranker(model_info, device)


def build_metrics_entry(
    results: list[dict],
    method_key: str,
    k_values: list[int],
    qrels_graded: dict[str, dict[str, int]],
) -> dict:
    filtered = [r for r in results if method_key in r.get("retrievals", {})]
    return compute_reranker_metrics(filtered, method_key, k_values, qrels_graded)


def print_results_table(
    metrics_map: dict[str, dict],
    metric_names: list[str],
    title: str = "RESULTS",
):
    methods = list(metrics_map.keys())
    col_width = 12

    print()
    print("=" * (20 + len(methods) * (col_width + 2)))
    print(f"  {title}")
    print("=" * (20 + len(methods) * (col_width + 2)))

    header = f"  {'Metric':<20}"
    for m in methods:
        header += f" {m:>{col_width}}"
    print(header)
    print("  " + "-" * (20 + len(methods) * (col_width + 2)))

    for metric_name in metric_names:
        row = f"  {metric_name:<20}"
        for m in methods:
            val = metrics_map[m].get(metric_name, float("nan"))
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * (20 + len(methods) * (col_width + 2)))


def main():
    parser = argparse.ArgumentParser(
        description="Exp-004: Cross-Encoder Reranker Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_FILE),
        help="Path to prepared data JSONL",
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=None,
        help="Run only a single model (skip full pipeline)",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default=None,
        help="Run only on one split (default: full pipeline)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for reranker inference (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for model inference",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "exp004"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip depth ablation (Phase 2)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Single model quick mode ──
    if args.model:
        return run_single_model(args)

    # ── Full pipeline ──
    print("=" * 70)
    print("  EXP-004: CROSS-ENCODER RERANKER COMPARISON")
    print("=" * 70)
    print(f"  Data:    {args.data}")
    print(f"  Models:  {len(MODELS)} ({', '.join(MODELS.keys())})")
    print(f"  Device:  {args.device}")
    print(f"  Batch:   {args.batch_size}")

    # ── Load prepared data ──
    all_data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    logger.info(f"Loaded {len(all_data)} entries from prepared data")

    val_data = [d for d in all_data if d["split"] == "val"]
    test_data = [d for d in all_data if d["split"] == "test"]
    logger.info(f"Validation: {len(val_data)}, Test: {len(test_data)}")

    graded_qrels_val = {d["qid"]: d["graded_qrels"] for d in val_data if d["graded_qrels"]}
    graded_qrels_test = {d["qid"]: d["graded_qrels"] for d in test_data if d["graded_qrels"]}

    # =========================================================================
    # PHASE 1: Full Model Comparison
    # =========================================================================
    print()
    print("=" * 70)
    print("  PHASE 1: Model Comparison (Validation Set, output K=50)")
    print("=" * 70)
    print(f"  {len(MODELS)} models x K=50 = {len(MODELS)} configs")
    print(f"  Primary metric: NDCG@10")

    model_ids = list(MODELS.keys())

    phase1_metrics = {}
    for model_id in model_ids:
        logger.info(f"[{model_id}] Running on validation set ({len(val_data)} queries) ...")
        score_fn, cleanup_fn = load_reranker(model_id, args.device)

        t0 = time.time()
        results = run_reranker_on_split(val_data, score_fn, args.batch_size, model_id)
        elapsed = time.time() - t0

        metrics = build_metrics_entry(
            results, model_id, EVAL_K_VALUES, graded_qrels_val
        )
        phase1_metrics[model_id] = metrics
        logger.info(f"[{model_id}] Done in {elapsed:.1f}s. NDCG@10={metrics.get('NDCG@10', 0):.4f}")

        cleanup_fn()

    # ── Phase 1 table ──
    p1_metric_names = [
        "NDCG@5", "NDCG@10", "NDCG@20",
        "NDCG@10_graded", "MRR", "Recall@10", "Recall@50",
        "Precision@5", "Precision@10",
    ]
    print_results_table(phase1_metrics, p1_metric_names, "Phase 1: Model Comparison (Val, K=50)")

    # ── Select top-3 ──
    ranking = sorted(
        model_ids,
        key=lambda mid: phase1_metrics[mid].get("NDCG@10", 0),
        reverse=True,
    )
    top3 = ranking[:3]
    print()
    print("  Phase 1 Ranking (by NDCG@10):")
    for rank, mid in enumerate(ranking, 1):
        flag = " ⚠️ contamin." if MODELS[mid]["contamination"] else ""
        ndcg10 = phase1_metrics[mid].get("NDCG@10", 0)
        print(f"    #{rank}: {mid:<14}  NDCG@10={ndcg10:.4f}  {MODELS[mid]['params']:>5}{flag}")

    print()
    print(f"  Top-3 advancing to Phase 2: {', '.join(top3)}")

    # =========================================================================
    # PHASE 2: Output Depth Ablation
    # =========================================================================
    if args.skip_phase2:
        print()
        logger.info("Phase 2 skipped (--skip-phase2)")
    else:
        print()
        print("=" * 70)
        print("  PHASE 2: Output Depth Ablation (Validation Set)")
        print("=" * 70)
        print(f"  Top-3 models x {len(OUTPUT_DEPTHS)} depths = {len(top3) * len(OUTPUT_DEPTHS)} configs")

        phase2_metrics = {}
        for model_id in top3:
            logger.info(f"[{model_id}] Depth ablation ...")
            score_fn, cleanup_fn = load_reranker(model_id, args.device)

            t0 = time.time()
            results_full = run_reranker_on_split(val_data, score_fn, args.batch_size, model_id)
            elapsed = time.time() - t0

            for depth in OUTPUT_DEPTHS:
                config_key = f"{model_id}_K{depth}"
                truncated = truncate_results(results_full, depth)
                phase2_metrics[config_key] = build_metrics_entry(
                    truncated, model_id, EVAL_K_VALUES, graded_qrels_val
                )
                ndcg10 = phase2_metrics[config_key].get("NDCG@10", 0)
                logger.info(f"  [{config_key}] NDCG@10={ndcg10:.4f}")

            logger.info(f"[{model_id}] Depth ablation done in {elapsed:.1f}s")
            cleanup_fn()

        # ── Phase 2 table ──
        p2_cols = ["NDCG@5", "NDCG@10", "NDCG@20", "NDCG@10_graded", "MRR", "Recall@10", "Precision@5"]
        config_keys = [f"{mid}_K{d}" for mid in top3 for d in OUTPUT_DEPTHS]
        p2_filtered = {k: phase2_metrics[k] for k in config_keys if k in phase2_metrics}
        print_results_table(p2_filtered, p2_cols, "Phase 2: Depth Ablation (Val)")

    # =========================================================================
    # PHASE 3: Test Set Final Evaluation
    # =========================================================================
    best_model_id = top3[0]
    best_depth = 50

    if not args.skip_phase2 and phase2_metrics:
        best_ndcg = -1.0
        for mid in top3:
            for depth in OUTPUT_DEPTHS:
                ck = f"{mid}_K{depth}"
                val = phase2_metrics.get(ck, {}).get("NDCG@10", -1.0)
                if val > best_ndcg:
                    best_ndcg = val
                    best_model_id = mid
                    best_depth = depth

    print()
    print("=" * 70)
    print("  PHASE 3: Test Set Final Evaluation")
    print("=" * 70)
    print(f"  Best config: {best_model_id}_K{best_depth}")
    print(f"  Test set: {len(test_data)} queries")

    score_fn, cleanup_fn = load_reranker(best_model_id, args.device)
    t0 = time.time()
    test_results = run_reranker_on_split(test_data, score_fn, args.batch_size, best_model_id)
    test_results = truncate_results(test_results, best_depth)
    test_metrics = build_metrics_entry(
        test_results, best_model_id, EVAL_K_VALUES, graded_qrels_test
    )
    elapsed = time.time() - t0
    logger.info(f"Test inference done in {elapsed:.1f}s")

    # ── Also compute coarse-ranking (RRF) baseline on test set ──
    rrf_test_results = build_rrf_baseline(test_data)
    rrf_metrics = build_metrics_entry(
        rrf_test_results, "rrf_baseline", EVAL_K_VALUES, graded_qrels_test
    )

    # ── Test set table ──
    test_compare = {
        f"rrf_coarse": rrf_metrics,
        f"{best_model_id}_K{best_depth}": test_metrics,
    }
    test_cols = [
        "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@10_graded",
        "MRR", "Recall@10", "Recall@20", "Recall@50",
        "Precision@5", "Precision@10",
        "Hit@5", "Hit@10",
    ]
    print_results_table(test_compare, test_cols, f"Test Set: RRF (coarse) vs {best_model_id} (reranked)")

    # ── Save results ──
    metrics_save = {
        "experiment": "exp-004",
        "phase1": {mid: phase1_metrics[mid] for mid in model_ids},
        "phase1_ranking": [
            {"rank": i + 1, "model": mid, "NDCG@10": phase1_metrics[mid].get("NDCG@10", 0)}
            for i, mid in enumerate(ranking)
        ],
        "best_config": {
            "model": best_model_id,
            "output_depth": best_depth,
        },
        "test": {
            "rrf_coarse": rrf_metrics,
            "reranked": test_metrics,
        },
    }
    if not args.skip_phase2 and phase2_metrics:
        metrics_save["phase2"] = {
            ck: {k: v for k, v in metrics.items()}
            for ck, metrics in phase2_metrics.items()
        }

    metrics_path = output_dir / "exp004_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_save, f, ensure_ascii=False, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # ── Final summary ──
    print()
    print("=" * 70)
    print("  EXP-004 COMPLETE")
    print("=" * 70)
    print(f"  Best model:          {best_model_id} ({MODELS[best_model_id]['name']})")
    print(f"  Best output depth:   {best_depth}")
    print(f"  Test NDCG@10:        {test_metrics.get('NDCG@10', 0):.4f}")
    print(f"  Test NDCG@10_graded: {test_metrics.get('NDCG@10_graded', 0):.4f}")
    print(f"  Test MRR:            {test_metrics.get('MRR', 0):.4f}")
    print(f"  RRF MRR (baseline):  {rrf_metrics.get('MRR', 0):.4f}")
    mrr_gain = test_metrics.get("MRR", 0) - rrf_metrics.get("MRR", 0)
    print(f"  MRR gain:            {mrr_gain:+.4f}")
    print(f"  Results saved to:    {output_dir}")
    print("=" * 70)

    cleanup_fn()
    return 0


def run_reranker_on_split(
    data: list[dict],
    score_fn,
    batch_size: int,
    model_id: str,
) -> list[dict]:
    all_pairs = []
    query_index = []

    for entry in data:
        query = entry["query"]
        for cand in entry["candidates"]:
            if cand["text"]:
                all_pairs.append((query, cand["text"]))
                query_index.append((entry["qid"], cand["pid"]))

    logger.info(f"[{model_id}] Scoring {len(all_pairs)} query-passage pairs ...")
    scores = score_fn(all_pairs, batch_size)

    qid_scored: dict[str, list] = {}
    for (qid, pid), score in zip(query_index, scores):
        qid_scored.setdefault(qid, []).append({
            "pid": pid,
            "score": round(score, 6),
        })

    for entry in data:
        qid = entry["qid"]
        existing = {item["pid"] for item in qid_scored.get(qid, [])}
        for cand in entry["candidates"]:
            if cand["pid"] not in existing:
                qid_scored.setdefault(qid, []).append({
                    "pid": cand["pid"],
                    "score": 0.0,
                })

    results = []
    for entry in data:
        qid = entry["qid"]
        retrievals = sorted(
            qid_scored.get(qid, []),
            key=lambda x: x["score"],
            reverse=True,
        )
        for rank, item in enumerate(retrievals, 1):
            item["rank"] = rank

        results.append({
            "qid": qid,
            "query": entry["query"],
            "relevant_pids": set(entry["relevant_pids"]),
            "retrievals": {model_id: retrievals},
        })

    return results


def truncate_results(results: list[dict], max_k: int) -> list[dict]:
    truncated = []
    for r in results:
        retrievals = {}
        for key, items in r.get("retrievals", {}).items():
            retrievals[key] = items[:max_k]
        truncated.append({
            **r,
            "retrievals": retrievals,
        })
    return truncated


def build_rrf_baseline(data: list[dict]) -> list[dict]:
    results = []
    for entry in data:
        retrievals = []
        for cand in entry["candidates"]:
            retrievals.append({
                "pid": cand["pid"],
                "score": cand["rrf_score"],
                "rank": cand["rrf_rank"],
            })
        results.append({
            "qid": entry["qid"],
            "query": entry["query"],
            "relevant_pids": set(entry["relevant_pids"]),
            "retrievals": {"rrf_baseline": retrievals},
        })
    return results


def run_single_model(args) -> int:
    logger.info(f"Single model mode: {args.model}")
    split = args.split or "val"

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry["split"] == split:
                    data.append(entry)

    if not data:
        logger.error(f"No {split} data found")
        return 1

    logger.info(f"Loaded {len(data)} entries for {split}")

    score_fn, cleanup_fn = load_reranker(args.model, args.device)
    results = run_reranker_on_split(data, score_fn, args.batch_size, args.model)

    graded = {d["qid"]: d["graded_qrels"] for d in data if d["graded_qrels"]}
    metrics = build_metrics_entry(results, args.model, EVAL_K_VALUES, graded)

    metric_names = [
        "NDCG@5", "NDCG@10", "NDCG@20", "NDCG@10_graded",
        "MRR", "Recall@10", "Recall@20", "Recall@50",
        "Precision@5", "Precision@10", "Hit@5", "Hit@10",
    ]
    print()
    for mn in metric_names:
        val = metrics.get(mn, float("nan"))
        print(f"  {mn:<20} {val:.4f}")

    cleanup_fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
