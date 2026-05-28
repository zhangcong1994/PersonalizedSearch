"""
Evaluate fine-tuned embedding models for exp-007.

Loads pre-built ChromaDB vector indexes and runs retrieval evaluation
on T2Ranking dev set. Reports Recall@k, MRR, NDCG side-by-side with deltas.

Index convention (same as build_t2ranking_index.py):
  {VECTOR_DB_DIR}/t2ranking/{model_short_name}/

Usage:
  # Quick sample with auto-derived index paths
  python scripts/exp007/evaluate_embedding.py --sample 500 --device cpu --offline

  # Full evaluation (needs GPU)
  python scripts/exp007/evaluate_embedding.py --device cuda --offline

  # Explicit index paths
  python scripts/exp007/evaluate_embedding.py \
      --baseline-vector-db data/vector_db/t2ranking/m3e-base \
      --vector-db data/vector_db/t2ranking/m3e-base-t2ranking-phase1 \
      --device cuda --offline
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    RAW_DATA_DIR, VECTOR_DB_DIR, DATA_ROOT,
    resolve_model_local_path, model_short_name,
)
from src.evaluation.data_loader import load_queries, load_qrels, load_qrels_graded
from src.evaluation.metrics import compute_reranker_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_FILE = T2RANKING_DIR / "queries.dev.tsv"
QRELS_FILE = T2RANKING_DIR / "qrels.retrieval.dev.tsv"
QRELS_GRADED_FILE = T2RANKING_DIR / "qrels.dev.tsv"
COLLECTION_NAME = "t2ranking_passages"

DEFAULT_BASELINE_MODEL = "moka-ai/m3e-base"
DEFAULT_FINETUNED_MODEL = "models/m3e-base-t2ranking-phase1"

EVAL_K_VALUES = [10, 20, 50]


def _resolve_index_path(model_id: str) -> Path:
    return VECTOR_DB_DIR / "t2ranking" / model_short_name(model_id)


def _resolve_model_path(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    local_path = resolve_model_local_path(model_id_or_path)
    if local_path is not None:
        resolved = str(local_path)
        logger.info(f"Resolved {model_id_or_path} → {resolved}")
        return resolved

    relative_to_data_root = (DATA_ROOT / model_id_or_path).resolve()
    if relative_to_data_root.is_dir():
        resolved = str(relative_to_data_root)
        logger.info(f"Resolved via DATA_ROOT: {model_id_or_path} → {resolved}")
        return resolved

    if not os.path.isabs(model_id_or_path):
        candidate = (Path.cwd() / model_id_or_path).resolve()
        if candidate.is_dir():
            resolved = str(candidate)
            logger.info(f"Resolved via CWD: {model_id_or_path} → {resolved}")
            return resolved

    return model_id_or_path


def load_index(model_id_or_path: str, vector_db_dir: str, device: str):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    index_path = Path(vector_db_dir)
    if not index_path.is_dir():
        raise FileNotFoundError(
            f"Vector index not found: {index_path}\n"
            f"  Build it first: python scripts/build_t2ranking_index.py "
            f"--model {model_id_or_path} --device {device}"
        )

    chroma_sqlite = index_path / "chroma.sqlite3"
    if not chroma_sqlite.exists():
        raise FileNotFoundError(
            f"Vector index appears incomplete (missing chroma.sqlite3): {chroma_sqlite}\n"
            f"  Rebuild it: python scripts/build_t2ranking_index.py "
            f"--model {model_id_or_path} --device {device} --rebuild"
        )

    model_path = _resolve_model_path(model_id_or_path)
    logger.info(f"Embedding model: {model_path}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(index_path),
    )
    count = vs._collection.count()
    logger.info(f"Vector index loaded: {count:,} docs in '{COLLECTION_NAME}'")

    if count == 0:
        raise RuntimeError(
            f"Vector index at {index_path} is empty (0 documents). Rebuild it."
        )

    return vs, count


def evaluate_model(
    vs,
    queries: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    qrels_graded: dict[str, dict[str, int]],
    model_name: str,
    top_k: int = 50,
) -> tuple[list[dict], float]:
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    results = []
    total_time = 0.0

    logger.info(f"Retrieving top-{top_k} for {len(queries)} queries...")

    for i, (qid, query_text) in enumerate(queries):
        t0 = time.time()
        docs = retriever.invoke(query_text)
        elapsed = time.time() - t0
        total_time += elapsed

        relevant = qrels.get(qid, set())
        retrievals = []
        for rank, doc in enumerate(docs, 1):
            pid = doc.metadata.get("pid", "?")
            retrievals.append({"pid": pid, "rank": rank})

        results.append({
            "qid": qid,
            "relevant_pids": relevant,
            "retrievals": {model_name: retrievals},
        })

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i + 1}/{len(queries)} "
                f"({total_time / (i + 1) * 1000:.0f}ms/q)"
            )

    avg_time = total_time / len(queries) if queries else 0
    logger.info(f"  done: {total_time:.1f}s total ({avg_time * 1000:.0f}ms/q)")

    metrics = compute_reranker_metrics(
        results, model_name, k_values=EVAL_K_VALUES, qrels_graded=qrels_graded
    )
    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned embedding models for exp-007"
    )
    parser.add_argument(
        "--model", default=DEFAULT_FINETUNED_MODEL,
        help="Fine-tuned model path or HF ID"
    )
    parser.add_argument(
        "--vector-db", default=None,
        help="Vector DB dir for fine-tuned model "
             "(default: {VECTOR_DB_DIR}/t2ranking/{model_short_name})"
    )
    parser.add_argument(
        "--baseline-model", default=DEFAULT_BASELINE_MODEL,
        help="Pretrained baseline model HF ID"
    )
    parser.add_argument(
        "--baseline-vector-db", default=None,
        help="Vector DB dir for baseline model "
             "(default: {VECTOR_DB_DIR}/t2ranking/{model_short_name})"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for model inference (cpu, cuda, cuda:0)"
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Evaluate on first N queries (0 = all ~24K)"
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Evaluate baseline model only"
    )
    parser.add_argument(
        "--finetuned-only", action="store_true",
        help="Evaluate fine-tuned model only"
    )
    parser.add_argument(
        "--no-graded-ndcg", action="store_true",
        help="Skip graded NDCG computation (faster)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Use HF offline mode"
    )
    args = parser.parse_args()

    do_baseline = not args.finetuned_only
    do_finetuned = not args.baseline_only

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    baseline_index_dir = args.baseline_vector_db or str(_resolve_index_path(args.baseline_model))
    finetuned_index_dir = args.vector_db or str(_resolve_index_path(args.model))

    logger.info("=" * 60)
    logger.info("  EXP-007 Embedding Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Data root: {DATA_ROOT}")
    logger.info(f"  Vector DB base: {VECTOR_DB_DIR}")
    if do_baseline:
        logger.info(f"  Baseline model:  {args.baseline_model}")
        logger.info(f"  Baseline index:  {baseline_index_dir}")
    if do_finetuned:
        logger.info(f"  Finetuned model: {args.model}")
        logger.info(f"  Finetuned index: {finetuned_index_dir}")
    logger.info(f"  Device:          {args.device}")
    logger.info(f"  Sample:          {args.sample if args.sample > 0 else 'all'}")
    logger.info(f"  K values:        {EVAL_K_VALUES}")
    logger.info(f"  Collection:      {COLLECTION_NAME}")
    logger.info("-" * 60)

    logger.info("Loading evaluation data...")
    all_queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)
    qrels_graded = {}
    if not args.no_graded_ndcg and QRELS_GRADED_FILE.exists():
        qrels_graded = load_qrels_graded(QRELS_GRADED_FILE)

    queries_with_qrels = [(qid, text) for qid, text in all_queries if qid in qrels]
    if args.sample > 0:
        queries_with_qrels = queries_with_qrels[:args.sample]
    logger.info(f"Queries with qrels: {len(queries_with_qrels)}")

    eval_tasks = []
    if do_baseline:
        eval_tasks.append(("pretrained", args.baseline_model, baseline_index_dir))
    if do_finetuned:
        eval_tasks.append(("finetuned", args.model, finetuned_index_dir))

    all_metrics = {}

    for tag, model_id, index_dir in eval_tasks:
        print()
        logger.info("=" * 60)
        logger.info(f"  Evaluating: {tag} ({model_id})")
        logger.info(f"  Index:      {index_dir}")
        logger.info("=" * 60)

        vs, count = load_index(model_id, index_dir, device=args.device)

        _, metrics = evaluate_model(
            vs, queries_with_qrels, qrels, qrels_graded, tag,
        )
        all_metrics[tag] = metrics

    print()
    print("=" * 80)
    print("  RESULTS: Pretrained vs Fine-tuned")
    print("=" * 80)

    metric_names = [
        "Recall@10", "Recall@20", "Recall@50",
        "MRR", "NDCG@10", "Hit@10", "Hit@50",
    ]
    if qrels_graded:
        metric_names.append("NDCG@10_graded")

    col_w = 12
    methods = list(all_metrics.keys())

    header = f"  {'Metric':<18}"
    for m in methods:
        header += f" {m:>{col_w}}"
    if len(methods) == 2:
        header += f" {'delta':>{col_w}}"
    print(header)
    print("  " + "-" * (18 + len(methods) * (col_w + 2) + (col_w + 2 if len(methods) == 2 else 0)))

    for metric_name in metric_names:
        row = f"  {metric_name:<18}"
        vals = []
        for m in methods:
            val = all_metrics[m].get(metric_name, float("nan"))
            vals.append(val)
            row += f" {val:>{col_w}.4f}"
        if len(vals) == 2:
            delta = vals[1] - vals[0]
            sign = "+" if delta >= 0 else ""
            row += f" {sign}{delta:>{col_w-1}.4f}"
        print(row)

    print("=" * 80)

    if len(methods) == 2:
        print()
        logger.info("Relative improvement:")
        for mn in ["Recall@10", "Recall@50", "MRR"]:
            if mn in all_metrics.get("pretrained", {}) and mn in all_metrics.get("finetuned", {}):
                base = all_metrics["pretrained"][mn]
                ft = all_metrics["finetuned"][mn]
                pct = (ft - base) / base * 100 if base > 0 else 0
                sign = "+" if pct >= 0 else ""
                logger.info(f"  {mn}: {base:.4f} → {ft:.4f} ({sign}{pct:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
