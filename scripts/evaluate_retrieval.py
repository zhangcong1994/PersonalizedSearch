import os
import sys
import re
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "data" / "raw" / "t2ranking" / "queries.dev.tsv"
QRELS_FILE = PROJECT_ROOT / "data" / "raw" / "t2ranking" / "qrels.retrieval.dev.tsv"
COLLECTION_FILE = PROJECT_ROOT / "data" / "raw" / "t2ranking" / "collection.tsv"

HTML_RE = re.compile(r"<[^>]*>")
TRUNCATE_LEN = 2000
MIN_TEXT_LEN = 10


def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = text.strip()
    return text


def load_queries(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    logger.info(f"Loaded {len(pairs)} queries from {path.name}")
    return pairs


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                qid, pid = parts[0], parts[1]
                qrels.setdefault(qid, set()).add(pid)
    logger.info(f"Loaded qrels: {len(qrels)} queries, {sum(len(v) for v in qrels.values())} pairs")
    return qrels


def load_passages(path: Path, max_passages: int = 0) -> tuple[list[str], list[str]]:
    pids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            text = clean_text(text)
            if len(text) < MIN_TEXT_LEN:
                continue
            if len(text) > TRUNCATE_LEN:
                text = text[:TRUNCATE_LEN]
            pids.append(pid)
            texts.append(text)
            if max_passages > 0 and len(pids) >= max_passages:
                break
    logger.info(f"Loaded {len(pids)} passages")
    return pids, texts


def build_bm25(texts: list[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), tokenized


def load_dense_retriever(vector_db_dir: str, collection_name: str, device: str = "cpu"):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    local_path = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
    model_name = str(local_path.resolve()) if local_path.is_dir() else "BAAI/bge-small-zh-v1.5"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=vector_db_dir,
    )
    count = vs._collection.count()
    logger.info(f"Dense retriever loaded: {count:,} docs in '{collection_name}'")

    return vs.as_retriever(search_type="similarity", search_kwargs={"k": 10}), count


def compute_metrics(results: list[dict], k_values: list[int] = None):
    if k_values is None:
        k_values = [1, 3, 5, 10]

    metrics = {}
    for k in k_values:
        recalls = []
        precisions = []
        reciprocal_ranks = []
        hits = 0

        for r in results:
            retrieved_pids = r["retrieved_pids"][:k]
            relevant = r["relevant_pids"]

            hits_in_k = sum(1 for pid in retrieved_pids if pid in relevant)
            recalls.append(hits_in_k / len(relevant) if relevant else 0.0)
            precisions.append(hits_in_k / k if k > 0 else 0.0)

            rr = 0.0
            for rank, pid in enumerate(retrieved_pids, 1):
                if pid in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            if hits_in_k > 0:
                hits += 1

        n = len(results) if results else 1
        metrics[f"Recall@{k}"] = sum(recalls) / n
        metrics[f"Precision@{k}"] = sum(precisions) / n
        metrics[f"Hit@{k}"] = hits / n
        if k == max(k_values):
            metrics["MRR"] = sum(reciprocal_ranks) / n

    return metrics


def evaluate(
    sample_size: int = 500,
    top_k: int = 10,
    vector_db_dir: str = None,
    collection_name: str = "t2ranking_passages",
    device: str = "cpu",
):
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    if sample_size > len(queries):
        sample_size = len(queries)
    sampled = queries[:sample_size]

    valid_queries = [(qid, text) for qid, text in sampled if qid in qrels and qrels[qid]]
    logger.info(f"Sampled queries with qrels: {len(valid_queries)} / {sample_size}")

    max_pid = 0
    for rel_set in qrels.values():
        for pid_str in rel_set:
            try:
                max_pid = max(max_pid, int(pid_str))
            except ValueError:
                pass
    logger.info(f"Max relevant pid in sampled qrels: {max_pid}")

    print()
    print("=" * 60)
    print("  Loading passages & building BM25...")
    print("=" * 60)
    t0 = time.time()
    pids, texts = load_passages(COLLECTION_FILE, max_passages=150000)
    load_time = time.time() - t0
    logger.info(f"Passage loading: {load_time:.1f}s")

    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    pool_pids = set(pids)

    pool_queries = []
    for qid, text in valid_queries:
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        missing = relevant - pool_pids
        if not missing:
            pool_queries.append((qid, text))
        elif len(relevant - missing) > 0:
            pool_queries.append((qid, text))
    logger.info(
        f"Queries with relevant passages in pool: {len(pool_queries)} / {len(valid_queries)}"
    )

    if not pool_queries:
        print()
        print("WARNING: No queries have relevant passages in the loaded pool!")
        print("Try loading more passages (--passages N) or rebuilding the vector DB.")
        return None, None

    t0 = time.time()
    bm25, tokenized = build_bm25(texts)
    bm25_time = time.time() - t0
    logger.info(f"BM25 build: {bm25_time:.1f}s ({len(pids)} docs)")

    print()
    print("=" * 60)
    print("  Loading Dense Retriever...")
    print("=" * 60)
    t0 = time.time()
    if vector_db_dir is None:
        vector_db_dir = str(PROJECT_ROOT / "data" / "vector_db" / "t2ranking" / "bge-small-zh-v1.5")
    dense_retriever, dense_count = load_dense_retriever(
        vector_db_dir, collection_name, device
    )
    dense_load_time = time.time() - t0
    logger.info(f"Dense retriever load: {dense_load_time:.1f}s")

    print()
    print("=" * 60)
    print(f"  Running BM25 vs Dense evaluation")
    print(f"  Queries: {len(pool_queries)} | Passages: {len(pids)}")
    print("=" * 60)

    bm25_results = []
    dense_results = []

    bm25_total_time = 0.0
    dense_total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())

        t0 = time.time()
        tokenized_query = query_text.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_elapsed = time.time() - t0
        bm25_total_time += bm25_elapsed

        top_indices = sorted(
            range(len(bm25_scores)), key=lambda j: bm25_scores[j], reverse=True
        )[:top_k]
        bm25_retrieved = [pids[j] for j in top_indices]
        bm25_results.append({"retrieved_pids": bm25_retrieved, "relevant_pids": relevant})

        t0 = time.time()
        dense_docs = dense_retriever.invoke(query_text)
        dense_elapsed = time.time() - t0
        dense_total_time += dense_elapsed

        dense_retrieved = [doc.metadata.get("pid", "?") for doc in dense_docs[:top_k]]
        dense_results.append({"retrieved_pids": dense_retrieved, "relevant_pids": relevant})

        if (i + 1) % 50 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"BM25 avg {bm25_total_time/(i+1)*1000:.0f}ms/q | "
                f"Dense avg {dense_total_time/(i+1)*1000:.0f}ms/q"
            )

    bm25_metrics = compute_metrics(bm25_results)
    dense_metrics = compute_metrics(dense_results)

    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'Metric':<16} {'BM25':>10} {'Dense':>10} {'Delta':>10}")
    print("  " + "-" * 50)

    for metric_name in ["Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"]:
        if metric_name in bm25_metrics:
            b = bm25_metrics[metric_name]
            d = dense_metrics[metric_name]
            delta = d - b
            print(f"  {metric_name:<16} {b:>10.4f} {d:>10.4f} {delta:>+10.4f}")

    print("  " + "-" * 50)
    print(f"  BM25 total search time:  {bm25_total_time:.1f}s ({bm25_total_time/len(pool_queries)*1000:.0f}ms/q)")
    print(f"  Dense total search time: {dense_total_time:.1f}s ({dense_total_time/len(pool_queries)*1000:.0f}ms/q)")
    print("=" * 70)

    return bm25_metrics, dense_metrics


def main():
    parser = argparse.ArgumentParser(description="T2Ranking retrieval evaluation: BM25 vs Dense")
    parser.add_argument("--sample", type=int, default=500, help="Number of queries to evaluate")
    parser.add_argument("--passages", type=int, default=150000, help="Number of passages to load")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument(
        "--vector-db",
        default=None,
        help="Vector DB directory (default: data/vector_db/t2ranking/bge-small-zh-v1.5)",
    )
    parser.add_argument("--collection-name", default="t2ranking_passages")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25")
    parser.add_argument("--bm25-only", action="store_true", help="Skip Dense")
    args = parser.parse_args()

    evaluate(
        sample_size=args.sample,
        top_k=args.top_k,
        vector_db_dir=args.vector_db,
        collection_name=args.collection_name,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
