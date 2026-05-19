"""
BM25S verification script for T2Ranking.

Validates that BM25S can:
1. Load and tokenize T2Ranking passages (Chinese, via jieba)
2. Build a BM25S index within reasonable time/memory
3. Execute queries with correct top-k results
4. Persist and reload the index (roundtrip)
5. Produce scores consistent with the existing ShardedBM25

Usage:
    python scripts/verify_bm25s.py --limit 10000   # verify with 10K passages
    python scripts/verify_bm25s.py --limit 100000  # verify with 100K passages
"""
import os
import sys
import time
import argparse
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_bm25s")

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.retrieval.bm25_store import _load_stopwords

COLLECTION_FILE = RAW_DATA_DIR / "t2ranking" / "collection.tsv"


def check_imports():
    logger.info("Checking imports...")
    ok = True
    for pkg in ["jieba", "bm25s", "numpy", "scipy"]:
        try:
            __import__(pkg)
            logger.info(f"  {pkg}: OK")
        except ImportError:
            logger.error(f"  {pkg}: MISSING — pip install {pkg}")
            ok = False
    return ok


def get_memory_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


# ═══════════════════════════════════════════════════════════
# Phase 1: Load data
# ═══════════════════════════════════════════════════════════

def load_passages(limit: int) -> tuple[list[str], list[str]]:
    from src.evaluation.data_loader import load_passages as _load

    logger.info(f"Loading up to {limit:,} passages from {COLLECTION_FILE}...")
    t0 = time.time()
    pids, texts = _load(COLLECTION_FILE, max_passages=limit, show_progress=True)
    elapsed = time.time() - t0
    total_chars = sum(len(t) for t in texts)
    avg_len = total_chars / len(texts) if texts else 0
    logger.info(
        f"Loaded {len(pids):,} passages in {elapsed:.1f}s "
        f"(avg {avg_len:.0f} chars/doc, {total_chars/1024/1024:.1f} MB)"
    )
    return pids, texts


# ═══════════════════════════════════════════════════════════
# Phase 2: Tokenize with jieba (mirror bm25_store logic)
# ═══════════════════════════════════════════════════════════

def tokenize_jieba(texts: list[str]) -> list[list[str]]:
    import jieba

    stopwords = _load_stopwords()
    logger.info(
        f"Tokenizing {len(texts):,} passages (jieba, stopwords={len(stopwords)})..."
    )
    t0 = time.time()
    tokenized = []
    for text in texts:
        tokens = jieba.lcut(text)
        tokens = [w.strip() for w in tokens
                  if w.strip() and len(w) > 1 and w not in stopwords]
        tokenized.append(tokens)
    elapsed = time.time() - t0
    n_tokens = sum(len(tk) for tk in tokenized)
    logger.info(
        f"Tokenized in {elapsed:.1f}s: {n_tokens:,} tokens, "
        f"avg {n_tokens//max(len(tokenized),1)} tokens/doc"
    )
    return tokenized


def tokenize_query(text: str) -> list[str]:
    import jieba

    stopwords = _load_stopwords()
    tokens = jieba.lcut(text)
    return [w.strip() for w in tokens
            if w.strip() and len(w) > 1 and w not in stopwords]


# ═══════════════════════════════════════════════════════════
# Phase 3: Build BM25S index
# ═══════════════════════════════════════════════════════════

def build_index(
    tokenized: list[list[str]],
    texts: list[str],
    method: str = "lucene",
    k1: float = 1.5,
    b: float = 0.75,
) -> "bm25s.BM25":
    import bm25s

    logger.info(f"Building BM25S index (method={method}, k1={k1}, b={b})...")
    t0 = time.time()

    retriever = bm25s.BM25(method=method, k1=k1, b=b, corpus=texts)
    retriever.index(tokenized)

    elapsed = time.time() - t0
    logger.info(
        f"Index built in {elapsed:.1f}s ({len(tokenized)/elapsed:.0f} docs/s)"
    )
    return retriever


# ═══════════════════════════════════════════════════════════
# Phase 4: Query benchmarks
# ═══════════════════════════════════════════════════════════

def benchmark_queries(
    retriever: "bm25s.BM25",
    texts: list[str],
    n_warmup: int = 50,
    n_bench: int = 200,
    top_k: int = 10,
):
    import random
    import numpy as np

    rng = random.Random(42)

    if len(texts) <= n_warmup + n_bench:
        sample_queries = texts
    else:
        sample_queries = rng.sample(texts, n_warmup + n_bench)

    warmup_queries = [tokenize_query(q) for q in sample_queries[:n_warmup]]
    bench_queries = [tokenize_query(q) for q in sample_queries[n_warmup:]]

    warmup_queries = [qt for qt in warmup_queries if qt]
    bench_queries = [qt for qt in bench_queries if qt]

    logger.info(f"Warming up with {len(warmup_queries)} queries...")
    for qt in warmup_queries:
        retriever.retrieve([qt], k=top_k)

    logger.info(f"Benchmarking {len(bench_queries)} queries (k={top_k})...")
    latencies = []
    for qt in bench_queries:
        t0 = time.perf_counter()
        retriever.retrieve([qt], k=top_k)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    logger.info(
        f"Latency (ms): mean={latencies.mean():.2f}, "
        f"p50={np.median(latencies):.2f}, "
        f"p95={np.percentile(latencies, 95):.2f}, "
        f"p99={np.percentile(latencies, 99):.2f}"
    )
    logger.info(f"QPS: {1000/latencies.mean():.1f} queries/s")

    return latencies.mean()


# ═══════════════════════════════════════════════════════════
# Phase 5: Save / load roundtrip
# ═══════════════════════════════════════════════════════════

def test_roundtrip(
    retriever: "bm25s.BM25",
    tokenized: list[list[str]],
    texts: list[str],
):
    import bm25s
    import numpy as np
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="bm25s_verify_")
    try:
        save_dir = Path(tmpdir) / "test_index"

        t0 = time.time()
        retriever.save(str(save_dir), corpus=texts)
        save_time = time.time() - t0
        save_size = sum(
            f.stat().st_size for f in save_dir.rglob("*") if f.is_file()
        )
        logger.info(
            f"Index saved in {save_time:.1f}s "
            f"({save_size/1024/1024:.1f} MB)"
        )

        t0 = time.time()
        reloaded = bm25s.BM25.load(str(save_dir), load_corpus=True)
        load_time = time.time() - t0
        logger.info(f"Index loaded in {load_time:.1f}s")

        query_tokens = tokenized[0][:5] if tokenized[0] else ["测试"]
        orig_docs, orig_scores = retriever.retrieve([query_tokens], k=5)
        reload_docs, reload_scores = reloaded.retrieve([query_tokens], k=5)

        scores_ok = np.allclose(orig_scores, reload_scores, atol=1e-6)
        docs_ok = bool((orig_docs == reload_docs).all())
        logger.info(f"Roundtrip scores match: {scores_ok}")
        logger.info(f"Roundtrip docs match:   {docs_ok}")

        for i in range(min(3, orig_scores.shape[1])):
            logger.info(
                f"  #{i+1}: score={orig_scores[0,i]:.4f}  "
                f"text={str(orig_docs[0,i])[:80]}..."
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
# Phase 6: Consistency check with existing ShardedBM25
# ═══════════════════════════════════════════════════════════

def check_consistency(
    retriever: "bm25s.BM25",
    tokenized: list[list[str]],
    texts: list[str],
):
    import random
    import numpy as np
    from src.retrieval.bm25_store import load as bm25_load, DEFAULT_STORE_DIR

    try:
        old_bm25, _ = bm25_load(DEFAULT_STORE_DIR)
    except FileNotFoundError:
        logger.warning(
            f"ShardedBM25 not found at {DEFAULT_STORE_DIR}, skipping check"
        )
        return

    if len(tokenized) != len(old_bm25.doc_len):
        logger.warning(
            f"Doc count mismatch: BM25S={len(tokenized)}, "
            f"ShardedBM25={len(old_bm25.doc_len)} — skipping check"
        )
        return

    logger.info(
        f"Comparing top-k rankings: BM25S vs ShardedBM25 "
        f"({len(tokenized)} docs)..."
    )

    rng = random.Random(42)
    sample_qt = rng.sample(tokenized, min(30, len(tokenized)))
    top_k = 20

    overlaps = []
    rank_corrs = []

    for qt in sample_qt:
        if not qt:
            continue

        old_scores = old_bm25.get_scores(qt)
        old_top_idx = np.argsort(old_scores)[::-1][:top_k].tolist()

        bm25s_docs, bm25s_scores = retriever.retrieve([qt], k=top_k)
        bm25s_top_texts = [str(doc) for doc in bm25s_docs[0].tolist()]

        bm25s_top_idx = []
        for doc_text in bm25s_top_texts:
            try:
                idx = texts.index(doc_text)
                bm25s_top_idx.append(idx)
            except ValueError:
                pass

        overlap = len(set(old_top_idx) & set(bm25s_top_idx))
        overlaps.append(overlap / top_k)

        if bm25s_top_idx:
            if len(old_top_idx) == len(set(bm25s_top_idx)):
                sp = sum(
                    (top_k - abs(a - b) + 1)
                    for a in old_top_idx
                    for b in bm25s_top_idx
                    if a == b
                )
                max_sp = sum(top_k - i for i in range(top_k))
                rank_corrs.append(sp / max_sp if max_sp > 0 else 0)

    if overlaps:
        logger.info(f"Rank overlap @{top_k}:  {np.mean(overlaps):.3f}")
    if rank_corrs:
        logger.info(f"Rank correlation:      {np.mean(rank_corrs):.3f}")
        if np.mean(rank_corrs) > 0.8:
            logger.info("Good — rankings are strongly correlated")
        else:
            logger.info(
                "Moderate divergence — expected due to "
                "Lucene vs Robertson BM25 variants"
            )


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Verify BM25S on T2Ranking passages",
    )
    parser.add_argument(
        "--limit", type=int, default=10000,
        help="Number of passages (default: 10000)",
    )
    parser.add_argument(
        "--method", default="robertson",
        choices=["lucene", "robertson", "atire", "bm25l", "bm25+"],
        help="BM25 variant (default: robertson, matching ShardedBM25)",
    )
    parser.add_argument(
        "--k1", type=float, default=1.5, help="BM25 k1 (default: 1.5)",
    )
    parser.add_argument(
        "--b", type=float, default=0.75, help="BM25 b (default: 0.75)",
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Top-K (default: 10)",
    )
    parser.add_argument(
        "--skip-consistency", action="store_true",
        help="Skip consistency check with ShardedBM25",
    )
    args = parser.parse_args()

    if not check_imports():
        return 1

    mem_before = get_memory_mb()

    print()
    print("=" * 65)
    print(f"  BM25S Verification — {args.limit:,} passages")
    print("=" * 65)
    print(f"  Collection: {COLLECTION_FILE}")
    print(f"  Method: {args.method}, k1={args.k1}, b={args.b}")
    if mem_before > 0:
        print(f"  Memory baseline: {mem_before:.0f} MB")
    print()

    # Phase 1
    logger.info("=" * 50)
    logger.info("Phase 1/6: Load T2Ranking passages")
    logger.info("=" * 50)
    pids, texts = load_passages(args.limit)

    # Phase 2
    logger.info("=" * 50)
    logger.info("Phase 2/6: Tokenize with jieba")
    logger.info("=" * 50)
    tokenized = tokenize_jieba(texts)

    # Phase 3
    logger.info("=" * 50)
    logger.info("Phase 3/6: Build BM25S index")
    logger.info("=" * 50)
    retriever = build_index(tokenized, texts, method=args.method,
                            k1=args.k1, b=args.b)

    mem_after = get_memory_mb()
    if mem_before > 0 and mem_after > 0:
        logger.info(
            f"Memory: {mem_before:.0f} → {mem_after:.0f} MB "
            f"(+{mem_after-mem_before:.0f} MB)"
        )

    # Phase 4
    logger.info("=" * 50)
    logger.info("Phase 4/6: Query benchmarks")
    logger.info("=" * 50)
    benchmark_queries(retriever, texts, top_k=args.top_k)

    # Phase 5
    logger.info("=" * 50)
    logger.info("Phase 5/6: Save/load roundtrip")
    logger.info("=" * 50)
    test_roundtrip(retriever, tokenized, texts)

    # Phase 6
    if not args.skip_consistency:
        logger.info("=" * 50)
        logger.info("Phase 6/6: Consistency with ShardedBM25")
        logger.info("=" * 50)
        check_consistency(retriever, tokenized, texts)
    else:
        logger.info("Phase 6/6: Skipped (--skip-consistency)")

    # Summary
    print()
    print("=" * 65)
    print("  Verification Complete")
    print("=" * 65)
    print(f"  Documents: {len(texts):,}")
    print(f"  Method:    {args.method}")
    print(f"  Memory:    {mem_after:.0f} MB" if mem_after > 0 else "")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
