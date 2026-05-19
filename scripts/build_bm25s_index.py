"""
Build a BM25S index for T2Ranking passages.

Usage:
    # Test with 10K passages
    python scripts/build_bm25s_index.py --limit 10000

    # Build full 2.3M index on server
    python scripts/build_bm25s_index.py
"""
import os
import sys
import time
import argparse
import math
import multiprocessing as mp
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.evaluation.data_loader import load_passages
from src.retrieval.bm25_store import _load_stopwords

COLLECTION_FILE = RAW_DATA_DIR / "t2ranking" / "collection.tsv"
DEFAULT_STORE_DIR = DATA_ROOT / "data" / "bm25s_index"

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ═══════════════════════════════════════════════════════════
# Multiprocessing tokenization (mirrors bm25_store pattern)
# ═══════════════════════════════════════════════════════════

_worker_stopwords = None


def _init_worker(stopwords_set):
    global _worker_stopwords
    _worker_stopwords = stopwords_set
    import jieba
    jieba.lcut("")


def _tokenize_worker(text):
    global _worker_stopwords
    import jieba

    tokens = jieba.lcut(text)
    sw = _worker_stopwords
    if sw:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1 and w not in sw]
    else:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1]
    return tokens


def tokenize_parallel(texts, n_jobs, stopwords):
    n_jobs = max(1, min(n_jobs, mp.cpu_count()))
    chunksize = max(500, math.ceil(len(texts) / (n_jobs * 20)))

    logger.info(
        f"Tokenizing {len(texts):,} passages "
        f"(jieba, stopwords={len(stopwords)}, workers={n_jobs})..."
    )
    t0 = time.time()

    with mp.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(stopwords,),
        maxtasksperchild=max(1, len(texts) // (n_jobs * 5)),
    ) as pool:
        iterator = pool.imap_unordered(
            _tokenize_worker, texts, chunksize=chunksize
        )
        if tqdm is not None:
            iterator = tqdm(
                iterator, total=len(texts), desc="Tokenizing",
                unit="docs", mininterval=2,
            )
        tokenized = list(iterator)

    elapsed = time.time() - t0
    n_tokens = sum(len(tk) for tk in tokenized)
    logger.info(
        f"Tokenized in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/s): "
        f"{n_tokens:,} tokens, avg {n_tokens//max(len(tokenized),1)} tokens/doc"
    )
    return tokenized


def tokenize_sequential(texts, stopwords):
    import jieba

    logger.info(
        f"Tokenizing {len(texts):,} passages "
        f"(jieba, stopwords={len(stopwords)}, single-process)..."
    )
    t0 = time.time()

    tokenized = []
    iterator = texts
    if tqdm is not None:
        iterator = tqdm(texts, desc="Tokenizing", unit="docs", mininterval=2)

    for text in iterator:
        tokens = jieba.lcut(text)
        tokens = [w.strip() for w in tokens
                  if w.strip() and len(w) > 1 and w not in stopwords]
        tokenized.append(tokens)

    elapsed = time.time() - t0
    n_tokens = sum(len(tk) for tk in tokenized)
    logger.info(
        f"Tokenized in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/s): "
        f"{n_tokens:,} tokens, avg {n_tokens//max(len(tokenized),1)} tokens/doc"
    )
    return tokenized


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build a BM25S index for T2Ranking passages",
    )
    parser.add_argument(
        "--collection", default=str(COLLECTION_FILE),
        help="Path to collection.tsv",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_STORE_DIR),
        help="Directory to save the BM25S index",
    )
    parser.add_argument(
        "--name", default="t2ranking",
        help="Name prefix for the index directory",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max passages to index (0 = all, use for testing)",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=-1,
        help="Parallel workers for jieba tokenization "
             "(-1 = all CPUs, 1 = single-process)",
    )
    parser.add_argument(
        "--method", default="robertson",
        choices=["lucene", "robertson", "atire", "bm25l", "bm25+"],
        help="BM25 variant (default: robertson)",
    )
    parser.add_argument(
        "--k1", type=float, default=1.5, help="BM25 k1 parameter",
    )
    parser.add_argument(
        "--b", type=float, default=0.75, help="BM25 b parameter",
    )
    args = parser.parse_args()

    import bm25s

    collection_path = Path(args.collection)
    if not collection_path.exists():
        logger.error(f"Collection file not found: {collection_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──
    logger.info(f"Loading passages from {collection_path}...")
    t0 = time.time()
    if args.limit > 0:
        pids, texts = load_passages(
            collection_path, max_passages=args.limit, show_progress=True
        )
    else:
        pids, texts = load_passages(collection_path, show_progress=True)
    load_time = time.time() - t0
    total_mb = sum(len(t) for t in texts) / (1024 * 1024)
    avg_len = sum(len(t) for t in texts) / max(len(texts), 1)
    logger.info(
        f"Loaded {len(texts):,} passages in {load_time:.1f}s "
        f"({total_mb:.1f} MB, avg {avg_len:.0f} chars/doc)"
    )

    # ── Tokenize ──
    stopwords = _load_stopwords()
    n_jobs = args.workers
    if n_jobs == 0:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = mp.cpu_count()

    if n_jobs > 1 and len(texts) > 1000:
        tokenized = tokenize_parallel(texts, n_jobs, stopwords)
    else:
        tokenized = tokenize_sequential(texts, stopwords)

    del texts

    # ── Build BM25S index ──
    logger.info(
        f"Building BM25S index (method={args.method}, k1={args.k1}, b={args.b})..."
    )
    t0 = time.time()

    retriever = bm25s.BM25(method=args.method, k1=args.k1, b=args.b)
    retriever.index(tokenized)

    index_time = time.time() - t0
    logger.info(
        f"Index built in {index_time:.1f}s ({len(tokenized)/index_time:.0f} docs/s)"
    )

    del tokenized

    # ── Save ──
    save_path = output_dir / args.name
    logger.info(f"Saving index to {save_path}...")
    t0 = time.time()

    retriever.save(str(save_path))

    save_time = time.time() - t0
    save_size = sum(
        f.stat().st_size for f in save_path.rglob("*") if f.is_file()
    )
    logger.info(
        f"Index saved in {save_time:.1f}s ({save_size/1024/1024:.1f} MB)"
    )

    # ── Summary ──
    print()
    print("=" * 55)
    print(f"  BM25S index built successfully")
    print("=" * 55)
    print(f"  Documents:  {len(pids):,}")
    print(f"  Method:     {args.method} (k1={args.k1}, b={args.b})")
    print(f"  Index path: {save_path}")
    print(f"  Index size: {save_size/1024/1024:.1f} MB")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())
