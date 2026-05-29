"""
Build a BM25S index for T2Ranking passages.

Usage:
    # Test with 10K passages
    python scripts/build_bm25s_index.py --limit 10000

    # Build full 2.3M index on server
    python scripts/build_bm25s_index.py -j 16

    # Sequential fallback (if multiprocessing fails)
    python scripts/build_bm25s_index.py -j 1
"""
import os
import sys
import time
import argparse
import math
import multiprocessing as mp
import traceback
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
# Multiprocessing tokenization
# ═══════════════════════════════════════════════════════════

def _get_mp_context():
    if sys.platform == "linux" or sys.platform == "darwin":
        return mp.get_context("spawn")
    return mp.get_context()


def _init_worker(stopwords_set):
    import jieba
    jieba.lcut("")


def _tokenize_one(text):
    import jieba

    if text is _ERROR_SENTINEL:
        return _ERROR_SENTINEL

    try:
        tokens = jieba.lcut(text)
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1]
        return tokens
    except Exception:
        return RuntimeError(traceback.format_exc())


_ERROR_SENTINEL = object()


def tokenize_parallel(texts, n_jobs, stopwords):
    ctx = _get_mp_context()
    n_jobs = max(1, min(n_jobs, mp.cpu_count()))

    n_total = len(texts)
    chunksize = max(200, math.ceil(n_total / (n_jobs * 40)))

    logger.info(
        f"Tokenizing {n_total:,} passages "
        f"(jieba, stopwords={len(stopwords)}, workers={n_jobs})..."
    )
    t0 = time.time()

    pool = ctx.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(stopwords,),
    )
    try:
        iterator = pool.imap_unordered(
            _tokenize_one, texts, chunksize=chunksize
        )
        if tqdm is not None:
            iterator = tqdm(iterator, total=n_total,
                            desc="Tokenizing", unit="docs", mininterval=2)

        tokenized = []
        errors = 0
        for result in iterator:
            if isinstance(result, Exception):
                errors += 1
                tokenized.append([])
            else:
                if stopwords:
                    result = [w for w in result if w not in stopwords]
                tokenized.append(result)

        if errors:
            logger.warning(f"{errors}/{n_total} docs failed during tokenization "
                           f"({100*errors/n_total:.2f}%)")
    finally:
        pool.terminate()
        pool.join()

    elapsed = time.time() - t0
    n_tokens = sum(len(tk) for tk in tokenized)
    assert len(tokenized) == n_total, \
        f"tokenized count mismatch: {len(tokenized)} != {n_total}"
    logger.info(
        f"Tokenized in {elapsed:.1f}s ({n_total/elapsed:.0f} docs/s): "
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

def _tokenize_texts(texts, n_jobs, stopwords):
    if n_jobs > 1 and len(texts) > 1000:
        return tokenize_parallel(texts, n_jobs, stopwords)
    else:
        return tokenize_sequential(texts, stopwords)


def main():
    import json

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
    parser.add_argument(
        "--docs-per-shard", type=int, default=300000,
        help="Docs per index shard to control peak memory (default: 300000). "
             "Lower = less RAM needed. Set to 0 to disable sharding.",
    )
    args = parser.parse_args()

    import bm25s

    collection_path = Path(args.collection)
    if not collection_path.exists():
        logger.error(f"Collection file not found: {collection_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve worker count ──
    n_jobs = args.workers
    if n_jobs == 0:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = mp.cpu_count()

    # ── Load ──
    logger.info(f"Loading passages from {collection_path}...")
    t_total = time.time()
    t0 = time.time()
    if args.limit > 0:
        pids, texts = load_passages(
            collection_path, max_passages=args.limit, show_progress=True
        )
    else:
        pids, texts = load_passages(collection_path, show_progress=True)
    load_time = time.time() - t0
    n_total = len(texts)
    total_mb = sum(len(t) for t in texts) / (1024 * 1024)
    avg_len = sum(len(t) for t in texts) / max(n_total, 1)
    logger.info(
        f"Loaded {n_total:,} passages in {load_time:.1f}s "
        f"({total_mb:.1f} MB, avg {avg_len:.0f} chars/doc)"
    )

    # ── Shard config ──
    shard_size = args.docs_per_shard
    if shard_size <= 0 or shard_size >= n_total:
        shard_size = n_total
    n_shards = math.ceil(n_total / shard_size)

    stopwords = _load_stopwords()

    # ── Build shards ──
    shard_base = output_dir / args.name
    shard_base.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Building {n_shards} shard(s) of ~{shard_size:,} docs each "
        f"(method={args.method}, k1={args.k1}, b={args.b})"
        f"\n  Output: {shard_base}"
    )

    total_index_time = 0.0
    total_save_time = 0.0
    total_save_size = 0
    shard_offsets = [0]

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n_total)
        shard_texts = texts[start:end]
        shard_pids = pids[start:end]

        logger.info(
            f"\n{'='*50}\n"
            f"Shard {shard_idx+1}/{n_shards}: docs {start:,}–{end:,} "
            f"({len(shard_texts):,} docs)\n"
            f"{'='*50}"
        )

        shard_tokens = _tokenize_texts(shard_texts, n_jobs, stopwords)
        del shard_texts

        t0 = time.time()
        retriever = bm25s.BM25(method=args.method, k1=args.k1, b=args.b)
        retriever.index(shard_tokens)
        index_time = time.time() - t0
        total_index_time += index_time
        logger.info(
            f"  Index built in {index_time:.1f}s "
            f"({len(shard_tokens)/index_time:.0f} docs/s)"
        )

        del shard_tokens

        shard_name = f"shard_{shard_idx:04d}"
        shard_path = shard_base / shard_name
        t0 = time.time()
        retriever.save(str(shard_path))
        save_time = time.time() - t0
        total_save_time += save_time
        shard_size_bytes = sum(
            f.stat().st_size for f in shard_path.rglob("*") if f.is_file()
        )
        total_save_size += shard_size_bytes
        logger.info(
            f"  Saved: {shard_path.name} "
            f"({save_time:.1f}s, {shard_size_bytes/1024/1024:.1f} MB)"
        )

        shard_offsets.append(shard_offsets[-1] + len(shard_pids))
        del retriever

    del texts

    # ── Save manifest ──
    manifest = {
        "version": 1,
        "name": args.name,
        "n_shards": n_shards,
        "n_docs": n_total,
        "shard_offsets": shard_offsets,
        "method": args.method,
        "k1": args.k1,
        "b": args.b,
    }
    manifest_path = shard_base / "shards.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"Manifest saved: {manifest_path}")

    # ── Summary ──
    t_total = time.time() - t_total
    print()
    print("=" * 60)
    print(f"  BM25S index built successfully")
    print("=" * 60)
    print(f"  Documents:     {n_total:,}")
    print(f"  Shards:        {n_shards} × ~{shard_size:,} docs")
    print(f"  Method:        {args.method} (k1={args.k1}, b={args.b})")
    print(f"  Index dir:     {shard_base}")
    print(f"  Total size:    {total_save_size/1024/1024:.1f} MB")
    print(f"  Total time:    {t_total:.0f}s "
          f"(build {total_index_time:.0f}s + save {total_save_time:.0f}s)")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
