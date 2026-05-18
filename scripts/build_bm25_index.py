import os
import sys
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

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.evaluation.data_loader import load_passages
from src.retrieval.bm25_store import build

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"
DEFAULT_STORE_DIR = DATA_ROOT / "data" / "bm25_index"


def main():
    parser = argparse.ArgumentParser(
        description="Build and save a BM25 index for T2Ranking passages",
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_STORE_DIR),
        help=f"Directory to save the BM25 index pickle (default: {DEFAULT_STORE_DIR})",
    )
    parser.add_argument(
        "--name", default="t2ranking",
        help="Base name for the saved index file (default: t2ranking)",
    )
    parser.add_argument(
        "--collection", default=str(COLLECTION_FILE),
        help="Path to collection.tsv (default: T2Ranking collection.tsv)",
    )
    parser.add_argument(
        "-j", "--workers", type=int, default=-1,
        help="Number of parallel workers for jieba tokenization (-1 = all CPUs, 1 = single-process, default: -1)",
    )
    args = parser.parse_args()

    collection_path = Path(args.collection)
    if not collection_path.exists():
        logger.error(f"Collection file not found: {collection_path}")
        return 1

    logger.info(f"Loading passages from {collection_path}...")
    t0 = time.time()
    pids, texts = load_passages(collection_path, show_progress=True)
    load_time = time.time() - t0
    logger.info(f"Loaded {len(pids):,} passages in {load_time:.1f}s")

    output_dir = Path(args.output_dir)
    pkl_path = build(texts, store_dir=output_dir, name=args.name, n_jobs=args.workers)

    total_time = time.time() - t0
    logger.info(f"Done in {total_time:.1f}s: {pkl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
