"""
Build ChromaDB vector index for exp-007 embedding evaluation.

Encodes all T2Ranking collection passages and stores them in a ChromaDB
index. Index path follows the same convention as build_t2ranking_index.py:
  {VECTOR_DB_DIR}/t2ranking/{model_short_name}/

Usage:
  # Build index for fine-tuned model
  python scripts/exp007/build_index.py --model models/m3e-base-t2ranking-phase1 --device cuda

  # Build index for pretrained baseline
  python scripts/exp007/build_index.py --model moka-ai/m3e-base --device cuda

  # Quick test on CPU (10K passages only)
  python scripts/exp007/build_index.py --model models/m3e-base-t2ranking-phase1 --max-passages 10000 --device cpu --offline
"""

import os
import sys
import re
import html
import time
import shutil
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import (
    DATA_ROOT, RAW_DATA_DIR, VECTOR_DB_DIR,
    resolve_model_local_path, model_short_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"
COLLECTION_NAME = "t2ranking_passages"
INDEX_BASE_DIR = VECTOR_DB_DIR / "t2ranking"

DEFAULT_BATCH_SIZE = 5000
DEFAULT_MODEL = "models/m3e-base-t2ranking-phase1"

HTML_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUA_RE = re.compile(r"[\uE000-\uF8FF\u200E\u200F\u202A-\u202E\uFEFF]+")


def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = CONTROL_RE.sub("", text)
    text = PUA_RE.sub("", text)
    return text.strip()


def resolve_model_path(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    local_path = resolve_model_local_path(model_id_or_path)
    if local_path is not None:
        logger.info(f"Resolved {model_id_or_path} → {local_path}")
        return str(local_path)

    relative = (DATA_ROOT / model_id_or_path).resolve()
    if relative.is_dir():
        logger.info(f"Resolved via DATA_ROOT: {model_id_or_path} → {relative}")
        return str(relative)

    if not os.path.isabs(model_id_or_path):
        candidate = (Path.cwd() / model_id_or_path).resolve()
        if candidate.is_dir():
            logger.info(f"Resolved via CWD: {model_id_or_path} → {candidate}")
            return str(candidate)

    return model_id_or_path


def load_passages_batched(max_passages: int = 0, batch_size: int = DEFAULT_BATCH_SIZE):
    pids_batch = []
    texts_batch = []
    total = 0

    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue

            pid, raw_text = parts[0], parts[1]
            text = clean_text(raw_text)
            if len(text) < 10:
                continue
            if len(text) > 2000:
                text = text[:2000]

            pids_batch.append(pid)
            texts_batch.append(text)
            total += 1

            if len(pids_batch) >= batch_size:
                yield pids_batch, texts_batch
                pids_batch = []
                texts_batch = []

            if max_passages > 0 and total >= max_passages:
                break

    if pids_batch:
        yield pids_batch, texts_batch

    logger.info(f"Total passages to index: {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB vector index for exp-007"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model path or HF ID"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device (cpu, cuda, cuda:0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Passages per insert batch (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete existing index and rebuild from scratch"
    )
    parser.add_argument(
        "--max-passages", type=int, default=0,
        help="Only index first N passages (0 = all ~2.3M)"
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="HF offline mode"
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    index_dir = INDEX_BASE_DIR / model_short_name(args.model)
    model_path = resolve_model_path(args.model)

    if args.rebuild and index_dir.is_dir():
        logger.info(f"Rebuild mode: deleting {index_dir}")
        shutil.rmtree(index_dir)

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    if index_dir.is_dir():
        embeddings_check = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": args.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
        )
        existing = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings_check,
            persist_directory=str(index_dir),
        )
        count = existing._collection.count()
        if count > 0:
            logger.info(f"Index already exists with {count:,} docs")
            logger.info(f"  Path: {index_dir}")
            logger.info(f"  Use --rebuild to overwrite")
            return 0

    logger.info("=" * 60)
    logger.info("  EXP-007 Vector Index Builder")
    logger.info("=" * 60)
    logger.info(f"  Model:        {args.model}")
    logger.info(f"  Model path:   {model_path}")
    logger.info(f"  Index dir:    {index_dir}")
    logger.info(f"  Device:       {args.device}")
    logger.info(f"  Batch size:   {args.batch_size:,}")
    logger.info(f"  Max passages: {args.max_passages if args.max_passages > 0 else 'all (~2.3M)'}")
    logger.info(f"  Collection:   {COLLECTION_NAME}")
    logger.info("-" * 60)

    logger.info("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={"device": args.device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )

    try:
        dim = embeddings.client.get_sentence_embedding_dimension()
    except AttributeError:
        dim = 768
    logger.info(f"Embedding dim: {dim}")

    logger.info("Initializing ChromaDB...")
    os.makedirs(index_dir, exist_ok=True)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(index_dir),
    )

    try:
        hnsw_batch = max(args.batch_size, 5000)
        meta = vectorstore._collection.metadata or {}
        meta.update({
            "hnsw:batch_size": hnsw_batch,
            "hnsw:sync_threshold": hnsw_batch * 10,
        })
        vectorstore._collection.modify(metadata=meta)
    except Exception:
        pass

    overall_start = time.time()
    total_stored = 0
    batch_no = 0

    for pids, texts in load_passages_batched(
        max_passages=args.max_passages,
        batch_size=args.batch_size,
    ):
        batch_no += 1
        batch_start = time.time()

        emb_start = time.time()
        vectors = embeddings.embed_documents(texts)
        emb_time = time.time() - emb_start

        ids = [f"t2r_{pid}" for pid in pids]
        metadatas = [{"pid": pid, "source": "T2Ranking"} for pid in pids]

        vectorstore._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        add_time = time.time() - emb_start

        total_stored += len(pids)
        batch_total = time.time() - batch_start
        speed = len(pids) / batch_total if batch_total > 0 else 0

        logger.info(
            f"Batch {batch_no}: {len(pids):,} docs "
            f"(enc {emb_time:.1f}s, add {add_time - emb_time:.1f}s, "
            f"{speed:.0f} docs/s) | "
            f"total {total_stored:,}"
        )

    overall_time = time.time() - overall_start
    logger.info(f"Index build complete: {total_stored:,} docs in {overall_time/60:.1f} min")

    print()
    print("=" * 60)
    print("  INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Model:  {args.model}")
    print(f"  Path:   {index_dir}")
    print(f"  Docs:   {total_stored:,}")
    print(f"  Time:   {overall_time/60:.1f} min")
    print(f"  Dim:    {dim}")
    print("=" * 60)
    print()
    print("  Evaluate with:")
    print(f"  python scripts/exp007/evaluate_embedding.py \\")
    print(f"      --baseline-model moka-ai/m3e-base \\")
    print(f"      --model {args.model} \\")
    print(f"      --device {args.device} --offline")

    return 0


if __name__ == "__main__":
    sys.exit(main())
