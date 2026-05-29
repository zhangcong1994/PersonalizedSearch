"""
Build FAISS vector index for exp-007 embedding evaluation.

Encodes all T2Ranking collection passages and stores them in a FAISS
IndexFlatIP. This is 5-10x faster than the ChromaDB incremental build
approach because:
  1. All passages are encoded in one pass (GPU batch encoding)
  2. FAISS IndexFlatIP is pure vector storage, no graph building overhead
  3. Vectors are saved to .npy for reuse

Index path: {VECTOR_DB_DIR}/t2ranking/{model_short_name}/
Contains: index.faiss, pids.json, vectors.npy

Usage:
  python scripts/exp007/build_faiss_index.py --model models/m3e-base-t2ranking-phase1 --device cuda

  python scripts/exp007/build_faiss_index.py --model moka-ai/m3e-base --device cuda

  python scripts/exp007/build_faiss_index.py --model models/m3e-base-t2ranking-phase1 \\
      --max-passages 10000 --device cpu --offline
"""

import os
import sys
import re
import html
import json
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
INDEX_BASE_DIR = VECTOR_DB_DIR / "t2ranking"

DEFAULT_BATCH_SIZE = 5000
DEFAULT_ENCODE_BATCH_SIZE = 1024
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
    if os.path.isabs(model_id_or_path) and os.path.isdir(model_id_or_path):
        return os.path.abspath(model_id_or_path)

    local_path = resolve_model_local_path(model_id_or_path)
    if local_path is not None:
        logger.info(f"Resolved {model_id_or_path} → {local_path}")
        return str(local_path)

    relative = (DATA_ROOT / model_id_or_path).resolve()
    if relative.is_dir():
        logger.info(f"Resolved via DATA_ROOT: {model_id_or_path} → {relative}")
        return str(relative)

    candidate = (Path.cwd() / model_id_or_path).resolve()
    if candidate.is_dir():
        logger.info(f"Resolved via CWD: {model_id_or_path} → {candidate}")
        return str(candidate)

    return model_id_or_path


def load_all_passages(max_passages: int = 0) -> tuple[list[str], list[str]]:
    pids: list[str] = []
    texts: list[str] = []
    total_expected = max_passages if max_passages > 0 else 2_300_000

    logger.info(f"Reading collection.tsv (target: {'all ~2.3M' if max_passages <= 0 else f'{max_passages:,}'})...")
    t_start = time.time()
    last_log_line = 0
    log_interval = 500_000

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

            pids.append(pid)
            texts.append(text)

            n = len(pids)
            if n - last_log_line >= log_interval:
                elapsed = time.time() - t_start
                speed = n / elapsed
                remaining = (total_expected - n) / speed if speed > 0 else 0
                logger.info(
                    f"  loaded {n:>10,} / ~{total_expected:,} passages "
                    f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, {speed:,.0f} docs/s)"
                )
                last_log_line = n

            if max_passages > 0 and n >= max_passages:
                break

    elapsed = time.time() - t_start
    logger.info(f"Total: {len(pids):,} passages loaded in {elapsed:.0f}s ({len(pids)/elapsed:,.0f} docs/s)")
    return pids, texts


def encode_all_passages(
    model,
    texts: list[str],
    batch_size: int = DEFAULT_ENCODE_BATCH_SIZE,
) -> "np.ndarray":
    import numpy as np

    all_vectors: list[np.ndarray] = []
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size

    logger.info(f"Encoding {total:,} passages ({num_batches:,} batches × {batch_size})...")
    t_start = time.time()
    last_log = 0

    for i, batch_start in enumerate(range(0, total, batch_size)):
        batch = texts[batch_start:batch_start + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vectors.append(vecs)

        n_done = batch_start + len(batch)
        if i == 0 or n_done - last_log >= 200_000 or batch_start + len(batch) >= total:
            elapsed = time.time() - t_start
            speed = n_done / elapsed if elapsed > 0 else 0
            remaining = (total - n_done) / speed if speed > 0 else 0
            pct = n_done / total * 100
            logger.info(
                f"  batch {i + 1:>5,}/{num_batches:,} | "
                f"{n_done:>10,}/{total:,} ({pct:.0f}%) | "
                f"~{remaining:.0f}s remaining | {speed:,.0f} docs/s"
            )
            last_log = n_done

    result = np.concatenate(all_vectors, axis=0)
    elapsed = time.time() - t_start
    logger.info(f"Encoded {total:,} passages → {result.shape} in {elapsed:.0f}s ({total/elapsed:,.0f} docs/s)")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index for exp-007"
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
        "--encode-batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE,
        help=f"Batch size for encoding (default: {DEFAULT_ENCODE_BATCH_SIZE})"
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
    parser.add_argument(
        "--no-save-vectors", action="store_true",
        help="Skip saving vectors.npy (saves ~7GB disk space)"
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

    faiss_index_file = index_dir / "index.faiss"
    pids_file = index_dir / "pids.json"
    vectors_file = index_dir / "vectors.npy"

    if faiss_index_file.exists() and pids_file.exists():
        logger.info(f"FAISS index already exists at {index_dir}")
        logger.info(f"  Use --rebuild to overwrite")
        return 0

    logger.info("=" * 60)
    logger.info("  EXP-007 FAISS Vector Index Builder")
    logger.info("=" * 60)
    logger.info(f"  Model:          {args.model}")
    logger.info(f"  Model path:     {model_path}")
    logger.info(f"  Index dir:      {index_dir}")
    logger.info(f"  Device:         {args.device}")
    logger.info(f"  Encode batch:   {args.encode_batch_size:,}")
    logger.info(f"  Max passages:   {args.max_passages if args.max_passages > 0 else 'all (~2.3M)'}")
    logger.info(f"  Save vectors:   {not args.no_save_vectors}")
    logger.info("-" * 60)

    logger.info("[1/5] Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path, device=args.device)
    dim = model.get_sentence_embedding_dimension()
    logger.info(f"  Embedding dim: {dim}")

    logger.info("[2/5] Reading passages from collection.tsv...")
    pids, texts = load_all_passages(max_passages=args.max_passages)

    logger.info("[3/5] Encoding all passages to vectors...")
    vectors = encode_all_passages(model, texts, batch_size=args.encode_batch_size)

    os.makedirs(index_dir, exist_ok=True)

    if not args.no_save_vectors:
        import numpy as np
        logger.info(f"  Saving vectors to {vectors_file}...")
        t0 = time.time()
        np.save(vectors_file, vectors)
        vsize_mb = vectors_file.stat().st_size / (1024 * 1024)
        logger.info(f"  saved {vectors.shape} ({vsize_mb:.0f} MB) in {time.time() - t0:.0f}s")

    logger.info(f"  Saving pids to {pids_file}...")
    with open(pids_file, "w", encoding="utf-8") as f:
        json.dump(pids, f, ensure_ascii=False)
    logger.info(f"  {len(pids):,} pids saved")

    logger.info("[4/5] Building FAISS index and saving to disk...")
    import numpy as np
    import faiss

    t0 = time.time()
    logger.info(f"  Converting to float32 ({vectors.shape[0]:,} × {vectors.shape[1]})...")
    vectors_f32 = vectors.astype(np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_f32)
    build_time = time.time() - t0
    logger.info(f"  IndexFlatIP built: {index.ntotal:,} vectors in {build_time:.0f}s")

    logger.info(f"Saving FAISS index to {faiss_index_file}...")
    faiss.write_index(index, str(faiss_index_file))

    isize_mb = faiss_index_file.stat().st_size / (1024 * 1024)
    logger.info(f"  index saved ({isize_mb:.0f} MB)")

    del vectors_f32, vectors

    logger.info("[5/5] Done!")
    print()
    print("=" * 60)
    print("  FAISS INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Path:     {index_dir}")
    print(f"  Docs:     {len(pids):,}")
    print(f"  Dim:      {dim}")
    print(f"  Files:")
    print(f"    index.faiss  ({isize_mb:.0f} MB)")
    if not args.no_save_vectors:
        vsize_mb_final = vectors_file.stat().st_size / (1024 * 1024)
        print(f"    vectors.npy  ({vsize_mb_final:.0f} MB)")
    print(f"    pids.json")
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
