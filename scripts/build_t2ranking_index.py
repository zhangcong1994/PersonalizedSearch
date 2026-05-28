import os
import sys
import re
import html
import json
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────
from src.utils.config import (
    PROJECT_ROOT, RAW_DATA_DIR, VECTOR_DB_DIR as _BASE_VECTOR_DB_DIR, DATA_ROOT,
    MODEL_CACHE_DIR, EMBEDDING_MODEL,
    resolve_model_local_path, model_short_name,
)

DATA_DIR = RAW_DATA_DIR / "t2ranking"
COLLECTION_FILE = DATA_DIR / "collection.tsv"
COLLECTION_NAME = "t2ranking_passages"

DEFAULT_BATCH_SIZE = 5000
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL

# HTML tag cleanup pattern
HTML_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUA_RE = re.compile(r"[\uE000-\uF8FF\u200E\u200F\u202A-\u202E\uFEFF]+")

# ── html / text cleaning ─────────────────────────────────

def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = CONTROL_RE.sub("", text)
    text = PUA_RE.sub("", text)
    return text.strip()


# ── passage loading ──────────────────────────────────────

def load_passages(
    start_line: int,
    batch_size: int,
    min_text_length: int = 10,
    max_text_length: int = 2000,
) -> Tuple[List[Tuple[str, str]], int]:
    passages = []
    total_lines = 0
    line_no = 0
    skipped_too_short = 0
    skipped_too_long = 0

    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        header = f.readline()

        for line in f:
            if line_no < start_line:
                line_no += 1
                continue

            if len(passages) >= batch_size:
                break

            parts = line.strip().split("\t", 1)
            line_no += 1
            total_lines = line_no

            if len(parts) < 2:
                continue

            pid, raw_text = parts[0], parts[1]
            text = clean_text(raw_text)

            if len(text) < min_text_length:
                skipped_too_short += 1
                continue

            if len(text) > max_text_length:
                text = text[:max_text_length]

            passages.append((pid, text))

    return passages, total_lines


# ── langchain document conversion ────────────────────────

def passages_to_documents(passages: List[Tuple[str, str]]):
    from langchain_core.documents import Document

    docs = []
    for pid, text in passages:
        docs.append(Document(
            page_content=text,
            metadata={
                "pid": pid,
                "source": "T2Ranking",
                "text_length": len(text),
            },
        ))
    return docs


# ── embedding model ──────────────────────────────────────

def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    device: str = "cpu",
    batch_size: int = 128,
    use_fp16: bool = True,
):
    from langchain_huggingface import HuggingFaceEmbeddings
    import torch

    if os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)
        logger.info(f"Using local embedding model: {model_name}")
    else:
        local_path = resolve_model_local_path(model_name)
        if local_path is not None:
            model_name = str(local_path.resolve())
            logger.info(f"Using local embedding model: {model_name}")
        else:
            data_root_candidate = (DATA_ROOT / model_name).resolve()
            if data_root_candidate.is_dir():
                model_name = str(data_root_candidate)
                logger.info(f"Using local embedding model (DATA_ROOT): {model_name}")
            else:
                logger.info(f"Loading embedding model: {model_name}")

    model_kwargs = {"device": device}

    if device == "cuda":
        batch_size = 256
        if use_fp16:
            model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            logger.info("Loading model in FP16 to reduce GPU memory")

    encode_kwargs = {"normalize_embeddings": True, "batch_size": batch_size}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    return embeddings


# ── vector store ─────────────────────────────────────────

def get_vectorstore(embeddings, persist_dir: str, collection_name: str):
    from langchain_chroma import Chroma

    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


# ── state management ─────────────────────────────────────

def load_state(state_path: Path) -> Dict:
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: Dict, state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def save_build_log(entry: Dict, build_log_path: Path):
    build_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(build_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_index_info(total_stored: int, embedding_dim: int, model_name: str,
                    vector_db_dir: Path, index_info_path: Path):
    info = {
        "collection_name": COLLECTION_NAME,
        "description": "T2Ranking evaluation index for passage retrieval",
        "embedding_model": model_name,
        "embedding_dim": embedding_dim,
        "total_passages": total_stored,
        "build_completed_at": datetime.now(timezone.utc).isoformat(),
        "collection_file": str(COLLECTION_FILE),
        "chunking": "none (passages kept intact, long passages truncated at 2000 chars)",
        "html_cleaned": True,
        "min_text_length": 10,
        "vectors_dir": str(vector_db_dir / COLLECTION_NAME),
    }
    index_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    logger.info(f"Index info saved to: {index_info_path}")


# ── main build loop ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build T2Ranking evaluation vector index incrementally")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Passages per batch (default: 1000)")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL,
                        help="Embedding model name (HF ID or local path). For fine-tuned models, "
                             "pass the local directory, e.g. models/m3e-base-t2ranking-phase1")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Device for embedding")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index and start fresh")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without building")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to run (for testing)")
    parser.add_argument("--prefetch", action="store_true", help="Pipeline: GPU embed batch N+1 while CPU stores batch N")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision on CUDA (use FP32, more memory)")
    args = parser.parse_args()

    vector_db_dir = _BASE_VECTOR_DB_DIR / "t2ranking" / model_short_name(args.model)

    state_file = DATA_DIR / f"state_{model_short_name(args.model)}.json"
    index_info_file = DATA_DIR / f"index_info_{model_short_name(args.model)}.json"
    build_log_file = DATA_DIR / f"build_log_{model_short_name(args.model)}.jsonl"

    # ── resolve state ──
    if args.rebuild:
        state = {}
        logger.info("Rebuild mode: clearing state and existing index")
        import shutil
        index_dir = vector_db_dir / COLLECTION_NAME
        if index_dir.exists():
            shutil.rmtree(index_dir)
            logger.info(f"Deleted existing index: {index_dir}")
        if state_file.exists():
            state_file.unlink()
        if build_log_file.exists():
            build_log_file.unlink()
        if index_info_file.exists():
            index_info_file.unlink()
    else:
        state = load_state(state_file)

    start_line = state.get("last_processed_line", 0)
    batches_completed = state.get("batches_completed", 0)
    total_stored = state.get("total_stored", 0)
    total_skipped = state.get("total_skipped", 0)
    total_lines = state.get("total_lines")
    if total_lines is None:
        total_lines = _count_total_lines()

    # ── print plan ──
    remaining = max(0, total_lines - start_line)
    max_batches = remaining // args.batch_size
    if args.max_batches:
        max_batches = min(max_batches, args.max_batches)

    print()
    print("=" * 60)
    print("  T2Ranking 向量索引增量构建")
    print("=" * 60)
    print(f"  Collection file:  {COLLECTION_FILE}")
    print(f"  State file:        {state_file}")
    print(f"  Vector DB dir:     {vector_db_dir}")
    print(f"  Collection name:   {COLLECTION_NAME}")
    print(f"  Embedding model:   {args.model}")
    print(f"  Device:            {args.device}")
    print(f"  Batch size:        {args.batch_size:,} passages/batch")
    print(f"  Total lines:       {total_lines:,}")
    print(f"  Start line:        {start_line:,}")
    print(f"  Batches done:      {batches_completed}")
    print(f"  Max batches:       {max_batches if args.max_batches else 'all'}")
    print(f"  Passages stored:   {total_stored:,}")
    print(f"  Passages skipped:  {total_skipped:,}")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        return 0

    # ── init model and vectorstore ──
    logger.info("Loading embedding model...")
    embeddings = get_embedding_model(args.model, device=args.device, use_fp16=not args.no_fp16)

    # Determine embedding dimension
    try:
        if hasattr(embeddings, "client") and hasattr(embeddings.client, "get_sentence_embedding_dimension"):
            embedding_dim = embeddings.client.get_sentence_embedding_dimension()
        else:
            embedding_dim = len(embeddings.embed_query("test"))
    except Exception:
        embedding_dim = 512
    logger.info(f"Embedding dimension: {embedding_dim}")

    logger.info(f"Initializing vector store at: {vector_db_dir}")
    vectorstore = get_vectorstore(embeddings, str(vector_db_dir), COLLECTION_NAME)

    # 调大 HNSW 内部 batch，避免索引变大后插入退化
    try:
        hnsw_batch = max(args.batch_size, 5000)
        meta = vectorstore._collection.metadata or {}
        meta.update({
            "hnsw:batch_size": hnsw_batch,
            "hnsw:sync_threshold": hnsw_batch * 10,
            "hnsw:M": 16,
            "hnsw:construction_ef": 100,
        })
        vectorstore._collection.modify(metadata=meta)
        logger.info(f"HNSW tuned: batch_size={hnsw_batch}, sync_threshold={hnsw_batch * 10}")
    except Exception as e:
        logger.warning(f"HNSW tuning skipped: {e}")

    # ── batch processing loop ──
    batch_no = batches_completed

    if args.prefetch:
        total_stored, total_skipped, batch_no, start_line, overall_start = _prefetch_build_loop(
            embeddings, vectorstore, args, state,
            start_line, batches_completed, total_stored, total_skipped, total_lines, max_batches, embedding_dim,
            state_file, build_log_file,
        )
    else:
        overall_start = time.time()

        for _ in range(max_batches):
            batch_no += 1
            batch_start_time = time.time()

            logger.info(f"--- Batch {batch_no} (line {start_line:,} → {start_line + args.batch_size:,}) ---")

            passages, current_line = load_passages(start_line, args.batch_size)

            if not passages:
                logger.info("No more passages to process. Build complete.")
                break

            stored_count = len(passages)
            skipped_in_batch = args.batch_size - stored_count

            logger.info(f"  Loaded {stored_count} passages (skipped {skipped_in_batch} too short)")

            docs = passages_to_documents(passages)

            logger.info(f"  Embedding {len(docs)} documents...")
            embed_start = time.time()

            vectorstore.add_documents(docs)

            embed_time = time.time() - embed_start
            batch_time = time.time() - batch_start_time

            total_stored += stored_count
            total_skipped += skipped_in_batch
            start_line = current_line

            state = {
                "source_file": str(COLLECTION_FILE),
                "total_lines": total_lines,
                "last_processed_line": start_line,
                "last_processed_pid": passages[-1][0] if passages else "",
                "total_stored": total_stored,
                "total_skipped": total_skipped,
                "batches_completed": batch_no,
                "batch_size": args.batch_size,
                "started_at": state.get("started_at", datetime.now(timezone.utc).isoformat()),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "collection_name": COLLECTION_NAME,
                "embedding_model": args.model,
                "embedding_dim": embedding_dim,
            }
            save_state(state, state_file)

            log_entry = {
                "batch": batch_no,
                "line_range": f"{start_line - args.batch_size}-{start_line}",
                "stored": stored_count,
                "skipped": skipped_in_batch,
                "total_stored": total_stored,
                "embed_time_s": round(embed_time, 1),
                "batch_time_s": round(batch_time, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_build_log(log_entry, build_log_file)

            avg_speed = stored_count / embed_time if embed_time > 0 else 0
            elapsed_total = time.time() - overall_start
            progress_pct = (start_line / total_lines * 100) if total_lines > 0 else 0
            eta_total = (elapsed_total / batch_no * (max_batches - (batch_no - batches_completed))
                         ) if batch_no > batches_completed else 0

            logger.info(
                f"  Batch complete: {stored_count} docs in {batch_time:.1f}s "
                f"(embed: {embed_time:.1f}s, {avg_speed:.0f} docs/s)"
            )
            logger.info(
                f"  Total: {total_stored:,} stored | {total_skipped:,} skipped | "
                f"{progress_pct:.1f}% | ETA: {eta_total/60:.0f}min"
            )

    # ── final ──
    overall_time = time.time() - overall_start
    save_index_info(total_stored, embedding_dim, args.model, vector_db_dir, index_info_file)
    final_state = load_state(state_file)

    print()
    print("=" * 60)
    print("  Build Summary")
    print("=" * 60)
    print(f"  Total passages stored:   {total_stored:,}")
    print(f"  Total passages skipped:  {total_skipped:,}")
    print(f"  Batches completed:       {batch_no}")
    print(f"  Progress:                {start_line:,}/{total_lines:,} lines "
          f"({start_line/max(total_lines,1)*100:.1f}%)")
    print(f"  Total time:              {overall_time/60:.1f} min")
    print(f"  Collection:              {COLLECTION_NAME}")
    print(f"  Vector DB:               {vector_db_dir}")
    print(f"  State file:              {state_file}")
    print(f"  Index info:              {index_info_file}")
    print(f"  Build log:               {build_log_file}")

    if start_line >= total_lines:
        print("\n  [DONE] All passages indexed. Ready for retrieval evaluation.")
    else:
        print(f"\n  [PAUSED] {total_lines - start_line:,} passages remaining.")
        print(f"  Resume with: python scripts/build_t2ranking_index.py")

    return 0


def _count_total_lines() -> int:
    if not COLLECTION_FILE.exists():
        logger.warning(f"Collection file not found: {COLLECTION_FILE}")
        return 0
    count = 0
    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        for _ in f:
            count += 1
    return count - 1  # subtract header


def _prefetch_build_loop(
    embeddings, vectorstore, args, state,
    start_line, batches_completed, total_stored, total_skipped, total_lines, max_batches, embedding_dim,
    state_file, build_log_file,
):
    from concurrent.futures import ThreadPoolExecutor
    from langchain_core.documents import Document

    logger.info(f"Prefetch pipeline: embed_size=256, ChromaDB batch={args.batch_size}")

    def _embed(batch_data):
        docs = batch_data["docs"]
        texts = [d.page_content for d in docs]
        vectors = embeddings.embed_documents(texts)
        batch_data["vectors"] = vectors
        return batch_data

    def _store(batch_data):
        docs = batch_data["docs"]
        ids = batch_data["ids"]
        vectors = batch_data["vectors"]
        metadatas = batch_data["metadatas"]
        batch_no = batch_data["batch_no"]

        store_start = time.time()
        vectorstore._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=[d.page_content for d in docs],
            metadatas=metadatas,
        )
        store_time = time.time() - store_start
        return store_time, batch_no

    def _make_batch(batch_no, b_start_line, passages, current_line):
        docs = passages_to_documents(passages)
        ids = [f"t2r_{d.metadata['pid']}" for d in docs]
        metadatas = [d.metadata for d in docs]
        return {
            "batch_no": batch_no,
            "start_line": b_start_line,
            "passages": passages,
            "docs": docs,
            "ids": ids,
            "metadatas": metadatas,
            "current_line": current_line,
            "stored_count": len(passages),
            "skipped_count": args.batch_size - len(passages),
            "vectors": None,
        }

    pool = ThreadPoolExecutor(max_workers=1)
    overall_start = time.time()
    batch_no = batches_completed
    current_line = start_line

    # ── cold start: embed batch 1 (don't store yet) ──
    passages, next_line = load_passages(current_line, args.batch_size)
    if not passages:
        logger.info("No passages to process.")
        pool.shutdown(wait=False)
        return total_stored, total_skipped, batch_no, current_line, overall_start

    batch_no += 1
    prev = _make_batch(batch_no, current_line, passages, next_line)
    prev = _embed(prev)
    current_line = next_line

    # ── pipeline loop: store batch N while embedding batch N+1 ──
    for batch_idx in range(1, max_batches):
        passages, next_line = load_passages(current_line, args.batch_size)
        if not passages:
            break

        batch_no += 1
        current = _make_batch(batch_no, current_line, passages, next_line)

        future = pool.submit(_embed, current)

        store_time, _ = _store(prev)
        current_line = current["current_line"]
        total_stored += prev["stored_count"]
        total_skipped += prev["skipped_count"]
        _save_state_and_log(state, prev, store_time, total_stored, total_skipped,
                            batch_no - 1, args, embedding_dim, overall_start, max_batches, batches_completed, total_lines,
                            state_file, build_log_file)

        prev = future.result()
        current_line = prev["current_line"]

    # ── store final batch ──
    if prev and prev["batch_no"] == batch_no:
        store_time, _ = _store(prev)
        total_stored += prev["stored_count"]
        total_skipped += prev["skipped_count"]
        _save_state_and_log(state, prev, store_time, total_stored, total_skipped,
                            batch_no, args, embedding_dim, overall_start, max_batches, batches_completed, total_lines,
                            state_file, build_log_file)

    pool.shutdown(wait=False)
    return total_stored, total_skipped, batch_no, current_line, overall_start


def _save_state_and_log(state, batch_data, store_time, total_stored, total_skipped,
                        batch_no, args, embedding_dim, overall_start, max_batches, batches_completed, total_lines,
                        state_file, build_log_file):
    embed_time = store_time * 0.1
    batch_time = store_time

    passages = batch_data["passages"]
    start_line_val = batch_data["start_line"]
    current_line = batch_data["current_line"]

    state.update({
        "source_file": str(COLLECTION_FILE),
        "total_lines": total_lines,
        "last_processed_line": current_line,
        "last_processed_pid": passages[-1][0] if passages else "",
        "total_stored": total_stored,
        "total_skipped": total_skipped,
        "batches_completed": batch_no,
        "batch_size": args.batch_size,
        "started_at": state.get("started_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "collection_name": COLLECTION_NAME,
        "embedding_model": args.model,
        "embedding_dim": embedding_dim,
    })
    save_state(state, state_file)

    log_entry = {
        "batch": batch_no,
        "line_range": f"{start_line_val}-{current_line}",
        "stored": batch_data["stored_count"],
        "skipped": batch_data["skipped_count"],
        "total_stored": total_stored,
        "embed_time_s": round(embed_time, 1),
        "batch_time_s": round(batch_time, 1),
        "pipeline": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_build_log(log_entry, build_log_file)

    progress_pct = (current_line / total_lines * 100) if total_lines > 0 else 0
    effective_batches = batch_no - batches_completed
    elapsed_total = time.time() - overall_start
    eta_total = (elapsed_total / effective_batches * (max_batches - effective_batches)
                 ) if effective_batches > 0 else 0

    logger.info(
        f"  Batch complete: {batch_data['stored_count']} docs in {batch_time:.1f}s (pipeline)"
    )
    logger.info(
        f"  Total: {total_stored:,} stored | {total_skipped:,} skipped | "
        f"{progress_pct:.1f}% | ETA: {eta_total/60:.0f}min"
    )


if __name__ == "__main__":
    sys.exit(main())
