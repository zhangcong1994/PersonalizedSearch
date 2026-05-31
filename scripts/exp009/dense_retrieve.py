"""
Exp-009 Step 2.3: Dense Retrieval (3 routes).

Loads a pre-built FAISS IndexFlatIP + SentenceTransformer model.
Encodes queries from three routes (original / rewritten / HyDE), runs FAISS
top-K search, then collects all unique retrieved pids to load their texts
from collection.tsv only once. Outputs three JSONL files.

Usage:
  python scripts/exp009/dense_retrieve.py \
      --model models/m3e-base-t2ranking-phase3-2/ep1/merged \
      --device cuda \
      --input-queries data/processed/exp009_sampled_queries.jsonl \
      --input-rewritten data/processed/exp009_rewritten_queries.jsonl \
      --input-hyde data/processed/exp009_hyde_answers.jsonl \
      --output-b0 data/processed/exp009_dense_B0.jsonl \
      --output-p2 data/processed/exp009_dense_P2.jsonl \
      --output-h2 data/processed/exp009_dense_H2.jsonl \
      --top-k 50
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR, VECTOR_DB_DIR, model_short_name
from src.evaluation.data_loader import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"


def load_queries_texts(path: Path) -> list[tuple[str, str]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append((obj["qid"], obj["query"]))
    return queries


def load_augment_texts(path: Path) -> dict[str, str]:
    mapping = {}
    if not path.exists():
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("rewritten") or obj.get("hyde") or obj.get("query", "")
            mapping[obj["qid"]] = key
    return mapping


def resolve_model_path(model_id: str) -> str:
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)
    relative = (DATA_ROOT / model_id).resolve()
    if relative.is_dir():
        return str(relative)
    candidate = (Path.cwd() / model_id).resolve()
    if candidate.is_dir():
        return str(candidate)
    return model_id


RouteSpec = tuple[str, str, str, str]
# (route_tag, query_source, input_arg_name, output_path)


def load_faiss_index(model_id: str, device: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    short_name = model_short_name(model_id)
    index_dir = VECTOR_DB_DIR / "t2ranking" / short_name
    faiss_file = index_dir / "index.faiss"
    pids_file = index_dir / "pids.json"

    if not faiss_file.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {faiss_file}\n"
            f"  Build: python scripts/exp007/build_faiss_index.py "
            f"--model {model_id} --device {device}"
        )

    model_path = resolve_model_path(model_id)
    logger.info(f"Loading embedding model: {model_path}")
    model = SentenceTransformer(model_path, device=device)
    if device != "cpu":
        model.half()
        logger.info("  model → FP16")

    logger.info(f"Loading pids from {pids_file}")
    with open(pids_file, "r", encoding="utf-8") as f:
        pids: list[str] = json.load(f)
    logger.info(f"  {len(pids):,} pids")

    logger.info(f"Loading FAISS index from {faiss_file}")
    index = faiss.read_index(str(faiss_file))
    logger.info(f"  {index.ntotal:,} docs, dim={index.d}")

    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        logger.info("  FAISS → GPU")
    except Exception:
        logger.info("  FAISS GPU not available, using CPU")

    return model, index, pids


def retrieve_route(
    model,
    index,
    pids: list[str],
    queries: list[tuple[str, str]],
    batch_size: int,
    top_k: int,
    route_name: str,
) -> list[dict]:
    import numpy as np

    results: list[dict] = []
    total = len(queries)
    t_start = time.time()

    logger.info(f"[{route_name}] encoding + searching {total} queries...")

    for batch_start in range(0, total, batch_size):
        batch = queries[batch_start:batch_start + batch_size]
        qids = [qid for qid, _ in batch]
        texts = [text for _, text in batch]

        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        scores, indices = index.search(vecs, top_k)

        for i, qid in enumerate(qids):
            route_results = []
            for rank_idx in range(top_k):
                idx = indices[i][rank_idx]
                if idx < 0:
                    break
                route_results.append({
                    "pid": pids[idx],
                    "score": round(float(scores[i][rank_idx]), 6),
                    "rank": rank_idx + 1,
                })
            results.append({
                "qid": qid,
                "query": texts[i],
                "results": route_results,
            })

        n_done = batch_start + len(batch)
        if n_done % (batch_size * 20) == 0 or n_done >= total:
            elapsed = time.time() - t_start
            rate = n_done / max(elapsed, 0.1)
            eta = (total - n_done) / max(rate, 0.01) / 60
            logger.info(f"  [{route_name}] {n_done}/{total} | "
                        f"{rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"[{route_name}] done: {total} queries in {elapsed/60:.1f}min "
                f"({total/max(elapsed,0.1):.1f}q/s)")
    return results


def load_passage_texts(pids_to_load: set[str]) -> dict[str, str]:
    texts: dict[str, str] = {}
    missing = set(pids_to_load)
    logger.info(f"Loading texts for {len(pids_to_load):,} unique pids from collection.tsv...")
    t_start = time.time()

    with open(COLLECTION_FILE, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if not missing:
                break
            parts = line.strip().split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid in missing:
                raw = clean_text(parts[1])
                texts[pid] = raw[:2000] if len(raw) > 2000 else raw
                missing.discard(pid)

    elapsed = time.time() - t_start
    logger.info(f"Loaded {len(texts):,}/{len(pids_to_load):,} texts in {elapsed:.1f}s "
                f"({len(missing):,} not found)")
    return texts


def collect_all_pids(all_results: list[list[dict]]) -> set[str]:
    pids = set()
    for route_results in all_results:
        for entry in route_results:
            for r in entry["results"]:
                pids.add(r["pid"])
    return pids


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = DATA_ROOT / p
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 Step 2.3: Dense Retrieval (3 routes)"
    )
    parser.add_argument("--model", required=True,
                        help="Embedding model path or HF ID")
    parser.add_argument("--device", default="cuda",
                        help="Device for encoding + FAISS search (default: cuda)")
    parser.add_argument("--input-queries", required=True,
                        help="Sampled queries JSONL")
    parser.add_argument("--input-rewritten", default=None,
                        help="Rewritten queries JSONL")
    parser.add_argument("--input-hyde", default=None,
                        help="HyDE answers JSONL")
    parser.add_argument("--output-b0", required=True,
                        help="Output JSONL: original query → dense retrieval")
    parser.add_argument("--output-p2", default=None,
                        help="Output JSONL: rewritten query → dense retrieval")
    parser.add_argument("--output-h2", default=None,
                        help="Output JSONL: HyDE query → dense retrieval")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Number of passages to retrieve per query (default: 50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Encode batch size (default: 256)")
    args = parser.parse_args()

    inp_queries = _resolve(args.input_queries)
    inp_rw_path = _resolve(args.input_rewritten) if args.input_rewritten else None
    inp_hy_path = _resolve(args.input_hyde) if args.input_hyde else None
    out_b0 = _resolve(args.output_b0)
    out_p2 = _resolve(args.output_p2) if args.output_p2 else None
    out_h2 = _resolve(args.output_h2) if args.output_h2 else None

    logger.info("=" * 60)
    logger.info("  Exp-009 Step 2.3: Dense Retrieval")
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Device:     {args.device}")
    logger.info(f"  Top-K:      {args.top_k}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info("=" * 60)

    queries = load_queries_texts(inp_queries)
    logger.info(f"Loaded {len(queries)} queries")

    rewritten_map = load_augment_texts(inp_rw_path) if inp_rw_path else {}
    hyde_map = load_augment_texts(inp_hy_path) if inp_hy_path else {}

    # ── load FAISS once ──
    model, index, pids = load_faiss_index(args.model, args.device)

    # ── build route specs ──
    routes: list[tuple[str, list[tuple[str, str]], Path]] = []

    # B0: original queries
    routes.append(("B0", queries, out_b0))

    # P2: rewritten queries
    if out_p2 and rewritten_map:
        rw_queries = [(qid, rewritten_map.get(qid, text)) for qid, text in queries]
        routes.append(("P2", rw_queries, out_p2))
    elif out_p2:
        logger.warning("--output-p2 specified but no --input-rewritten; skipping P2")

    # H2: HyDE answers
    if out_h2 and hyde_map:
        hy_queries = [(qid, hyde_map.get(qid, text)) for qid, text in queries]
        routes.append(("H2", hy_queries, out_h2))
    elif out_h2:
        logger.warning("--output-h2 specified but no --input-hyde; skipping H2")

    # ── retrieve each route ──
    all_route_results: list[list[dict]] = []
    for route_name, route_queries, output_path in routes:
        logger.info("-" * 40)
        logger.info(f"Route {route_name}: {len(route_queries)} queries → {output_path}")
        results = retrieve_route(model, index, pids, route_queries,
                                 args.batch_size, args.top_k, route_name)
        all_route_results.append(results)

    # ── load passage texts for all unique pids ──
    all_pids = collect_all_pids(all_route_results)
    pid_to_text = load_passage_texts(all_pids)

    # ── write JSONL with texts ──
    for (route_name, route_queries, output_path), results in zip(routes, all_route_results):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in results:
                for r in entry["results"]:
                    r["text"] = pid_to_text.get(r["pid"], "")
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                written += 1
        logger.info(f"[{route_name}] wrote {written} entries → {output_path}")

    # ── stats ──
    logger.info("-" * 40)
    total_passages = sum(
        len(entry["results"]) for results in all_route_results for entry in results
    )
    logger.info(f"Total passages retrieved: {total_passages:,}")
    logger.info(f"Unique pids: {len(all_pids):,}")
    logger.info("=" * 60)
    logger.info("  Step 2.3 complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
