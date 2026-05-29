"""
Exp-002: Query Rewriting Experiments

Unified entry point for all exp-002 experiments (E2a / E2b / E2c / E2d).
Route different strategies through a single --experiment flag.

Strategies:
  none       → original query, straight Dense retrieval
  single     → 1 LLM call → 1 rewritten query → Dense retrieval
  multi_query→ 1 LLM call → N sub-queries → N Dense retrievals → RRF fusion
  hyde       → 1 LLM call → fake answer → Dense retrieval with answer
  hyde_rrf   → 1 LLM call → fake answer → original + answer 2-way RRF
  prf        → no LLM → first-pass Dense → TF-IDF terms → second-pass Dense

Usage:
  python scripts/evaluate_exp002.py --experiment E2a-B1 --sample 2000 --device cuda --dense-only
  python scripts/evaluate_exp002.py --experiment E2b-M1 --sample 2000 --device cuda --dense-only
  python scripts/evaluate_exp002.py --experiment E2c-H2 --sample 2000 --device cuda --dense-only
  python scripts/evaluate_exp002.py --experiment E2d-P1 --sample 2000 --device cuda --dense-only
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

from src.utils.config import RAW_DATA_DIR, VECTOR_DB_DIR, DATA_ROOT, EMBEDDING_MODEL, resolve_model_local_path
from src.evaluation.data_loader import load_queries, load_qrels, load_passages
from src.evaluation.metrics import compute_metrics, print_comparison, get_metric_params
from src.evaluation.result_cache import save_results, load_results
from src.intent.query_rewrite_prompts import REGISTRY, get_experiment_config, list_experiments

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_FILE = T2RANKING_DIR / "queries.dev.tsv"
QRELS_FILE = T2RANKING_DIR / "qrels.retrieval.dev.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"
RESULTS_DIR = DATA_ROOT / "results"


def _get_llm(max_tokens: int = 128):
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=max_tokens,
    )


# ── LLM call + cache orchestration ───────────────────────


def _ensure_llm_output(
    experiment_id: str,
    queries: list[tuple[str, str]],
    cfg: dict,
    cache,
    llm_concurrency: int = 20,
) -> dict[str, str | list[str]]:
    from langchain_core.prompts import ChatPromptTemplate

    strategy = cfg["strategy"]
    if strategy in ("none", "prf"):
        return {}  # no LLM needed

    cached_all = cache.get_batch(experiment_id, [qid for qid, _ in queries])
    missing = [(qid, text) for qid, text in queries if qid not in cached_all]

    if not missing:
        logger.info(f"All {len(queries)} queries cached for {experiment_id}")
        return cached_all

    logger.info(
        f"Calling LLM for {len(missing)}/{len(queries)} queries "
        f"(cached: {len(cached_all)}, experiment={experiment_id}, "
        f"concurrency={llm_concurrency})"
    )

    llm = _get_llm(max_tokens=cfg["max_tokens"])

    if strategy in ("hyde", "hyde_rrf"):
        chain = ChatPromptTemplate.from_messages([("system", cfg["system"])]) | llm
    else:
        system_msg = cfg["system"]
        human_msg = cfg["human"]
        chain = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg),
        ]) | llm

    batch_size = llm_concurrency
    total_processed = 0

    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        inputs = [{"query": text} for _, text in chunk]

        results = chain.batch(inputs, config={"max_concurrency": batch_size}, return_exceptions=True)

        for (qid, text), result in zip(chunk, results):
            try:
                if isinstance(result, Exception):
                    raise result
                output = result.content.strip() if hasattr(result, "content") else str(result).strip()
                parsed = _parse_llm_output(output, cfg["output_parser"])
            except Exception as e:
                logger.warning(f"LLM call failed for qid={qid}: {e}, using original")
                parsed = text if cfg["output_parser"] == "text" else [text]

            cached_all[qid] = parsed
            cache.put(experiment_id, qid, text, parsed)
            total_processed += 1

        logger.info(f"  LLM progress: {total_processed}/{len(missing)}")

    logger.info(f"LLM calls complete: {total_processed} queries processed, {total_processed} cached")
    return cached_all


def _parse_llm_output(output: str, parser: str) -> str | list[str]:
    if parser == "text":
        return output.replace("\n", " ").strip()
    elif parser == "json_list":
        output = output.strip()
        if output.startswith("```"):
            output = output.split("\n", 1)[-1]
            if output.endswith("```"):
                output = output[:-3]
        obj = json.loads(output)
        return obj.get("sub_queries", [output])
    elif parser == "json_obj":
        output = output.strip()
        if output.startswith("```"):
            lines = output.split("\n")
            output = "\n".join(lines[1:-1])
        obj = json.loads(output)
        return obj.get("sub_queries", [output])
    else:
        return output


# ── dense retriever loader ────────────────────────────────


def _load_dense_retriever(vector_db_dir: str, collection_name: str, device: str,
                          search_type: str, model_id: str = None, top_k: int = 10):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    if model_id is None:
        model_id = EMBEDDING_MODEL

    local_path = resolve_model_local_path(model_id)
    model_name = str(local_path.resolve()) if local_path is not None else model_id

    logger.info(f"Dense retriever using embedding model: {model_name}")

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

    search_kwargs = {"k": top_k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = max(top_k * 2, 20)
        search_kwargs["lambda_mult"] = 0.5

    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs), vs, count


# ── strategy handlers ────────────────────────────────────


def _run_strategy_none(
    vs,
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    dense_search_type: str,
) -> list[dict]:
    results = []
    dense_total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": query_text, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        docs_with_scores = vs.similarity_search_with_score(query_text, k=top_k)
        dense_elapsed = time.time() - t0
        dense_total_time += dense_elapsed

        entry["retrievals"][dense_search_type] = [
            {"pid": doc.metadata.get("pid", "?"),
             "score": round(1.0 - float(score), 6),
             "rank": rank}
            for rank, (doc, score) in enumerate(docs_with_scores, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"Dense {dense_total_time/(i+1)*1000:.0f}ms/q"
            )

    logger.info(f"Dense total search time: {dense_total_time:.1f}s "
                f"({dense_total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_strategy_single(
    vs,
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    dense_search_type: str,
) -> list[dict]:
    results = []
    dense_total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        rewritten = llm_outputs.get(qid, original_text)
        if isinstance(rewritten, list):
            rewritten = rewritten[0] if rewritten else original_text

        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": rewritten, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        docs_with_scores = vs.similarity_search_with_score(rewritten, k=top_k)
        dense_elapsed = time.time() - t0
        dense_total_time += dense_elapsed

        entry["retrievals"][dense_search_type] = [
            {"pid": doc.metadata.get("pid", "?"),
             "score": round(1.0 - float(score), 6),
             "rank": rank}
            for rank, (doc, score) in enumerate(docs_with_scores, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"Dense {dense_total_time/(i+1)*1000:.0f}ms/q"
            )

    logger.info(f"Dense total search time: {dense_total_time:.1f}s "
                f"({dense_total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_strategy_multi_query(
    vs,
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    rrf_k: int,
) -> list[dict]:
    from collections import defaultdict

    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        sub_queries = llm_outputs.get(qid, [original_text])
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]

        relevant = qrels.get(qid, set())
        entry = {
            "qid": qid,
            "query": original_text,
            "relevant_pids": relevant,
            "retrievals": {},
        }

        t0 = time.time()
        rrf_scores: dict[str, float] = defaultdict(float)
        sub_query_top_k = top_k * 2

        for si, sq in enumerate(sub_queries):
            docs_with_scores = vs.similarity_search_with_score(sq, k=sub_query_top_k)
            sub_results = []
            for rank, (doc, dist) in enumerate(docs_with_scores, 1):
                pid = doc.metadata.get("pid", "?")
                score = round(1.0 - float(dist), 6)
                sub_results.append({"pid": pid, "score": score, "rank": rank})
                if rank <= sub_query_top_k:
                    rrf_scores[pid] += 1.0 / (rrf_k + rank)
            entry["retrievals"][f"sub_query_{si}"] = sub_results

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = time.time() - t0
        total_time += elapsed

        key = f"multi_query_rrf@k{rrf_k}"
        entry["retrievals"][key] = [
            {"pid": pid, "score": round(score, 6), "rank": rank}
            for rank, (pid, score) in enumerate(merged, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"MultiQuery {total_time/(i+1)*1000:.0f}ms/q"
            )

    logger.info(f"MultiQuery total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_strategy_hyde(
    vs,
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    top_k: int,
    rrf_k: int = None,
    with_original: bool = False,
) -> list[dict]:
    from collections import defaultdict

    rrf_k = rrf_k or 60
    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        fake_answer = llm_outputs.get(qid, original_text)
        if isinstance(fake_answer, list):
            fake_answer = fake_answer[0] if fake_answer else original_text

        relevant = qrels.get(qid, set())
        entry = {
            "qid": qid,
            "query": original_text,
            "relevant_pids": relevant,
            "retrievals": {},
        }

        t0 = time.time()
        if with_original:
            sub_query_top_k = top_k * 2

            hyde_docs = vs.similarity_search_with_score(fake_answer, k=sub_query_top_k)
            entry["retrievals"]["hyde_sub"] = [
                {"pid": doc.metadata.get("pid", "?"),
                 "score": round(1.0 - float(dist), 6),
                 "rank": rank}
                for rank, (doc, dist) in enumerate(hyde_docs, 1)
            ]

            orig_docs = vs.similarity_search_with_score(original_text, k=sub_query_top_k)
            entry["retrievals"]["original_sub"] = [
                {"pid": doc.metadata.get("pid", "?"),
                 "score": round(1.0 - float(dist), 6),
                 "rank": rank}
                for rank, (doc, dist) in enumerate(orig_docs, 1)
            ]

            rrf_scores: dict[str, float] = defaultdict(float)
            for rank, (doc, _) in enumerate(hyde_docs, 1):
                pid = doc.metadata.get("pid", "?")
                rrf_scores[pid] += 1.0 / (rrf_k + rank)
            for rank, (doc, _) in enumerate(orig_docs, 1):
                pid = doc.metadata.get("pid", "?")
                rrf_scores[pid] += 1.0 / (rrf_k + rank)

            merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            key = "hyde_rrf"
            entry["retrievals"][key] = [
                {"pid": pid, "score": round(score, 6), "rank": rank}
                for rank, (pid, score) in enumerate(merged, 1)
            ]
        else:
            hyde_docs = vs.similarity_search_with_score(fake_answer, k=top_k)
            key = "hyde"
            entry["retrievals"][key] = [
                {"pid": doc.metadata.get("pid", "?"),
                 "score": round(1.0 - float(dist), 6),
                 "rank": rank}
                for rank, (doc, dist) in enumerate(hyde_docs, 1)
            ]

        elapsed = time.time() - t0
        total_time += elapsed

        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"HyDE {total_time/(i+1)*1000:.0f}ms/q"
            )

    logger.info(f"HyDE total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_strategy_prf(
    vs,
    pool_queries: list[tuple[str, str]],
    passages: list[str],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    cfg: dict,
) -> list[dict]:
    import math
    from collections import Counter

    prf_feedback_k = cfg.get("prf_top_k", 50)
    num_terms = cfg.get("prf_num_terms", 5)
    weighted = cfg.get("prf_weighted", False)

    results = []
    total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {
            "qid": qid,
            "query": query_text,
            "relevant_pids": relevant,
            "retrievals": {},
        }

        t0 = time.time()

        first_docs = vs.similarity_search_with_score(query_text, k=prf_feedback_k)
        first_pass = [
            {"pid": doc.metadata.get("pid", "?"),
             "score": round(1.0 - float(dist), 6),
             "rank": rank}
            for rank, (doc, dist) in enumerate(first_docs, 1)
        ]
        entry["retrievals"]["prf_first_pass"] = first_pass

        feedback_pids = [item["pid"] for item in first_pass]
        feedback_indices = [pids.index(pid) for pid in feedback_pids if pid in pids]
        feedback_texts = [passages[j] for j in feedback_indices]

        query_terms = set(query_text)
        term_doc_counts: dict[str, Counter] = {}
        for ft in feedback_texts:
            for term in ft:
                if term in query_terms:
                    continue
                term_doc_counts.setdefault(term, Counter())[ft] += 1

        idf_cache = {}
        for term in term_doc_counts:
            df = sum(1 for t in feedback_texts if term in t)
            idf_cache[term] = math.log((len(feedback_texts) - df + 0.5) / (df + 0.5) + 1.0)

        if weighted:
            term_scores = {}
            for term, counter in term_doc_counts.items():
                tf_sum = sum(counter.values())
                term_scores[term] = tf_sum / len(feedback_texts) * idf_cache[term]
        else:
            term_scores = {}
            for term, counter in term_doc_counts.items():
                term_scores[term] = len(counter) * idf_cache[term]

        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [t for t, _ in sorted_terms[:num_terms]]

        expanded_query = query_text + " " + " ".join(expansion_terms) if expansion_terms else query_text

        second_docs = vs.similarity_search_with_score(expanded_query, k=top_k)
        key = f"prf_t{num_terms}"
        if weighted:
            key += "_w"
        entry["retrievals"][key] = [
            {"pid": doc.metadata.get("pid", "?"),
             "score": round(1.0 - float(dist), 6),
             "rank": rank}
            for rank, (doc, dist) in enumerate(second_docs, 1)
        ]

        elapsed = time.time() - t0
        total_time += elapsed

        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(
                f"  Progress: {i+1}/{len(pool_queries)} | "
                f"PRF {total_time/(i+1)*1000:.0f}ms/q"
            )

    logger.info(f"PRF total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


# ── BM25 strategy handlers ────────────────────────────────

def _load_bm25_index(bm25_index_dir: "Path", default_dir: "Path"):
    import bm25s
    from src.retrieval.bm25_store import load as sharded_load

    index_path = bm25_index_dir

    # 1. Check for sharded BM25S (shards.json)
    if (index_path / "shards.json").exists():
        logger.info(f"Loading sharded BM25S index from {index_path}...")
        from src.retrieval.bm25s_store import ShardedBM25S
        bm25 = ShardedBM25S.load(str(index_path))
        return bm25, None

    # 2. Check for single BM25S (params.index.json)
    if (index_path / "params.index.json").exists():
        logger.info(f"Loading single BM25S index from {index_path}...")
        bm25 = bm25s.BM25.load(str(index_path))
        return bm25, None

    # 3. Fall back to ShardedBM25
    logger.info(f"Loading ShardedBM25 index from {index_path}...")
    bm25, tokenized_corpus = sharded_load(index_path)
    return bm25, tokenized_corpus


def _topk_indices(scores: "np.ndarray", k: int) -> list[int]:
    import numpy as np

    if len(scores) <= k:
        return np.argsort(scores)[::-1].tolist()
    n = len(scores)
    partition_idx = np.argpartition(scores, n - k)[n - k:]
    topk = partition_idx[np.argsort(scores[partition_idx])[::-1]]
    return topk.tolist()


def _run_bm25_strategy_none(
    bm25,
    tokenized_corpus=None,
    pool_queries: list[tuple[str, str]] = None,
    pids: list[str] = None,
    qrels: dict[str, set[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    from src.retrieval.bm25_store import tokenize_query

    results = []
    total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": query_text, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        tokenized_query = tokenize_query(query_text)
        scores = bm25.get_scores(tokenized_query)
        top_idx = _topk_indices(scores, top_k)
        elapsed = time.time() - t0
        total_time += elapsed

        entry["retrievals"]["bm25"] = [
            {"pid": pids[j], "score": round(float(scores[j]), 6), "rank": rank}
            for rank, j in enumerate(top_idx, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25 {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_single(
    bm25,
    tokenized_corpus=None,
    llm_outputs: dict[str, str | list[str]] = None,
    pool_queries: list[tuple[str, str]] = None,
    pids: list[str] = None,
    qrels: dict[str, set[str]] = None,
    top_k: int = 10,
) -> list[dict]:
    from src.retrieval.bm25_store import tokenize_query

    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        rewritten = llm_outputs.get(qid, original_text)
        if isinstance(rewritten, list):
            rewritten = rewritten[0] if rewritten else original_text

        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": rewritten, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        tokenized_query = tokenize_query(rewritten)
        scores = bm25.get_scores(tokenized_query)
        top_idx = _topk_indices(scores, top_k)
        elapsed = time.time() - t0
        total_time += elapsed

        entry["retrievals"]["bm25"] = [
            {"pid": pids[j], "score": round(float(scores[j]), 6), "rank": rank}
            for rank, j in enumerate(top_idx, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25 {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_multi_query(
    bm25,
    tokenized_corpus=None,
    llm_outputs: dict[str, str | list[str]] = None,
    pool_queries: list[tuple[str, str]] = None,
    pids: list[str] = None,
    qrels: dict[str, set[str]] = None,
    top_k: int = 50,
    rrf_k: int = 60,
) -> list[dict]:
    from collections import defaultdict
    from src.retrieval.bm25_store import tokenize_query

    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        sub_queries = llm_outputs.get(qid, [original_text])
        if isinstance(sub_queries, str):
            sub_queries = [sub_queries]

        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": original_text, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        rrf_scores: dict[str, float] = defaultdict(float)
        sub_query_top_k = top_k * 2
        for si, sq in enumerate(sub_queries):
            tokenized_sq = tokenize_query(sq)
            sq_scores = bm25.get_scores(tokenized_sq)
            top_idx = _topk_indices(sq_scores, sub_query_top_k)
            sub_results = []
            for local_rank, j in enumerate(top_idx):
                pid = pids[j]
                rrf_scores[pid] += 1.0 / (rrf_k + local_rank + 1)
                sub_results.append({
                    "pid": pid,
                    "score": round(float(sq_scores[j]), 6),
                    "rank": local_rank + 1,
                })
            entry["retrievals"][f"bm25_sub_{si}"] = sub_results

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = time.time() - t0
        total_time += elapsed

        key = f"bm25_multi_query_rrf@k{rrf_k}"
        entry["retrievals"][key] = [
            {"pid": pid, "score": round(score, 6), "rank": rank}
            for rank, (pid, score) in enumerate(merged, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25-MQ {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 MultiQuery total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_hyde(
    bm25,
    tokenized_corpus=None,
    llm_outputs: dict[str, str | list[str]] = None,
    pool_queries: list[tuple[str, str]] = None,
    pids: list[str] = None,
    qrels: dict[str, set[str]] = None,
    top_k: int = 50,
    rrf_k: int = None,
    with_original: bool = False,
) -> list[dict]:
    from collections import defaultdict
    from src.retrieval.bm25_store import tokenize_query

    rrf_k = rrf_k or 60
    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        fake_answer = llm_outputs.get(qid, original_text)
        if isinstance(fake_answer, list):
            fake_answer = fake_answer[0] if fake_answer else original_text

        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": original_text, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()
        if with_original:
            sub_query_top_k = top_k * 2

            hyde_tokens = tokenize_query(fake_answer)
            hyde_scores = bm25.get_scores(hyde_tokens)
            hyde_top = _topk_indices(hyde_scores, sub_query_top_k)
            entry["retrievals"]["hyde_sub"] = [
                {"pid": pids[j], "score": round(float(hyde_scores[j]), 6), "rank": rank}
                for rank, j in enumerate(hyde_top, 1)
            ]

            orig_tokens = tokenize_query(original_text)
            orig_scores = bm25.get_scores(orig_tokens)
            orig_top = _topk_indices(orig_scores, sub_query_top_k)
            entry["retrievals"]["original_sub"] = [
                {"pid": pids[j], "score": round(float(orig_scores[j]), 6), "rank": rank}
                for rank, j in enumerate(orig_top, 1)
            ]

            rrf_score_map: dict[str, float] = defaultdict(float)
            for rank, j in enumerate(hyde_top):
                rrf_score_map[pids[j]] += 1.0 / (rrf_k + rank + 1)
            for rank, j in enumerate(orig_top):
                rrf_score_map[pids[j]] += 1.0 / (rrf_k + rank + 1)
            merged = sorted(rrf_score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
            key = "hyde_rrf"
            entry["retrievals"][key] = [
                {"pid": pid, "score": round(score, 6), "rank": rank}
                for rank, (pid, score) in enumerate(merged, 1)
            ]
        else:
            hyde_tokens = tokenize_query(fake_answer)
            hyde_scores = bm25.get_scores(hyde_tokens)
            hyde_top = _topk_indices(hyde_scores, top_k)
            key = "hyde"
            entry["retrievals"][key] = [
                {"pid": pids[j], "score": round(float(hyde_scores[j]), 6), "rank": rank}
                for rank, j in enumerate(hyde_top, 1)
            ]

        elapsed = time.time() - t0
        total_time += elapsed

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | "
                        f"BM25-HyDE {total_time/(i+1)*1000:.0f}ms/q")

        results.append(entry)

    logger.info(f"BM25 HyDE total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_prf(
    bm25,
    texts: list[str],
    pids: list[str],
    pool_queries: list[tuple[str, str]] = None,
    qrels: dict[str, set[str]] = None,
    top_k: int = 50,
    cfg: dict = None,
) -> list[dict]:
    import math
    from collections import Counter
    from src.retrieval.bm25_store import tokenize_query

    prf_feedback_k = cfg.get("prf_top_k", 50)
    num_terms = cfg.get("prf_num_terms", 5)
    weighted = cfg.get("prf_weighted", False)

    results = []
    total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {"qid": qid, "query": query_text, "relevant_pids": relevant, "retrievals": {}}

        t0 = time.time()

        query_tokens = tokenize_query(query_text)
        first_scores = bm25.get_scores(query_tokens)
        first_top = _topk_indices(first_scores, prf_feedback_k)

        entry["retrievals"]["prf_first_pass"] = [
            {"pid": pids[j], "score": round(float(first_scores[j]), 6), "rank": rank}
            for rank, j in enumerate(first_top, 1)
        ]

        feedback_texts = [texts[j] for j in first_top]

        query_terms = set(query_tokens)
        term_doc_counts: dict[str, Counter] = {}
        for ft in feedback_texts:
            ft_tokens = tokenize_query(ft)
            for term in ft_tokens:
                if term in query_terms or len(term) < 2:
                    continue
                term_doc_counts.setdefault(term, Counter())[ft] += 1

        idf_cache = {}
        for term in term_doc_counts:
            df = sum(1 for t in feedback_texts if term in t)
            idf_cache[term] = math.log((len(feedback_texts) - df + 0.5) / (df + 0.5) + 1.0)

        if weighted:
            term_scores = {}
            for term, counter in term_doc_counts.items():
                tf_sum = sum(counter.values())
                term_scores[term] = tf_sum / len(feedback_texts) * idf_cache[term]
        else:
            term_scores = {}
            for term, counter in term_doc_counts.items():
                term_scores[term] = len(counter) * idf_cache[term]

        sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        expansion_terms = [t for t, _ in sorted_terms[:num_terms]]

        if expansion_terms:
            expanded_tokens = query_tokens + expansion_terms
        else:
            expanded_tokens = query_tokens

        second_scores = bm25.get_scores(expanded_tokens)
        second_top = _topk_indices(second_scores, top_k)

        elapsed = time.time() - t0
        total_time += elapsed

        key = f"prf_t{num_terms}"
        if weighted:
            key += "_w"
        entry["retrievals"][key] = [
            {"pid": pids[j], "score": round(float(second_scores[j]), 6), "rank": rank}
            for rank, j in enumerate(second_top, 1)
        ]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | "
                        f"BM25-PRF {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 PRF total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


# ── main pipeline ────────────────────────────────────────


def _filter_pool_queries(
    valid_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
) -> list[tuple[str, str]]:
    pool_pids = set(pids)
    pool_queries = []
    for qid, text in valid_queries:
        relevant = qrels.get(qid, set())
        if not relevant:
            continue
        missing = relevant - pool_pids
        if not missing or len(relevant - missing) > 0:
            pool_queries.append((qid, text))
    logger.info(
        f"Queries with relevant passages in pool: {len(pool_queries)} / {len(valid_queries)}"
    )
    return pool_queries


def run_experiment(
    experiment_id: str,
    sample_size: int,
    top_k: int,
    device: str,
    vector_db_dir: str,
    collection_name: str,
    dense_search_type: str,
    save_path: str,
    model_id: str = None,
    llm_concurrency: int = 20,
    use_bm25: bool = False,
    bm25_index_dir: str = None,
):
    cfg = get_experiment_config(experiment_id)
    strategy = cfg["strategy"]
    logger.info(f"Experiment: {experiment_id} ({cfg['name']}, strategy={strategy})")

    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    if sample_size > len(queries):
        sample_size = len(queries)
    sampled = queries[:sample_size]

    valid_queries = [(qid, text) for qid, text in sampled if qid in qrels and qrels[qid]]
    logger.info(f"Sampled queries with qrels: {len(valid_queries)} / {sample_size}")

    from src.retrieval.rewrite_cache import RewriteCache
    cache = RewriteCache(RESULTS_DIR)
    llm_outputs = _ensure_llm_output(experiment_id, valid_queries, cfg, cache, llm_concurrency)

    pids, texts = load_passages(COLLECTION_FILE)
    pool_queries = _filter_pool_queries(valid_queries, pids, qrels)

    if not pool_queries:
        print("WARNING: No queries have relevant passages in the loaded pool!")
        return None

    if use_bm25:
        from src.retrieval.bm25_store import DEFAULT_STORE_DIR

        if bm25_index_dir is None:
            bm25_index_dir = str(DEFAULT_STORE_DIR)
        bm25_index_dir = Path(bm25_index_dir)

        bm25s_candidate = bm25_index_dir if (
            bm25_index_dir.exists() and (
                (bm25_index_dir / "params.index.json").exists() or
                (bm25_index_dir / "shards.json").exists()
            )
        ) else Path(str(DEFAULT_STORE_DIR).replace("bm25_index", "bm25s_index")) / "t2ranking"

        print()
        print("=" * 60)
        print("  Loading BM25 Index...")
        print("=" * 60)
        t0 = time.time()
        bm25, tokenized_corpus = _load_bm25_index(bm25s_candidate, DEFAULT_STORE_DIR)
        logger.info(f"BM25 index load: {time.time() - t0:.1f}s")

        print()
        print("=" * 60)
        print(f"  Running retrieval: {len(pool_queries)} queries x {len(pids)} passages")
        print(f"  Strategy: {strategy} | Experiment: {experiment_id} | BM25")
        print("=" * 60)

        if strategy == "none":
            results = _run_bm25_strategy_none(
                bm25, tokenized_corpus, pool_queries, pids, qrels, top_k,
            )
        elif strategy == "single":
            results = _run_bm25_strategy_single(
                bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
            )
        elif strategy == "multi_query":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_bm25_strategy_multi_query(
                bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k, rrf_k,
            )
        elif strategy == "hyde":
            results = _run_bm25_strategy_hyde(
                bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
                with_original=False,
            )
        elif strategy == "hyde_rrf":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_bm25_strategy_hyde(
                bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
                rrf_k=rrf_k, with_original=True,
            )
        elif strategy == "prf":
            results = _run_bm25_strategy_prf(
                bm25, texts, pids, pool_queries, qrels, top_k, cfg,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    else:
        print()
        print("=" * 60)
        print("  Loading Dense Retriever...")
        print("=" * 60)
        t0 = time.time()
        if vector_db_dir is None:
            vector_db_dir = str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5")
        dense_retriever, vs, dense_count = _load_dense_retriever(
            vector_db_dir, collection_name, device,
            search_type=dense_search_type, model_id=model_id, top_k=top_k,
        )
        logger.info(f"Dense retriever load: {time.time() - t0:.1f}s")

        print()
        print("=" * 60)
        print(f"  Running retrieval: {len(pool_queries)} queries x {len(pids)} passages")
        print(f"  Strategy: {strategy} | Experiment: {experiment_id}")
        print("=" * 60)

        if strategy == "none":
            results = _run_strategy_none(
                vs, pool_queries, pids, qrels, top_k, dense_search_type,
            )
        elif strategy == "single":
            results = _run_strategy_single(
                vs, llm_outputs, pool_queries, pids, qrels, top_k, dense_search_type,
            )
        elif strategy == "multi_query":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_strategy_multi_query(
                vs, llm_outputs, pool_queries, pids, qrels, top_k, rrf_k,
            )
        elif strategy == "hyde":
            results = _run_strategy_hyde(
                vs, llm_outputs, pool_queries, qrels, top_k, with_original=False,
            )
        elif strategy == "hyde_rrf":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_strategy_hyde(
                vs, llm_outputs, pool_queries, qrels, top_k,
                rrf_k=rrf_k, with_original=True,
            )
        elif strategy == "prf":
            results = _run_strategy_prf(
                vs, pool_queries, texts, pids, qrels, top_k, cfg,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    meta = {
        "experiment_id": experiment_id,
        "experiment_name": cfg["name"],
        "strategy": strategy,
        "sample_size": sample_size,
        "total_passages": len(pids),
        "pool_queries": len(pool_queries),
        "top_k": top_k,
        "dense_search_type": dense_search_type,
        "collection_name": collection_name,
        "dataset": "T2Ranking dev",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_results(results, save_path, meta)

    return results, meta


def run_experiments_batch(
    experiment_ids: list[str],
    sample_size: int,
    top_k: int,
    device: str,
    vector_db_dir: str,
    collection_name: str,
    dense_search_type: str,
    model_id: str = None,
    llm_concurrency: int = 20,
    use_bm25: bool = False,
    bm25_index_dir: str = None,
):
    cfgs = {}
    for eid in experiment_ids:
        cfgs[eid] = get_experiment_config(eid)

    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    if sample_size > len(queries):
        sample_size = len(queries)
    sampled = queries[:sample_size]

    valid_queries = [(qid, text) for qid, text in sampled if qid in qrels and qrels[qid]]
    logger.info(f"Sampled queries with qrels: {len(valid_queries)} / {sample_size}")

    from src.retrieval.rewrite_cache import RewriteCache
    cache = RewriteCache(RESULTS_DIR)

    pids, texts = load_passages(COLLECTION_FILE)
    pool_queries = _filter_pool_queries(valid_queries, pids, qrels)

    if not pool_queries:
        print("WARNING: No queries have relevant passages in the loaded pool!")
        return None

    if use_bm25:
        from src.retrieval.bm25_store import DEFAULT_STORE_DIR

        if bm25_index_dir is None:
            bm25_index_dir = str(DEFAULT_STORE_DIR)
        bm25_index_dir = Path(bm25_index_dir)

        bm25s_candidate = bm25_index_dir if (
            bm25_index_dir.exists() and (
                (bm25_index_dir / "params.index.json").exists() or
                (bm25_index_dir / "shards.json").exists()
            )
        ) else Path(str(DEFAULT_STORE_DIR).replace("bm25_index", "bm25s_index")) / "t2ranking"

        print()
        print("=" * 60)
        print("  Loading BM25 Index (shared across all experiments)...")
        print("=" * 60)
        t0 = time.time()
        bm25, tokenized_corpus = _load_bm25_index(bm25s_candidate, DEFAULT_STORE_DIR)
        logger.info(f"BM25 index load: {time.time() - t0:.1f}s")
    else:
        print()
        print("=" * 60)
        print("  Loading Dense Retriever (shared across all experiments)...")
        print("=" * 60)
        t0 = time.time()
        if vector_db_dir is None:
            vector_db_dir = str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5")
        dense_retriever, vs, dense_count = _load_dense_retriever(
            vector_db_dir, collection_name, device,
            search_type=dense_search_type, model_id=model_id, top_k=top_k,
        )
        logger.info(f"Dense retriever load: {time.time() - t0:.1f}s")

    mode_tag = "bm25" if use_bm25 else dense_search_type
    all_meta = {}

    for idx, experiment_id in enumerate(experiment_ids):
        cfg = cfgs[experiment_id]
        strategy = cfg["strategy"]

        print()
        print("=" * 60)
        print(f"  [{idx+1}/{len(experiment_ids)}] {experiment_id}: {cfg['name']}")
        print(f"  Strategy: {strategy} | {len(pool_queries)} queries x {len(pids)} passages")
        print("=" * 60)

        llm_outputs = _ensure_llm_output(
            experiment_id, valid_queries, cfg, cache, llm_concurrency,
        )

        if use_bm25:
            if strategy == "none":
                results = _run_bm25_strategy_none(
                    bm25, tokenized_corpus, pool_queries, pids, qrels, top_k,
                )
            elif strategy == "single":
                results = _run_bm25_strategy_single(
                    bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
                )
            elif strategy == "multi_query":
                rrf_k = cfg.get("rrf_k", 60)
                results = _run_bm25_strategy_multi_query(
                    bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k, rrf_k,
                )
            elif strategy == "hyde":
                results = _run_bm25_strategy_hyde(
                    bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
                    with_original=False,
                )
            elif strategy == "hyde_rrf":
                rrf_k = cfg.get("rrf_k", 60)
                results = _run_bm25_strategy_hyde(
                    bm25, tokenized_corpus, llm_outputs, pool_queries, pids, qrels, top_k,
                    rrf_k=rrf_k, with_original=True,
                )
            elif strategy == "prf":
                results = _run_bm25_strategy_prf(
                    bm25, texts, pids, pool_queries, qrels, top_k, cfg,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            if strategy == "none":
                results = _run_strategy_none(
                    vs, pool_queries, pids, qrels, top_k, dense_search_type,
                )
            elif strategy == "single":
                results = _run_strategy_single(
                    vs, llm_outputs, pool_queries, pids, qrels, top_k, dense_search_type,
                )
            elif strategy == "multi_query":
                rrf_k = cfg.get("rrf_k", 60)
                results = _run_strategy_multi_query(
                    vs, llm_outputs, pool_queries, pids, qrels, top_k, rrf_k,
                )
            elif strategy == "hyde":
                results = _run_strategy_hyde(
                    vs, llm_outputs, pool_queries, qrels, top_k, with_original=False,
                )
            elif strategy == "hyde_rrf":
                rrf_k = cfg.get("rrf_k", 60)
                results = _run_strategy_hyde(
                    vs, llm_outputs, pool_queries, qrels, top_k,
                    rrf_k=rrf_k, with_original=True,
                )
            elif strategy == "prf":
                results = _run_strategy_prf(
                    vs, pool_queries, texts, pids, qrels, top_k, cfg,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        meta = {
            "experiment_id": experiment_id,
            "experiment_name": cfg["name"],
            "strategy": strategy,
            "sample_size": sample_size,
            "total_passages": len(pids),
            "pool_queries": len(pool_queries),
            "top_k": top_k,
            "dense_search_type": dense_search_type,
            "collection_name": collection_name,
            "dataset": "T2Ranking dev",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_path = str(
            RESULTS_DIR / f"exp002_{experiment_id}_s{sample_size}_{mode_tag}.jsonl"
        )
        save_results(results, save_path, meta)

        retrievals_keys = set()
        for r in results:
            retrievals_keys.update(r["retrievals"].keys())

        metrics_map = {}
        k_values, metric_names = get_metric_params(top_k)
        for key in sorted(retrievals_keys):
            metrics_map[key] = compute_metrics(results, key, k_values=k_values)

        print_comparison(metrics_map, metric_names=metric_names)

        metrics_path = str(
            RESULTS_DIR / f"exp002_{experiment_id}_s{sample_size}_{mode_tag}_metrics.json"
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "metrics": metrics_map}, f, ensure_ascii=False, indent=2)

        all_meta[experiment_id] = metrics_map
        logger.info(f"Results: {save_path} | Metrics: {metrics_path}")

    _print_batch_summary(all_meta, experiment_ids, top_k)

    return all_meta


def _print_batch_summary(
    all_metrics: dict[str, dict],
    experiment_ids: list[str],
    top_k: int,
):
    print()
    print("=" * 80)
    print("  BATCH SUMMARY — All Experiments")
    print("=" * 80)

    _, metric_names = get_metric_params(top_k)

    extra_keys = set()
    for metrics_map in all_metrics.values():
        for method_key in metrics_map:
            for k in metrics_map[method_key]:
                if k not in metric_names:
                    extra_keys.add(k)
    all_names = metric_names + sorted(extra_keys)

    col_width = max(10, max(len(eid) for eid in experiment_ids) + 2)
    header = f"  {'Metric':<16}"
    for eid in experiment_ids:
        header += f" {eid:>{col_width}}"
    print(header)
    print("  " + "-" * (16 + len(experiment_ids) * (col_width + 1)))

    for metric_name in all_names:
        row = f"  {metric_name:<16}"
        for eid in experiment_ids:
            mm = all_metrics[eid]
            if not mm:
                row += f" {'--':>{col_width}}"
                continue
            primary = list(mm.keys())[0]
            val = mm[primary].get(metric_name, float("nan"))
            row += f" {val:>{col_width}.4f}"
        print(row)

    print("=" * 80)


def load_and_print(load_path: str, top_k: int = None):
    results, meta = load_results(load_path)

    if meta:
        print()
        print("=" * 60)
        print("  Cached Results Info")
        print("=" * 60)
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print("=" * 60)

    retrievals_keys = set()
    for r in results:
        retrievals_keys.update(r["retrievals"].keys())

    if top_k is not None:
        for r in results:
            for key in list(r["retrievals"].keys()):
                r["retrievals"][key] = r["retrievals"][key][:top_k]

    metrics_map = {}
    k_values, metric_names = get_metric_params(top_k)
    for key in sorted(retrievals_keys):
        metrics_map[key] = compute_metrics(results, key, k_values=k_values)

    print_comparison(metrics_map, metric_names=metric_names)
    return metrics_map


# ── CLI ───────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Exp-002: Query Rewriting Experiments (E2a / E2b / E2c / E2d)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available experiments:\n"
            + "\n".join(f"  {eid:12s}  {REGISTRY[eid]['name']}" for eid in list_experiments())
        ),
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment ID(s). Use comma to run multiple (e.g. E2a-B0,E2a-B1,E2a-P1). "
             "Batch mode loads the retriever once for all experiments.",
    )
    parser.add_argument("--sample", type=int, default=500, help="Number of queries to evaluate")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K for retrieval")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument(
        "--vector-db", default=None,
        help="Vector DB directory (default: data/vector_db/t2ranking/bge-small-zh-v1.5)",
    )
    parser.add_argument("--collection-name", default="t2ranking_passages")
    parser.add_argument(
        "--dense-strategy", default="similarity",
        choices=["similarity", "mmr"],
        help="Dense retrieval strategy",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="Embedding model HF id or short key (default: config.yaml indexing.embedding_model)",
    )
    parser.add_argument("--save", default=None, help="Save retrieval results to JSONL file")
    parser.add_argument("--load", default=None, help="Load cached results from JSONL (skip retrieval)")
    parser.add_argument(
        "--bm25", action="store_true",
        help="Use BM25 (sparse) retrieval instead of Dense (vector) retrieval",
    )
    parser.add_argument(
        "--bm25-index", default=None,
        help="Directory of pre-built BM25S or ShardedBM25 index "
             "(auto-detected; build with scripts/build_bm25s_index.py first)",
    )
    parser.add_argument(
        "--llm-concurrency", type=int, default=20,
        help="Max concurrent LLM API calls (default: 20). Set lower if hitting HTTP 429.",
    )
    parser.add_argument("--list", action="store_true", help="List available experiments and exit")

    args = parser.parse_args()

    if args.list:
        print("Available experiments:")
        for eid in list_experiments():
            cfg = REGISTRY[eid]
            print(f"  {eid:12s}  strategy={cfg['strategy']:12s}  {cfg['name']}")
        return 0

    experiment_ids = [eid.strip() for eid in args.experiment.split(",")]

    for eid in experiment_ids:
        if eid not in REGISTRY:
            print(f"ERROR: Unknown experiment '{eid}'")
            print(f"Available: {list_experiments()}")
            return 1

    if args.load:
        load_and_print(args.load, top_k=args.top_k)
        return 0

    if len(experiment_ids) > 1:
        print(f"Batch mode: {len(experiment_ids)} experiments, "
              f"retriever loaded once for all")
        run_experiments_batch(
            experiment_ids=experiment_ids,
            sample_size=args.sample,
            top_k=args.top_k,
            device=args.device,
            vector_db_dir=args.vector_db,
            collection_name=args.collection_name,
            dense_search_type=args.dense_strategy,
            model_id=args.embedding_model,
            llm_concurrency=args.llm_concurrency,
            use_bm25=args.bm25,
            bm25_index_dir=args.bm25_index,
        )
        return 0

    experiment_id = experiment_ids[0]

    save_path = args.save
    if not save_path:
        mode_tag = "bm25" if args.bm25 else args.dense_strategy
        save_path = str(
            RESULTS_DIR / f"exp002_{experiment_id}_s{args.sample}_{mode_tag}.jsonl"
        )

    output = run_experiment(
        experiment_id=experiment_id,
        sample_size=args.sample,
        top_k=args.top_k,
        device=args.device,
        vector_db_dir=args.vector_db,
        collection_name=args.collection_name,
        dense_search_type=args.dense_strategy,
        save_path=save_path,
        model_id=args.embedding_model,
        llm_concurrency=args.llm_concurrency,
        use_bm25=args.bm25,
        bm25_index_dir=args.bm25_index,
    )

    print()
    print(f"Results saved to: {save_path}")
    print(f"LLM cache: {RESULTS_DIR}/rewrite_cache/{experiment_id}.jsonl")

    if output is not None and output[0] is not None:
        results, meta = output

        retrievals_keys = set()
        for r in results:
            retrievals_keys.update(r["retrievals"].keys())

        metrics_map = {}
        k_values, metric_names = get_metric_params(args.top_k)
        for key in sorted(retrievals_keys):
            metrics_map[key] = compute_metrics(results, key, k_values=k_values)

        print_comparison(metrics_map, metric_names=metric_names)

        metrics_path = str(
            RESULTS_DIR / f"exp002_{experiment_id}_s{args.sample}_{args.dense_strategy}_metrics.json"
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": meta,
                "metrics": metrics_map,
            }, f, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
