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
from src.evaluation.metrics import compute_metrics, print_comparison
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
                          search_type: str, model_id: str = None):
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

    search_kwargs = {"k": 10}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = 20
        search_kwargs["lambda_mult"] = 0.5

    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs), count


# ── strategy handlers ────────────────────────────────────


def _run_strategy_none(
    dense_retriever,
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
        docs = dense_retriever.invoke(query_text)
        dense_elapsed = time.time() - t0
        dense_total_time += dense_elapsed

        entry["retrievals"][dense_search_type] = [
            doc.metadata.get("pid", "?") for doc in docs[:top_k]
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
    dense_retriever,
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
        docs = dense_retriever.invoke(rewritten)
        dense_elapsed = time.time() - t0
        dense_total_time += dense_elapsed

        entry["retrievals"][dense_search_type] = [
            doc.metadata.get("pid", "?") for doc in docs[:top_k]
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
    dense_retriever,
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    rrf_k: int,
) -> list[dict]:
    from src.retrieval.multi_query import MultiQueryRetriever

    mqr = MultiQueryRetriever(dense_retriever, pids, rrf_k=rrf_k)
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
        merged = mqr.retrieve(sub_queries, top_k=top_k, original_query=original_text)
        elapsed = time.time() - t0
        total_time += elapsed

        key = f"multi_query_rrf@k{rrf_k}"
        entry["retrievals"][key] = merged
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
    dense_retriever,
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    qrels: dict[str, set[str]],
    top_k: int,
    rrf_k: int = None,
    with_original: bool = False,
) -> list[dict]:
    from src.retrieval.hyde import HyDERetriever

    hyde = HyDERetriever(dense_retriever, rrf_k=rrf_k or 60)
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
            merged = hyde.retrieve_hyde_with_query(original_text, fake_answer, top_k=top_k)
            key = "hyde_rrf"
        else:
            merged = hyde.retrieve_hyde_only(fake_answer, top_k=top_k)
            key = "hyde"
        elapsed = time.time() - t0
        total_time += elapsed

        entry["retrievals"][key] = merged
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
    dense_retriever,
    pool_queries: list[tuple[str, str]],
    passages: list[str],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    cfg: dict,
) -> list[dict]:
    from src.retrieval.prf import PRFRetriever

    prf = PRFRetriever(
        dense_retriever,
        prf_top_k=cfg.get("prf_top_k", 20),
        num_terms=cfg.get("prf_num_terms", 5),
        weighted=cfg.get("prf_weighted", False),
    )
    results = []
    total_time = 0.0

    for i, (qid, original_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {
            "qid": qid,
            "query": original_text,
            "relevant_pids": relevant,
            "retrievals": {},
        }

        t0 = time.time()
        retrieved = prf.retrieve(original_text, passages, pids, top_k=top_k)
        elapsed = time.time() - t0
        total_time += elapsed

        key = f"prf_t{cfg.get('prf_num_terms', 5)}"
        if cfg.get("prf_weighted"):
            key += "_w"
        entry["retrievals"][key] = retrieved
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


def _run_bm25_strategy_none(
    bm25,
    tokenized_corpus: list[list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
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
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_k]
        elapsed = time.time() - t0
        total_time += elapsed

        entry["retrievals"]["bm25"] = [pids[j] for j in top_idx]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25 {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_single(
    bm25,
    tokenized_corpus: list[list[str]],
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
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
        top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_k]
        elapsed = time.time() - t0
        total_time += elapsed

        entry["retrievals"]["bm25"] = [pids[j] for j in top_idx]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25 {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 total search time: {total_time:.1f}s "
                f"({total_time/len(pool_queries)*1000:.0f}ms/q)")
    return results


def _run_bm25_strategy_multi_query(
    bm25,
    tokenized_corpus: list[list[str]],
    llm_outputs: dict[str, str | list[str]],
    pool_queries: list[tuple[str, str]],
    pids: list[str],
    qrels: dict[str, set[str]],
    top_k: int,
    rrf_k: int,
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
        for rank, sq in enumerate(sub_queries):
            tokenized_sq = tokenize_query(sq)
            sq_scores = bm25.get_scores(tokenized_sq)
            top_idx = sorted(range(len(sq_scores)), key=lambda j: sq_scores[j], reverse=True)[:top_k * 2]
            for local_rank, j in enumerate(top_idx):
                pid = pids[j]
                rrf_scores[pid] += 1.0 / (rrf_k + local_rank + 1)

        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        elapsed = time.time() - t0
        total_time += elapsed

        key = f"bm25_multi_query_rrf@k{rrf_k}"
        entry["retrievals"][key] = [pid for pid, _ in merged]
        results.append(entry)

        if (i + 1) % 100 == 0:
            logger.info(f"  Progress: {i+1}/{len(pool_queries)} | BM25-MQ {total_time/(i+1)*1000:.0f}ms/q")

    logger.info(f"BM25 MultiQuery total search time: {total_time:.1f}s "
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
        from src.retrieval.bm25_store import load as bm25_load, DEFAULT_STORE_DIR

        if bm25_index_dir is None:
            bm25_index_dir = str(DEFAULT_STORE_DIR)

        print()
        print("=" * 60)
        print("  Loading BM25 Index...")
        print("=" * 60)
        t0 = time.time()
        bm25, tokenized_corpus = bm25_load(Path(bm25_index_dir))
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
        else:
            raise ValueError(f"BM25 mode does not support strategy: {strategy} "
                             f"(use none/single/multi_query)")
    else:
        print()
        print("=" * 60)
        print("  Loading Dense Retriever...")
        print("=" * 60)
        t0 = time.time()
        if vector_db_dir is None:
            vector_db_dir = str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5")
        dense_retriever, dense_count = _load_dense_retriever(
            vector_db_dir, collection_name, device,
            search_type=dense_search_type, model_id=model_id,
        )
        logger.info(f"Dense retriever load: {time.time() - t0:.1f}s")

        print()
        print("=" * 60)
        print(f"  Running retrieval: {len(pool_queries)} queries x {len(pids)} passages")
        print(f"  Strategy: {strategy} | Experiment: {experiment_id}")
        print("=" * 60)

        if strategy == "none":
            results = _run_strategy_none(
                dense_retriever, pool_queries, pids, qrels, top_k, dense_search_type,
            )
        elif strategy == "single":
            results = _run_strategy_single(
                dense_retriever, llm_outputs, pool_queries, pids, qrels, top_k, dense_search_type,
            )
        elif strategy == "multi_query":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_strategy_multi_query(
                dense_retriever, llm_outputs, pool_queries, pids, qrels, top_k, rrf_k,
            )
        elif strategy == "hyde":
            results = _run_strategy_hyde(
                dense_retriever, llm_outputs, pool_queries, qrels, top_k, with_original=False,
            )
        elif strategy == "hyde_rrf":
            rrf_k = cfg.get("rrf_k", 60)
            results = _run_strategy_hyde(
                dense_retriever, llm_outputs, pool_queries, qrels, top_k,
                rrf_k=rrf_k, with_original=True,
            )
        elif strategy == "prf":
            results = _run_strategy_prf(
                dense_retriever, pool_queries, texts, pids, qrels, top_k, cfg,
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
    for key in sorted(retrievals_keys):
        metrics_map[key] = compute_metrics(results, key)

    print_comparison(metrics_map)
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
        help="Experiment ID (e.g. E2a-B1, E2b-M1, E2c-H2, E2d-P1)",
    )
    parser.add_argument("--sample", type=int, default=500, help="Number of queries to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
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
        help="Directory of pre-built BM25 index (default: data/bm25_index)",
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

    if args.experiment not in REGISTRY:
        print(f"ERROR: Unknown experiment '{args.experiment}'")
        print(f"Available: {list_experiments()}")
        return 1

    if args.load:
        load_and_print(args.load, top_k=args.top_k)
        return 0

    save_path = args.save
    if not save_path:
        mode_tag = "bm25" if args.bm25 else args.dense_strategy
        save_path = str(
            RESULTS_DIR / f"exp002_{args.experiment}_s{args.sample}_{mode_tag}.jsonl"
        )

    output = run_experiment(
        experiment_id=args.experiment,
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
    print(f"LLM cache: {RESULTS_DIR}/rewrite_cache/{args.experiment}.jsonl")

    if output is not None and output[0] is not None:
        results, meta = output

        retrievals_keys = set()
        for r in results:
            retrievals_keys.update(r["retrievals"].keys())

        metrics_map = {}
        for key in sorted(retrievals_keys):
            metrics_map[key] = compute_metrics(results, key)

        print_comparison(metrics_map)

        metrics_path = str(
            RESULTS_DIR / f"exp002_{args.experiment}_s{args.sample}_{args.dense_strategy}_metrics.json"
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
