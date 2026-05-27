"""
Exp-006: Multi-Round Retrieval with Gap Analysis

Data flow:
  1. Load exp-003 S4 results (Round 1: D-B0 + D-P2 + D-HyDE + B-B0 → RRF k=60)
  2. Gap analysis via DeepSeek → reformulated queries
  3. Round 2 Dense retrieval with reformulated queries (Dense only; BM25 optional)
  4. RRF fusion: Round 1 (4 routes) + Round 2 (1-3 routes) → final top-50
  5. Compute Recall@50 / Hit@50

Usage:
  # Phase 0: manual gap analysis on zero-hit queries (write prompt + output for inspection)
  python scripts/evaluate_exp006.py --phase 0 --n 5 --query-source zero_hit

  # Phase 1: 200 query smoke test
  python scripts/evaluate_exp006.py --phase 1 --n 200

  # Dry run: skip API calls, use dummy query expansion
  python scripts/evaluate_exp006.py --phase 1 --n 200 --dry-run
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.evaluation.data_loader import load_qrels, clean_text
from src.utils.config import DATA_ROOT
from src.utils.config import resolve_model_local_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = DATA_ROOT / "data" / "raw" / "t2ranking"
QRELS_FILE = RAW_DATA_DIR / "qrels.retrieval.dev.tsv"
COLLECTION_FILE = RAW_DATA_DIR / "collection.tsv"

EXP003_RESULTS = DATA_ROOT / "results" / "exp003" / "exp003_test_S4_K50_RRFk60.jsonl"
OUTPUT_DIR = DATA_ROOT / "results" / "exp006"
GAP_ANALYSIS_CACHE = OUTPUT_DIR / "gap_analysis_cache.jsonl"

VECTOR_DB_DIR = DATA_ROOT / "data" / "vector_db" / "t2ranking" / "m3e-base"
COLLECTION_NAME = "t2ranking_passages"
EMBEDDING_MODEL = "moka-ai/m3e-base"

ROUND1_ROUTE_COUNT = 4
ROUND2_PER_QUERY_K = 50
OUTPUT_TOP_K = 50
RRF_K = 60
ROUND1_TOP_FOR_ANALYSIS = 10

GAP_ANALYSIS_SYSTEM = """你是一个搜索失败诊断专家。用户提出查询后，搜索引擎返回了第一轮的 Top-10 段落。
请分析为什么这些结果可能没有满足用户的信息需求，并生成第二轮检索的改写查询。

## 分析维度
逐项判断是否存在以下问题：
1. **compound_split**（复合词拆分）：查询中的词组是否被错误拆分为单字/单词？如"马帮小鱼儿"被当成"马"+"鱼"、"石门出入通"被当成"出入"
2. **domain_misalignment**（领域偏离）：检索结果是否跑到了完全不相关的领域？如果有，用户真正关心的领域是什么？
3. **entity_rarity**（冷门专名）：查询中是否存在极低频的专有名词、缩写、俗称？语料库中它们可能以什么形式出现？
4. **granularity_mismatch**（粒度不匹配）：检索结果主题接近，但不够精确？（如查到小说简介，但用户问的是小说中具体人物列表）

## 改写策略
- **冷门专名**：不要只写专名本身，在其周围补充领域词、类别词（如 "司鱼" → "司鱼交友软件 社交APP 用户评价"）
- **被拆分的复合词**：在改写中保持词组的完整形式，不依赖分词器
- **领域偏离**：在改写中加入领域限定词或排除噪音词
- 生成 1 条改写查询（不要多条）

## 输出格式
仅输出一个 JSON 对象，不要包含 markdown 代码块标记或其他文字：

{
  "diagnosis": {
    "compound_split": false,
    "domain_misalignment": false,
    "entity_rarity": false,
    "granularity_mismatch": false,
    "summary": "一句话描述检索失败原因"
  },
  "reformulated_query": "改写查询（完整语义，补充领域词和消歧上下文，20-50字）",
  "negative_signals": ["检索时应避开的噪音词"]
}"""


def load_exp003_results(path: Path) -> list[dict]:
    """Load exp-003 S4 test set results."""
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "__meta__" in data:
                continue
            results.append(data)
    logger.info(f"Loaded {len(results)} results from {path.name}")
    return results


def load_collection(path: Path, max_lines: int = 0) -> dict[str, str]:
    """Load collection.tsv into a pid→text dict (for passage text lookup)."""
    pid_to_text: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            text = clean_text(text)
            if len(text) > 800:
                text = text[:800]
            if len(text) < 10:
                continue
            pid_to_text[pid] = text
            if max_lines > 0 and i >= max_lines:
                break
    logger.info(f"Loaded {len(pid_to_text)} passages")
    return pid_to_text


def load_dense_retriever(device: str = "cpu", vector_db_dir: str = ""):
    if not vector_db_dir:
        vector_db_dir = str(VECTOR_DB_DIR)
    model_id = EMBEDDING_MODEL
    local_path = resolve_model_local_path(model_id)
    model_name = str(local_path.resolve()) if local_path else model_id
    logger.info(f"Dense retriever using: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=vector_db_dir,
    )
    count = vs._collection.count()
    logger.info(f"Dense retriever loaded: {count:,} docs in '{COLLECTION_NAME}'")
    return vs, count


def dense_search(vs, query_text: str, top_k: int = 50) -> list[dict]:
    """Dense (vector) search, returns [{pid, score, rank}]."""
    results = vs.similarity_search_with_score(query_text, k=top_k)
    return [
        {"pid": doc.metadata.get("pid", ""), "score": round(float(score), 6), "rank": rank}
        for rank, (doc, score) in enumerate(results, 1)
    ]


def rrf_fuse(route_results: list[list[dict]], per_route_k: int, rrf_k: int, output_top_k: int) -> list[dict]:
    """RRF fusion across multiple retrieval routes."""
    rrf_scores: dict[str, float] = defaultdict(float)
    for retrievals in route_results:
        for item in retrievals[:per_route_k]:
            rrf_scores[item["pid"]] += 1.0 / (rrf_k + item["rank"])
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:output_top_k]
    return [
        {"pid": pid, "score": round(score, 6), "rank": rank}
        for rank, (pid, score) in enumerate(merged, 1)
    ]


def build_gap_analysis_prompt(query: str, top10_texts: list[str]) -> str:
    """Build the user prompt for gap analysis."""
    passage_block = ""
    for i, text in enumerate(top10_texts):
        truncated = text[:400] if len(text) > 400 else text
        passage_block += f"[{i + 1}] {truncated}\n\n"

    return f"""## 用户查询
{query}

## 第一轮检索 Top-10 段落
{passage_block}
## 任务
请分析检索缺口，生成改写查询。直接输出 JSON："""


def call_gap_analysis(
    query: str,
    top10_texts: list[str],
    client,
    model_name: str,
) -> dict:
    """Call DeepSeek API for gap analysis. Returns parsed JSON dict, or fallback on error."""
    user_prompt = build_gap_analysis_prompt(query, top10_texts)

    messages = [
        SystemMessage(content=GAP_ANALYSIS_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    try:
        response = client.invoke(messages)
        raw = response.content.strip()
    except Exception as e:
        logger.warning(f"API error: {e}")
        return _fallback_analysis()

    return _parse_gap_response(raw, query)


def _parse_gap_response(raw: str, query: str) -> dict:
    """Parse LLM output as JSON, with markdown code block removal."""
    raw = raw.strip()

    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return _validate_analysis(parsed)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return _validate_analysis(parsed)
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON from: {raw[:200]}...")
        return _fallback_analysis()


def _validate_analysis(parsed: dict) -> dict:
    """Validate and fill missing fields in gap analysis output."""
    if "diagnosis" not in parsed:
        parsed["diagnosis"] = {}
    diag = parsed["diagnosis"]
    for key in ("compound_split", "domain_misalignment", "entity_rarity", "granularity_mismatch"):
        if key not in diag:
            diag[key] = False
    if "summary" not in diag:
        diag["summary"] = "unknown"

    if "reformulated_query" not in parsed:
        parsed["reformulated_query"] = ""
    q = parsed["reformulated_query"]
    if not q:
        return _fallback_analysis()

    if "negative_signals" not in parsed:
        parsed["negative_signals"] = []

    return parsed


def _fallback_analysis() -> dict:
    """Return a safe fallback when gap analysis fails."""
    return {
        "diagnosis": {
            "compound_split": False,
            "domain_misalignment": False,
            "entity_rarity": False,
            "granularity_mismatch": False,
            "summary": "api_error",
        },
        "reformulated_query": "",
        "negative_signals": [],
        "_error": True,
    }


def process_query(
    entry: dict,
    vs,
    pid_to_text: dict[str, str],
    client,
    model_name: str,
    dry_run: bool,
    cache: dict,
) -> dict:
    """Process a single query through gap analysis → Round 2 retrieval → RRF fusion."""
    qid = entry["qid"]
    query = entry["query"]
    relevant_pids = set(entry.get("relevant_pids", []))

    round1_top10 = []
    rrf_key = "rrf@k60_perK50"
    rrf_items = entry.get(rrf_key, [])
    for item in rrf_items[:ROUND1_TOP_FOR_ANALYSIS]:
        round1_top10.append({"pid": item["pid"], "rank": item["rank"]})

    round1_routes: list[list[dict]] = []
    for route_key in ["D-B0@50", "D-P2@50", "D-HyDE@50", "B-B0@50"]:
        route_items = entry.get(route_key, [])
        round1_routes.append(route_items[:OUTPUT_TOP_K])

    gap_result = None
    if not dry_run:
        if qid in cache:
            gap_result = cache[qid]
        else:
            top10_texts = []
            for r in round1_top10:
                text = pid_to_text.get(r["pid"], "")
                top10_texts.append(text)
            gap_result = call_gap_analysis(query, top10_texts, client, model_name)

    round2_queries = []
    if gap_result and not gap_result.get("_error"):
        rq = gap_result.get("reformulated_query", "")
        if isinstance(rq, str) and rq.strip():
            round2_queries = [rq.strip()]
        elif isinstance(rq, dict):
            text = rq.get("text", rq.get("query", ""))
            if text and text.strip():
                round2_queries = [text.strip()]

    if dry_run and not round2_queries:
        words = query.split()
        if len(words) > 2:
            round2_queries = [" ".join(words[:len(words)//2]), " ".join(words)]
        else:
            round2_queries = [query]

    round2_routes: list[list[dict]] = []
    for rq in round2_queries:
        if rq.strip():
            results = dense_search(vs, rq, top_k=ROUND2_PER_QUERY_K)
            round2_routes.append(results)

    all_routes = round1_routes + round2_routes
    final_fused = rrf_fuse(all_routes, ROUND2_PER_QUERY_K, RRF_K, OUTPUT_TOP_K)

    return {
        "qid": qid,
        "query": query,
        "relevant_pids": sorted(relevant_pids),
        "round1_top10_pids": [r["pid"] for r in round1_top10],
        "gap_analysis": gap_result,
        "round2_queries": round2_queries,
        "round1_rrf_top50": [r["pid"] for r in rrf_items[:OUTPUT_TOP_K]],
        "round2_fused_top50": [r["pid"] for r in final_fused],
        "num_round2_routes": len(round2_routes),
    }


def compute_metrics(results: list[dict], key: str) -> dict:
    """Compute Recall@50 and Hit@50."""
    total = len(results)
    total_relevant = 0
    hit_count = 0
    recall_sum = 0.0

    for r in results:
        relevant = set(r["relevant_pids"])
        pids = set(r.get(key, []))
        hits = relevant & pids
        if hits:
            hit_count += 1
            recall_sum += len(hits) / max(len(relevant), 1)
        total_relevant += len(relevant)

    return {
        "Hit@50": round(hit_count / max(total, 1), 4),
        "Recall@50": round(recall_sum / max(total, 1), 4),
        "total_queries": total,
        "total_relevant": total_relevant,
        "hit_queries": hit_count,
    }


def save_results_jsonl(results: list[dict], path: Path, meta: dict | None = None):
    """Save results as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if meta:
            f.write(json.dumps({"__meta__": meta}, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(results)} results to {path}")


def main():
    parser = argparse.ArgumentParser(description="Exp-006: Multi-Round Retrieval with Gap Analysis")
    parser.add_argument("--phase", type=int, default=1, choices=[0, 1, 2],
                        help="Phase 0=manual case study, 1=200q smoke test, 2=full run")
    parser.add_argument("--n", type=int, default=200, help="Number of queries (Phase 1 default: 200)")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls, use trivial query expansion")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument("--query-source", default="all", choices=["all", "zero_hit"],
                        help="Which queries to use (phase 0 only)")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model name")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent API calls")
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--exp003-results", default=str(EXP003_RESULTS),
                        help="Path to exp-003 S4 JSONL results file")
    parser.add_argument("--vector-db", default=str(VECTOR_DB_DIR),
                        help="Path to ChromaDB vector DB directory")
    parser.add_argument("--collection", default=str(COLLECTION_FILE),
                        help="Path to collection.tsv")
    parser.add_argument("--no-collection", action="store_true", help="Skip collection.tsv loading (dry-run only)")
    parser.set_defaults(no_collection=True)
    args = parser.parse_args()

    # needed by HF tokenizer on Windows
    import multiprocessing
    multiprocessing.freeze_support()

    random.seed(args.seed)

    # Step 1: Load Round 1 results
    logger.info("=" * 60)
    logger.info("Step 1: Loading Round 1 (exp-003 S4) results")
    logger.info("=" * 60)
    all_entries = load_exp003_results(Path(args.exp003_results))

    # Step 2: Filter / sample queries
    if args.phase == 0 and args.query_source == "zero_hit":
        rrf_key = "rrf@k60_perK50"
        zero_hit_entries = []
        for e in all_entries:
            relevant = set(e.get("relevant_pids", []))
            rrf_pids = {item["pid"] for item in e.get(rrf_key, [])[:OUTPUT_TOP_K]}
            if not (relevant & rrf_pids):
                zero_hit_entries.append(e)
        logger.info(f"Zero-hit queries: {len(zero_hit_entries)} / {len(all_entries)}")
        entries = zero_hit_entries[:args.n]
    else:
        # Sample evenly across original indices
        sample_n = min(args.n, len(all_entries)) if args.phase != 2 else len(all_entries)
        entries = all_entries[:sample_n]

    logger.info(f"Processing {len(entries)} queries (phase={args.phase})")

    # Step 3: Load collection (for passage text)
    pid_to_text: dict[str, str] = {}
    if not args.dry_run or not args.no_collection:
        logger.info("Loading collection.tsv...")
        pid_to_text = load_collection(Path(args.collection))

    # Step 4: Load qrels
    qrels = load_qrels(QRELS_FILE)
    for e in entries:
        e.setdefault("relevant_pids", qrels.get(e["qid"], set()))

    # Step 5: Load Dense retriever
    logger.info("Loading Dense retriever...")
    vs, _ = load_dense_retriever(device=args.device, vector_db_dir=args.vector_db)

    # Step 6: Gap analysis (API calls)
    api_client = None
    cache: dict[str, dict] = {}

    if not args.dry_run:
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not set. Use --api-key or set env var.")
            sys.exit(1)

        llm = ChatOpenAI(
            model=args.model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            max_tokens=1024,
            temperature=0.1,
        )
        api_client = llm

        if GAP_ANALYSIS_CACHE.exists():
            with open(GAP_ANALYSIS_CACHE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        cached = json.loads(line)
                        cache[cached["qid"]] = cached["gap_analysis"]
            logger.info(f"Loaded {len(cache)} cached gap analyses")

    # Step 7: Process queries
    logger.info("=" * 60)
    logger.info(f"Processing {len(entries)} queries...")
    logger.info("=" * 60)

    results: list[dict] = []
    t_start = time.time()
    api_calls = 0

    if args.dry_run or api_client is None:
        for i, entry in enumerate(entries):
            r = process_query(entry, vs, pid_to_text, None, args.model,
                              dry_run=True, cache=cache)
            results.append(r)
            if (i + 1) % 50 == 0:
                logger.info(f"  Progress: {i + 1}/{len(entries)}")
    else:
        # Concurrent API calls
        cache_lock = __import__("threading").Lock()
        new_cache_entries: list[dict] = []

        def process_one(idx: int, entry: dict) -> tuple[int, dict]:
            nonlocal api_calls
            r = process_query(entry, vs, pid_to_text, api_client, args.model,
                              dry_run=False, cache=cache)
            api_calls += 1
            with cache_lock:
                new_cache_entries.append({"qid": entry["qid"], "gap_analysis": r["gap_analysis"]})
            return idx, r

        results_map: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            for i, entry in enumerate(entries):
                futures[executor.submit(process_one, i, entry)] = i

            for future in as_completed(futures):
                idx, r = future.result()
                results_map[idx] = r
                if len(results_map) % 20 == 0:
                    logger.info(f"  Progress: {len(results_map)}/{len(entries)}")

        results = [results_map[i] for i in range(len(entries))]

        # Save cache (deduped: existing + new)
        if new_cache_entries:
            GAP_ANALYSIS_CACHE.parent.mkdir(parents=True, exist_ok=True)
            all_entries = {k: {"qid": k, "gap_analysis": v} for k, v in cache.items()}
            for ce in new_cache_entries:
                all_entries[ce["qid"]] = ce
            with open(GAP_ANALYSIS_CACHE, "w", encoding="utf-8") as f:
                for ce in all_entries.values():
                    f.write(json.dumps(ce, ensure_ascii=False) + "\n")
            logger.info(f"Cached {len(new_cache_entries)} gap analyses")

    elapsed = time.time() - t_start
    logger.info(f"Processing complete: {len(results)} queries in {elapsed:.1f}s "
                f"({elapsed/len(results)*1000:.0f}ms/q), {api_calls} API calls")

    # Step 8: Compute metrics
    logger.info("=" * 60)
    logger.info("Metrics")
    logger.info("=" * 60)

    round1_metrics = compute_metrics(results, "round1_rrf_top50")
    round2_metrics = compute_metrics(results, "round2_fused_top50")

    logger.info(f"  Round 1 (exp-003 S4 RRF):")
    logger.info(f"    Recall@50 = {round1_metrics['Recall@50']:.4f}")
    logger.info(f"    Hit@50    = {round1_metrics['Hit@50']:.4f}")
    logger.info(f"  Round 1+2 (Multi-Round RRF):")
    logger.info(f"    Recall@50 = {round2_metrics['Recall@50']:.4f}")
    logger.info(f"    Hit@50    = {round2_metrics['Hit@50']:.4f}")
    logger.info(f"  Δ Recall@50 = {round2_metrics['Recall@50'] - round1_metrics['Recall@50']:+.4f}")
    logger.info(f"  Δ Hit@50    = {round2_metrics['Hit@50'] - round1_metrics['Hit@50']:+.4f}")

    avg_r2_routes = sum(r.get("num_round2_routes", 0) for r in results) / max(len(results), 1)
    logger.info(f"  Avg Round 2 routes: {avg_r2_routes:.1f}")

    # Step 9: Save results
    output_path = args.output
    if not output_path:
        tag = "phase0" if args.phase == 0 else f"n{args.n}"
        output_path = str(OUTPUT_DIR / f"exp006_{tag}.jsonl")
    meta = {
        "experiment_id": "exp-006",
        "phase": args.phase,
        "num_queries": len(results),
        "dry_run": args.dry_run,
        "model": args.model,
        "rrf_k": RRF_K,
        "output_top_k": OUTPUT_TOP_K,
        "round1_metrics": round1_metrics,
        "round2_metrics": round2_metrics,
    }
    save_results_jsonl(results, Path(output_path), meta)
    logger.info(f"Results saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
