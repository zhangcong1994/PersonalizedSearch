import os
import sys
import json
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

from src.utils.config import RAW_DATA_DIR, VECTOR_DB_DIR, DATA_ROOT, EMBEDDING_MODEL, resolve_model_local_path
from src.evaluation.data_loader import load_queries, load_qrels, load_passages

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_FILE = T2RANKING_DIR / "queries.dev.tsv"
QRELS_FILE = T2RANKING_DIR / "qrels.retrieval.dev.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"
RESULTS_DIR = DATA_ROOT / "results"


def build_bm25(texts: list[str]):
    from rank_bm25 import BM25Okapi

    tokenized = [t.split() for t in texts]
    return BM25Okapi(tokenized), tokenized


def load_dense_retriever(vector_db_dir: str, collection_name: str, device: str = "cpu",
                         search_type: str = "similarity", model_id: str = None):
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
    elif search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = 0.3

    return vs.as_retriever(search_type=search_type, search_kwargs=search_kwargs), count


# ── query rewriting ──────────────────────────────────────

QUERY_REWRITE_PROMPT = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 如果查询本身已经足够清晰完整（>15字），直接返回原查询
2. 如果查询是缩写、口语化、省略了关键信息，补全为完整的疑问句或陈述句
3. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
4. 不要编造查询中没有的信息

示例：
原始: 蜂巢取快递验证码摁错怎么办
改写: 蜂巢快递柜取件验证码输入错误如何重新获取

原始: 生产过后怎么还有一层肚子
改写: 产后腹部脂肪堆积原因及恢复平坦小腹的方法

原始: 考研英语一和英语二有什么区别
改写: 考研英语一和英语二的区别 考试内容 难度对比

原始: 怎么判断鱼卵是否活着
改写: 如何判断鱼卵是否存活 鱼卵活性检测方法

原始: 比特币和以太坊哪个更值得投资
改写: 比特币和以太坊哪个更值得投资

原始: 西红柿炒鸡蛋的正确做法是什么
改写: 西红柿炒鸡蛋的正确做法步骤

原始: 为什么晚上睡觉会磨牙
改写: 晚上睡觉磨牙的原因 夜磨牙症病因"""


def _get_rewrite_llm():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.1,
        max_tokens=128,
    )


def rewrite_queries(
    queries: list[tuple[str, str]],
    save_path: str,
    batch_size: int = 20,
) -> dict[str, str]:
    from langchain_core.prompts import ChatPromptTemplate

    existing = load_rewritten_queries(save_path)
    to_rewrite = [(qid, text) for qid, text in queries if qid not in existing]

    if not to_rewrite:
        logger.info(f"All {len(queries)} queries already cached in {save_path}")
        return existing

    logger.info(f"Rewriting {len(to_rewrite)} queries (cached: {len(existing)})...")
    llm = _get_rewrite_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("human", "原始: {query}\n改写:"),
    ])
    chain = prompt | llm

    new = {}
    for i in range(0, len(to_rewrite), batch_size):
        batch = to_rewrite[i:i + batch_size]
        for qid, text in batch:
            try:
                result = chain.invoke({"query": text})
                rewritten = result.content.strip() if hasattr(result, "content") else str(result).strip()
                new[qid] = rewritten if rewritten else text
            except Exception as e:
                logger.warning(f"Rewrite failed for qid={qid}: {e}, using original")
                new[qid] = text

        existing.update(new)
        save_rewritten_queries(existing, save_path)
        logger.info(f"  Rewrite progress: {min(i + batch_size, len(to_rewrite))}/{len(to_rewrite)}")

    return existing


def save_rewritten_queries(rewritten: dict[str, str], path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for qid in sorted(rewritten.keys(), key=lambda x: int(x)):
            f.write(json.dumps({"qid": qid, "rewritten": rewritten[qid]}, ensure_ascii=False) + "\n")


def load_rewritten_queries(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    result = {}
    with open(p, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            result[obj["qid"]] = obj["rewritten"]
    return result


# ── result cache ─────────────────────────────────────────
from src.evaluation.result_cache import save_results, load_results

# ── metrics ──────────────────────────────────────────────
from src.evaluation.metrics import compute_metrics, print_comparison


# ── main pipeline ────────────────────────────────────────

def run_and_save(
    sample_size: int,
    top_k: int,
    device: str,
    vector_db_dir: str,
    collection_name: str,
    dense_search_type: str,
    save_path: str,
    skip_bm25: bool = False,
    skip_dense: bool = False,
    rewrite_map: dict[str, str] = None,
    model_id: str = None,
):
    queries = load_queries(QUERIES_FILE)
    qrels = load_qrels(QRELS_FILE)

    if sample_size > len(queries):
        sample_size = len(queries)
    sampled = queries[:sample_size]

    valid_queries = [(qid, text) for qid, text in sampled if qid in qrels and qrels[qid]]
    logger.info(f"Sampled queries with qrels: {len(valid_queries)} / {sample_size}")

    if rewrite_map:
        valid_queries = [(qid, rewrite_map.get(qid, text)) for qid, text in valid_queries]

    pids, texts = load_passages(COLLECTION_FILE)
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

    if not pool_queries:
        print("WARNING: No queries have relevant passages in the loaded pool!")
        return None

    bm25 = None
    if not skip_bm25:
        print()
        print("=" * 60)
        print("  Building BM25...")
        print("=" * 60)
        t0 = time.time()
        bm25, tokenized = build_bm25(texts)
        logger.info(f"BM25 build: {time.time() - t0:.1f}s ({len(pids)} docs)")

    dense_retriever = None
    if not skip_dense:
        print()
        print("=" * 60)
        print("  Loading Dense Retriever...")
        print("=" * 60)
        t0 = time.time()
        if vector_db_dir is None:
            vector_db_dir = str(VECTOR_DB_DIR / "t2ranking" / "bge-small-zh-v1.5")
        dense_retriever, dense_count = load_dense_retriever(
            vector_db_dir, collection_name, device, search_type=dense_search_type,
            model_id=model_id,
        )
        logger.info(f"Dense retriever load: {time.time() - t0:.1f}s")

    print()
    print("=" * 60)
    print(f"  Running retrieval: {len(pool_queries)} queries × {len(pids)} passages")
    print("=" * 60)

    results = []
    bm25_total_time = 0.0
    dense_total_time = 0.0

    for i, (qid, query_text) in enumerate(pool_queries):
        relevant = qrels.get(qid, set())
        entry = {
            "qid": qid,
            "query": query_text,
            "relevant_pids": relevant,
            "retrievals": {},
        }

        if bm25 is not None:
            t0 = time.time()
            tokenized_query = query_text.split()
            scores = bm25.get_scores(tokenized_query)
            bm25_elapsed = time.time() - t0
            bm25_total_time += bm25_elapsed

            top_idx = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_k]
            entry["retrievals"]["bm25"] = [pids[j] for j in top_idx]

        if dense_retriever is not None:
            t0 = time.time()
            docs = dense_retriever.invoke(query_text)
            dense_elapsed = time.time() - t0
            dense_total_time += dense_elapsed

            entry["retrievals"][dense_search_type] = [
                doc.metadata.get("pid", "?") for doc in docs[:top_k]
            ]

        results.append(entry)

        if (i + 1) % 100 == 0:
            elapsed = time.time()
            parts = [f"Progress: {i+1}/{len(pool_queries)}"]
            if bm25 is not None:
                parts.append(f"BM25 {bm25_total_time/(i+1)*1000:.0f}ms/q")
            if dense_retriever is not None:
                parts.append(f"Dense {dense_total_time/(i+1)*1000:.0f}ms/q")
            logger.info("  " + " | ".join(parts))

    meta = {
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

    metrics_map = {}
    if not skip_bm25:
        metrics_map["BM25"] = compute_metrics(results, "bm25")
    if not skip_dense:
        metrics_map[dense_search_type] = compute_metrics(results, dense_search_type)
    print_comparison(metrics_map)

    if not skip_bm25:
        print(f"  BM25 total search time: {bm25_total_time:.1f}s ({bm25_total_time/len(pool_queries)*1000:.0f}ms/q)")
    if not skip_dense:
        print(f"  Dense total search time: {dense_total_time:.1f}s ({dense_total_time/len(pool_queries)*1000:.0f}ms/q)")

    return metrics_map


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


def main():
    parser = argparse.ArgumentParser(description="T2Ranking retrieval evaluation: BM25 vs Dense")
    parser.add_argument("--sample", type=int, default=500, help="Number of queries to evaluate")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--device", default="cpu", help="Device for embedding model")
    parser.add_argument(
        "--vector-db", default=None,
        help="Vector DB directory (default: data/vector_db/t2ranking/bge-small-zh-v1.5)",
    )
    parser.add_argument("--collection-name", default="t2ranking_passages")
    parser.add_argument("--dense-only", action="store_true", help="Skip BM25")
    parser.add_argument("--bm25-only", action="store_true", help="Skip Dense")
    parser.add_argument("--dense-strategy", default="similarity",
                        choices=["similarity", "mmr", "similarity_score_threshold"],
                        help="Dense retrieval strategy")
    parser.add_argument(
        "--embedding-model", default=None,
        help="Embedding model HF id or short key (default: config.yaml indexing.embedding_model = bge-small-zh-v1.5). "
             "Accepted: BAAI/bge-small-zh-v1.5, BAAI/bge-large-zh-v1.5, moka-ai/m3e-base, BAAI/bge-m3",
    )

    parser.add_argument("--save", default=None, help="Save retrieval results to JSONL file")
    parser.add_argument("--load", default=None, help="Load cached results from JSONL (skip retrieval)")

    parser.add_argument("--rewrite", action="store_true", help="Rewrite queries via DeepSeek API")
    parser.add_argument(
        "--rewrite-cache", default=None,
        help="Path for query rewrite cache (default: results/rewritten_queries.jsonl)",
    )

    args = parser.parse_args()

    if args.dense_only and args.bm25_only:
        print("ERROR: --dense-only and --bm25-only cannot be used together")
        return 1

    if args.load:
        load_and_print(args.load, top_k=args.top_k)
        return 0

    rewrite_cache = args.rewrite_cache or str(RESULTS_DIR / "rewritten_queries.jsonl")
    rewrite_map = None
    if args.rewrite:
        all_queries = load_queries(QUERIES_FILE)
        sampled_queries = all_queries[:args.sample]
        rewrite_map = rewrite_queries(sampled_queries, rewrite_cache)

    save_path = args.save
    if not save_path:
        strategy_tag = "dense" if args.dense_only else ("bm25" if args.bm25_only else "hybrid")
        if args.rewrite:
            strategy_tag = "rewrite_" + strategy_tag
        save_path = str(
            RESULTS_DIR /
            f"retrieval_{strategy_tag}_s{args.sample}_{args.dense_strategy}.jsonl"
        )

    if args.save and Path(args.save).exists():
        logger.warning(f"Save path already exists, will overwrite: {args.save}")

    run_and_save(
        sample_size=args.sample,
        top_k=args.top_k,
        device=args.device,
        vector_db_dir=args.vector_db,
        collection_name=args.collection_name,
        dense_search_type=args.dense_strategy,
        save_path=save_path,
        skip_bm25=args.dense_only,
        skip_dense=args.bm25_only,
        rewrite_map=rewrite_map,
        model_id=args.embedding_model,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
