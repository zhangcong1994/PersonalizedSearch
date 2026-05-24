"""
Exp-005 数据准备脚本。

从 exp-004 精排结果中提取 top-K passages，生成统一格式的生成阶段输入数据。

支持两种模式：
  - reranker: 使用 exp-004 精排后的 top-K 结果（主实验）
  - ideal: 使用 qrels 中的理想排序（上限测试）

输出：JSONL 文件，每行包含 query_id, query_text, passages

用法：
  python scripts/prepare_exp005_data.py --mode reranker --top-k 10
  python scripts/prepare_exp005_data.py --mode ideal --top-k 10 --n 200
"""

import os
import sys
import json
import logging
import argparse
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR
from src.evaluation.data_loader import load_queries, load_qrels, load_passages, clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 预定义查询类型关键词（仅用于后验标签标注，不参与采样决策）
QUERY_TYPE_KEYWORDS = {
    "factoid": ["什么时候", "多少", "谁", "哪里", "哪一年", "日期", "年龄", "多高",
                "多长", "多大", "什么时间", "何时", "几年", "多远"],
    "comparison": ["区别", "对比", "不同", "差异", " vs ", "比较", "哪个更好",
                   "哪个更", "优缺点", "优劣", "相似", "一样"],
    "concept": ["什么是", "定义", "概念", "解释", "什么意思", "原理", "指的是",
                "是什么", "含义"],
    "how_to": ["如何", "怎么", "怎样", "步骤", "方法", "做法", "教程", "指南",
               "操作", "配置", "设置"],
    "open_ended": ["影响", "未来", "趋势", "前景", "发展", "历史", "历程",
                   "原因", "为什么", "为何", "作用", "意义", "重要性"],
}


def classify_query(query_text: str) -> str:
    """基于关键词规则对查询进行粗分类。"""
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        if any(kw in query_text for kw in keywords):
            return qtype
    if query_text.endswith("？") or query_text.endswith("?"):
        return "open_ended"
    return "factoid"


def build_ideal_passages(
    queries: list[tuple[str, str]],
    qrels: dict[str, set[str] | dict[str, int]],
    passage_map: dict[str, str],
    top_k: int = 10,
) -> list[dict]:
    """为每条查询构建理想排序（oracle）的 passages。"""
    results = []
    for qid, qtext in queries:
        rel = qrels.get(qid, {})

        if isinstance(rel, dict):
            sorted_pids = sorted(rel.keys(), key=lambda p: rel[p], reverse=True)
        else:
            sorted_pids = list(rel)

        passages = []
        for rank, pid in enumerate(sorted_pids[:top_k], 1):
            text = passage_map.get(pid, "")
            if text:
                passages.append({
                    "pid": pid,
                    "text": text,
                    "rank": rank,
                    "source": "ideal",
                })

        if len(passages) < top_k:
            logger.debug(f"  {qid}: only {len(passages)} relevant passages (need {top_k})")

        results.append({
            "query_id": qid,
            "query_text": qtext,
            "passages": passages,
        })

    return results


def build_reranker_passages(
    queries: list[tuple[str, str]],
    reranker_results: list[dict],
    passage_map: dict[str, str],
    top_k: int = 10,
    method_key: str = None,
) -> list[dict]:
    """从精排结果中提取 top-K passages。"""
    reranker_map = {}
    for r in reranker_results:
        qid = r.get("qid", r.get("query_id", ""))
        retrievals = r.get("retrievals", r.get("passages", []))
        if method_key and isinstance(retrievals, dict):
            retrievals = retrievals.get(method_key, [])
        reranker_map[qid] = retrievals

    results = []
    for qid, qtext in queries:
        retrievals = reranker_map.get(qid, [])
        passages = []
        for rank, item in enumerate(retrievals[:top_k], 1):
            if isinstance(item, dict):
                pid = item.get("pid", "")
            else:
                pid = str(item)

            text = passage_map.get(pid, "")
            if not text:
                text = item.get("text", "") if isinstance(item, dict) else ""
            if text:
                passages.append({
                    "pid": pid,
                    "text": text,
                    "rank": rank,
                    "source": "reranker",
                })

        results.append({
            "query_id": qid,
            "query_text": qtext,
            "passages": passages,
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-005 Data Preparation")
    parser.add_argument("--mode", type=str, default="reranker",
                        choices=["reranker", "ideal"],
                        help="Passage source mode")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of passages per query")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of queries to sample")
    parser.add_argument("--output", "-o", type=str,
                        default=str(DATA_ROOT / "results" / "exp005" / "input_queries.jsonl"),
                        help="Output JSONL file")
    parser.add_argument("--reranker-file", type=str, default=None,
                        help="Exp-004 reranker results JSONL (for reranker mode)")
    parser.add_argument("--method-key", type=str, default=None,
                        help="Method key in reranker results dict")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()

    collection_file = RAW_DATA_DIR / "t2ranking" / "collection.tsv"

    if args.mode == "ideal":
        queries_file = RAW_DATA_DIR / "t2ranking" / "queries.dev.tsv"
        qrels_file = RAW_DATA_DIR / "t2ranking" / "qrels.retrieval.dev.tsv"

        logger.info("Loading data...")
        all_queries = load_queries(queries_file)
        qrels = load_qrels(qrels_file)

        logger.info(f"Sampling {args.n} queries (simple random)...")
        random.seed(args.seed)
        queries = random.sample(all_queries, args.n)

        logger.info("Loading passages...")
        pids, texts = load_passages(collection_file, show_progress=True)
        passage_map = dict(zip(pids, texts))
        logger.info(f"  Passage map: {len(passage_map)} entries")

        results = build_ideal_passages(queries, qrels, passage_map, args.top_k)

    elif args.mode == "reranker":
        if not args.reranker_file:
            logger.error("--reranker-file is required for reranker mode")
            sys.exit(1)

        logger.info("Loading reranker data...")
        with open(args.reranker_file, "r", encoding="utf-8") as f:
            reranker_data = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(reranker_data)} reranker results")

        all_queries = [(r["qid"], r.get("query", r.get("query_text", ""))) for r in reranker_data
                       if r.get("qid")]
        logger.info(f"Extracted {len(all_queries)} queries from reranker data")

        logger.info(f"Sampling {args.n} queries (simple random)...")
        random.seed(args.seed)
        queries = random.sample(all_queries, min(args.n, len(all_queries)))

        logger.info("Loading passages...")
        pids, texts = load_passages(collection_file, show_progress=True)
        passage_map = dict(zip(pids, texts))
        logger.info(f"  Passage map: {len(passage_map)} entries")

        results = build_reranker_passages(
            queries, reranker_data, passage_map, args.top_k, args.method_key
        )

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(results)} prepared queries to {output_path}")

    type_counts = {}
    for item in results:
        qtype = classify_query(item["query_text"])
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    logger.info(f"Query type distribution: {type_counts}")
