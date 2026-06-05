"""
Exp-012 Phase 0.5: 从 exp-009 的 5000 条查询中分层抽样，join 检索结果，
输出与 generate_multi_sample.py 兼容的 JSONL 格式。

输入:
  - data/processed/exp009_sampled_queries.jsonl  （5000 条，含 strata）
  - data/processed/exp009_reranked_top10.jsonl   （5000 条检索结果）

  exp-009 原始分层配比: T1(富信息)=35%, T2(中等)=25%, T3(贫信息)=40%

输出:
  - data/processed/exp012_sampled_queries.jsonl
    格式: {"query_id": str, "query_text": str, "passages": [...]}
    兼容 scripts/exp011/generate_multi_sample.py 的输入格式

用法:
    python scripts/exp012/sample_exp009_queries.py
    python scripts/exp012/sample_exp009_queries.py --total 2000 --seed 42
    python scripts/exp012/sample_exp009_queries.py --total 500 --output data/processed/exp012_sampled_500.jsonl
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 默认路径 ────────────────────────────────────────────────

DEFAULT_SAMPLED = DATA_ROOT / "data" / "processed" / "exp009_sampled_queries.jsonl"
DEFAULT_RERANKED = DATA_ROOT / "data" / "processed" / "exp009_reranked_top10.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "data" / "processed" / "exp012_sampled_queries.jsonl"

# 分层配比（与 exp-009 一致）
SAMPLE_RATIOS = {"T1": 0.35, "T2": 0.25, "T3": 0.40}


def load_jsonl(filepath: Path) -> list[dict]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {filepath.name}")
    return records


def stratified_sample(
    sampled_queries: list[dict],
    reranked_results: list[dict],
    total: int,
    seed: int,
) -> list[dict]:
    """
    1. 按 strata 分组
    2. 按比率计算每层采样数
    3. 随机采样
    4. join 检索结果
    5. 转换为 exp011 兼容格式
    """
    # ── 构建检索结果索引 ──
    reranked_index: dict[str, dict] = {}
    for r in reranked_results:
        reranked_index[r["qid"]] = r

    # ── 按 strata 分组 ──
    buckets: dict[str, list[dict]] = {"T1": [], "T2": [], "T3": []}
    for q in sampled_queries:
        stratum = q.get("stratum", "unknown")
        if stratum not in buckets:
            continue
        # 必须有对应的检索结果
        if q["qid"] not in reranked_index:
            continue
        buckets[stratum].append(q)

    # 打印可用量
    logger.info("Available per stratum:")
    for s in ["T1", "T2", "T3"]:
        logger.info(f"  {s}: {len(buckets[s])}")

    # ── 分层采样 ──
    rng = random.Random(seed)
    sampled: list[dict] = []
    stratum_counts: dict[str, int] = {}

    for s in ["T1", "T2", "T3"]:
        target = int(total * SAMPLE_RATIOS[s])
        available = len(buckets[s])
        actual = min(target, available)
        stratum_counts[s] = actual

        chosen = rng.sample(buckets[s], actual)
        for d in chosen:
            d["_stratum"] = s
        sampled.extend(chosen)

        note = ""
        if target > available:
            note = f" [WARN: wanted {target}, only {available}]"
        logger.info(f"  {s}: {actual} / {available} sampled{note}")

    total_sampled = sum(stratum_counts.values())
    logger.info(f"  Total: {total_sampled} (target: {total})")

    if total_sampled < total:
        shortage = total - total_sampled
        # 从各层补充
        remaining = [q for q in sampled_queries
                     if q["qid"] in reranked_index
                     and not any(q["qid"] == s["qid"] for s in sampled)]
        extra = rng.sample(remaining, min(shortage, len(remaining)))
        for d in extra:
            d["_stratum"] = d.get("stratum", "unknown")
        sampled.extend(extra)
        logger.info(f"  Supplemented {len(extra)} from remaining pool → final {len(sampled)}")

    # ── 后验：各层 num_positive 均值 ──
    for s in ["T1", "T2", "T3"]:
        vals = [d["num_positive"] for d in sampled if d["_stratum"] == s]
        if vals:
            mean = sum(vals) / len(vals)
            logger.info(f"  {s}: np mean={mean:.2f}, n={len(vals)}")

    # ── join 检索结果 + 格式转换 ──
    output = []
    skipped_no_reranked = 0
    for d in sampled:
        qid = d["qid"]
        reranked = reranked_index.get(qid)
        if not reranked:
            skipped_no_reranked += 1
            continue

        query_text = d.get("query", "")
        if not query_text:
            query_text = reranked.get("query", "")

        # 提取 passages（只取 pid, rank, text）
        passages = []
        for result in reranked.get("results", []):
            passages.append({
                "pid": result["pid"],
                "rank": result["rank"],
                "text": result["text"],
            })

        output.append({
            "query_id": qid,
            "query_text": query_text,
            "passages": passages,
            # 以下字段仅用于后验分析，不影响生成
            "_stratum": d.get("_stratum"),
            "_num_positive": d.get("num_positive"),
            "_source": "exp009_train",
        })

    if skipped_no_reranked:
        logger.warning(f"Skipped {skipped_no_reranked} queries without reranked results")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Exp-012: Stratified sample from exp-009 and format for multi-sample generation"
    )
    parser.add_argument(
        "--total", type=int, default=2000,
        help="Total queries to sample (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/processed/exp012_sampled_queries.jsonl)",
    )
    parser.add_argument(
        "--sampled-queries", type=str, default=None,
        help="Path to exp009 sampled queries JSONL",
    )
    parser.add_argument(
        "--reranked", type=str, default=None,
        help="Path to exp009 reranked top-10 JSONL",
    )
    args = parser.parse_args()

    sampled_path = Path(args.sampled_queries) if args.sampled_queries else DEFAULT_SAMPLED
    reranked_path = Path(args.reranked) if args.reranked else DEFAULT_RERANKED
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    if not sampled_path.exists():
        logger.error(f"Sampled queries not found: {sampled_path}")
        return 1
    if not reranked_path.exists():
        logger.error(f"Reranked results not found: {reranked_path}")
        return 1

    sampled_queries = load_jsonl(sampled_path)
    reranked_results = load_jsonl(reranked_path)

    # 检查 join 率
    reranked_ids = {r["qid"] for r in reranked_results}
    sampled_ids = {q["qid"] for q in sampled_queries}
    overlap = reranked_ids & sampled_ids
    logger.info(f"Join rate: {len(overlap)}/{len(sampled_ids)} have reranked results")

    output = stratified_sample(sampled_queries, reranked_results, args.total, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(output)} queries to {output_path}")

    # 打印摘要
    strata_dist = Counter(r["_stratum"] for r in output)
    logger.info(f"Final stratum distribution: {dict(strata_dist)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
