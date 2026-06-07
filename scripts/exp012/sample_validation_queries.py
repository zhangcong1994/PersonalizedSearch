"""
Exp-012 Phase: 从 exp-009 剩余 3000 条查询中分层抽样验证集。

排除 exp-012 已用的 2000 条，从剩余 3000 条中按比例抽样，
输出与 generate_multi_sample.py 兼容的 JSONL 格式。

用法:
    python scripts/exp012/sample_validation_queries.py
    python scripts/exp012/sample_validation_queries.py --total 300 --seed 123
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

SAMPLED_POOL = DATA_ROOT / "data" / "processed" / "exp009_sampled_queries.jsonl"
RERANKED = DATA_ROOT / "data" / "processed" / "exp009_reranked_top10.jsonl"
EXP012_USED = DATA_ROOT / "data" / "processed" / "exp012_sampled_queries.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "data" / "processed" / "exp012_validation_queries.jsonl"

# 分层配比
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


def load_used_ids(filepath: Path) -> set[str]:
    """从 exp012_sampled_queries.jsonl 提取已用的 query_id。"""
    ids = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["query_id"])
    logger.info(f"Loaded {len(ids)} used query_ids from {filepath.name}")
    return ids


def sample_validation(
    pool: list[dict],
    reranked_results: list[dict],
    used_ids: set[str],
    total: int,
    seed: int,
) -> list[dict]:
    """分层抽样验证集，排除已用于训练的 query。"""

    # ── 检索结果索引 ──
    reranked_index: dict[str, dict] = {r["qid"]: r for r in reranked_results}

    # ── 按 strata 分组，排除已用的 + 无检索结果的 ──
    buckets: dict[str, list[dict]] = {"T1": [], "T2": [], "T3": []}
    excluded_used = 0
    excluded_no_rerank = 0

    for q in pool:
        qid = q["qid"]
        stratum = q.get("stratum", "unknown")
        if stratum not in buckets:
            continue
        if qid in used_ids:
            excluded_used += 1
            continue
        if qid not in reranked_index:
            excluded_no_rerank += 1
            continue
        buckets[stratum].append(q)

    logger.info(f"Excluded: {excluded_used} used-in-exp012, {excluded_no_rerank} no-reranked")
    logger.info("Available per stratum (after exclusion):")
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

    # ── 后验：各层 num_positive 均值 ──
    for s in ["T1", "T2", "T3"]:
        vals = [d["num_positive"] for d in sampled if d["_stratum"] == s]
        if vals:
            mean = sum(vals) / len(vals)
            logger.info(f"  {s}: np mean={mean:.2f}, n={len(vals)}")

    # ── join 检索结果 + 格式转换 ──
    output = []
    for d in sampled:
        qid = d["qid"]
        reranked = reranked_index[qid]

        query_text = d.get("query", "")
        if not query_text:
            query_text = reranked.get("query", "")

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
            "_stratum": d.get("_stratum"),
            "_num_positive": d.get("num_positive"),
            "_source": "exp012_validation",
        })

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Exp-012: Stratified sample validation set from remaining exp-009 queries"
    )
    parser.add_argument("--total", type=int, default=300, help="Total queries (default: 300)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--pool", type=str, default=None, help="Full 5000 pool JSONL")
    parser.add_argument("--reranked", type=str, default=None, help="Reranked results JSONL")
    parser.add_argument("--used", type=str, default=None, help="exp012 used queries JSONL")
    args = parser.parse_args()

    pool_path = Path(args.pool) if args.pool else SAMPLED_POOL
    reranked_path = Path(args.reranked) if args.reranked else RERANKED
    used_path = Path(args.used) if args.used else EXP012_USED
    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT

    # 验证输入文件
    for p, label in [(pool_path, "pool"), (reranked_path, "reranked"), (used_path, "used")]:
        if not p.exists():
            logger.error(f"{label} not found: {p}")
            return 1

    pool = load_jsonl(pool_path)
    reranked = load_jsonl(reranked_path)
    used_ids = load_used_ids(used_path)

    output = sample_validation(pool, reranked, used_ids, args.total, args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(output)} queries to {output_path}")

    strata_dist = Counter(r["_stratum"] for r in output)
    logger.info(f"Stratum distribution: {dict(strata_dist)}")

    # 打印下一步命令
    logger.info("")
    logger.info("Next steps:")
    gen_out = f"results/exp012/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42_validation.jsonl"
    logger.info(f"  1. Generate: python scripts/exp011/generate_multi_sample.py "
                f"--input-file {output_path} --output-dir results/exp012/generation/ "
                f"--sample-size {len(output)}")
    logger.info(f"  2. Judge: Use existing exp011 judge pipeline on {gen_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
