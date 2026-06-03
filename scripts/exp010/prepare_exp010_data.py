"""
Exp-010 数据准备 —— 从 198 条评估数据中分层抽样 50 条。

用于 Phase 4 消融分析，控制 API 成本（5 个消融变体 × 50 条 = 250 条 Judge ≈ ¥3-4，
比 5 × 198 = 990 条 ≈ ¥12-15 更经济）。

抽样策略：
  - 分层维度：基线总分（高/中/低三组） + 是否 D 级
  - 保证原始 198 条中不同"难度"的 case 都被代表
  - 使用基线 Judge 评分结果（如果存在）来做分层依据；
    如果不存在基线评分，使用简单随机抽样

输出：results/exp010/queries_50.jsonl

用法：
  python scripts/exp010/prepare_exp010_data.py
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42
SAMPLE_SIZE = 50

INPUT_QUERIES = DATA_ROOT / "data" / "exp005_queries.jsonl"
OUTPUT_FILE = DATA_ROOT / "results" / "exp010" / "queries_50.jsonl"

# 可能的基线评分来源
BASELINE_JUDGE = [
    DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-4b-nothink-v0_judged.jsonl",
    DATA_ROOT / "results" / "exp005" / "judge_scores" / "qwen3-4b-nothink_judged.jsonl",
]


def stratified_sample(
    query_data: list[dict],
    judge_data: list[dict] | None,
    n: int,
) -> list[dict]:
    """分层抽样：按总分分高/中/低三组 + D 级单独一组。"""
    random.seed(SEED)

    if judge_data is None or len(judge_data) == 0:
        logger.warning("No baseline judge results found. Using random sampling.")
        return random.sample(query_data, min(n, len(query_data)))

    # 构建 qid → score 映射
    qid_score = {}
    qid_grade = {}
    for jd in judge_data:
        qid = jd.get("query_id", "")
        agg = jd.get("aggregated", {})
        total = agg.get("total_score", None)
        grade = agg.get("grade", "")
        if qid and total is not None:
            qid_score[qid] = total
            qid_grade[qid] = grade

    if not qid_score:
        logger.warning("No valid scores in judge results. Using random sampling.")
        return random.sample(query_data, min(n, len(query_data)))

    # 按总分分层: 高(>65), 中(50-65), 低(<50)
    high, mid, low, d_grade = [], [], [], []
    for item in query_data:
        qid = item.get("query_id", "")
        score = qid_score.get(qid)
        grade = qid_grade.get(qid, "")
        if grade == "D":
            d_grade.append(item)
        elif score is not None:
            if score > 65:
                high.append(item)
            elif score >= 50:
                mid.append(item)
            else:
                low.append(item)

    # 无评分映射的 item 随机分配
    unmapped = [item for item in query_data
                if item.get("query_id", "") not in qid_score]
    random.shuffle(unmapped)
    for idx, item in enumerate(unmapped):
        if idx % 3 == 0:
            high.append(item)
        elif idx % 3 == 1:
            mid.append(item)
        else:
            low.append(item)

    logger.info(
        f"Strata: high={len(high)}, mid={len(mid)}, "
        f"low={len(low)}, D-grade={len(d_grade)}"
    )

    # 按比例分配: 各层按原始比例抽样，D 级保证至少 5 条
    total = len(query_data)
    n_d = max(5, int(n * len(d_grade) / total)) if d_grade else 0
    remaining = n - n_d
    n_high = int(remaining * len(high) / max(total, 1))
    n_mid = int(remaining * len(mid) / max(total, 1))
    n_low = remaining - n_high - n_mid

    sampled = []
    sampled.extend(random.sample(d_grade, min(n_d, len(d_grade))))
    sampled.extend(random.sample(high, min(n_high, len(high))))
    sampled.extend(random.sample(mid, min(n_mid, len(mid))))
    sampled.extend(random.sample(low, min(n_low, len(low))))

    # 如果不够，随机补充
    if len(sampled) < n:
        remaining_pool = [item for item in query_data if item not in sampled]
        sampled.extend(random.sample(remaining_pool, min(n - len(sampled), len(remaining_pool))))

    random.shuffle(sampled)
    return sampled[:n]


def main():
    if not INPUT_QUERIES.exists():
        logger.error(f"Input file not found: {INPUT_QUERIES}")
        sys.exit(1)

    query_data = []
    with open(INPUT_QUERIES, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query_data.append(json.loads(line))
    logger.info(f"Loaded {len(query_data)} queries from {INPUT_QUERIES}")

    # 尝试加载基线评分
    judge_data = None
    for p in BASELINE_JUDGE:
        if p.exists():
            judge_data = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    judge_data.append(json.loads(line))
            logger.info(f"Loaded {len(judge_data)} baseline scores from {p}")
            break

    if judge_data is None:
        logger.info("No baseline scores found. Using simple random sampling.")

    sampled = stratified_sample(query_data, judge_data, SAMPLE_SIZE)
    logger.info(f"Sampled {len(sampled)} queries")

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
