"""
Exp-009 阶段一：训练集检索难度统计 + 分层抽样。

模式:
  --mode stats    统计训练集检索难度分布（直方图 + 交叉表 + 建议阈值）
  --mode sample   分层抽样（基于数据驱动的 T1/T2/T3 阈值）

stats 模式输出:
  终端打印：分布直方图 + 交叉表 + 建议阈值
  可选 JSON 文件：全量 query 难度明细

sample 模式输出:
  data/processed/exp009_sampled_queries.jsonl  (默认路径)

用法:
  python scripts/exp009/sample_training_queries.py --mode stats
  python scripts/exp009/sample_training_queries.py --mode sample
  python scripts/exp009/sample_training_queries.py --mode sample --total 5000 --seed 42
"""

import os
import sys

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
import random
import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import RAW_DATA_DIR, DATA_ROOT
from src.evaluation.data_loader import load_qrels_graded

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_TRAIN_FILE = T2RANKING_DIR / "queries.train.tsv"
QRELS_RETRIEVAL_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
QRELS_GRADED_FILE = T2RANKING_DIR / "qrels.train.tsv"

DEFAULT_SAMPLE_OUTPUT = DATA_ROOT / "data" / "processed" / "exp009_sampled_queries.jsonl"

# 分层阈值（2026-05-31 数据驱动确认）
#   T1 富信息: num_positive >= 3  —— 有充足相关 passage
#   T2 中等:   1 <= num_positive <= 2 —— 有少量相关 passage
#   T3 贫信息: num_positive = 0       —— 无任何标注相关 passage
STRATA_DEFS = [
    {"name": "T1", "min_np": 3,  "max_np": 999},
    {"name": "T2", "min_np": 1,  "max_np": 2},
    {"name": "T3", "min_np": 0,  "max_np": 0},
]

# 采样配比（详见 exp-009-generation-sft.yaml stage1_sampling.strata）
SAMPLE_RATIOS = {"T1": 0.35, "T2": 0.25, "T3": 0.40}


# =========================================================================
# 数据加载 + 难度计算（stats 和 sample 共用）
# =========================================================================

def load_queries_dict(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def load_retrieval_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                qrels.setdefault(parts[0], set()).add(parts[1])
    return qrels


def compute_difficulty(
    queries: dict[str, str],
    retrieval_qrels: dict[str, set[str]],
    graded_qrels: dict[str, dict[str, int]],
) -> list[dict]:
    results = []
    for qid in queries:
        positives = retrieval_qrels.get(qid, set())
        num_positive = len(positives)

        grades = []
        graded = graded_qrels.get(qid, {})
        for pid in positives:
            if pid in graded:
                grades.append(graded[pid])

        if grades:
            max_grade = max(grades)
            avg_grade = round(sum(grades) / len(grades), 2)
        else:
            max_grade = 0
            avg_grade = 0.0

        results.append({
            "qid": qid,
            "num_positive": num_positive,
            "max_grade": max_grade,
            "avg_grade": avg_grade,
        })

    return results


def classify_stratum(d: dict) -> str:
    np = d["num_positive"]
    if np >= 3:
        return "T1"
    elif np >= 1:
        return "T2"
    else:
        return "T3"


def run_analysis() -> tuple[dict[str, str], list[dict]]:
    """加载数据 + 计算难度，返回 (queries_dict, difficulty_list)。"""
    logger.info("Loading queries.train.tsv ...")
    queries = load_queries_dict(QUERIES_TRAIN_FILE)

    logger.info("Loading qrels.retrieval.train.tsv ...")
    retrieval_qrels = load_retrieval_qrels(QRELS_RETRIEVAL_FILE)

    logger.info("Loading qrels.train.tsv (graded) ...")
    graded_qrels = load_qrels_graded(QRELS_GRADED_FILE)

    logger.info("Computing difficulty per query ...")
    difficulty = compute_difficulty(queries, retrieval_qrels, graded_qrels)

    return queries, difficulty


# =========================================================================
# --mode stats
# =========================================================================

def _histogram(values: list[int], bins: list[int]) -> dict[str, int]:
    counts = {f"{lo}-{hi}": 0 for lo, hi in zip(bins, bins[1:])}
    for v in values:
        for lo, hi in zip(bins, bins[1:]):
            if lo <= v < hi:
                counts[f"{lo}-{hi}"] += 1
                break
        else:
            counts.setdefault(f">={bins[-1]}", 0)
            counts[f">={bins[-1]}"] += 1
    return counts


def print_histogram(title: str, counts: dict[str, int], total: int):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    max_label_len = max(len(k) for k in counts)
    for label, count in counts.items():
        bar = "█" * max(1, int(count / total * 50))
        pct = count / total * 100
        print(f"  {label:>{max_label_len}}  {count:>8,}  ({pct:5.1f}%)  {bar}")


def run_stats(queries: dict[str, str], difficulty: list[dict], output_path: str | None):
    total = len(difficulty)
    num_pos_values = [d["num_positive"] for d in difficulty]
    max_grade_values = [d["max_grade"] for d in difficulty]
    avg_grade_values = [d["avg_grade"] for d in difficulty]

    print(f"\n  总 query 数:      {len(queries):,}")
    print(f"  有 qrels 的 query: {total:,}")

    # num_positive 分布
    np_bins = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 100]
    np_h = _histogram(num_pos_values, np_bins)
    print_histogram("num_positive 分布 (检索 qrels 中相关 passage 数)", np_h, total)
    print(f"\n  num_positive 均值: {sum(num_pos_values)/total:.1f}")
    print(f"  num_positive 中位数: {sorted(num_pos_values)[total//2]}")
    print(f"  最小值/最大值: {min(num_pos_values)} / {max(num_pos_values)}")

    # max_grade 分布
    mg_counts = Counter(max_grade_values)
    print(f"\n{'─' * 60}")
    print(f"  max_grade 分布 (分级 qrels 中最高标签 0-3)")
    print(f"{'─' * 60}")
    for grade in [0, 1, 2, 3]:
        count = mg_counts.get(grade, 0)
        bar = "█" * max(1, int(count / total * 50))
        print(f"  grade={grade}  {count:>8,}  ({count/total*100:5.1f}%)  {bar}")

    # avg_grade 分布
    ag_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ag_labels = ["0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0"]
    ag_h = {label: 0 for label in ag_labels}
    for v in avg_grade_values:
        for lo, hi, label in zip(ag_bins, ag_bins[1:], ag_labels):
            if lo <= v < hi:
                ag_h[label] += 1
                break
        else:
            ag_h.setdefault("3.0", 0)
            ag_h["3.0"] += 1
    print_histogram("avg_grade 分布", ag_h, total)

    # 交叉表
    np_ranges = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, 50), (50, 999)]
    cross = defaultdict(lambda: defaultdict(int))
    for d in difficulty:
        np = d["num_positive"]
        mg = d["max_grade"]
        for lo, hi in np_ranges:
            if lo <= np < hi:
                row_label = f"np={lo}-{hi-1}" if hi < 999 else f"np>={lo}"
                cross[row_label][mg] += 1
                break

    print(f"\n{'─' * 60}")
    print(f"  交叉表: num_positive × max_grade")
    print(f"{'─' * 60}")
    header = f"  {'np 范围':<14}" + "".join(f"  grade={g:>4}" for g in [0, 1, 2, 3]) + "  total"
    print(header)
    print(f"  {'─' * 14}" + "─" * (6 * 4 + 7))
    for lo, hi in np_ranges:
        label = f"np={lo}-{hi-1}" if hi < 999 else f"np>={lo}"
        row_total = sum(cross[label].values())
        cols = "".join(f"  {cross[label][g]:>8,}" for g in [0, 1, 2, 3])
        print(f"  {label:<14}{cols}  {row_total:>8,}")

    # 建议阈值
    t3_count = sum(1 for d in difficulty if classify_stratum(d) == "T3")
    t2_count = sum(1 for d in difficulty if classify_stratum(d) == "T2")
    t1_count = sum(1 for d in difficulty if classify_stratum(d) == "T1")

    print(f"\n{'─' * 60}")
    print(f"  分层建议 (基于当前统计)")
    print(f"{'─' * 60}")
    print(f"""
  关键发现:
    - num_positive 中位数 = 2，均值 = 2.9，最大值 = 16
    - num_positive > 0 时 max_grade 全部 >= 2（不存在 grade=1 的 query）
    - 因此 T1/T2/T3 的划分只需看 num_positive，max_grade 几乎无区分度

  建议分层阈值:

    T3 贫信息: num_positive = 0
      → {t3_count:,} 条 ({t3_count/total*100:.1f}%)  — 查询完全没有标注相关 passage
      → 采样目标: {SAMPLE_RATIOS['T3']*100:.0f}% (~{int(t3_count*SAMPLE_RATIOS['T3']/t3_count*t3_count):,} 条)，从 {t3_count/total*100:.1f}% 过采样

    T2 中等:   num_positive = 1-2
      → {t2_count:,} 条 ({t2_count/total*100:.1f}%)
      → 采样目标: {SAMPLE_RATIOS['T2']*100:.0f}% (~1250 条)，保持自然比例

    T1 富信息: num_positive >= 3
      → {t1_count:,} 条 ({t1_count/total*100:.1f}%)
      → 采样目标: {SAMPLE_RATIOS['T1']*100:.0f}% (~1750 条)，从 {t1_count/total*100:.1f}% 降采样

  检查:
    - T3 占比应在 10%-25% → {'[OK]' if 10 <= t3_count/total*100 <= 25 else '[WARN] ' + str(round(t3_count/total*100, 1)) + '%'}
    - T1 占比应在 40%-60% → {'[OK]' if 40 <= t1_count/total*100 <= 60 else '[WARN] ' + str(round(t1_count/total*100, 1)) + '%'}
    - 三层总和应 = 100%   → {'[OK]' if t1_count + t2_count + t3_count == total else '[WARN]'}
""")

    if output_path:
        op = Path(output_path)
        if not op.is_absolute():
            op = DATA_ROOT / op
        op.parent.mkdir(parents=True, exist_ok=True)
        with open(op, "w", encoding="utf-8") as f:
            json.dump(difficulty, f, ensure_ascii=False)
        logger.info(f"全量明细已保存至: {op}")

    print(f"{'─' * 60}")
    print(f"  统计完成。请根据以上数据调整 T1/T2/T3 阈值。")
    print(f"{'─' * 60}")


# =========================================================================
# --mode sample
# =========================================================================

def run_sample(
    queries: dict[str, str],
    difficulty: list[dict],
    total: int,
    seed: int,
    output_path: str,
):
    # 分桶
    buckets: dict[str, list[dict]] = {"T1": [], "T2": [], "T3": []}
    for d in difficulty:
        s = classify_stratum(d)
        buckets[s].append(d)

    # 打印自然分布
    print(f"\n  自然分布:")
    for s in ["T1", "T2", "T3"]:
        b = buckets[s]
        print(f"    {s}: {len(b):,} ({len(b)/len(difficulty)*100:.1f}%)")

    # 计算每层采样数
    rng = random.Random(seed)
    sampled: list[dict] = []
    stratum_counts: dict[str, int] = {}

    print(f"\n  采样结果 (seed={seed}, total={total}):")
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
            note = f" [WARN: target {target}, only {available} available]"
        pct = actual / total * 100
        print(f"    {s}: {actual:>6,} ({pct:.1f}%) ← target {target}{note}")

    rng.shuffle(sampled)

    total_sampled = sum(stratum_counts.values())
    print(f"    {'─' * 20}")
    print(f"    总计: {total_sampled:,} (目标 {total:,})")

    if total_sampled < total:
        shortage = total - total_sampled
        print(f"    [WARN] 缺口 {shortage:,} 条——某层可用量不足目标")
        remaining = [d for d in difficulty
                     if not any(d["qid"] == s["qid"] for s in sampled)]
        extra = rng.sample(remaining, min(shortage, len(remaining)))
        for d in extra:
            d["_stratum"] = classify_stratum(d)
        sampled.extend(extra)
        print(f"    → 补抽 {len(extra):,} 条，最终 {len(sampled):,}")

    # 写入 JSONL
    op = Path(output_path)
    if not op.is_absolute():
        op = DATA_ROOT / op
    op.parent.mkdir(parents=True, exist_ok=True)

    with open(op, "w", encoding="utf-8") as f:
        for d in sampled:
            qid = d["qid"]
            out = {
                "qid": qid,
                "query": queries[qid],
                "stratum": d["_stratum"],
                "num_positive": d["num_positive"],
                "max_grade": d["max_grade"],
                "avg_grade": d["avg_grade"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    logger.info(f"Sampled {len(sampled):,} queries → {op}")

    # 后验统计：层内 num_positive 均值
    print(f"\n  后验检查（各层内 num_positive 均值 ± 标准差）:")
    for s in ["T1", "T2", "T3"]:
        vals = [d["num_positive"] for d in sampled if d["_stratum"] == s]
        if vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean)**2 for v in vals) / len(vals)
            std = var ** 0.5
            print(f"    {s}: mean={mean:.2f}, std={std:.2f}, min={min(vals)}, max={max(vals)}, n={len(vals)}")


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 阶段一：训练集检索难度统计 + 分层抽样"
    )
    parser.add_argument("--mode", choices=["stats", "sample"], default="stats",
                        help="stats: 打印分布; sample: 分层抽样到 JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（stats: 全量明细JSON; sample: 抽样JSONL）")
    parser.add_argument("--total", type=int, default=5000,
                        help="sample 模式的总采样数 (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    args = parser.parse_args()

    # ── 加载 + 计算 ──
    print("=" * 60)
    print(f"  Exp-009 阶段一: {'检索难度统计' if args.mode == 'stats' else '分层抽样'}")
    print("=" * 60)

    queries, difficulty = run_analysis()

    if args.mode == "stats":
        run_stats(queries, difficulty, args.output)
    elif args.mode == "sample":
        output = args.output or str(DEFAULT_SAMPLE_OUTPUT)
        run_sample(queries, difficulty, args.total, args.seed, output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
