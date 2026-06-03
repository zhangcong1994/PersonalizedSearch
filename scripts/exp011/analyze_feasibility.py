"""
Exp-011 Phase 2: RL/DPO 可行性分析。

从多样本 Judge 评分中计算 best-of-N 与基线的对比，回答核心问题：
  "更好的答案是否存在于模型的输出分布中？"

输入：
  - 多样本 + Judge 评分 JSONL（generations + scores 合并后）
  - 基线评分（T=0.3 single sample）

输出：
  - 每 query 的 best-of-N vs 基线 统计
  - 跨 query 聚合统计
  - RL/DPO 可行性判定

用法:
    python scripts/exp011/analyze_feasibility.py \
        --multi results/exp011/judge_scores/qwen3-4b-nothink_t0.8_n5_judged.jsonl \
        --baseline results/exp005/judge_scores/qwen3-4b-nothink_judged.jsonl \
        --output results/exp011/analysis/

    # 同时分析多个模型
    python scripts/exp011/analyze_feasibility.py --all
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp011"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
MULTI_SCORES_DIR = RESULTS_DIR / "judge_scores"
BASELINE_SCORES_DIR = DATA_ROOT / "results" / "exp005" / "judge_scores"

# 核心 6 维度
CORE6_DIMS = [
    "veracity",
    "safety",
    "relevance",
    "synthesis_quality",
    "citation_quality",
    "user_experience",
]


def load_judged_results(filepath: Path) -> list[dict]:
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    logger.info(f"Loaded {len(entries)} judge results from {filepath.name}")
    return entries


def group_by_query(judged_entries: list[dict]) -> dict[str, list[dict]]:
    """按原始 query_id 分组多采样结果。

    多采样文件使用复合 query_id（如 1869_s0, 1869_s1），
    这里还原为原始 query_id 后再分组。
    """
    groups = defaultdict(list)
    for entry in judged_entries:
        qid = entry.get("query_id", "unknown")
        # 还原原始 query_id：去掉 _sN 后缀
        if "_s" in qid:
            parts = qid.rsplit("_s", 1)
            if len(parts) == 2 and parts[1].isdigit():
                qid = parts[0]
        groups[qid].append(entry)
    return dict(groups)


def extract_score(entry: dict, dim: str = None) -> Optional[float]:
    """从一条 Judge 结果中提取评分。
    
    - dim=None: 返回 total_score (0-100)
    - dim=str: 返回单个维度分数 (1-4)
    """
    agg = entry.get("aggregation", {})
    if dim is None:
        score = agg.get("total_score", -1)
        return score if score >= 0 else None
    
    scores = entry.get("scores", {})
    if dim in scores:
        val = scores[dim]
        if isinstance(val, dict):
            return int(val.get("score", 0))
        return int(val)
    return None


def analyze_model(
    multi_scores: list[dict],
    baseline_scores: list[dict],
    model_name: str,
) -> dict:
    """分析单个模型的 best-of-N 表现。"""
    multi_by_qid = group_by_query(multi_scores)
    baseline_by_qid = {e["query_id"]: e for e in baseline_scores}

    # 找到两套数据中共同存在的 query
    common_qids = set(multi_by_qid.keys()) & set(baseline_by_qid.keys())
    logger.info(f"  Common queries: {len(common_qids)}")

    if len(common_qids) == 0:
        logger.error("  No common queries found between multi-sample and baseline!")
        return {}

    # -- Per-query 统计 --
    per_query = []
    for qid in sorted(common_qids):
        samples = multi_by_qid[qid]
        baseline = baseline_by_qid[qid]

        sample_scores = [extract_score(s) for s in samples]
        sample_scores = [s for s in sample_scores if s is not None]
        baseline_score = extract_score(baseline)

        if baseline_score is None or len(sample_scores) == 0:
            continue

        best_score = max(sample_scores)
        mean_score = sum(sample_scores) / len(sample_scores)
        std_score = (sum((s - mean_score) ** 2 for s in sample_scores) / len(sample_scores)) ** 0.5
        pass_count = sum(1 for s in sample_scores if s >= 60)
        best_score = max(sample_scores)

        # Per-dimension best-of-N
        dim_bests = {}
        for dim in CORE6_DIMS:
            dim_scores = [extract_score(s, dim) for s in samples]
            dim_scores = [s for s in dim_scores if s is not None]
            baseline_dim = extract_score(baseline, dim)
            if dim_scores:
                dim_bests[dim] = {
                    "best": max(dim_scores),
                    "mean": sum(dim_scores) / len(dim_scores),
                    "baseline": baseline_dim,
                    "delta_best": max(dim_scores) - baseline_dim if baseline_dim is not None else None,
                }

        per_query.append({
            "qid": qid,
            "num_samples": len(sample_scores),
            "baseline_score": baseline_score,
            "best_score": best_score,
            "mean_score": round(mean_score, 2),
            "std_score": round(std_score, 2),
            "pass_count": pass_count,
            "delta_best_vs_baseline": round(best_score - baseline_score, 2),
            "dim_analysis": dim_bests,
            "sample_scores": sample_scores,
        })

    if len(per_query) == 0:
        logger.error("  No valid query result pairs!")
        return {}

    # -- 聚合统计 --
    baseline_scores_all = [q["baseline_score"] for q in per_query]
    best_scores_all = [q["best_score"] for q in per_query]
    mean_scores_all = [q["mean_score"] for q in per_query]
    deltas = [q["delta_best_vs_baseline"] for q in per_query]

    # 判定：best-of-5 比基线提高 > 5 分的 query 占比
    improved_queries = [q for q in per_query if q["delta_best_vs_baseline"] > 5]
    regressed_queries = [q for q in per_query if q["delta_best_vs_baseline"] < -5]
    stable_queries = [q for q in per_query if abs(q["delta_best_vs_baseline"]) <= 5]

    # 平均 std（测量输出多样性）
    avg_std = sum(q["std_score"] for q in per_query) / len(per_query)

    # 基线及格率 vs best-of-N 及格率
    baseline_pass_rate = sum(1 for s in baseline_scores_all if s >= 60) / len(baseline_scores_all)
    best_pass_rate = sum(1 for s in best_scores_all if s >= 60) / len(best_scores_all)

    result = {
        "model": model_name,
        "num_queries": len(per_query),
        "mean_samples_per_query": sum(q["num_samples"] for q in per_query) / len(per_query),

        # 核心对比
        "mean_baseline": round(sum(baseline_scores_all) / len(baseline_scores_all), 2),
        "mean_best_of_n": round(sum(best_scores_all) / len(best_scores_all), 2),
        "mean_mean_of_n": round(sum(mean_scores_all) / len(mean_scores_all), 2),
        "mean_delta_best_vs_baseline": round(sum(deltas) / len(deltas), 2),
        "median_delta_best_vs_baseline": round(sorted(deltas)[len(deltas) // 2], 2),
        "mean_std": round(avg_std, 2),

        "baseline_pass_rate": round(baseline_pass_rate * 100, 1),
        "best_pass_rate": round(best_pass_rate * 100, 1),

        # 分布
        "n_improved": len(improved_queries),
        "pct_improved": round(len(improved_queries) / len(per_query) * 100, 1),
        "n_regressed": len(regressed_queries),
        "pct_regressed": round(len(regressed_queries) / len(per_query) * 100, 1),
        "n_stable": len(stable_queries),
        "pct_stable": round(len(stable_queries) / len(per_query) * 100, 1),

        # 分维度分析
        "dim_summary": _compute_dim_summary(per_query),

        # 可行性判定
        "feasibility_verdict": _feasibility_verdict(per_query, avg_std),

        "per_query": per_query,
    }

    return result


def _compute_dim_summary(per_query: list[dict]) -> dict:
    """聚合分维度 best-of-N delta。"""
    dim_deltas = {dim: [] for dim in CORE6_DIMS}
    for q in per_query:
        for dim in CORE6_DIMS:
            if dim in q["dim_analysis"]:
                d = q["dim_analysis"][dim]["delta_best"]
                if d is not None:
                    dim_deltas[dim].append(d)

    summary = {}
    for dim in CORE6_DIMS:
        deltas = dim_deltas[dim]
        if deltas:
            summary[dim] = {
                "mean_delta": round(sum(deltas) / len(deltas), 3),
                "median_delta": round(sorted(deltas)[len(deltas) // 2], 3),
                "pct_positive": round(sum(1 for d in deltas if d > 0) / len(deltas) * 100, 1),
            }
    return summary


def _feasibility_verdict(per_query: list[dict], avg_std: float) -> dict:
    """基于数据给出 RL/DPO 可行性判定。"""
    deltas = [q["delta_best_vs_baseline"] for q in per_query]
    mean_delta = sum(deltas) / len(deltas)
    pct_improved = sum(1 for d in deltas if d > 5) / len(deltas) * 100

    # 判定逻辑
    if mean_delta > 5 and pct_improved > 40:
        verdict = "LIKELY_FEASIBLE"
        reasoning = (
            f"Best-of-N 平均比基线高 {mean_delta:.1f} 分，{pct_improved:.0f}% 的 query 显著改善。"
            "模型分布中存在更好的答案，RL/DPO 有望将这些优质输出概率提高。"
        )
    elif mean_delta > 2 and avg_std > 3:
        verdict = "POSSIBLY_FEASIBLE"
        reasoning = (
            f"Best-of-N 平均比基线高 {mean_delta:.1f} 分，输出多样性中等（std={avg_std:.1f}）。"
            "RL 有一定空间，但收益可能有限。建议先试 DPO on teacher data。"
        )
    elif avg_std < 2:
        verdict = "UNLIKELY_FEASIBLE"
        reasoning = (
            f"输出方差极小（std={avg_std:.1f}），温度 0.8 下模型近乎确定性。"
            "模型没有足够的探索空间，RL 无法从分布中筛选出更好的答案。"
            "瓶颈不在训练方法，在模型能力上限。"
        )
    else:
        verdict = "UNLIKELY_FEASIBLE"
        reasoning = (
            f"Best-of-N 平均仅比基线高 {mean_delta:.1f} 分，{pct_improved:.0f}% 的 query 显著改善。"
            "高温度没有提升答案质量的天花板，模型在 T=0.3 下已接近最优。"
            "建议优先尝试更换更大模型（8B+）或优化 prompt。"
        )

    return {"verdict": verdict, "reasoning": reasoning}


def print_report(result: dict):
    """打印分析报告。"""
    if not result:
        logger.warning("Empty result, skipping report.")
        return

    print("\n" + "=" * 70)
    print(f"  RL/DPO Feasibility Report: {result['model']}")
    print("=" * 70)

    print(f"\n  Overview ({result['num_queries']} queries, "
          f"~{result['mean_samples_per_query']:.0f} samples/query):")
    print(f"    Baseline (T=0.3)  mean:  {result['mean_baseline']}")
    print(f"    Mean-of-N (T=0.8) mean:  {result['mean_mean_of_n']}")
    print(f"    Best-of-N (T=0.8) mean:  {result['mean_best_of_n']}")
    print(f"    Delta (Best - Baseline):  +{result['mean_delta_best_vs_baseline']}")
    print(f"    Median Delta:             {result['median_delta_best_vs_baseline']}")
    print(f"    Mean std (per query):      {result['mean_std']}")

    print(f"\n  Pass Rate:")
    print(f"    Baseline (T=0.3):  {result['baseline_pass_rate']}%")
    print(f"    Best-of-N (T=0.8): {result['best_pass_rate']}%")

    print(f"\n  Query分布:")
    print(f"    显著改善 (Δ>5):  {result['n_improved']:3d} ({result['pct_improved']}%)")
    print(f"    基本持平 (|Δ|≤5): {result['n_stable']:3d} ({result['pct_stable']}%)")
    print(f"    显著退化 (Δ<-5): {result['n_regressed']:3d} ({result['pct_regressed']}%)")

    if "dim_summary" in result:
        print(f"\n  分维度 Best-of-N Delta:")
        for dim, stats in result["dim_summary"].items():
            sign = "+" if stats["mean_delta"] >= 0 else ""
            print(f"    {dim:25s}: {sign}{stats['mean_delta']:.2f}  "
                  f"(median={stats['median_delta']:+.2f}, "
                  f"positive={stats['pct_positive']:.0f}%)")

    verdict = result.get("feasibility_verdict", {})
    print(f"\n  {'=' * 50}")
    print(f"  VERDICT: {verdict.get('verdict', 'UNKNOWN')}")
    print(f"  {verdict.get('reasoning', '')}")
    print(f"  {'=' * 50}")
    print()


def load_baseline_scores(model_id: str, prompt_version: str = "v0") -> Optional[Path]:
    """查找基线评分文件。

    查找逻辑：
    1. 先查 exp-011/judge_scores/（如果有同 prompt 的基线）
    2. 再查 exp-010/judge_scores/（prompt 工程实验的基线）
    3. 最后查 exp-005/judge_scores/（原始基线，仅 v0 prompt）
    """
    candidates = [
        # exp-011 自己的目录（用户可以通过 --baseline 直接指定）
        # exp-010 的 prompt 工程结果
        DATA_ROOT / "results" / "exp010" / "judge_scores" /
        f"{model_id}-{prompt_version}_judged.jsonl",
        DATA_ROOT / "results" / "exp005" / "judge_scores" /
        f"{model_id}-{prompt_version}_judged.jsonl",
        # exp-005 原始基线（v0 prompt）
        BASELINE_SCORES_DIR / f"{model_id}_judged.jsonl",
    ]
    if "-nothink" in model_id:
        base = model_id.replace("-nothink", "")
        # exp-010 无 -nothink 后缀的变体
        candidates.insert(1, DATA_ROOT / "results" / "exp010" / "judge_scores" /
                          f"{base}-{prompt_version}_judged.jsonl")
        candidates.insert(2, DATA_ROOT / "results" / "exp005" / "judge_scores" /
                          f"{base}-{prompt_version}_judged.jsonl")
        candidates.append(BASELINE_SCORES_DIR / f"{base}_judged.jsonl")

    for path in candidates:
        if path.exists():
            return path
    return None


def find_multi_scores(model_id: str, prompt_version: str = "v0") -> Optional[Path]:
    """查找多样本 Judge 评分文件（来自 exp-011）。

    命名约定：{model}_{prompt}_t0.8_n5_s42_judged.jsonl
    """
    if not MULTI_SCORES_DIR.exists():
        return None
    # 按命名约定查找
    prefix_old = f"{model_id}_t0.8"          # 旧格式：无 prompt version
    prefix_new = f"{model_id}_{prompt_version}_t0.8"  # 新格式

    for f in sorted(MULTI_SCORES_DIR.iterdir()):
        if not f.name.endswith("_judged.jsonl"):
            continue
        if f.name.startswith(prefix_new):
            return f
        if f.name.startswith(prefix_old) and prompt_version == "v0":
            return f
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Exp-011: RL/DPO feasibility analysis"
    )
    parser.add_argument(
        "--multi", type=str, default=None,
        help="Path to multi-sample judge scores JSONL",
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline judge scores JSONL (T=0.3 single sample)",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Model ID to auto-find files (e.g., qwen3-4b-nothink)",
    )
    parser.add_argument(
        "--prompt-version", type=str, default="v0",
        help="Prompt version used (e.g., v0, v1-full, v2, v3). Used for auto-finding files. Default: v0",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Analyze all available multi-sample results",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for analysis results (default: results/exp011/analysis/)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Only print report, don't save JSON",
    )
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else ANALYSIS_DIR

    # -- 自动发现模式 --
    to_analyze = []
    if args.all:
        score_dir = DATA_ROOT / "results" / "exp005" / "judge_scores"
        # 查找所有多样本评分文件（文件名包含 "t0.8"）
        # 这里不自动发现，而是让用户通过 --multi 指定
        logger.warning("--all mode: auto-discovery not implemented. Use --model-id instead.")
        return 1

    if args.model_id:
        # 自动查找对应文件
        found_multi = find_multi_scores(args.model_id, args.prompt_version)
        if not found_multi:
            logger.error(
                f"No multi-sample judge file found for {args.model_id} "
                f"(prompt={args.prompt_version}). "
                f"Looking for: {args.model_id}_{args.prompt_version}_t0.8_*_judged.jsonl "
                f"in {MULTI_SCORES_DIR}"
            )
            return 1

        baseline_path = args.baseline if args.baseline else load_baseline_scores(args.model_id, args.prompt_version)
        if baseline_path and isinstance(baseline_path, str):
            baseline_path = Path(baseline_path)
        if not baseline_path:
            logger.error(f"No baseline judge file found for {args.model_id} (prompt={args.prompt_version})")
            return 1

        to_analyze.append((args.model_id, found_multi, baseline_path))
    elif args.multi and args.baseline:
        to_analyze.append(("model", Path(args.multi), Path(args.baseline)))
    else:
        logger.error(
            "Specify either:\n"
            "  --model-id qwen3-4b-nothink --prompt-version v1-full  (auto-find)\n"
            "  --multi <file> --baseline <file>  (manual)"
        )
        return 1

    # -- 逐模型分析 --
    all_results = {}
    for model_id, multi_path, baseline_path in to_analyze:
        if not multi_path.exists():
            logger.error(f"Multi-sample file not found: {multi_path}")
            continue
        if not baseline_path.exists():
            logger.error(f"Baseline file not found: {baseline_path}")
            continue

        multi_entries = load_judged_results(multi_path)
        baseline_entries = load_judged_results(baseline_path)

        result = analyze_model(multi_entries, baseline_entries, model_id)
        if result:
            print_report(result)
            all_results[model_id] = result

    # -- 保存 --
    if not args.no_save and all_results:
        os.makedirs(output_dir, exist_ok=True)
        output_path = output_dir / "rl_feasibility_report.json"
        # 去掉 per_query 细节以减小文件体积
        save_data = {}
        for mid, r in all_results.items():
            save_data[mid] = {k: v for k, v in r.items() if k != "per_query"}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Report saved to {output_path}")

        # 同时保存带 per_query 的完整数据
        full_path = output_dir / "rl_feasibility_full.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Full data saved to {full_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
