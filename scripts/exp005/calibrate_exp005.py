"""
Exp-005 人机校准工具。

提供：
  1. 生成人工标注 CSV 模板
  2. 计算 Judge-Human 一致性指标（Cohen's Kappa, Agreement Rate, Spearman's ρ）
  3. 差异案例分析
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import DATA_ROOT
from src.evaluation.judge_prompts import GEN_STAGE_DIMS, GEN_STAGE_DIM_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005"

# 人类标注的核心 6 个维度（简化自 10 维）
HUMAN_ANNOTATION_DIMS = [
    "veracity",
    "synthesis_quality",
    "provenance",
    "user_experience",
    "safety",
    "refusal_judgment",
]

HUMAN_DIM_LABELS_CN = {
    "veracity": "信息准确性 (1-4)",
    "synthesis_quality": "信息综合质量 (1-4)",
    "provenance": "引文与可信度 (1-4)",
    "user_experience": "用户体验 (1-4)",
    "safety": "安全合规 (1-4)",
    "refusal_judgment": "拒答与边界判断 (1-4)",
    "overall": "综合质量 (1-4)",
}


def generate_annotation_template(
    generations_file: Path,
    output_file: Path,
    sample_size: int = 50,
    seed: int = 42,
):
    """
    从生成结果文件中抽取样本，生成人工标注 CSV 模板。

    CSV 列：
      query_id, query_text, answer, passages_summary,
      veracity, synthesis_quality, provenance, user_experience,
      safety, refusal_judgment, overall, notes
    """
    import random

    with open(generations_file, "r", encoding="utf-8") as f:
        all_gens = [json.loads(line) for line in f if line.strip()]

    if len(all_gens) > sample_size:
        random.seed(seed)
        sampled = random.sample(all_gens, sample_size)
    else:
        sampled = all_gens

    fieldnames = (
        ["query_id", "query_text", "answer", "passages_summary"]
        + HUMAN_ANNOTATION_DIMS
        + ["overall", "notes"]
    )

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in sampled:
            passages = item.get("passages", [])
            passages_summary = " | ".join(
                p.get("text", str(p))[:100] + "..."
                for p in passages[:5]
            )

            row = {
                "query_id": item.get("query_id", ""),
                "query_text": item.get("query_text", item.get("query", "")),
                "answer": item.get("answer", ""),
                "passages_summary": passages_summary,
                "notes": "",
            }
            for dim in HUMAN_ANNOTATION_DIMS:
                row[dim] = ""
            row["overall"] = ""

            writer.writerow(row)

    logger.info(f"Annotation template saved to {output_file} ({len(sampled)} samples)")


def load_annotations(annotation_file: Path) -> dict[str, dict]:
    """
    加载人工标注 CSV。

    Returns:
        {query_id: {dim: score, ..., "overall": score, "notes": str}}
    """
    annotations = {}
    with open(annotation_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row.get("query_id", "").strip()
            if not qid:
                continue
            ann = {}
            for dim in HUMAN_ANNOTATION_DIMS + ["overall"]:
                val = row.get(dim, "").strip()
                if val:
                    try:
                        ann[dim] = int(val)
                    except ValueError:
                        ann[dim] = None
                else:
                    ann[dim] = None
            ann["notes"] = row.get("notes", "").strip()
            annotations[qid] = ann
    return annotations


def load_judge_scores(judge_file: Path) -> dict[str, dict]:
    """加载 Judge 评分结果。"""
    scores = {}
    with open(judge_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            qid = item.get("query_id", "")
            scores[qid] = item.get("scores", {})
    return scores


def cohens_kappa(human_scores: list[int | None], judge_scores: list[int | None]) -> dict:
    """
    计算 Cohen's Kappa。

    Returns:
        {"kappa": float, "n_valid": int}
    """
    valid_pairs = [
        (h, j) for h, j in zip(human_scores, judge_scores)
        if h is not None and j is not None
    ]
    n = len(valid_pairs)
    if n < 5:
        return {"kappa": float("nan"), "n_valid": n}

    from collections import Counter

    observed = Counter()
    for h, j in valid_pairs:
        observed[(h, j)] += 1

    total = sum(observed.values())
    p_o = sum(observed[(k, k)] for k in range(1, 5)) / total

    human_marginal = Counter(h for h, _ in valid_pairs)
    judge_marginal = Counter(j for _, j in valid_pairs)
    p_e = sum(
        (human_marginal.get(k, 0) / total) * (judge_marginal.get(k, 0) / total)
        for k in range(1, 5)
    )

    if p_e == 1.0:
        return {"kappa": 1.0 if p_o == 1.0 else 0.0, "n_valid": n}

    kappa = (p_o - p_e) / (1 - p_e)
    return {"kappa": round(kappa, 4), "n_valid": n}


def exact_agreement_rate(human_scores: list[int | None], judge_scores: list[int | None]) -> dict:
    """精确一致率（±0 档）"""
    valid = [(h, j) for h, j in zip(human_scores, judge_scores)
             if h is not None and j is not None]
    if not valid:
        return {"rate": float("nan"), "n_valid": 0}
    exact = sum(1 for h, j in valid if h == j)
    return {"rate": round(exact / len(valid), 4), "n_valid": len(valid)}


def adjacent_agreement_rate(human_scores: list[int | None], judge_scores: list[int | None]) -> dict:
    """相邻一致率（±1 档）"""
    valid = [(h, j) for h, j in zip(human_scores, judge_scores)
             if h is not None and j is not None]
    if not valid:
        return {"rate": float("nan"), "n_valid": 0}
    adjacent = sum(1 for h, j in valid if abs(h - j) <= 1)
    return {"rate": round(adjacent / len(valid), 4), "n_valid": len(valid)}


def spearman_rho(human_scores: list[int | None], judge_scores: list[int | None]) -> dict:
    """Spearman 秩相关系数"""
    from scipy.stats import spearmanr

    valid = [(h, j) for h, j in zip(human_scores, judge_scores)
             if h is not None and j is not None]
    if len(valid) < 4:
        return {"rho": float("nan"), "p_value": float("nan"), "n_valid": len(valid)}

    h_arr = [h for h, _ in valid]
    j_arr = [j for _, j in valid]

    if len(set(h_arr)) == 1 or len(set(j_arr)) == 1:
        return {"rho": float("nan"), "p_value": 1.0, "n_valid": len(valid)}

    rho, p = spearmanr(h_arr, j_arr)
    return {"rho": round(float(rho), 4), "p_value": round(float(p), 4), "n_valid": len(valid)}


def compute_calibration(
    human_file: Path,
    judge_file: Path,
) -> dict:
    """
    计算完整的人机校准报告。

    Returns:
        {
            "per_dimension": {dim: {kappa, exact_agree, adjacent_agree, spearman}},
            "human_granularity": {dim: {n_1, n_2, n_3, n_4}},  # 人类评分分布
            "judge_granularity": {dim: {n_1, n_2, n_3, n_4}},  # Judge 评分分布
            "divergent_cases": [{qid, dim, human, judge}],     # 分歧最大的案例
        }
    """
    human_ann = load_annotations(human_file)
    judge_scores = load_judge_scores(judge_file)

    common_qids = set(human_ann.keys()) & set(judge_scores.keys())
    logger.info(f"Common queries: {len(common_qids)}")

    human_dist = {dim: {i: 0 for i in range(1, 5)} for dim in HUMAN_ANNOTATION_DIMS}
    judge_dist = {dim: {i: 0 for i in range(1, 5)} for dim in HUMAN_ANNOTATION_DIMS}

    per_dim = {}
    divergent_cases = []

    for dim in HUMAN_ANNOTATION_DIMS:
        h_scores = []
        j_scores = []

        for qid in common_qids:
            h_val = human_ann[qid].get(dim)
            if isinstance(h_val, (int, float)):
                human_dist[dim][int(h_val)] = human_dist[dim].get(int(h_val), 0) + 1

            j_obj = judge_scores[qid].get(dim, {})
            if isinstance(j_obj, dict):
                j_val = int(j_obj.get("score", 0))
            else:
                j_val = int(j_obj) if j_obj else 0
            judge_dist[dim][j_val] = judge_dist[dim].get(j_val, 0) + 1

            if h_val is not None and j_val > 0:
                h_scores.append(int(h_val))
                j_scores.append(j_val)

                if abs(int(h_val) - j_val) >= 2:
                    divergent_cases.append({
                        "query_id": qid,
                        "dimension": dim,
                        "human_score": int(h_val),
                        "judge_score": j_val,
                        "diff": abs(int(h_val) - j_val),
                    })

        per_dim[dim] = {
            "kappa": cohens_kappa(h_scores, j_scores),
            "exact_agreement": exact_agreement_rate(h_scores, j_scores),
            "adjacent_agreement": adjacent_agreement_rate(h_scores, j_scores),
            "spearman": spearman_rho(h_scores, j_scores),
        }

    divergent_cases.sort(key=lambda x: x["diff"], reverse=True)

    return {
        "per_dimension": per_dim,
        "human_distribution": human_dist,
        "judge_distribution": judge_dist,
        "divergent_cases": divergent_cases[:20],
        "n_common_queries": len(common_qids),
    }


def print_calibration_report(report: dict):
    """打印人机校准报告。"""
    print()
    print("=" * 70)
    print("  HUMAN-JUDGE CALIBRATION REPORT")
    print("=" * 70)
    print(f"  Common queries: {report['n_common_queries']}")
    print()

    for dim in HUMAN_ANNOTATION_DIMS:
        label = HUMAN_DIM_LABELS_CN.get(dim, dim)
        metrics = report["per_dimension"][dim]
        k = metrics["kappa"]["kappa"]
        exact = metrics["exact_agreement"]["rate"]
        adj = metrics["adjacent_agreement"]["rate"]
        sp = metrics["spearman"]["rho"]

        k_str = f"{k:.3f}" if not (isinstance(k, float) and (k != k)) else "N/A"
        exact_str = f"{exact:.1%}" if not (isinstance(exact, float) and (exact != exact)) else "N/A"
        adj_str = f"{adj:.1%}" if not (isinstance(adj, float) and (adj != adj)) else "N/A"
        sp_str = f"{sp:.3f}" if not (isinstance(sp, float) and (sp != sp)) else "N/A"

        k_status = (
            "✅ EXCELLENT" if (isinstance(k, float) and k >= 0.80) else
            "✅ GOOD" if (isinstance(k, float) and k >= 0.60) else
            "⚠️ MODERATE" if (isinstance(k, float) and k >= 0.40) else
            "❌ POOR" if (isinstance(k, float) and not (k != k)) else "N/A"
        )

        print(f"  {label}:")
        print(f"    Kappa={k_str} {k_status}")
        print(f"    Exact={exact_str}  Adjacent={adj_str}  Spearman={sp_str}")
        print()

    print("  Top 5 Most Divergent Cases:")
    for case in report["divergent_cases"][:5]:
        label = HUMAN_DIM_LABELS_CN.get(case["dimension"], case["dimension"])
        print(f"    {case['query_id']}: {label} "
              f"Human={case['human_score']} vs Judge={case['judge_score']} "
              f"(diff={case['diff']})")
    print("=" * 70)


def compute_inter_annotator_agreement(
    annotator1_file: Path,
    annotator2_file: Path,
) -> dict:
    """计算两名人标注者之间的一致性。"""
    ann1 = load_annotations(annotator1_file)
    ann2 = load_annotations(annotator2_file)

    common_qids = set(ann1.keys()) & set(ann2.keys())

    per_dim = {}
    for dim in HUMAN_ANNOTATION_DIMS + ["overall"]:
        scores1 = []
        scores2 = []
        for qid in common_qids:
            v1 = ann1[qid].get(dim)
            v2 = ann2[qid].get(dim)
            if v1 is not None and v2 is not None:
                scores1.append(int(v1))
                scores2.append(int(v2))

        per_dim[dim] = {
            "kappa": cohens_kappa(scores1, scores2),
            "exact_agreement": exact_agreement_rate(scores1, scores2),
            "n_pairs": len(scores1),
        }

    return {
        "per_dimension": per_dim,
        "n_common_queries": len(common_qids),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp-005 Human Calibration Tools")
    sub = parser.add_subparsers(dest="command")

    gen_parser = sub.add_parser("template", help="Generate annotation CSV template")
    gen_parser.add_argument("--input", "-i", type=str, required=True,
                            help="Generation results JSONL file")
    gen_parser.add_argument("--output", "-o", type=str, required=True,
                            help="Output CSV file")
    gen_parser.add_argument("--n", type=int, default=50,
                            help="Number of samples (default: 50)")

    cal_parser = sub.add_parser("calibrate", help="Compute calibration metrics")
    cal_parser.add_argument("--human", type=str, required=True,
                            help="Human annotation CSV file")
    cal_parser.add_argument("--judge", type=str, required=True,
                            help="Judge scores JSONL file")
    cal_parser.add_argument("--output", "-o", type=str, default=None,
                            help="Output JSON report file")

    iaa_parser = sub.add_parser("iaa", help="Compute inter-annotator agreement")
    iaa_parser.add_argument("--a1", type=str, required=True,
                            help="Annotator 1 CSV file")
    iaa_parser.add_argument("--a2", type=str, required=True,
                            help="Annotator 2 CSV file")

    args = parser.parse_args()

    if args.command == "template":
        generate_annotation_template(
            Path(args.input), Path(args.output), args.n
        )

    elif args.command == "calibrate":
        report = compute_calibration(
            Path(args.human), Path(args.judge)
        )
        print_calibration_report(report)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Report saved to {args.output}")

    elif args.command == "iaa":
        report = compute_inter_annotator_agreement(
            Path(args.a1), Path(args.a2)
        )
        print()
        print(f"Inter-Annotator Agreement ({report['n_common_queries']} common queries):")
        for dim, metrics in report["per_dimension"].items():
            label = HUMAN_DIM_LABELS_CN.get(dim, dim)
            k = metrics["kappa"]["kappa"]
            exact = metrics["exact_agreement"]["rate"]
            k_str = f"{k:.3f}" if not (isinstance(k, float) and (k != k)) else "N/A"
            exact_str = f"{exact:.1%}" if not (isinstance(exact, float) and (exact != exact)) else "N/A"
            print(f"  {label}: Kappa={k_str}, Exact={exact_str} (n={metrics['n_pairs']})")

    else:
        parser.print_help()
