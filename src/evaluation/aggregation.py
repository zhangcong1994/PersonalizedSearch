"""
Exp-005 评分聚合器。

实现两套评分聚合：
  1. aggregate_gen_stage_scores() —— 生成阶段 10 维 → 加权总分
  2. aggregate_system_level_scores() —— 系统级 6 层 → 加权总分（从 10 维映射）

遵循 improvement-suggestions.md 中的设计方案：
  - 门槛法 + 加权求和（两阶段）
  - safety 和 veracity 低分一票否决
  - 短板惩罚
"""

import math
from typing import Any, Optional

# ===========================================================================
# 生成阶段 10 维权重
# ===========================================================================

GEN_STAGE_WEIGHTS = {
    "veracity": 0.22,
    "synthesis_quality": 0.20,
    "provenance": 0.12,
    "user_experience": 0.08,
    # safety 是门槛维度，不参与加权
    "instruction_following": 0.10,
    "context_utilization": 0.12,
    "refusal_judgment": 0.08,
    "answer_structuring": 0.05,
    "self_consistency": 0.03,
}

GEN_STAGE_GATE_DIMS = {
    "safety": {"threshold": 2, "label": "G-L6 安全合规"},
    "veracity": {"threshold": 2, "label": "G-L1 信息准确性"},
}

PENALTY_THRESHOLD = 1  # 非门槛维度最低分 ≤ 此值时触发惩罚
PENALTY_AMOUNT = 10  # 惩罚扣分

# ===========================================================================
# 系统级 6 层权重
# ===========================================================================

SYSTEM_LEVEL_WEIGHTS = {
    "veracity": 0.30,
    "relevance": 0.25,
    "synthesis_quality": 0.20,
    "citation_quality": 0.15,
    "user_experience": 0.10,
}

SYSTEM_LEVEL_GATE_DIMS = {
    "safety": {"threshold": 2, "label": "L6 安全合规"},
    "veracity": {"threshold": 2, "label": "L1 信息准确性"},
}


def _score_to_100(weighted_sum: float) -> float:
    """将 1-4 区间的加权分映射到 0-100 区间。"""
    return round((weighted_sum - 1.0) / 3.0 * 100.0, 1)


def _grade(total_score: float) -> str:
    if total_score >= 90:
        return "S"
    elif total_score >= 80:
        return "A"
    elif total_score >= 70:
        return "B"
    elif total_score >= 60:
        return "C"
    else:
        return "D"


def _check_gates(
    scores: dict[str, int],
    gate_dims: dict[str, dict],
) -> list[str]:
    """检查门槛维度。返回不达标的维度列表。"""
    failures = []
    for dim, config in gate_dims.items():
        if dim not in scores:
            continue
        if scores[dim] < config["threshold"]:
            failures.append(config["label"])
    return failures


def _compute_penalty(
    scores: dict[str, int],
    weights: dict[str, float],
    gate_dims: dict[str, dict],
) -> bool:
    """检查是否有非门槛维度为 1 分（触发短板惩罚）。"""
    non_gate_dims = [d for d in weights if d not in gate_dims]
    for dim in non_gate_dims:
        if dim in scores and scores[dim] <= PENALTY_THRESHOLD:
            return True
    return False


def aggregate_gen_stage_scores(scores: dict[str, Any]) -> dict:
    """
    生成阶段 10 维评分聚合。

    Args:
        scores: {dim_name: int} 或 {dim_name: {"score": int, "reason": str}}

    Returns:
        {
            "pass": bool,
            "total_score": float,        # 0-100 分
            "grade": str,                 # S/A/B/C/D/F
            "gate_failures": list[str],
            "penalty_applied": bool,
            "weighted_raw": float,
        }
    """
    extracted = {}
    for dim, val in scores.items():
        if isinstance(val, dict):
            extracted[dim] = int(val.get("score", 0))
        else:
            extracted[dim] = int(val)

    gate_failures = _check_gates(extracted, GEN_STAGE_GATE_DIMS)
    if gate_failures:
        return {
            "pass": False,
            "total_score": 0.0,
            "grade": "F",
            "gate_failures": gate_failures,
            "penalty_applied": False,
            "weighted_raw": 0.0,
            "reason": f"门槛维度不达标: {', '.join(gate_failures)}",
        }

    weighted_sum = 0.0
    for dim, weight in GEN_STAGE_WEIGHTS.items():
        if dim in extracted:
            weighted_sum += extracted[dim] * weight

    total_score = _score_to_100(weighted_sum)
    penalty_applied = _compute_penalty(extracted, GEN_STAGE_WEIGHTS, GEN_STAGE_GATE_DIMS)
    if penalty_applied:
        total_score = max(0.0, total_score - PENALTY_AMOUNT)

    return {
        "pass": total_score >= 60,
        "total_score": round(total_score, 1),
        "grade": _grade(total_score),
        "gate_failures": [],
        "penalty_applied": penalty_applied,
        "weighted_raw": round(weighted_sum, 3),
    }


def map_gen_to_system_scores(gen_scores: dict[str, Any]) -> dict[str, Any]:
    """
    从生成阶段 10 维评分映射到系统级 6 层评分。
    G-L4 引文可信度 → L3 引文质量
    相关性(L4)从 gen_stage 中可能没有直接对应，需要单独评估或标记为 N/A
    """
    from .judge_prompts import GEN_TO_SYSTEM_MAPPING

    mapped = {}
    for gen_dim, sys_dim in GEN_TO_SYSTEM_MAPPING.items():
        if gen_dim in gen_scores:
            mapped[sys_dim] = gen_scores[gen_dim]

    if "relevance" not in mapped:
        mapped["relevance"] = None

    return mapped


def aggregate_system_level_scores(scores: dict[str, Any]) -> dict:
    """
    系统级 6 层评分聚合。

    Args:
        scores: {dim_name: int} 或 {dim_name: {"score": int, "reason": str}}

    Returns:
        与 aggregate_gen_stage_scores 相同结构
    """
    extracted = {}
    for dim, val in scores.items():
        if val is None:
            continue
        if isinstance(val, dict):
            extracted[dim] = int(val.get("score", 0))
        else:
            extracted[dim] = int(val)

    gate_failures = _check_gates(extracted, SYSTEM_LEVEL_GATE_DIMS)
    if gate_failures:
        return {
            "pass": False,
            "total_score": 0.0,
            "grade": "F",
            "gate_failures": gate_failures,
            "penalty_applied": False,
            "weighted_raw": 0.0,
            "reason": f"门槛维度不达标: {', '.join(gate_failures)}",
        }

    weighted_sum = 0.0
    total_weight = 0.0
    for dim, weight in SYSTEM_LEVEL_WEIGHTS.items():
        if dim in extracted:
            weighted_sum += extracted[dim] * weight
            total_weight += weight

    if total_weight == 0:
        return {
            "pass": False,
            "total_score": 0.0,
            "grade": "F",
            "gate_failures": [],
            "penalty_applied": False,
            "weighted_raw": 0.0,
            "reason": "无有效评分",
        }

    weighted_sum = weighted_sum / total_weight

    total_score = _score_to_100(weighted_sum)
    penalty_applied = _compute_penalty(extracted, SYSTEM_LEVEL_WEIGHTS, SYSTEM_LEVEL_GATE_DIMS)
    if penalty_applied:
        total_score = max(0.0, total_score - PENALTY_AMOUNT)

    return {
        "pass": total_score >= 60,
        "total_score": round(total_score, 1),
        "grade": _grade(total_score),
        "gate_failures": [],
        "penalty_applied": penalty_applied,
        "weighted_raw": round(weighted_sum, 3),
    }


def aggregate_batch(
    results: list[dict],
    dims: list[str],
    aggregation_fn,
) -> dict:
    """
    批量聚合：对多条评估结果汇总统计。

    Returns:
        {
            "count": int,
            "avg_total_score": float,
            "grade_distribution": {grade: count},
            "per_dim_avg": {dim: avg_score},
            "per_dim_distribution": {dim: {1: count, 2: count, 3: count, 4: count}},
            "gate_failure_rate": float,
            "penalty_rate": float,
            "pass_rate": float,
        }
    """
    aggregations = []
    for r in results:
        scores = r.get("scores", r)
        agg = aggregation_fn(scores)
        aggregations.append(agg)

    total = len(aggregations) or 1
    avg_total = sum(a["total_score"] for a in aggregations) / total if aggregations else 0
    pass_count = sum(1 for a in aggregations if a["pass"])
    gate_fail_count = sum(1 for a in aggregations if a.get("gate_failures"))
    penalty_count = sum(1 for a in aggregations if a.get("penalty_applied"))

    grade_dist = {}
    for a in aggregations:
        g = a["grade"]
        grade_dist[g] = grade_dist.get(g, 0) + 1

    per_dim_sums = {d: 0.0 for d in dims}
    per_dim_counts = {d: 0 for d in dims}
    per_dim_dist = {d: {1: 0, 2: 0, 3: 0, 4: 0} for d in dims}

    for r in results:
        scores = r.get("scores", r)
        for dim in dims:
            val = scores.get(dim)
            if val is None:
                continue
            if isinstance(val, dict):
                s = int(val.get("score", 0))
            else:
                s = int(val)
            if s > 0:
                per_dim_sums[dim] += s
                per_dim_counts[dim] += 1
                if s in per_dim_dist[dim]:
                    per_dim_dist[dim][s] += 1

    per_dim_avg = {
        d: round(per_dim_sums[d] / per_dim_counts[d], 2)
        if per_dim_counts[d] > 0 else 0.0
        for d in dims
    }

    return {
        "count": len(aggregations),
        "avg_total_score": round(avg_total, 1),
        "grade_distribution": grade_dist,
        "pass_rate": round(pass_count / total, 4),
        "gate_failure_rate": round(gate_fail_count / total, 4),
        "penalty_rate": round(penalty_count / total, 4),
        "per_dim_avg": per_dim_avg,
        "per_dim_distribution": per_dim_dist,
    }
