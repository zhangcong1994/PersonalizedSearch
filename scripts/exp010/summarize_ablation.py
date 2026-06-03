"""
Exp-010 消融结果汇总 —— 读取 Phase 4 所有消融变体的 Judge 结果，打印贡献分析表。

用法：
  python scripts/exp010/summarize_ablation.py

前提：所有消融变体已跑完 run_phase.py（generations + judge）。
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from scripts.exp010.run_phase import load_judge_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

JUDGE_DIR = DATA_ROOT / "results" / "exp010" / "judge_scores"

CORE6_LABELS = {
    "veracity": "准确性",
    "safety": "安全性",
    "relevance": "相关性",
    "synthesis_quality": "整合质量",
    "citation_quality": "引文质量",
    "user_experience": "用户体验",
}

# 消融变体列表（按预期贡献排序）
ABLATION_ORDER = [
    ("v0", "基线（50 条）"),
    ("v3", "Phase 1+2+3 全部（50 条）"),
    ("abl-no-cot", "v3 − 分步指令"),
    ("abl-no-contrast", "v3 − 对比示例"),
    ("abl-no-rules", "v3 − 硬规则区"),
    ("abl-no-refusal", "v3 − 分级拒答"),
]


def main():
    results: dict[str, dict] = {}
    missing = []

    for version_id, version_label in ABLATION_ORDER:
        # 找到对应的 judged 文件
        candidates = list(JUDGE_DIR.glob(f"*-{version_id}_judged.jsonl"))
        if not candidates:
            missing.append(version_id)
            continue

        filepath = candidates[0]
        results[version_id] = load_judge_results(filepath)
        results[version_id]["label"] = version_label

    if missing:
        logger.warning(f"Missing judge results for: {missing}")

    if "v3" not in results:
        logger.error("v3 (full prompt) results required for ablation analysis")
        sys.exit(1)

    v3 = results["v3"]
    v0 = results.get("v0")

    # ── 打印汇总表 ──
    print()
    print("=" * 90)
    print("  Exp-010 Phase 4: 消融分析")
    print("=" * 90)
    print()
    print(f"  {'Variant':<28s} {'Avg':>6s} {'Pass%':>7s}", end="")
    for label in CORE6_LABELS.values():
        print(f"  {label:<6s}", end="")
    print(f"  {'Δ总分':>6s}")
    print(f"  {'-' * 28} {'-' * 6} {'-' * 7}", end="")
    for _ in CORE6_LABELS:
        print(f"  {'-' * 6}", end="")
    print(f"  {'-' * 6}")

    for version_id, version_label in ABLATION_ORDER:
        r = results.get(version_id)
        if r is None:
            continue

        dim_avgs = r.get("dim_avgs", {})

        print(
            f"  {version_label:<28s} {r['avg_score']:>6.1f} "
            f"{r['pass_pct']:>6.1f}%",
            end="",
        )
        for dim in CORE6_LABELS:
            print(f"  {dim_avgs.get(dim, 0):>6.2f}", end="")

        # Δ 总分 vs v3
        delta = r["avg_score"] - v3["avg_score"]
        sign = "+" if delta >= 0 else ""
        print(f"  {sign}{delta:>5.1f}")
    print()

    # ── 消融边际贡献 ──
    print("  " + "=" * 70)
    print("  消融边际贡献 (各改动移除后导致的分数变化)")
    print("  " + "=" * 70)
    print(f"  {'移除的改动':<20s} {'Δ 总分':>8s} {'Δ 整合':>8s}", end="")
    for label in list(CORE6_LABELS.values())[3:]:
        print(f"  {'Δ ' + label:>8s}", end="")
    print()
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8}", end="")
    for _ in range(3):
        print(f"  {'-' * 8}", end="")
    print()

    for version_id, version_label in ABLATION_ORDER:
        if version_id in ("v0", "v3") or not version_id.startswith("abl-"):
            continue

        r = results.get(version_id)
        if r is None:
            continue

        removal_name = version_label.replace("v3 − ", "")
        v3_dim = v3.get("dim_avgs", {})
        r_dim = r.get("dim_avgs", {})

        delta_total = v3["avg_score"] - r["avg_score"]
        delta_synth = v3_dim.get("synthesis_quality", 0) - r_dim.get("synthesis_quality", 0)

        # 移除这个改动后分数下降了 → 说明这个改动是正向的（贡献 = 下降幅度）
        sign = "+" if delta_total >= 0 else ""
        print(
            f"  {removal_name:<20s} {sign}{delta_total:>7.1f}  {sign}{delta_synth:>7.2f}",
            end="",
        )
        for dim in ["citation_quality", "user_experience"]:
            d = v3_dim.get(dim, 0) - r_dim.get(dim, 0)
            sign2 = "+" if d >= 0 else ""
            print(f"  {sign2}{d:>7.2f}", end="")
        print()

    print()

    # ── v0 参考 ──
    if v0:
        print(f"  v0 基线 (50 条): Avg={v0['avg_score']:.1f}, Pass%={v0['pass_pct']:.1f}%")
        gain = v3["avg_score"] - v0["avg_score"]
        sign = "+" if gain >= 0 else ""
        print(f"  v3 (全改动) vs v0: {sign}{gain:.1f} 分")
    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
