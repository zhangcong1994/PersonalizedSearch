"""
Exp-010 Phase Runner —— 一键运行单 Phase 的完整评估流程。

流程:
  1. 生成: 调用 generate_exp010.py
     → results/exp010/generations/{model}-{prompt_version}.jsonl
  2. Judge: 调用 run_judge_exp005.py（复用 exp-005 Judge 体系）
     → results/exp010/judge_scores/{model}-{prompt_version}_judged.jsonl
  3. 对比: 加载基线评分，打印维度级差异表

用法:
  # 完整流程（生成 + Judge + 对比）
  python scripts/exp010/run_phase.py --model qwen3-4b-nonthink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl

  # 用 reasoner 做 Judge
  python scripts/exp010/run_phase.py --model qwen3-4b-nonthink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl \\
      --judge-model deepseek-reasoner

  # 只做生成
  python scripts/exp010/run_phase.py --model qwen3-4b-nothink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl --gen-only

  # 只做 Judge（已有生成结果时）
  python scripts/exp010/run_phase.py --model qwen3-4b-nothink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl --judge-only

  # 只做对比分析（已有生成+Judge结果时）
  python scripts/exp010/run_phase.py --model qwen3-4b-nothink \\
      --prompt-version v1-full --compare-only

  # 强制重新生成（忽略缓存）
  python scripts/exp010/run_phase.py --model qwen3-4b-nothink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl --force

  # 限制条数（快速试跑，如 20 条）
  python scripts/exp010/run_phase.py --model qwen3-4b-nothink \\
      --prompt-version v1-full --input data/exp005_queries.jsonl --max 20
"""

import os
import sys
import json
import subprocess
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, PROJECT_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp010"
GEN_DIR = RESULTS_DIR / "generations"
JUDGE_DIR = RESULTS_DIR / "judge_scores"

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
GENERATE_SCRIPT = SCRIPTS_DIR / "exp010" / "generate_exp010.py"
JUDGE_SCRIPT = SCRIPTS_DIR / "exp005" / "run_judge_exp005.py"

# 6 维标签
CORE6_LABELS = {
    "veracity": "准确性",
    "safety": "安全性",
    "relevance": "相关性",
    "synthesis_quality": "整合质量",
    "citation_quality": "引文质量",
    "user_experience": "用户体验",
}


def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)


def check_baseline_exists(model: str) -> Path | None:
    """检查基线 Judge 结果是否存在。优先级: exp010 > exp005。"""
    candidates = [
        JUDGE_DIR / f"{model}-v0_judged.jsonl",
        DATA_ROOT / "results" / "exp005" / "judge_scores" / f"{model}_judged.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_generation(
    model: str,
    prompt_version: str,
    input_file: Path,
    max_queries: int = 0,
    force: bool = False,
    vllm_url: str = "http://localhost:8000/v1",
) -> Path:
    """
    调用 generate_exp010.py 生成答案。
    返回生成结果文件路径。
    """
    output_file = GEN_DIR / f"{model}-{prompt_version}.jsonl"
    ensure_dir(GEN_DIR)

    if output_file.exists() and not force:
        logger.info(f"[SKIP] Generation already exists: {output_file}")
        return output_file

    cmd = [
        sys.executable, str(GENERATE_SCRIPT),
        "--model", model,
        "--prompt-version", prompt_version,
        "--input", str(input_file),
        "--output-dir", str(GEN_DIR),
        "--vllm-url", vllm_url,
    ]
    if max_queries > 0:
        cmd.extend(["--max", str(max_queries)])

    logger.info(f"[GENERATE] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Generation failed with code {result.returncode}")
        sys.exit(1)

    logger.info(f"[GENERATE] Done → {output_file}")
    return output_file


def run_judge(
    gen_file: Path,
    model: str,
    prompt_version: str,
    force: bool = False,
    judge_model: str = "deepseek-chat",
) -> Path:
    """
    调用 run_judge_exp005.py 进行两批 Judge 评分。
    返回评分结果文件路径。
    """
    output_file = JUDGE_DIR / f"{model}-{prompt_version}_judged.jsonl"
    ensure_dir(JUDGE_DIR)

    if output_file.exists() and not force:
        logger.info(f"[SKIP] Judge already exists: {output_file}")
        return output_file

    cmd = [
        sys.executable, str(JUDGE_SCRIPT),
        "--input", str(gen_file),
        "--output", str(output_file),
        "--judge-model", judge_model,
        "--concurrency", "10",
    ]

    logger.info(f"[JUDGE] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        logger.error(f"Judge failed with code {result.returncode}")
        sys.exit(1)

    logger.info(f"[JUDGE] Done → {output_file}")
    return output_file


def load_judge_results(filepath: Path) -> dict:
    """加载 Judge 评分结果，提取摘要统计。"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        return {"count": 0}

    dim_sums = {dim: 0.0 for dim in CORE6_LABELS}
    dim_counts = {dim: 0 for dim in CORE6_LABELS}
    total_scores = []
    grade_counts = {g: 0 for g in ["S", "A", "B", "C", "D", "F"]}
    pass_count = 0

    for r in records:
        scores = r.get("scores", {})
        agg = r.get("aggregated", {})
        total = agg.get("total_score", None)
        grade = agg.get("grade", "?")

        if total is not None:
            total_scores.append(total)
        if grade in grade_counts:
            grade_counts[grade] += 1
        if agg.get("pass", False):
            pass_count += 1

        for dim in CORE6_LABELS:
            if dim in scores:
                dim_sums[dim] += scores[dim]
                dim_counts[dim] += 1

    n = len(records)
    dim_avgs = {
        dim: (dim_sums[dim] / dim_counts[dim] if dim_counts[dim] > 0 else 0)
        for dim in CORE6_LABELS
    }
    total_avg = sum(total_scores) / len(total_scores) if total_scores else 0

    return {
        "count": n,
        "avg_score": round(total_avg, 1),
        "pass_pct": round(100 * pass_count / n, 1) if n > 0 else 0,
        "dim_avgs": {d: round(v, 2) for d, v in dim_avgs.items()},
        "grade_dist": grade_counts,
    }


def print_comparison(baseline: dict, current: dict, prompt_version: str):
    """打印维度级对比表。"""
    if baseline.get("count", 0) == 0 or current.get("count", 0) == 0:
        logger.warning("Cannot compare: missing baseline or current results")
        return

    b_dim = baseline.get("dim_avgs", {})
    c_dim = current.get("dim_avgs", {})

    print()
    print("=" * 75)
    print(f"  Exp-010 Phase Comparison: baseline (v0) vs {prompt_version}")
    print("=" * 75)
    print()
    print(f"  {'Model':<25s} {'Avg':>6s} {'Pass%':>7s}", end="")
    for label in CORE6_LABELS.values():
        print(f"  {label:<6s}", end="")
    print()
    print(f"  {'-' * 25} {'-' * 6} {'-' * 7}", end="")
    for _ in CORE6_LABELS:
        print(f"  {'-' * 6}", end="")
    print()

    print(
        f"  {'baseline (v0)':<25s} {baseline['avg_score']:>6.1f} "
        f"{baseline['pass_pct']:>6.1f}%",
        end="",
    )
    for dim in CORE6_LABELS:
        print(f"  {b_dim.get(dim, 0):>6.2f}", end="")
    print()

    print(
        f"  {prompt_version:<25s} {current['avg_score']:>6.1f} "
        f"{current['pass_pct']:>6.1f}%",
        end="",
    )
    for dim in CORE6_LABELS:
        print(f"  {c_dim.get(dim, 0):>6.2f}", end="")
    print()

    print(f"  {'Δ':<25s} ", end="")
    delta_avg = current["avg_score"] - baseline["avg_score"]
    delta_pass = current["pass_pct"] - baseline["pass_pct"]
    sign_avg = "+" if delta_avg >= 0 else ""
    sign_pass = "+" if delta_pass >= 0 else ""
    print(f"{sign_avg}{delta_avg:>5.1f} {sign_pass}{delta_pass:>6.1f}%", end="")
    for dim in CORE6_LABELS:
        delta = c_dim.get(dim, 0) - b_dim.get(dim, 0)
        sign = "+" if delta >= 0 else ""
        print(f"  {sign}{delta:>5.2f}", end="")
    print()

    print()
    print("  Grade Distribution:")
    for grade in ["S", "A", "B", "C", "D", "F"]:
        b_count = baseline.get("grade_dist", {}).get(grade, 0)
        c_count = current.get("grade_dist", {}).get(grade, 0)
        print(f"    {grade}: baseline={b_count:3d}  {prompt_version}={c_count:3d}")
    print()
    print("=" * 75)


def run_phase(
    model: str,
    prompt_version: str,
    input_file: Path,
    max_queries: int = 0,
    force: bool = False,
    gen_only: bool = False,
    judge_only: bool = False,
    compare_only: bool = False,
    vllm_url: str = "http://localhost:8000/v1",
    judge_model: str = "deepseek-chat",
):
    """运行单 Phase 完整流程。"""
    ensure_dir(GEN_DIR)
    ensure_dir(JUDGE_DIR)

    # ── 定位文件 ──
    gen_file = GEN_DIR / f"{model}-{prompt_version}.jsonl"
    judge_file = JUDGE_DIR / f"{model}-{prompt_version}_judged.jsonl"

    # ── 纯对比模式 ──
    if compare_only:
        baseline_path = check_baseline_exists(model)
        if baseline_path is None:
            logger.error(f"No baseline found for {model}. Run v0 first.")
            sys.exit(1)
        if not judge_file.exists():
            logger.error(f"Judge result not found: {judge_file}")
            sys.exit(1)
        baseline = load_judge_results(baseline_path)
        current = load_judge_results(judge_file)
        print_comparison(baseline, current, prompt_version)
        return

    # ── 生成 ──
    if not judge_only:
        gen_file = run_generation(
            model=model,
            prompt_version=prompt_version,
            input_file=input_file,
            max_queries=max_queries,
            force=force,
            vllm_url=vllm_url,
        )

    if gen_only:
        logger.info(f"[DONE] Generation only: {gen_file}")
        return

    # ── Judge ──
    judge_file = run_judge(
        gen_file=gen_file,
        model=model,
        prompt_version=prompt_version,
        force=force,
        judge_model=judge_model,
    )

    # ── 对比 ──
    baseline_path = check_baseline_exists(model)
    if baseline_path:
        baseline = load_judge_results(baseline_path)
        current = load_judge_results(judge_file)
        print_comparison(baseline, current, prompt_version)
    else:
        logger.warning(
            f"No baseline found. Run: python scripts/exp010/run_phase.py "
            f"--model {model} --prompt-version v0 --input {input_file}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-010 Phase Runner")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID (e.g., qwen3-4b-nothink)")
    parser.add_argument("--prompt-version", type=str, required=True,
                        help="Prompt version (e.g., v0, v1-full, v2, v3)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL with queries and passages")
    parser.add_argument("--max", type=int, default=0,
                        help="Max queries to process (0=all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation and re-judge")
    parser.add_argument("--gen-only", action="store_true",
                        help="Only run generation, skip judge + compare")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run judge + compare (skip generation)")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only print comparison (skip generation and judge)")
    parser.add_argument("--vllm-url", type=str,
                        default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--judge-model", type=str,
                        default="deepseek-chat",
                        help="Judge LLM model (deepseek-chat / deepseek-reasoner)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    run_phase(
        model=args.model,
        prompt_version=args.prompt_version,
        input_file=input_path,
        max_queries=args.max,
        force=args.force,
        gen_only=args.gen_only,
        judge_only=args.judge_only,
        compare_only=args.compare_only,
        vllm_url=args.vllm_url,
        judge_model=args.judge_model,
    )
