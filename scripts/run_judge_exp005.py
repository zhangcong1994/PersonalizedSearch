"""
Exp-005 Judge 执行器。

批量调用 Judge LLM 对生成结果进行多维度评分。
支持：
  - 生成阶段 10 维评估（默认）
  - 系统级 6 层评估
  - 结果缓存
  - 限流与重试
  - 进度显示
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intent.api_client import APIClientFactory
from src.evaluation.judge_prompts import (
    GEN_STAGE_SYSTEM_PROMPT,
    SYSTEM_LEVEL_SYSTEM_PROMPT,
    GEN_STAGE_DIMS,
    SYSTEM_LEVEL_DIMS,
    build_gen_stage_judge_input,
    parse_judge_response,
)
from src.evaluation.aggregation import (
    aggregate_gen_stage_scores,
    aggregate_system_level_scores,
    aggregate_batch,
)
from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # 指数退避基数（秒）


def load_generations(filepath: Path) -> list[dict]:
    """加载生成结果 JSONL 文件。"""
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} generations from {filepath.name}")
    return results


def judge_single(
    client,
    query: str,
    passages: list[dict],
    answer: str,
    system_prompt_text: str = None,
    judge_mode: str = "gen_stage",
    model_name: str = None,
) -> dict | None:
    """
    对单条生成结果进行 Judge 评分。

    Args:
        client: LLM API client
        query: 用户查询
        passages: 检索文档列表
        answer: LLM 生成的答案
        system_prompt_text: LLM 生成时使用的 system prompt
        judge_mode: "gen_stage" 或 "system_level"
        model_name: 生成模型的名称（用于日志和结果记录）

    Returns:
        {"scores": {...}, "aggregation": {...}, "raw_response": "..."} 或 None
    """
    if judge_mode == "gen_stage":
        system_prompt = GEN_STAGE_SYSTEM_PROMPT
        user_message = build_gen_stage_judge_input(
            query, passages, answer, system_prompt_text
        )
    else:
        system_prompt = SYSTEM_LEVEL_SYSTEM_PROMPT
        user_message = build_gen_stage_judge_input(query, passages, answer)

    full_prompt = f"{system_prompt}\n\n{user_message}"

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = client.generate(full_prompt)
            parsed = parse_judge_response(raw_response)

            if parsed is None:
                logger.warning(
                    f"Judge response parse failed (attempt {attempt + 1}). "
                    f"Response preview: {raw_response[:200]}..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
                continue

            if judge_mode == "gen_stage":
                aggregation = aggregate_gen_stage_scores(parsed)
            else:
                aggregation = aggregate_system_level_scores(parsed)

            return {
                "scores": parsed,
                "aggregation": aggregation,
                "raw_response": raw_response,
            }

        except Exception as e:
            logger.warning(f"Judge API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** (attempt + 1))

    logger.error(f"Judge failed after {MAX_RETRIES} attempts")
    return None


def run_judge(
    generations_file: Path,
    output_file: Path,
    judge_mode: str = "gen_stage",
    judge_model: str = "deepseek-chat",
    judge_api_key: Optional[str] = None,
    start_idx: int = 0,
    max_queries: int = 0,
    force: bool = False,
):
    """
    批量 Judge 评估。

    Args:
        generations_file: 生成结果 JSONL 文件
        output_file: 输出结果 JSONL 文件
        judge_mode: "gen_stage" 或 "system_level"
        judge_model: Judge LLM 模型名
        judge_api_key: API key
        start_idx: 起始索引（断点续跑）
        max_queries: 最大评估条数（0 = 全部）
        force: 是否强制重新评估（忽略已有缓存）
    """
    generations = load_generations(generations_file)
    total = len(generations)

    if max_queries > 0:
        generations = generations[:max_queries]
    if start_idx > 0:
        generations = generations[start_idx:]

    existing_results = {}
    if output_file.exists() and not force:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get("query_id", "")
                existing_results[qid] = r

    client = APIClientFactory.create(
        "deepseek" if "deepseek" in judge_model else "openai",
        api_key=judge_api_key,
        model=judge_model,
        max_tokens=2048,
        temperature=0.0,
    )

    processed = 0
    skipped = 0
    failed = 0

    logger.info(
        f"Starting judge evaluation: {len(generations)} queries, "
        f"mode={judge_mode}, judge={judge_model}"
    )

    os.makedirs(output_file.parent, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, gen in enumerate(generations):
            qid = gen.get("query_id", f"unknown-{i}")
            query_text = gen.get("query_text", gen.get("query", ""))

            if qid in existing_results:
                skipped += 1
                if skipped % 50 == 0:
                    logger.info(f"  Skipped {skipped} cached results so far...")
                continue

            passages = gen.get("passages", gen.get("context_docs", []))
            if isinstance(passages, list) and passages and not isinstance(passages[0], dict):
                passages = [
                    {"pid": f"doc-{j}", "text": str(p), "rank": j + 1}
                    for j, p in enumerate(passages)
                ]

            answer = gen.get("answer", "")
            model_name = gen.get("model_id", gen.get("model", "unknown"))
            system_prompt_text = gen.get("system_prompt", None)

            result = judge_single(
                client, query_text, passages, answer,
                system_prompt_text, judge_mode, model_name,
            )

            if result is None:
                failed += 1
                result = {"scores": {}, "aggregation": {"total_score": -1}, "error": True}

            output = {
                "query_id": qid,
                "query_text": query_text,
                "model": model_name,
                "judge_model": judge_model,
                "judge_mode": judge_mode,
                "scores": result.get("scores", {}),
                "aggregation": result.get("aggregation", {}),
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1
            if processed % 10 == 0:
                logger.info(
                    f"  Processed {processed}/{len(generations)} "
                    f"(skipped {skipped}, failed {failed})"
                )
            elif processed % 50 == 0:
                _print_progress_summary(output_file, judge_mode)

    logger.info(
        f"Judge evaluation complete: {processed} processed, "
        f"{skipped} skipped, {failed} failed"
    )

    summary = compute_judge_summary(output_file, judge_mode)
    _print_summary(summary, judge_mode)

    summary_file = output_file.with_suffix(".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to {summary_file}")


def _print_progress_summary(output_file: Path, judge_mode: str):
    """打印中间进度摘要。"""
    summary = compute_judge_summary(output_file, judge_mode)
    dims = GEN_STAGE_DIMS if judge_mode == "gen_stage" else SYSTEM_LEVEL_DIMS
    parts = [f"avg={summary.get('avg_total_score', '?')}"]
    for d in dims[:5]:
        avg = summary.get("per_dim_avg", {}).get(d, "?")
        parts.append(f"{d[:6]}={avg}")
    logger.info(f"  Progress: {', '.join(parts)}")


def compute_judge_summary(results_file: Path, judge_mode: str) -> dict:
    """从已评估的结果文件计算汇总统计。"""
    results = load_generations(results_file)
    if not results:
        return {"count": 0}

    dims = GEN_STAGE_DIMS if judge_mode == "gen_stage" else SYSTEM_LEVEL_DIMS
    agg_fn = aggregate_gen_stage_scores if judge_mode == "gen_stage" else aggregate_system_level_scores

    return aggregate_batch(results, dims, agg_fn)


def _print_summary(summary: dict, judge_mode: str):
    """打印评估汇总。"""
    dims = GEN_STAGE_DIMS if judge_mode == "gen_stage" else SYSTEM_LEVEL_DIMS
    dim_labels = (
        __import__("src.evaluation.judge_prompts", fromlist=["GEN_STAGE_DIM_LABELS"]).GEN_STAGE_DIM_LABELS
        if judge_mode == "gen_stage"
        else __import__("src.evaluation.judge_prompts", fromlist=["SYSTEM_LEVEL_DIM_LABELS"]).SYSTEM_LEVEL_DIM_LABELS
    )

    print()
    print("=" * 60)
    print(f"  JUDGE EVALUATION SUMMARY ({judge_mode})")
    print("=" * 60)
    print(f"  Samples:        {summary.get('count', 0)}")
    print(f"  Avg Score:      {summary.get('avg_total_score', 0):.1f}")
    print(f"  Pass Rate:      {summary.get('pass_rate', 0):.1%}")
    print(f"  Gate Failures:  {summary.get('gate_failure_rate', 0):.1%}")
    print(f"  Penalty Rate:   {summary.get('penalty_rate', 0):.1%}")
    print()
    print(f"  Grade Distribution:")
    for g in ["S", "A", "B", "C", "D", "F"]:
        count = summary.get("grade_distribution", {}).get(g, 0)
        bar = "█" * max(1, count // (max(1, summary.get('count', 1)) // 40 + 1))
        print(f"    {g}: {count:4d}  {bar}")

    print()
    print(f"  Per-Dimension Averages:")
    for dim in dims:
        label = dim_labels.get(dim, dim)
        avg = summary.get("per_dim_avg", {}).get(dim, 0)
        print(f"    {label:<30s} {avg:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-005 Judge Evaluation Runner")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="生成结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="评估结果输出路径（默认：results/exp005/judge_scores/<input_name>.jsonl）",
    )
    parser.add_argument(
        "--mode", type=str, default="gen_stage", choices=["gen_stage", "system_level"],
        help="评估模式（默认：gen_stage）",
    )
    parser.add_argument(
        "--judge-model", type=str, default="deepseek-chat",
        help="Judge LLM 模型",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API Key（默认从环境变量读取）",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="起始索引（断点续跑）",
    )
    parser.add_argument(
        "--max", type=int, default=0,
        help="最大评估条数（0=全部）",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="强制重新评估（忽略已有缓存）",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = RESULTS_DIR / "judge_scores" / f"{input_path.stem}_judged.jsonl"

    run_judge(
        generations_file=input_path,
        output_file=output_path,
        judge_mode=args.mode,
        judge_model=args.judge_model,
        judge_api_key=args.api_key,
        start_idx=args.start,
        max_queries=args.max,
        force=args.force,
    )
