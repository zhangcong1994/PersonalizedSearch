"""
Exp-005 Judge 执行器（两批滚动评估）。

每条生成答案分两批评估：
  Batch1（门槛组）: 准确性 + 安全性 + 相关性
  Batch2（质量组）: 整合质量 + 引文质量 + 用户体验

每条答案调用 Judge LLM 两次（每批一次），合并为 6 维核心评分。
支持：结果缓存、限流与重试、进度显示。
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.intent.api_client import APIClientFactory
from src.evaluation.judge_prompts import (
    get_batch_system_prompt,
    build_gen_stage_judge_input,
    parse_judge_response,
    ALL_CORE_DIMS,
)
from src.evaluation.aggregation import (
    aggregate_core6_scores,
    aggregate_batch,
    CORE6_DIM_LABELS,
)
from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2


def load_generations(filepath: Path) -> list[dict]:
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} generations from {filepath.name}")
    return results


def judge_single_batch(
    client,
    query: str,
    passages: list[dict],
    answer: str,
    batch: int,
) -> dict | None:
    """
    执行单批次 Judge 评分（Batch 1 或 Batch 2）。

    Returns:
        {"scores": {...}, "raw_response": "..."} 或 None
    """
    system_prompt = get_batch_system_prompt(batch)
    user_message = build_gen_stage_judge_input(query, passages, answer, batch=batch)
    full_prompt = f"{system_prompt}\n\n{user_message}"

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = client.generate(full_prompt)
            parsed = parse_judge_response(raw_response)

            if parsed is None:
                logger.warning(
                    f"Batch{batch} parse failed (attempt {attempt + 1}). "
                    f"Preview: {raw_response[:200]}..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
                continue

            return {"scores": parsed, "raw_response": raw_response}

        except Exception as e:
            logger.warning(f"Batch{batch} API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** (attempt + 1))

    logger.error(f"Batch{batch} failed after {MAX_RETRIES} attempts")
    return None


def judge_both_batches(
    client,
    query: str,
    passages: list[dict],
    answer: str,
) -> dict:
    """
    执行两批 Judge 评分并合并结果。

    Returns:
        {
            "scores": {all 6 dims merged},
            "aggregation": aggregate_core6_scores result,
            "batch1_raw": "...",
            "batch2_raw": "...",
            "error": bool,
        }
    """
    result1 = judge_single_batch(client, query, passages, answer, batch=1)
    result2 = judge_single_batch(client, query, passages, answer, batch=2)

    if result1 is None or result2 is None:
        return {
            "scores": {},
            "aggregation": {"total_score": -1},
            "error": True,
        }

    all_scores = {}
    all_scores.update(result1.get("scores", {}))
    all_scores.update(result2.get("scores", {}))

    aggregation = aggregate_core6_scores(all_scores)

    return {
        "scores": all_scores,
        "aggregation": aggregation,
        "batch1_raw": result1.get("raw_response", ""),
        "batch2_raw": result2.get("raw_response", ""),
        "error": False,
    }


def run_judge(
    generations_file: Path,
    output_file: Path,
    judge_model: str = "deepseek-chat",
    judge_api_key: Optional[str] = None,
    start_idx: int = 0,
    max_queries: int = 0,
    force: bool = False,
):
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
        f"Starting 2-batch judge: {len(generations)} queries, "
        f"judge={judge_model}, 2 calls/query → ~{len(generations) * 2} calls"
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

            judge_result = judge_both_batches(
                client, query_text, passages, answer,
            )

            if judge_result["error"]:
                failed += 1

            output = {
                "query_id": qid,
                "query_text": query_text,
                "model": model_name,
                "judge_model": judge_model,
                "scores": judge_result.get("scores", {}),
                "aggregation": judge_result.get("aggregation", {}),
            }

            out_f.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_f.flush()

            processed += 1
            if processed % 5 == 0:
                logger.info(
                    f"  Processed {processed}/{len(generations)} "
                    f"(skipped {skipped}, failed {failed})"
                )
            if processed % 25 == 0:
                _print_progress_summary(output_file)

    logger.info(
        f"Judge complete: {processed} processed, "
        f"{skipped} skipped, {failed} failed"
    )

    summary = compute_judge_summary(output_file)
    _print_summary(summary)

    summary_file = output_file.with_suffix(".summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary saved to {summary_file}")


def _print_progress_summary(output_file: Path):
    summary = compute_judge_summary(output_file)
    parts = [f"avg={summary.get('avg_total_score', '?')}"]
    for d in ALL_CORE_DIMS:
        avg = summary.get("per_dim_avg", {}).get(d, "?")
        parts.append(f"{d[:6]}={avg}")
    logger.info(f"  Progress: {', '.join(parts)}")


def compute_judge_summary(results_file: Path) -> dict:
    results = load_generations(results_file)
    if not results:
        return {"count": 0}
    return aggregate_batch(results, ALL_CORE_DIMS, aggregate_core6_scores)


def _print_summary(summary: dict):
    print()
    print("=" * 60)
    print("  JUDGE EVALUATION SUMMARY (6-dim core, 2-batch)")
    print("=" * 60)
    print(f"  Samples:        {summary.get('count', 0)}")
    print(f"  Avg Score:      {summary.get('avg_total_score', 0):.1f}")
    print(f"  Pass Rate:      {summary.get('pass_rate', 0):.1%}")
    print(f"  Gate Failures:  {summary.get('gate_failure_rate', 0):.1%}")
    print(f"  Penalty Rate:   {summary.get('penalty_rate', 0):.1%}")
    print()
    print("  Grade Distribution:")
    for g in ["S", "A", "B", "C", "D", "F"]:
        count = summary.get("grade_distribution", {}).get(g, 0)
        bar = "█" * max(1, count // (max(1, summary.get("count", 1)) // 40 + 1))
        print(f"    {g}: {count:4d}  {bar}")

    print()
    print("  Per-Dimension Averages:")
    for dim in ALL_CORE_DIMS:
        label = CORE6_DIM_LABELS.get(dim, dim)
        avg = summary.get("per_dim_avg", {}).get(dim, 0)
        print(f"    {label:<25s} {avg:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-005 2-Batch Judge Runner")
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="生成结果 JSONL 文件路径",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="评估结果输出路径（默认：results/exp005/judge_scores/<input_name>_judged.jsonl）",
    )
    parser.add_argument(
        "--judge-model", type=str, default="deepseek-chat",
        help="Judge LLM 模型（默认 deepseek-chat）",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API Key（默认从环境变量 DEEPSEEK_API_KEY 读取）",
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
        judge_model=args.judge_model,
        judge_api_key=args.api_key,
        start_idx=args.start,
        max_queries=args.max,
        force=args.force,
    )
