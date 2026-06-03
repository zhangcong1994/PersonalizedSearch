"""
Exp-011 Phase 1.5: Judge 评分 —— RL/DPO 可行性验证。

复用 exp-005 的 Judge pipeline（双批 6 维评分），对多样本生成结果评分。

用法:
    # 对单个多样本文件评分
    python scripts/exp011/run_judge.py \
        --input results/exp011/generation/qwen3-4b-nothink_t0.8_n5_s42.jsonl \
        --judge-model deepseek-chat

    # 批量评分（自动发现 exp011/generation/ 下的所有多样本文件）
    python scripts/exp011/run_judge.py --all
"""

import os
import sys
import json
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

RESULTS_DIR = DATA_ROOT / "results" / "exp011"
GENERATION_DIR = RESULTS_DIR / "generation"
JUDGE_DIR = RESULTS_DIR / "judge_scores"


def run_judge_on_file(
    generations_file: Path,
    output_file: Path,
    judge_model: str = "deepseek-chat",
    concurrency: int = 3,
    force: bool = False,
):
    """对单个多样本生成文件运行 Judge 评分。"""
    from scripts.exp005.run_judge_exp005 import run_judge

    logger.info(f"Judge: {generations_file.name} -> {output_file.name}")
    logger.info(f"  Judge model={judge_model}, concurrency={concurrency}")

    run_judge(
        generations_file=generations_file,
        output_file=output_file,
        judge_model=judge_model,
        concurrency=concurrency,
        stagger_delay=0.5,
        force=force,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Exp-011: Judge scoring for multi-sample generations"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to multi-sample generation JSONL file",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for judge scores (default: derived from input)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="deepseek-chat",
        choices=["deepseek-chat", "deepseek-reasoner", "glm-4.7", "glm-4-flash"],
        help="Judge model to use (default: deepseek-chat)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run judge on all multi-sample generation files in exp011/generation/",
    )
    parser.add_argument(
        "--concurrency", type=int, default=3,
        help="Max concurrent Judge API calls (default: 3)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-judging even if cached results exist",
    )
    args = parser.parse_args()

    # -- 自动发现模式 --
    if args.all:
        if not GENERATION_DIR.exists():
            logger.error(f"Generation directory not found: {GENERATION_DIR}")
            return 1

        gen_files = sorted(GENERATION_DIR.glob("*_t0.8_*.jsonl"))
        if not gen_files:
            logger.error(f"No multi-sample generation files found in {GENERATION_DIR}")
            return 1

        logger.info(f"Found {len(gen_files)} multi-sample generation files to judge:")
        for gf in gen_files:
            logger.info(f"  - {gf.name}")

        os.makedirs(JUDGE_DIR, exist_ok=True)
        for gen_file in gen_files:
            output_file = JUDGE_DIR / f"{gen_file.stem}_judged.jsonl"
            if output_file.exists() and not args.force:
                logger.info(f"  SKIP (cached): {output_file.name}")
                continue
            run_judge_on_file(
                gen_file, output_file, args.judge_model, args.concurrency, args.force,
            )
        logger.info("All judges done.")
        return 0

    # -- 单文件模式 --
    if not args.input:
        logger.error("Specify --input <file> or --all")
        return 1

    gen_file = Path(args.input)
    if not gen_file.exists():
        logger.error(f"Generation file not found: {gen_file}")
        return 1

    if args.output:
        output_file = Path(args.output)
    else:
        os.makedirs(JUDGE_DIR, exist_ok=True)
        output_file = JUDGE_DIR / f"{gen_file.stem}_judged.jsonl"

    run_judge_on_file(
        gen_file, output_file, args.judge_model, args.concurrency, args.force,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
