"""
Exp-009 阶段八：SFT 模型评估（vLLM 生成 + Judge 评分）。

prompt 与 exp-005 完全一致（src/generation/prompts.py），
确保 Judge 评分可比：SFT 模型 vs 基线模型。

前置条件：vLLM 服务已启动
    vllm serve models/qwen3-4b-t2ranking-sft/merged --host 0.0.0.0 --port 8000

用法：
    # 仅生成（不跑 Judge）
    python scripts/exp009/eval_sft_model.py --generate-only

    # 一条龙：生成 + Judge（默认）
    python scripts/exp009/eval_sft_model.py

    # 仅跑 Judge（已有生成结果，因 API 错误断点续跑）
    python scripts/exp009/eval_sft_model.py --judge-only
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from src.generation.prompts import PromptManager, get_default_prompts
from src.intent.api_client import LangChainLLMClient
from langchain_openai import ChatOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005"
INPUT_QUERIES = RESULTS_DIR / "input_queries.jsonl"
GENERATIONS_DIR = RESULTS_DIR / "generations"

MODEL_ID = "qwen3-4b-sft"
MODEL_PATH = str(DATA_ROOT / "models" / "qwen3-4b-t2ranking-sft" / "merged")


def load_input_queries(filepath: Path) -> list[dict]:
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    logger.info(f"Loaded {len(queries)} input queries from {filepath.name}")
    return queries


def build_prompt(
    query_text: str,
    passages: list[dict],
    prompt_manager: PromptManager,
) -> tuple[str, str]:
    """与 generate_exp005.py 完全一致的 prompt 构造逻辑。"""
    system_prompt = prompt_manager.get_system_prompt()

    context_parts = []
    for i, p in enumerate(passages):
        pid = p.get("pid", f"doc-{i}")
        rank = p.get("rank", i + 1)
        text = p.get("text", "")
        context_parts.append(f"[{rank}] 来源: {pid}\n{text[:800]}")

    context = "\n\n".join(context_parts)
    user_prompt = prompt_manager.build_user_prompt(query_text, context)

    return system_prompt, user_prompt


def generate_vllm(
    model_name: str,
    query_data: list[dict],
    prompt_manager: PromptManager,
    output_file: Path,
    vllm_url: str,
):
    llm = ChatOpenAI(
        model=model_name,
        api_key="not-needed",
        base_url=vllm_url,
        max_tokens=1024,
        temperature=0.3,
    )
    client = LangChainLLMClient(llm)

    os.makedirs(output_file.parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            system_prompt, user_prompt = build_prompt(query_text, passages, prompt_manager)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            t0 = time.time()
            try:
                answer = client.generate(full_prompt)
            except Exception as e:
                logger.warning(f"  Error on query {qid}: {e}")
                answer = f"[GENERATION ERROR: {e}]"
            elapsed_ms = int((time.time() - t0) * 1000)

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": MODEL_ID,
                "answer": answer.strip(),
                "passages": passages,
                "system_prompt": system_prompt,
                "generation_time_ms": elapsed_ms,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{len(query_data)} ({elapsed_ms}ms)")

    logger.info(f"Saved {len(query_data)} generations to {output_file}")


def run_judge(generations_file: Path):
    from scripts.exp005.run_judge_exp005 import run_judge as judge_main

    output_path = RESULTS_DIR / "judge_scores" / f"{generations_file.stem}_judged.jsonl"

    logger.info(f"Running Judge on {generations_file.name}...")
    judge_main(
        generations_file=generations_file,
        output_file=output_path,
        judge_model="deepseek-chat",
        concurrency=3,
        stagger_delay=0.5,
    )


def main():
    parser = argparse.ArgumentParser(description="Exp-009 SFT Model Evaluation")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate answers, skip Judge")
    parser.add_argument("--judge-only", action="store_true",
                        help="Only run Judge on existing generations (skip generation)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation even if output exists")
    args = parser.parse_args()

    if not INPUT_QUERIES.exists():
        logger.error(f"Input queries not found: {INPUT_QUERIES}")
        logger.error("Run prepare_exp005_data.py first to create input queries.")
        return 1

    query_data = load_input_queries(INPUT_QUERIES)
    prompt_manager = get_default_prompts()

    output_file = GENERATIONS_DIR / f"{MODEL_ID}.jsonl"

    if not args.judge_only:
        if output_file.exists() and not args.force:
            logger.info(f"Skipping generation (cached at {output_file})")
        else:
            logger.info(f"Generating {MODEL_ID} via vLLM at {args.vllm_url}...")
            generate_vllm(
                model_name=MODEL_PATH,
                query_data=query_data,
                prompt_manager=prompt_manager,
                output_file=output_file,
                vllm_url=args.vllm_url,
            )

    if args.generate_only:
        logger.info("generate-only mode: skipping Judge.")
        return 0

    if output_file.exists():
        run_judge(output_file)
    else:
        logger.error(f"Generation output not found: {output_file}")
        return 1

    logger.info("=" * 60)
    logger.info("  Evaluation complete!")
    logger.info(f"  Scores: {RESULTS_DIR / 'judge_scores'}")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
