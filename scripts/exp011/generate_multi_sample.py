"""
Exp-011 Phase 1: 多样本生成 —— RL/DPO 可行性验证。

对每条 query 在温度 0.8 下生成 N 个样本，用于回答：
  "更好的答案是否存在于模型的输出分布中？"

实现逻辑：
  - 从 exp-005 198 条输入数据中抽样 M 条
  - 每条 query 调用 API/vLLM N 次（温度=0.8），每次独立采样
  - 输出格式与 generate_exp005.py 完全兼容，每行一个 sample

输出兼容现有 Judge pipeline（run_judge_exp005.py），每行是一条独立的
生成记录，额外包含 sample_id 字段区分重复采样。

用到的模型配置沿袭 generate_exp005.py 的 MODEL_CONFIGS。

用法:
    # vLLM 模式（本地模型）
    python scripts/exp011/generate_multi_sample.py \
        --model qwen3-4b-nothink \
        --vllm-url http://localhost:8000/v1 \
        --num-samples 5 \
        --temperature 0.8 \
        --seed 42

    # 两个模型依次跑
    python scripts/exp011/generate_multi_sample.py \
        --model qwen3-4b-nothink \
        --vllm-url http://localhost:8000/v1 \
        --num-samples 5 --temperature 0.8 --seed 42

    python scripts/exp011/generate_multi_sample.py \
        --model qwen3-8b-nothink \
        --vllm-url http://localhost:8001/v1 \
        --num-samples 5 --temperature 0.8 --seed 42
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from src.generation.prompts_v2 import PromptV2Manager, get_prompt_manager, list_versions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ===========================================================================
# 配置
# ===========================================================================

RESULTS_DIR = DATA_ROOT / "results" / "exp011"
GENERATIONS_DIR = RESULTS_DIR / "generation"
INPUT_QUERIES = DATA_ROOT / "results" / "exp005" / "input_queries.jsonl"

# 模型配置摘要（从 generate_exp005.py 移植）
MODEL_CONFIGS = {
    "qwen3-4b-nothink": {
        "hf_id": "Qwen/Qwen3-4B",
        "backend": "vllm",
        "params": "4B",
        "max_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen3-8b-nothink": {
        "hf_id": "Qwen/Qwen3-8B",
        "backend": "vllm",
        "params": "8B",
        "max_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
}


# ===========================================================================
# 工具函数
# ===========================================================================

def load_input_queries(filepath: Path, sample_size: Optional[int] = None, seed: int = 42) -> list[dict]:
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    if sample_size and sample_size < len(queries):
        rng = random.Random(seed)
        rng.shuffle(queries)
        queries = queries[:sample_size]
        logger.info(f"Sampled {sample_size} queries from {len(queries) + sample_size} total (seed={seed})")
    else:
        logger.info(f"Loaded {len(queries)} input queries from {filepath.name}")

    return queries


def build_prompt(
    query_text: str,
    passages: list[dict],
    prompt_manager: PromptV2Manager,
) -> tuple[str, str]:
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


# ===========================================================================
# 生成
# ===========================================================================

def generate_multi_vllm(
    model_config: dict,
    model_id: str,
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    vllm_url: str,
    temperature: float,
    num_samples: int,
    seed: int,
):
    from langchain_openai import ChatOpenAI

    llm_kwargs = dict(
        model=model_config["hf_id"],
        api_key="not-needed",
        base_url=vllm_url,
        max_tokens=model_config.get("max_tokens", 1024),
        temperature=temperature,
        max_retries=6,
        timeout=180.0,
    )
    extra_body = model_config.get("extra_body")
    if extra_body:
        llm_kwargs["model_kwargs"] = {"extra_body": extra_body}

    llm = ChatOpenAI(**llm_kwargs)

    os.makedirs(output_file.parent, exist_ok=True)

    total_ok = 0
    total_err = 0
    overall_start = time.time()

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            if (i + 1) % 10 == 0:
                elapsed = time.time() - overall_start
                queries_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  [{i+1:3d}/{len(query_data)}] queries done, "
                    f"{total_ok} ok / {total_err} err, "
                    f"{queries_per_sec:.1f} q/s"
                )

            system_prompt, user_prompt = build_prompt(query_text, passages, prompt_manager)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            for sample_idx in range(num_samples):
                try:
                    # 每次独立调用，T=0.8 时 vLLM 自然产生不同输出
                    response = llm.invoke(full_prompt)
                    answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
                except Exception as e:
                    logger.warning(f"  Error on qid={qid} sample={sample_idx}: {e}")
                    total_err += 1
                    time.sleep(1.0)  # 出错后稍等再发下一个请求
                    continue

                # 请求间短暂间隔，避免 vLLM 请求堆积
                if sample_idx < num_samples - 1:
                    time.sleep(0.5)

                # 使用复合 ID 避免 Judge 缓存去重（同一条 query 的多个 sample 需要独立评分）
                composite_qid = f"{qid}_s{sample_idx}"
                result = {
                    "query_id": composite_qid,
                    "original_query_id": qid,
                    "query_text": query_text,
                    "model_id": model_id,
                    "sample_id": sample_idx,
                    "answer": answer,
                    "passages": passages,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                }
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                total_ok += 1

    total_elapsed = time.time() - overall_start
    logger.info(
        f"Multi-sample generation done: {total_ok} answers ({len(query_data)} queries x "
        f"~{num_samples} samples), {total_err} errors, "
        f"{total_elapsed:.1f}s elapsed"
    )
    logger.info(f"Saved to {output_file}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp-011: Multi-sample generation for RL feasibility check"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model ID to use (must be in MODEL_CONFIGS)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API URL",
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of samples per query (default: 5)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature (default: 0.8)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=50,
        help="Number of queries to sample from dev set (default: 50; use 0 for all 198)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling and generation (default: 42)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-generation even if output exists",
    )
    parser.add_argument(
        "--prompt-version", type=str, default="v0",
        choices=list_versions(),
        help="Prompt version to use (from prompts_v2.py). Default: v0 (exp-005 baseline)",
    )
    args = parser.parse_args()

    # -- 路径与采样 --
    sample_size = args.sample_size if args.sample_size > 0 else None
    if not INPUT_QUERIES.exists():
        logger.error(f"Input queries not found: {INPUT_QUERIES}")
        logger.error("Run prepare_exp005_data.py first.")
        return 1

    query_data = load_input_queries(INPUT_QUERIES, sample_size, args.seed)
    model_config = MODEL_CONFIGS[args.model]

    output_file = (
        GENERATIONS_DIR /
        f"{args.model}_{args.prompt_version}_t{args.temperature}_n{args.num_samples}_s{args.seed}.jsonl"
    )

    if output_file.exists() and not args.force:
        logger.info(f"Skipping generation (cached at {output_file}). Use --force to regenerate.")
        return 0

    # -- Prompt --
    prompt_mgr = get_prompt_manager(args.prompt_version)
    logger.info(f"Using prompt version: {args.prompt_version}")

    # -- 生成 --
    logger.info(
        f"Multi-sample generation: model={args.model}, "
        f"t={args.temperature}, n={args.num_samples}, "
        f"queries={len(query_data)}, seed={args.seed}, "
        f"prompt={args.prompt_version}"
    )

    generate_multi_vllm(
        model_config=model_config,
        model_id=args.model,
        query_data=query_data,
        prompt_manager=prompt_mgr,
        output_file=output_file,
        vllm_url=args.vllm_url,
        temperature=args.temperature,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
