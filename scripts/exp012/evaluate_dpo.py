"""
Exp-012: DPO 模型评估 —— vLLM (LoRA adapter 模式) 推理 + Judge 评分。

prompt 格式与训练时完全一致（[{rank}] 来源: {pid}），确保干净对比。

前置条件：
  vllm serve Qwen/Qwen3-8B \\
      --enable-lora \\
      --lora-modules dpo-pilot=/root/autodl-tmp/models/exp012-dpo-pilot \\
      --max-lora-rank 16 \\
      --host 0.0.0.0 --port 8000

用法:
  # 一条龙：生成 + Judge
  python scripts/exp012/evaluate_dpo.py

  # 仅生成
  python scripts/exp012/evaluate_dpo.py --generate-only

  # 仅跑 Judge（已有生成结果）
  python scripts/exp012/evaluate_dpo.py --judge-only
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
from src.generation.prompts_v2 import PromptV2Manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp012"
GENERATIONS_DIR = RESULTS_DIR / "generations"
JUDGE_DIR = RESULTS_DIR / "judge_scores"
INPUT_QUERIES = DATA_ROOT / "results" / "exp012" / "eval_queries.jsonl"  # 148 条，排除训练 query

# ── 配置 ────────────────────────────────────────────────────

MODEL_ID = "qwen3-8b-dpo-v1"
# LoRA adapter 模式：vLLM serve Qwen/Qwen3-8B --enable-lora --lora-modules dpo-pilot=/path/to/adapter
# API 调用时的 model 名 = lora module 名
VLLM_MODEL_NAME = "dpo-pilot"

# 推理参数
TEMPERATURE = 0.3
MAX_TOKENS = 1024
PROMPT_VERSION = "v1-full"


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
    prompt_manager: PromptV2Manager,
) -> str:
    """重建完整 prompt，格式与训练时一致：[{rank}] 来源: {pid}"""
    system_prompt = prompt_manager.get_system_prompt()

    context_parts = []
    for p in passages:
        pid = p.get("pid", "unknown")
        rank = p.get("rank", 1)
        text = p.get("text", "")
        context_parts.append(f"[{rank}] 来源: {pid}\n{text[:800]}")

    context = "\n\n".join(context_parts)
    user_prompt = (
        f"参考资料:\n{context}\n\n"
        f"用户问题: {query_text}\n\n"
        f"请根据以上参考资料回答问题："
    )

    return f"{system_prompt}\n\n{user_prompt}"


def generate_vllm(
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    vllm_url: str,
):
    """vLLM 推理，生成完整的答案。"""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=VLLM_MODEL_NAME,
        api_key="not-needed",
        base_url=vllm_url,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )

    os.makedirs(output_file.parent, exist_ok=True)

    total = len(query_data)
    errors = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            full_prompt = build_prompt(query_text, passages, prompt_manager)

            t0 = time.time()
            try:
                response = llm.invoke(full_prompt)
                answer = response.content.strip() if hasattr(response, "content") else str(response).strip()
            except Exception as e:
                logger.warning(f"  Error on qid={qid}: {e}")
                answer = f"[ERROR: {e}]"
                errors += 1

            elapsed_ms = int((time.time() - t0) * 1000)

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": MODEL_ID,
                "answer": answer,
                "passages": passages,
                "system_prompt": prompt_manager.get_system_prompt(),
                "temperature": TEMPERATURE,
                "generation_time_ms": elapsed_ms,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 20 == 0:
                logger.info(f"  [{i+1:3d}/{total}] done, {errors} errors, {elapsed_ms}ms last")

    logger.info(f"Saved {total} generations to {output_file.name} ({errors} errors)")


def run_judge(generations_file: Path):
    """复用 exp-005 的 Judge pipeline（deepseek-chat, 6 维两批）。"""
    from scripts.exp005.run_judge_exp005 import run_judge as judge_main

    os.makedirs(JUDGE_DIR, exist_ok=True)
    output_file = JUDGE_DIR / f"{generations_file.stem}_judged.jsonl"

    logger.info(f"Running Judge on {generations_file.name}...")
    judge_main(
        generations_file=generations_file,
        output_file=output_file,
        judge_model="deepseek-chat",
        concurrency=3,
        stagger_delay=0.5,
    )
    logger.info(f"Scores saved to {output_file.name}")


def main():
    parser = argparse.ArgumentParser(description="Exp-012: DPO Model Evaluation")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL (需先启动 vLLM + LoRA adapter)")
    parser.add_argument("--generate-only", action="store_true",
                        help="仅生成，不跑 Judge")
    parser.add_argument("--judge-only", action="store_true",
                        help="仅跑 Judge（已有生成结果）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成")
    parser.add_argument(
        "--input", type=str, default=str(INPUT_QUERIES),
        help="输入 query JSONL 路径（默认：148 条 eval set，已排除训练 query）",
    )
    args = parser.parse_args()

    input_queries = Path(args.input)
    if not input_queries.exists():
        logger.error(f"Input queries not found: {input_queries}")
        return 1

    query_data = load_input_queries(input_queries)
    prompt_manager = PromptV2Manager(PROMPT_VERSION)

    output_file = GENERATIONS_DIR / f"{MODEL_ID}.jsonl"

    # ── 生成 ──
    if not args.judge_only:
        if output_file.exists() and not args.force:
            logger.info(f"Skipping generation (cached at {output_file})")
        else:
            logger.info("=" * 60)
            logger.info(f"  Generating answers: {MODEL_ID}")
            logger.info(f"  vLLM model: {VLLM_MODEL_NAME}, Prompt: v1-full, T={TEMPERATURE}")
            logger.info(f"  Queries: {len(query_data)}")
            logger.info("=" * 60)
            generate_vllm(query_data, prompt_manager, output_file, args.vllm_url)

    if args.generate_only:
        logger.info("--generate-only: skipping Judge.")
        return 0

    # ── Judge ──
    if output_file.exists():
        run_judge(output_file)
    else:
        logger.error(f"Generation output not found: {output_file}")
        return 1

    logger.info("=" * 60)
    logger.info("  Evaluation complete!")
    logger.info(f"  Scores:    {JUDGE_DIR}")
    logger.info("=" * 60)

    # ── 与基线对比（同一 148 条子集）──
    _print_comparison(input_queries, JUDGE_DIR / f"{output_file.stem}_judged.jsonl")
    return 0


def _print_comparison(eval_queries_file: Path, dpo_judge_file: Path):
    """在 DPO 结果和基线间做同 query 子集对比。"""
    baseline_files = [
        # exp-010 Phase 1: 8B-nonthink + v1-full
        DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-8b-nothink-v1-full_judged.jsonl",
        # exp-010 v0 基线 (8B + 旧 prompt)
        DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-8b-nothink-v0_judged.jsonl",
    ]

    # 加载 DPO 结果
    if not dpo_judge_file.exists():
        return
    dpo_scores = {}
    with open(dpo_judge_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r["query_id"]
            score = r.get("aggregation", {}).get("total_score")
            if score is not None:
                dpo_scores[qid] = score

    # 加载评估 query ID 列表
    eval_qids = set()
    with open(eval_queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            eval_qids.add(json.loads(line)["query_id"])

    dpo_mean = sum(dpo_scores.values()) / max(len(dpo_scores), 1)
    dpo_pass = sum(1 for s in dpo_scores.values() if s >= 60) / max(len(dpo_scores), 1) * 100

    print(f"\n  {'='*50}")
    print(f"  DPO 模型 ({len(dpo_scores)} 条):")
    print(f"    均分: {dpo_mean:.1f}")
    print(f"    Pass%: {dpo_pass:.1f}%")

    for bf in baseline_files:
        if not bf.exists():
            continue
        baseline_scores = {}
        with open(bf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r["query_id"]
                if qid not in eval_qids:
                    continue
                score = r.get("aggregation", {}).get("total_score")
                if score is not None:
                    baseline_scores[qid] = score

        if not baseline_scores:
            continue

        bl_mean = sum(baseline_scores.values()) / len(baseline_scores)
        bl_pass = sum(1 for s in baseline_scores.values() if s >= 60) / len(baseline_scores) * 100

        # 只对比两条结果都有的 query
        common = set(dpo_scores.keys()) & set(baseline_scores.keys())
        if len(common) < 10:
            continue

        common_dpo = sum(dpo_scores[q] for q in common) / len(common)
        common_bl = sum(baseline_scores[q] for q in common) / len(common)
        delta = common_dpo - common_bl

        label = bf.stem
        print(f"\n  基线 {label}:")
        print(f"    ({len(common)} common queries)")
        print(f"    基线均分: {common_bl:.1f}  →  DPO: {common_dpo:.1f}  (Δ = {delta:+.1f})")
        print(f"    基线 Pass%: {bl_pass:.1f}%  →  DPO Pass%: {dpo_pass:.1f}%")

    print(f"  {'='*50}\n")


if __name__ == "__main__":
    sys.exit(main())
