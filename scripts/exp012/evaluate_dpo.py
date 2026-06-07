"""
Exp-012: DPO 模型评估 —— vLLM HTTP API 生成 + Judge 评分。

仅支持 vLLM HTTP API 推理。需先在另一个终端启动 vLLM 服务。

用法:
  # 0) 先启动 vLLM 服务（另一个终端）：
  #    基线: vllm serve <Qwen3-8B本地路径> --port 8000 --served-model-name qwen3-8b
  #    DPO:  vllm serve <models/exp012-dpo-pilot/merged> --port 8000 --served-model-name qwen3-8b

  # 1) 基线（纯基座）
  python scripts/exp012/evaluate_dpo.py --baseline --vllm-model qwen3-8b

  # 2) DPO（跑完自动打印配对对比报告）
  python scripts/exp012/evaluate_dpo.py --vllm-model qwen3-8b

  # 仅生成 / 仅 Judge / 强制重新生成
  python scripts/exp012/evaluate_dpo.py --generate-only --vllm-model qwen3-8b
  python scripts/exp012/evaluate_dpo.py --judge-only
  python scripts/exp012/evaluate_dpo.py --force --vllm-model qwen3-8b
"""

import os
import sys
import json
import time
import argparse
import logging
import statistics
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
INPUT_QUERIES = DATA_ROOT / "data" / "processed" / "exp012_validation_queries.jsonl"  # 300 条

# ── 配置 ────────────────────────────────────────────────────

TEMPERATURE = 0.3
MAX_TOKENS = 1024
PROMPT_VERSION = "v1-full"


# ── Prompt 构造 ─────────────────────────────────────────────

def clean_answer(text: str) -> str:
    """移除答案中回显的 prompt 模板片段。"""
    text = text.strip()
    if not text:
        return text

    stripped = text.lstrip()
    if stripped.startswith("【参考资料】"):
        for marker in ["\n【回答】", "【回答】"]:
            idx = stripped.find(marker)
            if idx >= 0:
                after = stripped[idx + len(marker):].strip().lstrip("\n").lstrip()
                if after:
                    return after
        for marker in ["\n【核心结论】", "【核心结论】", "\n【核心答案】", "【核心答案】"]:
            idx = stripped.find(marker)
            if idx > 10:
                after = stripped[idx:].strip()
                if after:
                    return after
    return text


def build_user_content(query_text: str, passages: list[dict]) -> str:
    """构造 user 消息内容（[{rank}] 来源: {pid} 格式，与训练时一致）。"""
    context_parts = []
    for p in passages:
        pid = p.get("pid", "unknown")
        rank = p.get("rank", 1)
        text = p.get("text", "")
        context_parts.append(f"[{rank}] 来源: {pid}\n{text[:800]}")

    context = "\n\n".join(context_parts)
    return (
        f"参考资料:\n{context}\n\n"
        f"用户问题: {query_text}\n\n"
        f"请根据以上参考资料回答问题："
    )


# ── 加载数据 ──────────────────────────────────────────────

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


# ── vLLM HTTP API 生成 ───────────────────────────────────

def generate(
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    model_id: str,
    vllm_url: str,
    vllm_model: str,
):
    """通过 vLLM HTTP API 批量生成答案（与 generate_multi_sample.py 一致）。"""
    from openai import OpenAI

    os.makedirs(output_file.parent, exist_ok=True)

    client = OpenAI(api_key="not-needed", base_url=vllm_url)
    system_prompt = prompt_manager.get_system_prompt()

    logger.info(f"Generating {len(query_data)} answers via {vllm_url} (model={vllm_model})...")
    t_start = time.time()

    errors = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            full_prompt = f"{system_prompt}\n\n{build_user_content(query_text, passages)}"
            messages = [{"role": "user", "content": full_prompt}]

            try:
                response = client.chat.completions.create(
                    model=vllm_model,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                answer = response.choices[0].message.content.strip() if response.choices else ""
                answer = clean_answer(answer)
            except Exception as e:
                logger.warning(f"  Error on qid={qid}: {e}")
                answer = ""
                errors += 1

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": model_id,
                "answer": answer,
                "passages": passages,
                "system_prompt": system_prompt,
                "temperature": TEMPERATURE,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                qps = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  [{i+1:3d}/{len(query_data)}] {qps:.1f} q/s, {errors} errors")

    elapsed_s = time.time() - t_start
    logger.info(
        f"Generation done in {elapsed_s:.0f}s "
        f"({len(query_data)/elapsed_s:.1f} q/s), {errors} errors"
    )
    logger.info(f"Saved to {output_file.name}")


# ── Judge ────────────────────────────────────────────────

def run_judge(generations_file: Path, judge_model: str = "deepseek-reasoner"):
    """复用 exp-005 的 Judge pipeline（deepseek-reasoner, 6 维两批）。"""
    from scripts.exp005.run_judge_exp005 import run_judge as judge_main

    os.makedirs(JUDGE_DIR, exist_ok=True)
    output_file = JUDGE_DIR / f"{generations_file.stem}_judged.jsonl"

    logger.info(f"Running Judge on {generations_file.name} (model={judge_model})...")
    judge_main(
        generations_file=generations_file,
        output_file=output_file,
        judge_model=judge_model,
        concurrency=12,
        stagger_delay=0.5,
    )
    logger.info(f"Scores saved to {output_file.name}")


# ── 对比 ────────────────────────────────────────────────

def _print_comparison(dpo_judge_file: Path, baseline_judge_file: Path | None = None):
    """打印 DPO vs 基线的配对对比报告。"""
    if not dpo_judge_file.exists():
        logger.warning(f"DPO judge file not found: {dpo_judge_file}")
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

    dpo_mean = sum(dpo_scores.values()) / max(len(dpo_scores), 1)
    dpo_pass = sum(1 for s in dpo_scores.values() if s >= 60) / max(len(dpo_scores), 1) * 100

    print(f"\n  {'='*60}")
    print(f"  DPO 模型 ({len(dpo_scores)} 条):  均分={dpo_mean:.1f}  Pass%={dpo_pass:.1f}%")

    if baseline_judge_file and baseline_judge_file.exists():
        baseline_scores = {}
        with open(baseline_judge_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r["query_id"]
                score = r.get("aggregation", {}).get("total_score")
                if score is not None:
                    baseline_scores[qid] = score

        common = set(dpo_scores.keys()) & set(baseline_scores.keys())
        if len(common) >= 10:
            common_dpo = sum(dpo_scores[q] for q in common) / len(common)
            common_bl = sum(baseline_scores[q] for q in common) / len(common)
            delta = common_dpo - common_bl
            bl_pass = sum(1 for q in common if baseline_scores[q] >= 60) / len(common) * 100

            deltas = [dpo_scores[q] - baseline_scores[q] for q in common]
            win = sum(1 for d in deltas if d > 0)
            tie = sum(1 for d in deltas if d == 0)
            lose = sum(1 for d in deltas if d < 0)
            delta_mean = statistics.mean(deltas)
            delta_stderr = statistics.stdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else 0

            print(f"\n  Baseline (基座, {len(common)} common queries):")
            print(f"    均分: {common_bl:.1f}  Pass%: {bl_pass:.1f}%")
            print(f"\n  配对对比 (n={len(common)})")
            print(f"    Delta 均值 (DPO - Baseline):  {delta:+.1f}")
            print(f"    Delta 均值 +/- 1.96*SE:       {delta_mean:+.1f} +/- {1.96*delta_stderr:.1f}")
            print(f"    Win / Tie / Lose:              {win} / {tie} / {lose}")
            print(f"    Win rate:                      {win/len(common)*100:.1f}%")
            print(f"    Regression rate:               {lose/len(common)*100:.1f}%")
        else:
            print(f"  [WARN] Too few common queries ({len(common)}) for paired comparison")
    else:
        print(f"  [NOTE] 无同 query 集基线文件，无法做配对对比")

    print(f"  {'='*60}\n")


# ── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-012: DPO Model Evaluation (vLLM HTTP API)")
    parser.add_argument("--generate-only", action="store_true",
                        help="仅生成，跳过 Judge")
    parser.add_argument("--judge-only", action="store_true",
                        help="仅跑 Judge（已有生成结果）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新生成，忽略缓存")
    parser.add_argument("--baseline", action="store_true",
                        help="仅用基座模型推理（输出文件名标记为 baseline）")
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API URL (默认: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--vllm-model", type=str, default="qwen3-8b",
        help="vLLM 服务里注册的模型名 (默认: qwen3-8b)",
    )
    parser.add_argument(
        "--input", type=str, default=str(INPUT_QUERIES),
        help="输入 query JSONL 路径",
    )
    args = parser.parse_args()

    input_queries = Path(args.input)
    if not input_queries.exists():
        logger.error(f"Input queries not found: {input_queries}")
        return 1

    model_id = "qwen3-8b-baseline" if args.baseline else "qwen3-8b-dpo-v1"
    query_data = load_input_queries(input_queries)
    prompt_manager = PromptV2Manager(PROMPT_VERSION)

    input_stem = input_queries.stem
    output_file = GENERATIONS_DIR / f"{input_stem}_{model_id}.jsonl"

    if not args.judge_only:
        if output_file.exists() and not args.force:
            logger.info(f"Skipping generation (cached at {output_file})")
        else:
            logger.info("=" * 60)
            logger.info(f"  Generating: {model_id}")
            logger.info(f"  API:      {args.vllm_url}  model={args.vllm_model}")
            logger.info(f"  Prompt:   {PROMPT_VERSION}, T={TEMPERATURE}")
            logger.info(f"  Queries:  {len(query_data)}")
            logger.info("=" * 60)
            generate(query_data, prompt_manager, output_file, model_id,
                     args.vllm_url, args.vllm_model)

    if args.generate_only:
        logger.info("--generate-only: skipping Judge.")
        return 0

    if output_file.exists():
        run_judge(output_file)
    else:
        logger.error(f"Generation output not found: {output_file}")
        return 1

    logger.info("=" * 60)
    logger.info("  Evaluation complete!")
    logger.info(f"  Scores: {JUDGE_DIR}")
    logger.info("=" * 60)

    # DPO 跑完自动配对对比
    if not args.baseline:
        baseline_judge_file = JUDGE_DIR / f"{input_stem}_qwen3-8b-baseline_judged.jsonl"
        _print_comparison(
            JUDGE_DIR / f"{output_file.stem}_judged.jsonl",
            baseline_judge_file if baseline_judge_file.exists() else None,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
