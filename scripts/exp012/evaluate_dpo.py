"""
Exp-012: DPO 模型评估 —— QLoRA 基座 + LoRA adapter 直接推理 + Judge 评分。

不需要 vLLM（绕开 Qwen3 tokenizer 兼容性问题）。
直接加载 4-bit 基座 + LoRA adapter，用 model.generate() 逐条推理。

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

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, MODEL_CACHE_DIR
from src.generation.prompts_v2 import PromptV2Manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp012"
GENERATIONS_DIR = RESULTS_DIR / "generations"
JUDGE_DIR = RESULTS_DIR / "judge_scores"
INPUT_QUERIES = DATA_ROOT / "results" / "exp012" / "eval_queries.jsonl"  # 148 条

# ── 配置 ────────────────────────────────────────────────────

MODEL_ID = "qwen3-8b-dpo-v1"
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = DATA_ROOT / "models" / "exp012-dpo-pilot"

TEMPERATURE = 0.3
MAX_TOKENS = 1024
PROMPT_VERSION = "v1-full"


# ── 模型路径解析（与 train_dpo.py 一致）─────────────────────

def resolve_base_model(base_model_id: str, cache_dir: Path) -> str:
    """找到本地已缓存的基座模型路径。"""
    hf_cache_name = f"models--{base_model_id.replace('/', '--')}"
    hf_cache_path = cache_dir / hf_cache_name
    if hf_cache_path.exists():
        snaps = sorted((hf_cache_path / "snapshots").iterdir())
        if snaps:
            return str(snaps[-1])

    local_name = base_model_id.split("/")[-1]
    local_path = cache_dir / local_name
    if (local_path / "config.json").exists():
        return str(local_path)

    return base_model_id


# ── Prompt 构造（与训练时一致）─────────────────────────────

def clean_answer(text: str) -> str:
    """移除答案中回显的 prompt 模板片段。

    基座模型有时会把 prompt 结构吐回到答案开头：
      【参考资料】
      [1] 来源: 123  ...
      【用户问题】
      原始问题
      【回答】
      <实际答案内容>

    清洗规则：【参考资料】 开头 → 找后面第一个 【回答】/【核心结论】/【核心答案】切断。
    """
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


def build_prompt(
    query_text: str,
    passages: list[dict],
    prompt_manager: PromptV2Manager,
) -> str:
    """[{rank}] 来源: {pid} 格式，与训练时完全一致。"""
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


# ── Transformers 直接推理 ─────────────────────────────────

def generate_transformers(
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    base_model_path: str,
    adapter_path: Path,
):
    """加载 QLoRA 基座 + LoRA adapter，逐条推理生成答案。"""
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from peft import PeftModel

    os.makedirs(output_file.parent, exist_ok=True)

    # --- 加载 tokenizer ---
    logger.info(f"Loading tokenizer from {base_model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, cache_dir=str(MODEL_CACHE_DIR),
    )

    # --- 加载 4-bit 基座 ---
    logger.info("Loading 4-bit base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    # --- 加载 LoRA adapter ---
    logger.info(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    # --- 逐条推理 ---
    total = len(query_data)
    errors = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            full_prompt = build_prompt(query_text, passages, prompt_manager)

            messages = [{"role": "user", "content": full_prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            t0 = time.time()
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_TOKENS,
                        temperature=TEMPERATURE,
                        do_sample=True if TEMPERATURE > 0 else False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # 只取生成部分
                generated = outputs[0][inputs["input_ids"].shape[1]:]
                answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
                answer = clean_answer(answer)  # 去掉可能的 prompt 回显
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

            if (i + 1) % 10 == 0:
                eta = (elapsed_ms * (total - i - 1)) // 1000
                logger.info(f"  [{i+1:3d}/{total}] {elapsed_ms}ms, ETA {eta}s, {errors} errors")

    logger.info(f"Saved {total} generations to {output_file.name} ({errors} errors)")


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
        concurrency=3,
        stagger_delay=0.5,
    )
    logger.info(f"Scores saved to {output_file.name}")


# ── 对比 ────────────────────────────────────────────────

def _print_comparison(eval_queries_file: Path, dpo_judge_file: Path):
    """在 DPO 结果和基线间做同 query 子集对比。"""
    baseline_files = [
        DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-8b-nothink-v1-full_judged.jsonl",
        DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-8b-nothink-v0_judged.jsonl",
    ]

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

        common = set(dpo_scores.keys()) & set(baseline_scores.keys())
        if len(common) < 10:
            continue

        common_dpo = sum(dpo_scores[q] for q in common) / len(common)
        common_bl = sum(baseline_scores[q] for q in common) / len(common)
        delta = common_dpo - common_bl

        label = bf.stem
        print(f"\n  基线 {label}:")
        print(f"    ({len(common)} common queries)")
        print(f"    基线均分: {common_bl:.1f}  ->  DPO: {common_dpo:.1f}  (Delta = {delta:+.1f})")

    print(f"  {'='*50}\n")


# ── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-012: DPO Model Evaluation (transformers direct)")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--input", type=str, default=str(INPUT_QUERIES),
        help="输入 query JSONL 路径（默认：148 条 eval set）",
    )
    parser.add_argument(
        "--base-model", type=str, default=BASE_MODEL,
        help="基座模型 HF ID 或本地路径",
    )
    parser.add_argument(
        "--adapter", type=str, default=str(ADAPTER_PATH),
        help="LoRA adapter 目录",
    )
    args = parser.parse_args()

    input_queries = Path(args.input)
    if not input_queries.exists():
        logger.error(f"Input queries not found: {input_queries}")
        return 1

    adapter_path = Path(args.adapter)
    if not adapter_path.exists() and not args.judge_only:
        logger.error(f"Adapter not found: {adapter_path}")
        return 1

    base_model_path = resolve_base_model(args.base_model, MODEL_CACHE_DIR)
    logger.info(f"Base model: {base_model_path}")

    query_data = load_input_queries(input_queries)
    prompt_manager = PromptV2Manager(PROMPT_VERSION)

    output_file = GENERATIONS_DIR / f"{MODEL_ID}.jsonl"

    if not args.judge_only:
        if output_file.exists() and not args.force:
            logger.info(f"Skipping generation (cached at {output_file})")
        else:
            logger.info("=" * 60)
            logger.info(f"  Generating: {MODEL_ID}")
            logger.info(f"  Base: {base_model_path}")
            logger.info(f"  Adapter: {adapter_path}")
            logger.info(f"  Prompt: v1-full, T={TEMPERATURE}")
            logger.info(f"  Queries: {len(query_data)}")
            logger.info("=" * 60)
            generate_transformers(query_data, prompt_manager, output_file,
                                  base_model_path, adapter_path)

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

    _print_comparison(input_queries, JUDGE_DIR / f"{output_file.stem}_judged.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
