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
import statistics
from pathlib import Path
from typing import Optional

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
INPUT_QUERIES = DATA_ROOT / "data" / "processed" / "exp012_validation_queries.jsonl"  # 300 条

# ── 配置 ────────────────────────────────────────────────────

MODEL_ID = "qwen3-8b-dpo-v1"
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = DATA_ROOT / "models" / "exp012-dpo-pilot"
DPO_MERGED_PATH = DATA_ROOT / "models" / "exp012-dpo-pilot" / "merged"  # vLLM 用合并模型

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
    """[{rank}] 来源: {pid} 格式，与训练时完全一致。
    transformers 路径用此函数（system prompt 拼进 user message）。
    """
    system_prompt = prompt_manager.get_system_prompt()
    user_prompt = _build_user_content(query_text, passages)
    return f"{system_prompt}\n\n{user_prompt}"


def _build_user_content(query_text: str, passages: list[dict]) -> str:
    """构造纯 user 消息内容（不含 system prompt）。
    vLLM 路径使用此函数，system prompt 放在独立 role=system 消息中。
    """
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


# ── Transformers 直接推理 ─────────────────────────────────

def generate_transformers(
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    base_model_path: str,
    adapter_path: Optional[Path] = None,
):
    """加载 4-bit 基座（可选 + LoRA adapter），逐条推理生成答案。

    adapter_path=None 时仅用基座模型（baseline 模式）。
    """
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

    # --- 可选：加载 LoRA adapter ---
    if adapter_path is not None:
        logger.info(f"Loading LoRA adapter from {adapter_path} ...")
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
    else:
        logger.info("No adapter — using base model (baseline mode)")
        model = base_model
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


# ── vLLM 批量推理 ────────────────────────────────────────

def generate_vllm(
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    model_path: str,
    model_id: str,
):
    """用 vLLM offline batch inference 批量生成答案（比 transformers 逐条快 5-10x）。"""
    from vllm import LLM, SamplingParams

    os.makedirs(output_file.parent, exist_ok=True)

    # --- 构造所有 prompt（system + user 两段式）---
    logger.info(f"Building {len(query_data)} prompts...")
    system_prompt = prompt_manager.get_system_prompt()
    all_messages = []
    for item in query_data:
        query_text = item.get("query_text", item.get("query", ""))
        passages = item.get("passages", [])
        user_prompt = _build_user_content(query_text, passages)
        all_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

    # --- 加载 vLLM 模型 ---
    logger.info(f"Loading vLLM model from {model_path} ...")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.80,
        max_model_len=8192,
        max_num_seqs=16,
        enforce_eager=True,
    )

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # --- 批量推理 ---
    logger.info(f"Generating {len(query_data)} answers via vLLM...")
    t_start = time.time()

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": False},
        use_tqdm=True,
    )

    elapsed_s = time.time() - t_start
    logger.info(f"vLLM generation done in {elapsed_s:.0f}s ({len(query_data)/elapsed_s:.1f} q/s)")

    # --- 写入输出 ---
    errors = 0
    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, (item, output) in enumerate(zip(query_data, outputs)):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            if output.outputs:
                answer = output.outputs[0].text.strip()
                answer = clean_answer(answer)
            else:
                logger.warning(f"  Empty output for qid={qid}")
                answer = ""
                errors += 1

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": model_id,
                "answer": answer,
                "passages": passages,
                "system_prompt": prompt_manager.get_system_prompt(),
                "temperature": TEMPERATURE,
                "generation_time_ms": 0,  # batch 模式无法分条计时
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(query_data)} generations to {output_file.name} ({errors} errors)")


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
    """打印 DPO vs 基线的配对对比报告。

    优先使用同 query 集的 baseline judge 文件（paired comparison）；
    若未提供，回退到 exp-010 历史基线（不同 query 集，仅供参考）。
    """
    if not dpo_judge_file.exists():
        logger.warning(f"DPO judge file not found: {dpo_judge_file}")
        return

    # 加载 DPO scores
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

    # ── 配对对比（同 query 集）──
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

            # per-query delta 统计
            deltas = [dpo_scores[q] - baseline_scores[q] for q in common]
            win = sum(1 for d in deltas if d > 0)
            tie = sum(1 for d in deltas if d == 0)
            lose = sum(1 for d in deltas if d < 0)
            delta_mean = statistics.mean(deltas)
            delta_stderr = statistics.stdev(deltas) / (len(deltas) ** 0.5) if len(deltas) > 1 else 0

            print(f"\n  Baseline (基座, {len(common)} common queries):")
            print(f"    均分: {common_bl:.1f}  Pass%: {bl_pass:.1f}%")
            print(f"\n  ── 配对对比 (n={len(common)}) ──")
            print(f"    Δ 均值 (DPO − Baseline):  {delta:+.1f}")
            print(f"    Δ 均值 ± 1.96*SE:         {delta_mean:+.1f} ± {1.96*delta_stderr:.1f}")
            print(f"    Win / Tie / Lose:          {win} / {tie} / {lose}")
            print(f"    Win rate:                  {win/len(common)*100:.1f}%")
            print(f"    Regression rate:           {lose/len(common)*100:.1f}%")
        else:
            print(f"  [WARN] Too few common queries ({len(common)}) for paired comparison")

    else:
        print(f"  [NOTE] 无同 query 集基线文件，无法做配对对比")

    # ── 回退：exp-010 历史基线（仅参考）──
    history_files = [
        DATA_ROOT / "results" / "exp010" / "judge_scores" / "qwen3-8b-nothink-v1-full_judged.jsonl",
    ]
    for hf in history_files:
        if not hf.exists():
            continue
        hist_scores = {}
        with open(hf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r["query_id"]
                score = r.get("aggregation", {}).get("total_score")
                if score is not None:
                    hist_scores[qid] = score
        common = set(dpo_scores.keys()) & set(hist_scores.keys())
        if len(common) < 10:
            continue
        common_dpo = sum(dpo_scores[q] for q in common) / len(common)
        common_hist = sum(hist_scores[q] for q in common) / len(common)
        delta = common_dpo - common_hist
        print(f"\n  历史基线 {hf.stem}:")
        print(f"    ({len(common)} common queries)")
        print(f"    历史均分: {common_hist:.1f}  →  DPO: {common_dpo:.1f}  (Δ={delta:+.1f})")

    print(f"  {'='*60}\n")


# ── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-012: DPO Model Evaluation")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--judge-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--baseline", action="store_true",
                        help="仅用基座模型推理（不加载 LoRA adapter）")
    parser.add_argument("--use-vllm", action="store_true",
                        help="使用 vLLM 批量推理（比 transformers 逐条快 5-10x）")
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
        help="LoRA adapter 目录（--baseline 或 --use-vllm 时忽略）",
    )
    parser.add_argument(
        "--dpo-merged", type=str, default=str(DPO_MERGED_PATH),
        help="DPO 合并模型路径（--use-vllm 非 baseline 模式使用）",
    )
    args = parser.parse_args()

    input_queries = Path(args.input)
    if not input_queries.exists():
        logger.error(f"Input queries not found: {input_queries}")
        return 1

    # baseline 模式：不用 adapter
    if args.baseline:
        model_id = "qwen3-8b-baseline"
        adapter_path = None
    else:
        model_id = MODEL_ID
        adapter_path = Path(args.adapter)
        if not adapter_path.exists() and not args.judge_only and not args.use_vllm:
            logger.error(f"Adapter not found: {adapter_path}")
            return 1

    base_model_path = resolve_base_model(args.base_model, MODEL_CACHE_DIR)
    logger.info(f"Base model: {base_model_path}")

    query_data = load_input_queries(input_queries)
    prompt_manager = PromptV2Manager(PROMPT_VERSION)

    # 输出文件名包含输入文件名前缀 + model_id
    input_stem = input_queries.stem
    output_file = GENERATIONS_DIR / f"{input_stem}_{model_id}.jsonl"

    if not args.judge_only:
        if output_file.exists() and not args.force:
            logger.info(f"Skipping generation (cached at {output_file})")
        elif args.use_vllm:
            # ── vLLM 路径 ──
            if args.baseline:
                vllm_model_path = base_model_path  # 基座 HF 路径
            else:
                vllm_model_path = args.dpo_merged
                if not Path(vllm_model_path).exists():
                    logger.error(
                        f"DPO merged model not found: {vllm_model_path}\n"
                        f"  Run merge_adapter.py first, or train_dpo.py without --no-merge"
                    )
                    return 1
            logger.info("=" * 60)
            logger.info(f"  Generating via vLLM: {model_id}")
            logger.info(f"  Model:  {vllm_model_path}")
            logger.info(f"  Prompt: v1-full, T={TEMPERATURE}")
            logger.info(f"  Queries: {len(query_data)}")
            logger.info("=" * 60)
            generate_vllm(query_data, prompt_manager, output_file, vllm_model_path, model_id)
        else:
            # ── Transformers 路径 ──
            logger.info("=" * 60)
            logger.info(f"  Generating via Transformers: {model_id}")
            logger.info(f"  Base: {base_model_path}")
            logger.info(f"  Adapter: {adapter_path if adapter_path else 'NONE (baseline)'}")
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

    # 找同 query 集的 baseline judge 文件做配对对比
    baseline_judge_file = JUDGE_DIR / f"{input_stem}_qwen3-8b-baseline_judged.jsonl"
    dpo_judge_file = JUDGE_DIR / f"{output_file.stem}_judged.jsonl"
    if args.baseline:
        # baseline 模式不打印对比（等 DPO 也跑完再比）
        pass
    else:
        _print_comparison(dpo_judge_file, baseline_judge_file if baseline_judge_file.exists() else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
