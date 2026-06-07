"""
验证 DPO LoRA adapter 推理是否正常（transformers + PEFT，不加 merge）。

对比两种模式：
  1. Baseline（纯 4-bit 基座，不用 LoRA）
  2. DPO（4-bit 基座 + LoRA adapter）

用前 N 条 query 分别跑，对比输出差异。
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.utils.config import DATA_ROOT, MODEL_CACHE_DIR
from src.generation.prompts_v2 import PromptV2Manager

# ── 配置 ──
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = DATA_ROOT / "models" / "exp012-dpo-pilot"
INPUT_FILE = DATA_ROOT / "data" / "processed" / "exp012_validation_queries.jsonl"
TEMPERATURE = 0.3
MAX_TOKENS = 1024

def build_user_content(query_text, passages):
    parts = []
    for p in passages:
        pid = p.get("pid", "unknown")
        rank = p.get("rank", 1)
        text = p.get("text", "")
        parts.append(f"[{rank}] 来源: {pid}\n{text[:800]}")
    ctx = "\n\n".join(parts)
    return f"参考资料:\n{ctx}\n\n用户问题: {query_text}\n\n请根据以上参考资料回答问题："


def main():
    parser = argparse.ArgumentParser(description="Debug: 验证 DPO LoRA adapter 推理")
    parser.add_argument("--num", type=int, default=5, help="跑几条 query (default: 5)")
    parser.add_argument("--baseline-only", action="store_true", help="只跑 baseline，不加载 LoRA")
    parser.add_argument("--dpo-only", action="store_true", help="只跑 DPO，不跑 baseline")
    args = parser.parse_args()

    run_baseline = not args.dpo_only
    run_dpo = not args.baseline_only

    # 加载数据
    queries = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num:
                break
            queries.append(json.loads(line.strip()))

    prompt_mgr = PromptV2Manager("v1-full")
    system_prompt = prompt_mgr.get_system_prompt()

    # 找本地 base model
    cache = MODEL_CACHE_DIR
    name = f"models--{BASE_MODEL.replace('/', '--')}"
    snap = sorted((cache / name / "snapshots").iterdir())[-1]
    base_path = str(snap)

    # ── 加载 ──
    print(f"Loading tokenizer from {base_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)

    print("Loading 4-bit base model ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if run_dpo:
        print(f"Loading LoRA from {ADAPTER_PATH} ...")
        dpo_model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
        dpo_model.eval()
        base.eval()  # baseline 也用同一个，不需重新加载

    # ── 推理 ──
    for i, item in enumerate(queries):
        qid = item["query_id"]
        query_text = item.get("query_text", item.get("query", ""))
        passages = item.get("passages", [])

        user = build_user_content(query_text, passages)
        full = f"{system_prompt}\n\n{user}"
        msgs = [{"role": "user", "content": full}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(base.device)

        print(f"\n{'='*60}")
        print(f"[{i+1}/{args.num}] qid={qid}  query={query_text[:50]}...")

        if run_baseline:
            with torch.no_grad():
                outputs = base.generate(
                    **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0, pad_token_id=tokenizer.eos_token_id,
                )
            gen = outputs[0][inputs["input_ids"].shape[1]:]
            ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
            print(f"\n  [BASELINE] ({len(ans)} chars):")
            print(f"  {ans[:400]}")

        if run_dpo:
            with torch.no_grad():
                outputs = dpo_model.generate(
                    **inputs, max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                    do_sample=TEMPERATURE > 0, pad_token_id=tokenizer.eos_token_id,
                )
            gen = outputs[0][inputs["input_ids"].shape[1]:]
            ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
            print(f"\n  [DPO] ({len(ans)} chars):")
            print(f"  {ans[:400]}")


if __name__ == "__main__":
    main()
