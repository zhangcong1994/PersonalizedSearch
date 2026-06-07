"""
将 exp012 教师答案转换为 exp011 多样本生成兼容格式，供 Judge 评分。

输入: data/processed/exp012_teacher_answers.jsonl
      {"qid": ..., "query": ..., "passages": [...], "teacher_answer": "...", "model": "...", ...}

输出: results/exp012/generation/qwen3.6-plus-teacher_v1-full_t0.3_n1_s42.jsonl
      {"query_id": "xxx_s0", "original_query_id": "xxx", "sample_id": 0,
       "query_text": "...", "answer": "...", "passages": [...], "system_prompt": "...",
       "model_id": "qwen3.6-plus-teacher", "temperature": 0.3}

用法: python scripts/exp012/convert_teacher_for_judge.py
"""

import os, sys, json
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from src.generation.prompts_v2 import PromptV2Manager

INPUT_FILE = DATA_ROOT / "data" / "processed" / "exp012_teacher_answers.jsonl"
OUTPUT_FILE = DATA_ROOT / "results" / "exp012" / "generation" / "qwen3.6-plus-teacher_v1-full_t0.3_n1_s42.jsonl"

INPUT_FILE = Path(INPUT_FILE)
OUTPUT_FILE = Path(OUTPUT_FILE)

if not INPUT_FILE.exists():
    print(f"ERROR: {INPUT_FILE} not found")
    sys.exit(1)

prompt_mgr = PromptV2Manager("v1-full")
system_prompt = prompt_mgr.get_system_prompt()

with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
    lines = [json.loads(l) for l in f_in if l.strip()]

print(f"Loaded {len(lines)} teacher answers")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
    for r in lines:
        qid = r["qid"]
        answer = r.get("teacher_answer", "")
        if not answer.strip():
            continue

        out = {
            "query_id": f"{qid}_s0",
            "original_query_id": qid,
            "query_text": r["query"],
            "model_id": "qwen3.6-plus-teacher",
            "sample_id": 0,
            "answer": answer,
            "passages": r["passages"],
            "system_prompt": system_prompt,
            "temperature": 0.3,
        }
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        written += 1

print(f"Converted {written} entries -> {OUTPUT_FILE}")
print(f"\nNext: python scripts/exp011/run_judge.py --input {OUTPUT_FILE} --output results/exp012/judge_scores/qwen3.6-plus-teacher_v1-full_judged.jsonl --judge-model deepseek-chat")
