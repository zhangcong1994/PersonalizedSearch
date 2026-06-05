"""Show V1 vs V3 generated text for specific query IDs."""
import json, sys

targets = sys.argv[1:] if len(sys.argv) > 1 else ["1496", "319", "778"]

gen_v1 = {}
gen_v3 = {}
with open("results/exp010/generations/qwen3-8b-nothink-v1-full.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line.strip())
        gen_v1[r["query_id"]] = r.get("answer", "")

with open("results/exp010/generations/qwen3-8b-nothink-v3.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line.strip())
        gen_v3[r["query_id"]] = r.get("answer", "")

sep = "-" * 60
for qid in targets:
    v1_text = gen_v1.get(qid, "NOT FOUND")
    v3_text = gen_v3.get(qid, "NOT FOUND")
    print(f"=== Query [{qid}] ===")
    print(f"V1 ({len(v1_text)} chars):")
    print(v1_text[:300])
    print()
    print(f"V3 ({len(v3_text)} chars):")
    print(v3_text[:300])
    print()
    print(sep)
    print()
