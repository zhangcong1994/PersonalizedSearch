"""快速检查 exp-011 judged 数据格式。"""
import json, sys
sys.path.insert(0, ".")

from src.utils.config import DATA_ROOT

judge_file = DATA_ROOT / "results/exp011/judge_scores/qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl"

with open(judge_file, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total records: {len(lines)}")

# 第一条记录的所有 key
r0 = lines[0]
print(f"\nKeys: {sorted(r0.keys())}")
print(f"query_id: {r0['query_id']}")
print(f"original_query_id: {r0.get('original_query_id')}")
print(f"sample_id: {r0.get('sample_id')}")
print(f"model_id: {r0.get('model_id')}")
print(f"temperature: {r0.get('temperature')}")
print(f"has system_prompt: {'system_prompt' in r0}")
print(f"has passages: {'passages' in r0}")
print(f"has query_text: {'query_text' in r0}")
print(f"passages count: {len(r0.get('passages', []))}")

# aggregation
agg = r0.get("aggregation", {})
print(f"\naggregation keys: {sorted(agg.keys())}")
print(f"total_score: {agg.get('total_score')}")

# scores
scores = r0.get("scores", {})
print(f"\nscores keys: {sorted(scores.keys())}")
for dim in ["veracity", "safety", "relevance", "synthesis_quality", "citation_quality", "user_experience"]:
    s = scores.get(dim)
    if isinstance(s, dict):
        reason = s.get("reason", "")[:80]
        print(f"  {dim}: score={s.get('score')}, reason={reason}")
    else:
        print(f"  {dim}: {s}")

# answer
answer = r0.get("answer", "")
print(f"\nanswer length: {len(answer)} chars")
print(f"answer preview: {answer[:200]}")

# 按 original_query_id 分组统计
from collections import Counter
qid_counts = Counter()
for line in lines:
    qid = line.get("original_query_id", line["query_id"].rsplit("_s", 1)[0])
    qid_counts[qid] += 1
print(f"\nPer-query sample counts: min={min(qid_counts.values())}, max={max(qid_counts.values())}, unique queries={len(qid_counts)}")
print(f"Count distribution: {Counter(qid_counts.values())}")
