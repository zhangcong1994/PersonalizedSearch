"""Compare V1 and V3 judged results, find cases where V3 degraded."""
import json

def load(filepath):
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            agg = r.get("aggregated", {})
            results[r["query_id"]] = {
                "query": r.get("query_text", "?"),
                "total": agg.get("total_score", 0),
                "grade": agg.get("grade", "?"),
                "dims": {k: v.get("score", 0) for k, v in r.get("scores", {}).items() if isinstance(v, dict)},
                "pass_": agg.get("pass", False),
            }
    return results

import os
os.chdir(r"e:\Users\czhang\trae_projects\PersonalizedSearch")

v1 = load("results/exp010/judge_scores/qwen3-8b-nothink-v1-full_judged.jsonl")
v3 = load("results/exp010/judge_scores/qwen3-8b-nothink-v3_judged.jsonl")

# Worsened cases (V3 < V1 by >= 3 points)
worse = []
# Improved cases (V3 > V1 by >= 3 points)
better = []
for qid in v1:
    if qid not in v3:
        continue
    d = v1[qid]["total"] - v3[qid]["total"]
    if d >= 3:
        worse.append((qid, d, v1[qid], v3[qid]))
    elif d <= -3:
        better.append((qid, -d, v1[qid], v3[qid]))

worse.sort(key=lambda x: -x[1])
better.sort(key=lambda x: -x[1])

# Find cases where accuracy dropped by 2+
acc_drops = []
for qid in v1:
    if qid not in v3:
        continue
    va = v1[qid]["dims"].get("veracity", 0)
    vb = v3[qid]["dims"].get("veracity", 0)
    if va - vb >= 2:
        acc_drops.append((qid, va, vb, v1[qid], v3[qid]))

acc_drops.sort(key=lambda x: x[2] - x[1])

dim_labels = {
    "veracity": "准确性",
    "safety": "安全性",
    "relevance": "相关性",
    "synthesis_quality": "整合质量",
    "citation_quality": "引文质量",
    "user_experience": "用户体验",
}

print(f"Accuracy dropped by 2+ in {len(acc_drops)} cases")
print()
print("=== 准确性暴跌 case ===")
print()
for qid, va, vb, vi, vt in acc_drops[:8]:
    query = vi["query"]
    if len(query) > 80:
        query = query[:80] + "..."
    print(f"Query [{qid}]: {query}")
    print(f"  V1: {vi['total']:.1f} ({vi['grade']})  V3: {vt['total']:.1f} ({vt['grade']})")
    print(f"  准确性: {va} -> {vb}  ({vb-va:+d})")
    for dim in vi["dims"]:
        dv = vt["dims"].get(dim, 0) - vi["dims"].get(dim, 0)
        if dv != 0:
            label = dim_labels.get(dim, dim)
            print(f"    {label}: {vi['dims'].get(dim,0)} -> {vt['dims'].get(dim,0)} ({dv:+d})")
    print()

# Also find V3 improved accuracy cases
acc_gains = []
for qid in v1:
    if qid not in v3:
        continue
    va = v1[qid]["dims"].get("veracity", 0)
    vb = v3[qid]["dims"].get("veracity", 0)
    if vb - va >= 2:
        acc_gains.append((qid, va, vb, v1[qid], v3[qid]))

print(f"Accuracy improved by 2+ in {len(acc_gains)} cases")
