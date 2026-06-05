"""Show V1 vs V3 where V3 improved significantly."""
import json, os
os.chdir(r"e:\Users\czhang\trae_projects\PersonalizedSearch")

def load_judged(filepath):
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
                "reasons": {k: v.get("reason", "")[:200] for k, v in r.get("scores", {}).items() if isinstance(v, dict)},
            }
    return results

v1 = load_judged("results/exp010/judge_scores/qwen3-8b-nothink-v1-full_judged.jsonl")
v3 = load_judged("results/exp010/judge_scores/qwen3-8b-nothink-v3_judged.jsonl")

dim_labels = {
    "veracity": "准确性", "safety": "安全性", "relevance": "相关性",
    "synthesis_quality": "整合质量", "citation_quality": "引文质量", "user_experience": "用户体验",
}

# Cases where V3 > V1 by total score >= 5
better = []
for qid in v1:
    if qid not in v3:
        continue
    d = v3[qid]["total"] - v1[qid]["total"]
    if d >= 3:
        better.append((qid, d, v1[qid], v3[qid]))
better.sort(key=lambda x: -x[1])

# Also: cases where UX or relevance improved by 2+
ux_better = []
rel_better = []
syn_better = []
for qid in v1:
    if qid not in v3:
        continue
    for label, lst, dim_key in [
        ("用户体验", ux_better, "user_experience"),
        ("相关性", rel_better, "relevance"),
        ("整合质量", syn_better, "synthesis_quality"),
    ]:
        dv = v3[qid]["dims"].get(dim_key, 0) - v1[qid]["dims"].get(dim_key, 0)
        if dv >= 2:
            lst.append((qid, dv, v1[qid], v3[qid], dim_key))

print(f"V3 > V1 by >=3 total score: {len(better)} cases")
print(f"UX improved by >=2: {len(ux_better)} cases")
print(f"Relevance improved by >=2: {len(rel_better)} cases")
print(f"Synthesis improved by >=2: {len(syn_better)} cases")
print()

print("=== 总分改善 >=3 的 case ===")
print()
for qid, delta, vi, vt in better[:6]:
    query = vi["query"]
    if len(query) > 80:
        query = query[:80] + "..."
    print(f"Query [{qid}]: {query}")
    print(f"  V1: {vi['total']:.1f} ({vi['grade']}) -> V3: {vt['total']:.1f} ({vt['grade']})  (+{delta:.1f})")
    for dim in ["veracity", "relevance", "synthesis_quality", "citation_quality", "user_experience", "safety"]:
        dv = vt["dims"].get(dim, 0) - vi["dims"].get(dim, 0)
        if dv > 0:
            label = dim_labels.get(dim, dim)
            print(f"    {label}: {vi['dims'].get(dim,0)} -> {vt['dims'].get(dim,0)} (+{dv})")
    for dim in ["relevance", "user_experience"]:
        if vt["dims"].get(dim, 0) > vi["dims"].get(dim, 0):
            print(f"    V3 Judge({dim_labels.get(dim,dim)}): {vt['reasons'].get(dim, '')[:180]}")
    print()

ux_better.sort(key=lambda x: -x[1])
print("=== UX 暴涨的 case ===")
print()
for qid, dv, vi, vt, dim_key in ux_better[:5]:
    query = vi["query"]
    if len(query) > 80:
        query = query[:80] + "..."
    print(f"Query [{qid}]: {query}")
    print(f"  V1: {vi['dims'].get('user_experience',0)} -> V3: {vt['dims'].get('user_experience',0)}")
    print(f"  V3 Judge(UX): {vt['reasons'].get('user_experience', '')[:200]}")
    print()

rel_better.sort(key=lambda x: -x[1])
print("=== 相关性暴涨的 case ===")
print()
for qid, dv, vi, vt, dim_key in rel_better[:5]:
    query = vi["query"]
    if len(query) > 80:
        query = query[:80] + "..."
    print(f"Query [{qid}]: {query}")
    print(f"  V1: {vi['dims'].get('relevance',0)} -> V3: {vt['dims'].get('relevance',0)}")
    print(f"  V3 Judge(Relevance): {vt['reasons'].get('relevance', '')[:200]}")
    print()
