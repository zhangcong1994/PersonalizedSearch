"""分析 exp-011 生成答案中的格式污染模式"""
import json

GEN = "results/exp011/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl"
JUDGE = "results/exp011/judge_scores/qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl"

# 加载 judge 分数
judge = {}
with open(JUDGE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        judge[r["query_id"]] = r.get("aggregation", {}).get("total_score", 0)

# 检查生成答案的格式
patterns = {}
total = 0
samples = []
with open(GEN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        ans = r.get("answer", "")
        qid = r.get("query_id", "")
        score = judge.get(qid, 0)
        total += 1

        # 检测各种模式
        matched = False
        if ans.startswith("【参考资料】"):
            patterns["start_ckzl"] = patterns.get("start_ckzl", 0) + 1
            matched = True
        if "【用户问题】" in ans[:200]:
            patterns["has_user_q"] = patterns.get("has_user_q", 0) + 1
            matched = True
        if "【回答】" in ans[:200]:
            patterns["has_answer_tag"] = patterns.get("has_answer_tag", 0) + 1
            matched = True
        if ans.startswith("【核心结论】"):
            patterns["start_core"] = patterns.get("start_core", 0) + 1
            matched = True
        if ans.startswith("【核心答案】"):
            patterns["start_core_a"] = patterns.get("start_core_a", 0) + 1
            matched = True

        if matched and len(samples) < 5:
            samples.append((qid, score, ans[:350]))

print(f"Total answers: {total}")
print(f"\nPattern counts:")
for k, v in sorted(patterns.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/total*100:.1f}%)")

print(f"\nSample contaminated answers:")
for qid, score, ans in samples:
    print(f"\n  [{qid}] score={score:.0f}")
    print(f"    {ans}")
