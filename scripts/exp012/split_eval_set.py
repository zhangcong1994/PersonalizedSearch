"""找出 exp-011 从 198 条中抽走的 50 条 query_id，输出剩余 148 条供评估使用。"""
import json, sys, random
sys.path.insert(0, ".")

from src.utils.config import DATA_ROOT

INPUT_QUERIES = DATA_ROOT / "results" / "exp005" / "input_queries.jsonl"
OUTPUT_TRAIN = DATA_ROOT / "results" / "exp012" / "train_query_ids.json"
OUTPUT_EVAL = DATA_ROOT / "results" / "exp012" / "eval_queries.jsonl"

# 加载全量 198 条
all_queries = []
with open(INPUT_QUERIES, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        all_queries.append(json.loads(line))

print(f"Total queries: {len(all_queries)}")

# 用与 exp-011 完全相同的抽样逻辑：sampled 50, seed=42
SAMPLE_SIZE = 50
SEED = 42

rng = random.Random(SEED)
indices = list(range(len(all_queries)))
rng.shuffle(indices)
train_indices = set(indices[:SAMPLE_SIZE])

# 找出训练用的 50 条
train_qids = []
eval_queries = []
for i, q in enumerate(all_queries):
    if i in train_indices:
        train_qids.append(q["query_id"])
    else:
        eval_queries.append(q)

print(f"Train queries: {len(train_qids)}")
print(f"  Sample train qids: {train_qids[:5]}...")
print(f"Eval queries (clean): {len(eval_queries)}")

# 保存训练 query_id 列表
import os
os.makedirs(OUTPUT_TRAIN.parent, exist_ok=True)
with open(OUTPUT_TRAIN, "w") as f:
    json.dump(sorted(train_qids), f)
print(f"Saved train qids to {OUTPUT_TRAIN}")

# 保存评估 query 集
with open(OUTPUT_EVAL, "w", encoding="utf-8") as f:
    for q in eval_queries:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"Saved {len(eval_queries)} eval queries to {OUTPUT_EVAL}")
