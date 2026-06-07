"""
将教师生成的回答和 Judge 评分合并到学生的多样本文件中，
以便 construct_dpo_pairs.py 把教师回答当作第 6 个候选样本。

输入:
  - results/exp012/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl  (学生 8140 条)
  - results/exp012/judge_scores/qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl  (学生评分)
  - results/exp012/generation/qwen3.6-plus-teacher_v1-full_t0.3_n1_s42.jsonl  (教师 1997 条)
  - results/exp012/judge_scores/qwen3.6-plus-teacher_v1-full_judged.jsonl  (教师评分)

输出:
  - results/exp012/generation/qwen3-8b-plus-teacher_v1-full_t0.8_n5+1_s42.jsonl
  - results/exp012/judge_scores/qwen3-8b-plus-teacher_v1-full_t0.8_n5+1_s42_judged.jsonl

用法: python scripts/exp012/merge_teacher_student.py
"""

import os, sys, json
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

R = DATA_ROOT / "results" / "exp012"

STUDENT_GEN = R / "generation" / "qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl"
STUDENT_JUDGE = R / "judge_scores" / "qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl"
TEACHER_GEN = R / "generation" / "qwen3.6-plus-teacher_v1-full_t0.3_n1_s42.jsonl"
TEACHER_JUDGE = R / "judge_scores" / "qwen3.6-plus-teacher_v1-full_judged.jsonl"

MERGED_GEN = R / "generation" / "qwen3-8b-plus-teacher_v1-full_t0.8_n5+1_s42.jsonl"
MERGED_JUDGE = R / "judge_scores" / "qwen3-8b-plus-teacher_v1-full_t0.8_n5+1_s42_judged.jsonl"

# ── 加载 ──

def count_lines(path):
    if not path.exists():
        return 0, set()
    qids = set()
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qids.add(r.get("original_query_id", r["query_id"].rsplit("_s", 1)[0]))
            count += 1
    return count, qids

student_gen_n, student_qids = count_lines(STUDENT_GEN)
teacher_gen_n, teacher_qids = count_lines(TEACHER_GEN)
student_judge_n, _ = count_lines(STUDENT_JUDGE)
teacher_judge_n, _ = count_lines(TEACHER_JUDGE)

print(f"Student gen: {student_gen_n} records, {len(student_qids)} unique queries")
print(f"Teacher gen: {teacher_gen_n} records, {len(teacher_qids)} unique queries")
print(f"Student judge: {student_judge_n} records")
print(f"Teacher judge: {teacher_judge_n} records")

# 检查 query 重叠
overlap = student_qids & teacher_qids
print(f"\nQuery overlap: {len(overlap)} teacher queries also in student set")
print(f"Teacher queries NOT in student: {len(teacher_qids - student_qids)}")
print(f"Student queries NOT in teacher: {len(student_qids - teacher_qids)}")

# ── 合并 generation ──
print(f"\nMerging generation files -> {MERGED_GEN.name}")
MERGED_GEN.parent.mkdir(parents=True, exist_ok=True)

# 先删除可能残留的文件（上次 shutil.copy 留下的只读文件）
if MERGED_GEN.exists():
    MERGED_GEN.chmod(0o666)
    MERGED_GEN.unlink()

# 直接读取 + 写入（避免 shutil.copy 的 Windows 文件锁）
appended = 0
total = 0
with open(MERGED_GEN, "w", encoding="utf-8") as f_out:
    # 写学生
    with open(STUDENT_GEN, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.strip():
                f_out.write(line)
                total += 1
    # 追写教师（只保留 overlap 部分）
    with open(TEACHER_GEN, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("original_query_id", r["query_id"].rsplit("_s", 1)[0])
            if qid in student_qids:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                appended += 1
                total += 1

print(f"  Written {total} records ({appended} from teacher)")

# ── 合并 judge ──
print(f"Merging judge files -> {MERGED_JUDGE.name}")
MERGED_JUDGE.parent.mkdir(parents=True, exist_ok=True)

if MERGED_JUDGE.exists():
    MERGED_JUDGE.chmod(0o666)
    MERGED_JUDGE.unlink()

appended = 0
total = 0
with open(MERGED_JUDGE, "w", encoding="utf-8") as f_out:
    with open(STUDENT_JUDGE, "r", encoding="utf-8") as f_in:
        for line in f_in:
            if line.strip():
                f_out.write(line)
                total += 1
    with open(TEACHER_JUDGE, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("original_query_id", r["query_id"].rsplit("_s", 1)[0])
            if qid in student_qids:
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                appended += 1
                total += 1

print(f"  Written {total} records ({appended} from teacher)")

# ── 验证 ──
gen_count, gen_qids = count_lines(MERGED_GEN)
judge_count, judge_qids = count_lines(MERGED_JUDGE)
print(f"\nVerification:")
print(f"  Merged gen: {gen_count} records, {len(gen_qids)} unique queries")
print(f"  Merged judge: {judge_count} records, {len(judge_qids)} unique queries")
print(f"  Query overlap (gen ∩ judge): {len(gen_qids & judge_qids)}")

# per-query 样本数
from collections import Counter
gen_per_q = Counter()
with open(MERGED_GEN, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        qid = r.get("original_query_id", r["query_id"].rsplit("_s", 1)[0])
        gen_per_q[qid] += 1

cnt_dist = Counter(gen_per_q.values())
print(f"  Per-query sample distribution: {dict(sorted(cnt_dist.items()))}")

print(f"\nDone. Next:")
print(f"  python scripts/exp012/construct_dpo_pairs.py --generation {MERGED_GEN} --judge {MERGED_JUDGE} --output-dir data/processed/exp012")
