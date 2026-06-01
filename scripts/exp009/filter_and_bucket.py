"""
Exp-009 阶段四+五：质量过滤 + 检索质量分桶 + (可选) 幻觉检测。

流程:
  python scripts/exp009/filter_and_bucket.py                    # 仅规则过滤+分桶
  python scripts/exp009/filter_and_bucket.py --hallu-check      # 规则过滤 + 幻觉检测

输入: data/processed/exp009_teacher_answers.jsonl
输出:
  data/processed/exp009_filtered_bucketed.jsonl  (过滤后 + 分桶标注 + hallu 结果)
  data/processed/exp009_discarded.jsonl          (被丢弃的条目)

质量过滤规则:
  1. 过短        len(answer) < 20  → 丢弃
  2. 意外拒答     拒答关键词 + len<=400 + top-10 有 relevant passage  → 丢弃
  3. 无引用       "[来源" not in answer  → 丢弃
  4. 幻觉检测     仅当 --hallu-check，deepseek-chat PASS/FAIL  → FAIL 丢弃
  合理拒答       拒答关键词 + len<=400 + top-10 无 relevant  → 保留（训练拒答能力）

分桶指标:
  search_coverage = |top-10 pids ∩ retrieval_qrels| / min(10, |retrieval_qrels|)
  best_grade      = max(graded_qrels[pid] for pid in top-10 if pid in graded_qrels)

分桶:
  桶 A (检索好):  coverage >= 0.5 AND best_grade >= 2
  桶 B (检索中):  0.2 <= coverage < 0.5 OR best_grade == 1
  桶 C (检索差):  coverage < 0.2 OR top-10 全部无 qrels 标注
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径 ──────────────────────────────────────────────────

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QRELS_RETRIEVAL_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
QRELS_GRADED_FILE = T2RANKING_DIR / "qrels.train.tsv"

INPUT_FILE = DATA_ROOT / "data" / "processed" / "exp009_teacher_answers.jsonl"
OUTPUT_FILE = DATA_ROOT / "data" / "processed" / "exp009_filtered_bucketed.jsonl"
DISCARD_FILE = DATA_ROOT / "data" / "processed" / "exp009_discarded.jsonl"

# ── 拒答关键词 ────────────────────────────────────────────

REFUSAL_KW = [
    "无法确定", "无法回答", "没有提供", "无法提供",
    "资料中未", "资料中没有", "没有提及",
    "未提及", "无相关信息", "没有相关信息",
    "没能找到", "没有找到", "未找到",
    "参考资料中未",
    "未涉及", "没有涉及",
    "无法判断", "无法确认",
    "没有直接提供",
]

REFUSAL_MAX_LEN = 400


# ── 数据加载 ──────────────────────────────────────────────

def load_answers(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_retrieval_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                qrels.setdefault(parts[0], set()).add(parts[1])
    return qrels


def load_graded_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 4:
                qid, _, pid, label = parts[0], parts[1], parts[2], int(parts[3])
            elif len(parts) == 3:
                qid, pid, label = parts[0], parts[1], int(parts[2])
            else:
                continue
            if label > 0:
                qrels.setdefault(qid, {})[pid] = label
    return qrels


# ── 质量过滤 ──────────────────────────────────────────────

def is_refusal_kw(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KW)


def compute_filter(
    entry: dict,
    retrieval_qrels: dict[str, set[str]],
) -> tuple[str, int]:
    """
    返回 (discard_reason or "", relevant_hit_count).
    "" = 通过, 非空 = 丢弃原因.
    """
    qid = entry["qid"]
    answer = entry["teacher_answer"]
    top10_pids = [p["pid"] for p in entry.get("passages", [])]

    # 规则 1: 过短
    if len(answer) < 20:
        return "too_short", 0

    relevant = retrieval_qrels.get(qid, set())
    hits = len([pid for pid in top10_pids if pid in relevant])

    has_kw = is_refusal_kw(answer)

    # 规则 2: 意外拒答（短拒答 + top-10 有 relevant）
    if has_kw and len(answer) <= REFUSAL_MAX_LEN and hits > 0:
        return "accidental_refusal", hits

    # 规则 3: 无引用
    if "[来源" not in answer:
        return "no_citation", hits

    return "", hits


# ── 检索质量分桶 ──────────────────────────────────────────

def compute_bucket(
    entry: dict,
    retrieval_qrels: dict[str, set[str]],
    graded_qrels: dict[str, dict[str, int]],
) -> str:
    qid = entry["qid"]
    top10_pids = [p["pid"] for p in entry.get("passages", [])]
    relevant = retrieval_qrels.get(qid, set())
    graded = graded_qrels.get(qid, {})

    n_relevant = len(relevant)
    hits = len([pid for pid in top10_pids if pid in relevant])
    coverage = hits / min(10, n_relevant) if n_relevant > 0 else 0.0

    best_grade = 0
    for pid in top10_pids:
        g = graded.get(pid, 0)
        if g > best_grade:
            best_grade = g

    if coverage >= 0.5 and best_grade >= 2:
        return "A"
    elif coverage >= 0.2:
        return "B"
    elif best_grade == 1:
        return "B"
    else:
        return "C"


# ── 幻觉检测 ──────────────────────────────────────────────

HALLU_SYSTEM = """你是一个事实核查专家。你的任务是判断 AI 生成的回答是否忠实地基于给定的参考资料。

规则:
1. 逐条检查回答中的每个事实性陈述（实体、数字、日期、因果关系、步骤等）
2. 判断该陈述是否能在参考资料中找到依据
3. 如果所有事实陈述都有资料支撑 → PASS
4. 如果存在任何编造、曲解、或资料中没有的事实在回答中呈现 → FAIL
5. 合理推断不扣分：回答中基于常识的合理推论（如"二楼夏天会更热""孕妇可以适量食用"），即使在资料中没有逐字对应，不算 FAIL
6. 诚实拒答不扣分：回答主动声明"资料中未提及X"、"根据现有资料无法确定"，这是诚实的表现，不算 FAIL
7. 表述偏差不扣分：轻微的同义词替换、表述顺序差异（如"防锈涂层"与"涂层防锈"），事实本身一致即算通过
8. 张冠李戴必扣分：把资料中关于A的描述安到B上，属于曲解

输出格式（严格 JSON）:
{"verdict": "PASS"|"FAIL", "reason": "一句话说明原因。如果是 PASS，简要说明检查了哪些关键事实。如果是 FAIL，明确指出哪个具体陈述在资料中找不到依据或曲解了资料。"}
"""

HALLU_USER = """请判断以下 AI 回答是否忠于给定的参考资料。

---
【参考资料】
{passages}

【用户问题】
{query}

【AI 回答】
{answer}
---
"""


def format_passages_hallu(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages):
        pid = p.get("pid", "?")
        text = p.get("text", "")
        lines.append(f"[{i+1}] pid={pid}\n{text}")
    return "\n\n".join(lines)


def check_hallu(entry: dict) -> tuple[str, str]:
    """返回 (verdict, reason)."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set (required for --hallu-check)")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
        max_tokens=256,
    )

    user_msg = HALLU_USER.format(
        passages=format_passages_hallu(entry.get("passages", [])),
        query=entry["query"],
        answer=entry["teacher_answer"],
    )

    response = llm.invoke([
        SystemMessage(content=HALLU_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    raw = response.content.strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        return result.get("verdict", "?"), result.get("reason", raw)
    except json.JSONDecodeError:
        if "PASS" in raw.upper() and "FAIL" not in raw.upper():
            return "PASS", raw
        elif "FAIL" in raw.upper():
            return "FAIL", raw
        else:
            return "?", raw


def run_hallu_check(entries: list[dict], workers: int = 8) -> tuple[list[dict], list[dict]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info("-" * 60)
    logger.info(f"  HALLUCINATION DETECTION (deepseek-chat, {workers} concurrent)")
    logger.info(f"  Checking {len(entries):,} entries...")

    kept = []
    discarded = []
    pass_count = 0
    fail_count = 0
    error_count = 0
    t_start = time.time()

    def process_one(idx: int, entry: dict) -> tuple[int, dict, str, str]:
        try:
            verdict, reason = check_hallu(entry)
        except Exception as e:
            verdict, reason = "ERROR", str(e)
        return idx, entry, verdict, reason

    futures_map: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i, entry in enumerate(entries):
            fut = executor.submit(process_one, i, entry, )
            futures_map[fut] = i

        for fut in as_completed(futures_map):
            idx, entry, verdict, reason = fut.result()
            entry["_hallu_verdict"] = verdict
            entry["_hallu_reason"] = reason

            if verdict == "PASS":
                kept.append(entry)
                pass_count += 1
            elif verdict == "FAIL":
                entry["_discard_reason"] = "hallu_fail"
                entry["_hallu_reason"] = reason
                discarded.append(entry)
                fail_count += 1
            else:
                kept.append(entry)
                error_count += 1

            done = pass_count + fail_count + error_count
            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 0.1)
                eta = (len(entries) - done) / max(rate, 0.01) / 60
                logger.info(f"  Progress: {done}/{len(entries)}  PASS={pass_count}  FAIL={fail_count}  "
                            f"{rate:.1f}q/s  ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    total = pass_count + fail_count + error_count
    logger.info(f"  Hallu check done in {elapsed/60:.1f}min: "
                f"PASS={pass_count} FAIL={fail_count} ERROR={error_count}")
    return kept, discarded


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exp-009 Stage 4+5: Filter & Bucket")
    parser.add_argument("--hallu-check", action="store_true",
                        help="Run hallucination detection via deepseek-chat (~5 CNY for full set)")
    parser.add_argument("--hallu-workers", type=int, default=8,
                        help="Concurrent workers for hallucination detection (default: 8)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Exp-009 Stage 4+5: Filter & Bucket")
    if args.hallu_check:
        logger.info("  Mode: rule-based filter + hallucination detection")
    else:
        logger.info("  Mode: rule-based filter only")
    logger.info("=" * 60)

    logger.info("Loading answers...")
    answers = load_answers(INPUT_FILE)
    logger.info(f"  Loaded {len(answers):,} teacher answers")

    logger.info("Loading retrieval qrels...")
    retrieval_qrels = load_retrieval_qrels(QRELS_RETRIEVAL_FILE)
    logger.info(f"  Loaded {len(retrieval_qrels):,} queries with retrieval qrels")

    logger.info("Loading graded qrels...")
    graded_qrels = load_graded_qrels(QRELS_GRADED_FILE)
    logger.info(f"  Loaded {len(graded_qrels):,} queries with graded qrels")

    # ── 规则过滤 ──
    kept = []
    discarded = []
    discard_reasons: dict[str, int] = {}

    for entry in answers:
        reason, hits = compute_filter(entry, retrieval_qrels)
        if reason:
            entry["_discard_reason"] = reason
            discarded.append(entry)
            discard_reasons[reason] = discard_reasons.get(reason, 0) + 1
        else:
            kept.append(entry)

    logger.info("-" * 60)
    logger.info("  RULE-BASED FILTER SUMMARY")
    logger.info(f"  Total input:        {len(answers):,}")
    logger.info(f"  Kept:               {len(kept):,} ({len(kept)/len(answers)*100:.1f}%)")
    logger.info(f"  Discarded:          {len(discarded):,} ({len(discarded)/len(answers)*100:.1f}%)")
    for reason, count in sorted(discard_reasons.items()):
        logger.info(f"    - {reason}:       {count}")

    # ── 幻觉检测（可选）──
    if args.hallu_check:
        logger.info("")
        hallu_kept, hallu_discarded = run_hallu_check(kept, workers=args.hallu_workers)
        kept = hallu_kept
        discarded.extend(hallu_discarded)
        discard_reasons["hallu_fail"] = len(hallu_discarded)

    # ── 分桶 ──
    bucket_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    for entry in kept:
        bucket = compute_bucket(entry, retrieval_qrels, graded_qrels)
        entry["bucket"] = bucket
        bucket_counts[bucket] += 1

    # ── 统计 ──
    logger.info("-" * 60)
    logger.info("  FINAL SUMMARY")
    logger.info(f"  Kept:               {len(kept):,} ({len(kept)/len(answers)*100:.1f}%)")
    logger.info(f"  Discarded:          {len(discarded):,} ({len(discarded)/len(answers)*100:.1f}%)")
    for reason, count in sorted(discard_reasons.items()):
        logger.info(f"    - {reason}:       {count}")
    logger.info("-" * 60)
    logger.info("  BUCKET DISTRIBUTION (kept)")
    for b in ["A", "B", "C"]:
        count = bucket_counts[b]
        pct = count / max(len(kept), 1) * 100
        logger.info(f"    Bucket {b}:       {count:,} ({pct:.1f}%)")

    # ── 写输出 ──
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in kept:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Filtered + bucketed → {OUTPUT_FILE}")

    with open(DISCARD_FILE, "w", encoding="utf-8") as f:
        for entry in discarded:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Discarded           → {DISCARD_FILE}")

    logger.info("=" * 60)
    logger.info("  Stage 4+5 complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
