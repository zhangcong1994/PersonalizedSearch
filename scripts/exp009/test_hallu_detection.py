"""Exp-009: Hallucination detection pilot test (50 samples).

用 deepseek-chat 检查教师答案是否忠于给定的检索结果。
二分类：PASS（所有事实陈述有资料依据）/ FAIL（存在编造或曲解）。

用法:
  python scripts/exp009/test_hallu_detection.py
  python scripts/exp009/test_hallu_detection.py --n 100
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_FILE = DATA_ROOT / "data" / "processed" / "exp009_filtered_bucketed.jsonl"
OUTPUT_FILE = DATA_ROOT / "data" / "processed" / "exp009_hallu_test_results.jsonl"

# ── Prompt ────────────────────────────────────────────────

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
{"verdict": "PASS"|"FAIL", "reason": "一句话说明扣分原因。如果是 PASS，简要说明检查了哪些关键事实并确认有依据。如果是 FAIL，明确指出哪个具体陈述在资料中找不到依据或曲解了资料。"}
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


# ── Helpers ───────────────────────────────────────────────

def format_passages(passages: list[dict]) -> str:
    lines = []
    for i, p in enumerate(passages):
        pid = p.get("pid", "?")
        text = p.get("text", "")
        lines.append(f"[{i+1}] pid={pid}\n{text}")
    return "\n\n".join(lines)


def call_hallu_judge(
    query: str,
    answer: str,
    passages: list[dict],
) -> tuple[str, str]:
    """返回 (verdict, reason)."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.0,
        max_tokens=256,
    )

    user_msg = HALLU_USER.format(
        passages=format_passages(passages),
        query=query,
        answer=answer,
    )

    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([
        SystemMessage(content=HALLU_SYSTEM),
        HumanMessage(content=user_msg),
    ])

    raw = response.content.strip()
    # 尝试解析 JSON
    try:
        # 去掉可能的 markdown 代码块标记
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        return result.get("verdict", "?"), result.get("reason", raw)
    except json.JSONDecodeError:
        # 尝试从文本中提取 PASS/FAIL
        if "PASS" in raw.upper() and "FAIL" not in raw.upper():
            return "PASS", raw
        elif "FAIL" in raw.upper():
            return "FAIL", raw
        else:
            return "?", raw


# ── Main ─────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hallucination detection pilot")
    parser.add_argument("--n", type=int, default=50, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--buckets", nargs="*", default=["A", "B", "C"],
                        help="Which buckets to sample from")
    args = parser.parse_args()

    # ── 加载过滤后的数据 ──
    logger.info(f"Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info(f"  Total: {len(data):,} entries")

    # ── 分层采样 ──
    by_bucket = {b: [e for e in data if e.get("bucket") == b] for b in args.buckets}
    for b in args.buckets:
        logger.info(f"  Bucket {b}: {len(by_bucket[b]):,}")

    random.seed(args.seed)
    n_per = args.n // len(args.buckets)
    samples = []
    for b in args.buckets:
        pool = by_bucket[b]
        take = min(n_per, len(pool))
        samples.extend(random.sample(pool, take))
    # 如果除不尽，补足
    while len(samples) < args.n:
        b = random.choice(args.buckets)
        pool = by_bucket[b]
        remaining = [e for e in pool if e not in samples]
        if remaining:
            samples.append(random.choice(remaining))

    logger.info(f"Sampled {len(samples)} entries for hallu detection")
    logger.info("-" * 60)

    # ── 逐条检测 ──
    results = []
    pass_count = 0
    fail_count = 0
    fail_examples = []

    for i, entry in enumerate(samples):
        qid = entry["qid"]
        query = entry["query"]
        answer = entry["teacher_answer"]
        passages = entry.get("passages", [])

        try:
            verdict, reason = call_hallu_judge(query, answer, passages)
        except Exception as e:
            verdict, reason = "ERROR", str(e)

        if verdict == "PASS":
            pass_count += 1
        elif verdict == "FAIL":
            fail_count += 1
            if len(fail_examples) < 8:
                fail_examples.append({
                    "qid": qid,
                    "query": query,
                    "answer": answer,
                    "passages": passages,
                    "reason": reason,
                })
        else:
            # UNKNOWN
            pass

        results.append({
            "qid": qid,
            "query": query,
            "bucket": entry.get("bucket", "?"),
            "answer_len": len(answer),
            "verdict": verdict,
            "reason": reason,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(samples)}  PASS={pass_count}  FAIL={fail_count}")

    # ── 保存结果 ──
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # ── 总结 ──
    print()
    print("=" * 70)
    print("  HALLUCINATION DETECTION RESULTS")
    print("=" * 70)
    print(f"  Samples:     {len(samples)}")
    print(f"  PASS:        {pass_count} ({pass_count/len(samples)*100:.1f}%)")
    print(f"  FAIL:        {fail_count} ({fail_count/len(samples)*100:.1f}%)")

    # 按桶拆分
    for b in args.buckets:
        bucket_results = [r for r in results if r.get("bucket") == b]
        if bucket_results:
            p = sum(1 for r in bucket_results if r["verdict"] == "PASS")
            f = sum(1 for r in bucket_results if r["verdict"] == "FAIL")
            print(f"    Bucket {b}:       PASS={p}  FAIL={f}  ({p/len(bucket_results)*100:.0f}% pass)")

    # ── 打印 FAIL 样例 ──
    print()
    print("=" * 70)
    print(f"  FAIL EXAMPLES ({len(fail_examples)} shown)")
    print("=" * 70)
    for i, ex in enumerate(fail_examples):
        print(f"\n--- FAIL [{i+1}] qid={ex['qid']} ---")
        print(f"  Query:   {ex['query'][:80]}")
        print(f"  Reason:  {ex['reason'][:200]}")
        ans = ex["answer"]
        print(f"  Answer ({len(ans)} chars): {ans[:200]}...")

    print()


if __name__ == "__main__":
    main()
