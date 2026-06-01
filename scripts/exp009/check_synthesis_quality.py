"""
Exp-009: 整合质量抽样评估（synthesis quality）。

用 deepseek-chat 对随机抽取的 30 条教师答案做单维度整合质量评分（1-4 分）。
目的是判断教师答案是否存在大量"简单罗列"（1-2 分），如果是则需要全量评估+过滤。

用法:
  python scripts/exp009/check_synthesis_quality.py
  python scripts/exp009/check_synthesis_quality.py --n 50
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path

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

SYNTH_SYSTEM = """你是 AI 搜索答案质量的评估专家。你现在只评估 **一个维度：信息整合质量 (Synthesis Quality)**。

## 核心问题
LLM 是否对多篇检索文档进行了有机整合？还是只是简单罗列？

## 评分标准（1-4 分，整数）

| 分数 | 标签 | 详细定义 |
|------|------|---------|
| 1 | 简单罗列 | 未做任何整合。答案形如"文档A说X。文档B说Y。"信息被机械排列，用户需自己做综合。 |
| 2 | 浅层整合 | 做了基本的归类，但缺乏深度对比和推理。存在信息重复。文档间矛盾未作处理。 |
| 3 | 有效整合 | 将多文档信息融合为连贯整体。核心信息被提炼（非原文照搬）。能从多文档中推出合理综合结论。 |
| 4 | 深度综合 | 多源信息组织为层次分明的知识结构。文档间矛盾被明确指出、对比分析。产出任意单一文档都不具备的认知价值。 |

## 评分步骤
1. 数一下回答中实际引用了几个不同的资料来源
2. 判断引用方式：是"来源1说...来源2说..."（罗列），还是信息被融合成连贯段落（整合）？
3. 判断是否处理了多文档的矛盾或互补信息
4. 给出 1-4 分整数评分

## 重要提示
- 如果检索只提供了 1-2 篇相关文档，LLM 自然无法做多源综合 → 给 3 分作为基线
- 如果检索提供了多篇但 LLM 只用了一篇 → 应扣分
- 评估的是 LLM 的整合能力，不是检索结果的质量

## 输出格式（严格 JSON）
{"score": 3, "reason": "一句话简要说明评分理由"}
"""

SYNTH_USER = """请评估以下 AI 回答的整合质量。

【用户问题】
{query}

【AI 回答】
{answer}

【回答中引用的资料来源数量】
{source_count} 个
"""


def count_sources(answer: str) -> int:
    import re
    matches = re.findall(r'\[来源[:\s]*(\d+(?:[,，\s]*\d+)*)\]', answer)
    sources = set()
    for m in matches:
        for num in re.findall(r'\d+', m):
            sources.add(int(num))
    return len(sources)


def score_synthesis(entry: dict) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

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

    source_count = count_sources(entry["teacher_answer"])
    user_msg = SYNTH_USER.format(
        query=entry["query"],
        answer=entry["teacher_answer"],
        source_count=source_count,
    )

    response = llm.invoke([
        SystemMessage(content=SYNTH_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        return {"score": int(result.get("score", 0)), "reason": result.get("reason", raw)}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0, "reason": raw}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Synthesis quality sampling")
    parser.add_argument("--n", type=int, default=30, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info(f"Loading: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    logger.info(f"  Total: {len(data):,} entries")

    # 分层采样（每桶等比例）
    by_bucket = {"A": [], "B": [], "C": []}
    for e in data:
        b = e.get("bucket", "C")
        if b in by_bucket:
            by_bucket[b].append(e)

    random.seed(args.seed)
    n_per = max(1, args.n // 3)
    samples = []
    for b in ["A", "B", "C"]:
        pool = by_bucket[b]
        samples.extend(random.sample(pool, min(n_per, len(pool))))

    logger.info(f"Sampled {len(samples)} entries ({len(by_bucket['A'][:n_per])}A + "
                f"{len(by_bucket['B'][:n_per])}B + {len(by_bucket['C'][:n_per])}C)")
    logger.info("-" * 60)

    # 逐条评估
    scores = []
    score_dist = {1: 0, 2: 0, 3: 0, 4: 0}
    examples_by_score = {1: [], 2: [], 3: [], 4: []}

    for i, entry in enumerate(samples):
        try:
            result = score_synthesis(entry)
        except Exception as e:
            result = {"score": 0, "reason": str(e)}

        s = result["score"]
        reason = result["reason"]
        scores.append(s)

        if s in score_dist:
            score_dist[s] += 1

        bucket = entry.get("bucket", "?")
        source_count = count_sources(entry["teacher_answer"])
        logger.info(f"  [{i+1:2d}/{len(samples)}] qid={entry['qid']} bucket={bucket} "
                    f"sources={source_count} score={s}")

        if s in (1, 2) and len(examples_by_score[s]) < 3:
            examples_by_score[s].append({
                "qid": entry["qid"],
                "query": entry["query"][:60],
                "bucket": bucket,
                "sources": source_count,
                "score": s,
                "reason": reason,
                "answer_preview": entry["teacher_answer"][:250],
            })
        elif s in (3, 4) and len(examples_by_score[s]) < 2:
            examples_by_score[s].append({
                "qid": entry["qid"],
                "query": entry["query"][:60],
                "bucket": bucket,
                "sources": source_count,
                "score": s,
                "reason": reason,
                "answer_preview": entry["teacher_answer"][:200],
            })

    # 汇总
    avg = sum(s for s in scores if s > 0) / max(1, sum(1 for s in scores if s > 0))
    print()
    print("=" * 60)
    print("  SYNTHESIS QUALITY RESULTS")
    print("=" * 60)
    print(f"  Samples:         {len(samples)}")
    print(f"  Avg score:       {avg:.2f}")
    print(f"  Score distribution:")
    for s in [1, 2, 3, 4]:
        count = score_dist[s]
        bar = "#" * count
        pct = count / len(samples) * 100
        print(f"    {s}分:  {count:>3} ({pct:5.1f}%)  {bar}")
    print()
    print(f"  Low-quality (1-2): {score_dist[1]+score_dist[2]}/{len(samples)} "
          f"({(score_dist[1]+score_dist[2])/len(samples)*100:.1f}%)")
    print(f"  High-quality (3-4): {score_dist[3]+score_dist[4]}/{len(samples)} "
          f"({(score_dist[3]+score_dist[4])/len(samples)*100:.1f}%)")

    # 低分示例
    for s in [1, 2]:
        if examples_by_score[s]:
            print()
            print(f"--- Score={s} examples ---")
            for ex in examples_by_score[s]:
                print(f"\n  qid={ex['qid']} bucket={ex['bucket']} sources={ex['sources']}")
                print(f"  Query:  {ex['query']}")
                print(f"  Reason: {ex['reason'][:200]}")
                print(f"  Answer: {ex['answer_preview']}...")

    # 高分示例
    for s in [4, 3]:
        if examples_by_score[s]:
            print()
            print(f"--- Score={s} examples ---")
            for ex in examples_by_score[s]:
                print(f"\n  qid={ex['qid']} bucket={ex['bucket']} sources={ex['sources']}")
                print(f"  Query:  {ex['query']}")
                print(f"  Reason: {ex['reason'][:200]}")
                print(f"  Answer: {ex['answer_preview']}...")

    # 建议
    print()
    low_pct = (score_dist[1] + score_dist[2]) / len(samples) * 100
    if low_pct > 20:
        print("[WARN] Low-quality ratio > 20%, full-scale filtering recommended")
    elif low_pct > 10:
        print("[INFO] Low-quality ratio 10-20%, optionally filter score=1 only")
    else:
        print("[OK] Low-quality ratio < 10%, synthesis quality is good, no extra filtering needed")
    print()


if __name__ == "__main__":
    main()
