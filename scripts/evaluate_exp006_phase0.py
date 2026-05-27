"""
Exp-006 Phase 0: Manual Gap Analysis Validation

Pick a few zero-hit queries, call DeepSeek thinking mode to do gap analysis,
and print the full prompt + response for human inspection.

Usage:
  python scripts/evaluate_exp006_phase0.py --n 3
  python scripts/evaluate_exp006_phase0.py --n 3 --no-api  # just print prompts
  python scripts/evaluate_exp006_phase0.py --qids 68,155   # specific QIDs
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

os.environ["HF_HUB_OFFLINE"] = "1"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.evaluation.data_loader import clean_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP003_RESULTS = PROJECT_ROOT / "results" / "exp003" / "exp003_test_S4_K50_RRFk60.jsonl"
COLLECTION_FILE = PROJECT_ROOT / "data" / "raw" / "t2ranking" / "collection.tsv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp006"

GAP_ANALYSIS_SYSTEM = """你是一个搜索失败诊断专家。用户提出查询后，搜索引擎返回了第一轮的 Top-10 段落。
请分析为什么这些结果可能没有满足用户的信息需求，并生成第二轮检索的改写查询。

## 分析维度
逐项判断是否存在以下问题：
1. **compound_split**（复合词拆分）：查询中的词组是否被错误拆分为单字/单词？
2. **domain_misalignment**（领域偏离）：检索结果是否跑到了完全不相关的领域？
3. **entity_rarity**（冷门专名）：查询中是否存在极低频的专有名词、缩写、俗称？
4. **granularity_mismatch**（粒度不匹配）：检索结果主题接近，但不够精确？

## 改写策略
- **冷门专名**：不要只写专名本身，在其周围补充领域词、类别词
  （如 "司鱼" → "司鱼交友软件 社交APP 用户评价 下载"）
- **被拆分的复合词**：在改写中保持词组的完整形式
- **领域偏离**：在改写中加入领域限定词或排除噪音词
- 生成 1 条改写查询（不要多条）

## 输出格式
仅输出一个 JSON 对象，不要包含 markdown 代码块标记：

{
  "diagnosis": {
    "compound_split": false,
    "domain_misalignment": false,
    "entity_rarity": false,
    "granularity_mismatch": false,
    "summary": "一句话诊断失败原因"
  },
  "reformulated_query": "改写查询（完整语义，补充领域词和消歧上下文，20-50字）",
  "negative_signals": ["检索时应避开的噪音词"]
}"""


def load_exp003_results(path: Path) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if "__meta__" in data:
                continue
            results.append(data)
    return results


def load_collection_texts(pids: set[str], collection_path: Path) -> dict[str, str]:
    pid_to_text: dict[str, str] = {}
    with open(collection_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            if pid not in pids:
                continue
            text = clean_text(parts[1])
            if len(text) > 500:
                text = text[:500]
            pid_to_text[pid] = text
            if len(pid_to_text) >= len(pids):
                break
    return pid_to_text


def find_zero_hit_queries(results: list[dict]) -> list[dict]:
    rrf_key = "rrf@k60_perK50"
    zero_hit = []
    for e in results:
        relevant = set(e.get("relevant_pids", []))
        rrf_pids = {item["pid"] for item in e.get(rrf_key, [])[:50]}
        if not (relevant & rrf_pids):
            zero_hit.append(e)
    return zero_hit


def build_user_prompt(query: str, top10_passages: list[tuple[str, str]]) -> str:
    passage_block = ""
    for i, (pid, text) in enumerate(top10_passages):
        passage_block += f"[{i + 1}] pid={pid}\n{text}\n\n"

    return f"""## 用户查询
{query}

## 第一轮检索 Top-10 段落
{passage_block}
## 任务
请分析检索缺口，生成改写查询。直接输出 JSON："""


def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    elif raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"_parse_error": True, "_raw": raw[:500]}


def main():
    parser = argparse.ArgumentParser(description="Exp-006 Phase 0: Manual Gap Analysis")
    parser.add_argument("--n", type=int, default=3, help="Number of zero-hit queries to sample")
    parser.add_argument("--qids", type=str, default=None, help="Comma-separated QIDs to analyze")
    parser.add_argument("--no-api", action="store_true", help="Skip API call, just print prompt")
    parser.add_argument("--model", default="deepseek-chat", help="DeepSeek model")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key")
    parser.add_argument("--thinking", action="store_true", help="Use deepseek-reasoner (thinking mode)")
    parser.add_argument("--full-text", action="store_true", help="Print full passage texts (not truncated)")
    args = parser.parse_args()

    logger.info("Loading data...")
    all_results = load_exp003_results(EXP003_RESULTS)
    zero_hit = find_zero_hit_queries(all_results)
    logger.info(f"Zero-hit queries: {len(zero_hit)} / {len(all_results)}")

    if args.qids:
        target_qids = set(args.qids.split(","))
        selected = [e for e in zero_hit if e["qid"] in target_qids]
    else:
        import random
        random.seed(42)
        selected = random.sample(zero_hit, min(args.n, len(zero_hit)))

    logger.info(f"Selected {len(selected)} queries for analysis")

    needed_pids: set[str] = set()
    for e in selected:
        for item in e.get("rrf@k60_perK50", [])[:10]:
            needed_pids.add(item["pid"])
    pid_to_text = load_collection_texts(needed_pids, COLLECTION_FILE)
    logger.info(f"Loaded {len(pid_to_text)} passage texts")

    api_client = None
    if not args.no_api:
        api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("DEEPSEEK_API_KEY not set")
            sys.exit(1)
        model = "deepseek-reasoner" if args.thinking else args.model
        api_client = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            max_tokens=1024,
            temperature=0.1,
        )
        logger.info(f"API client initialized: {model}")

    for idx, entry in enumerate(selected):
        qid = entry["qid"]
        query = entry["query"]
        relevant = entry.get("relevant_pids", [])

        top10 = []
        for item in entry.get("rrf@k60_perK50", [])[:10]:
            pid = item["pid"]
            text = pid_to_text.get(pid, "(text not found)")
            top10.append((pid, text))

        print()
        print("=" * 80)
        print(f"  [{idx + 1}/{len(selected)}]  QID: {qid}")
        print(f"  Query: {query}")
        print(f"  Qrels count: {len(relevant)}")
        print("=" * 80)

        print()
        print("--- Round 1 Top-10 Passages ---")
        for i, (pid, text) in enumerate(top10):
            display = text if args.full_text else (text[:200] + "..." if len(text) > 200 else text)
            print(f"  [{i + 1}] pid={pid}: {display}")

        print()
        print("--- Relevant Passages (qrels) ---")
        for pid in relevant:
            text = pid_to_text.get(pid, "(text not found)")
            display = text if args.full_text else (text[:200] + "..." if len(text) > 200 else text)
            print(f"  pid={pid}: {display}")

        if api_client:
            user_prompt = build_user_prompt(query, top10)
            messages = [
                SystemMessage(content=GAP_ANALYSIS_SYSTEM),
                HumanMessage(content=user_prompt),
            ]

            print()
            print("--- Calling DeepSeek API ---")
            try:
                response = api_client.invoke(messages)
                raw = response.content.strip()
                print(f"\n  Raw response ({len(raw)} chars):\n{raw}\n")

                parsed = parse_response(raw)
                print("  Parsed JSON:")
                print(json.dumps(parsed, ensure_ascii=False, indent=2))

            except Exception as e:
                logger.error(f"API error: {e}")
        else:
            print()
            print("--- Prompt (would send to LLM) ---")
            user_prompt = build_user_prompt(query, top10)
            print(f"\n[SYSTEM]\n{GAP_ANALYSIS_SYSTEM}\n")
            print(f"[USER]\n{user_prompt}")

    # Save results
    output_path = OUTPUT_DIR / "phase0_gap_analysis.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_data = []
    for e in selected:
        save_data.append({
            "qid": e["qid"],
            "query": e["query"],
            "relevant_pids": e.get("relevant_pids", []),
            "rrf_top10_pids": [item["pid"] for item in e.get("rrf@k60_perK50", [])[:10]],
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved query list to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
