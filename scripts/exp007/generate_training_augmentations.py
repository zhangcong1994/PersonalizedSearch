"""
Phase 2: Query Rewriting + HyDE Pseudo-Answer Generation

Generates rewritten queries and HyDE pseudo-answers for all training queries
using local vLLM (OpenAI-compatible API). Results are cached to JSONL files
for idempotent re-runs and consumed by merge_training_data_phase2.py.

Prompt sources:
  - Rewrite: adapted from exp-002 E2a-P2 (domain few-shot), with one change:
    removed ">15字直接返回原查询" because in multi-route RRF (exp-003 S4),
    the original query already has its own route (D-B0). Rewrite must always
    produce a different formulation.
  - HyDE:    identical to exp-002 E2c-H2 (encyclopedia-style 100-150 word answer)

Quality filter: only format checks (empty, too long, no Chinese), no semantic
filtering — training distribution must match online inference distribution.

Usage:
  # Quick test with 10 queries (dry-run)
  python scripts/exp007/generate_training_augmentations.py --sample 10

  # Full generation on server (vLLM must be running on port 8000)
  python scripts/exp007/generate_training_augmentations.py --llm-url http://localhost:8000/v1

  # Only rewrite queries
  python scripts/exp007/generate_training_augmentations.py --task rewrite

  # Only HyDE
  python scripts/exp007/generate_training_augmentations.py --task hyde

Output:
  {DATA_ROOT}/data/processed/exp007_rewritten_queries.jsonl
  {DATA_ROOT}/data/processed/exp007_hyde_answers.jsonl

  Each line: {"qid": "xxx", "original": "原始查询", "rewritten": "改写查询"}
  Each line: {"qid": "xxx", "original": "原始查询", "hyde": "假答案文本"}
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QUERIES_TRAIN_FILE = T2RANKING_DIR / "queries.train.tsv"

OUTPUT_DIR = DATA_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REWRITE_OUTPUT = OUTPUT_DIR / "exp007_rewritten_queries.jsonl"
HYDE_OUTPUT = OUTPUT_DIR / "exp007_hyde_answers.jsonl"

# ── Prompts ──────────────────────────────────────────────

# Adapted from exp-002 E2a-P2. Key change: removed ">15字直接返回原查询"
# because original query already has its own route (D-B0) in multi-route RRF.
REWRITE_SYSTEM = """你是一个搜索查询优化器。用户输入的是真实搜索引擎中的短查询，请将其扩展为更完整、更具体的搜索词，以提高信息检索的召回率。

规则：
1. 缩写、口语化、省略了关键信息的，补全为完整的疑问句或陈述句
2. 补充正式术语和同义词，扩展核心概念词
3. 保持改写后的查询为一行纯文本，不要加引号、编号、解释
4. 不编造查询中没有的信息
5. 始终输出改写版本——原查询已有单独的检索通道，改写必须产生不同的表述

示例：
原始: 为什么脸上老长痘痘
改写: 面部反复长痘的原因 痤疮成因 皮肤护理方法

原始: 怎么看电脑配置
改写: 如何查看电脑硬件配置 查看CPU型号内存大小显卡型号方法

原始: 苹果和安卓哪个好用
改写: 苹果iOS和安卓Android系统优缺点对比 适用人群

原始: 什么是区块链
改写: 区块链技术定义 分布式账本原理 去中心化特点

原始: 中国未来经济发展趋势
改写: 中国经济发展趋势 未来增长动力 产业结构转型

原始: 武汉有什么好玩的
改写: 武汉旅游景点推荐 武汉必去好玩的地方

原始: 怎么减肥最快
改写: 快速减肥方法 科学减重饮食运动计划

原始: 5G和4G有什么区别
改写: 5G和4G的区别 网速延迟应用场景对比"""

REWRITE_HUMAN = "原始: {query}\n改写:"

# Identical to exp-002 E2c-H1 / E2c-H2
HYDE_SYSTEM = """你是一个知识渊博的助手。请根据用户的问题，生成一段 100-150 字的回答。不需要保证完全准确，但应包含与该问题相关的关键概念、术语和背景信息。

要求：
1. 使用正式、信息丰富的语言
2. 包含与问题相关的专业术语和关键概念
3. 回答长度为 100-150 字
4. 就像你在写一个百科全书条目

用户问题: {query}
回答:"""


# ── Quality filters (format only, no semantic filtering) ──

def is_valid_rewrite(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"
    text = text.strip()
    if len(text) > 100:
        return False, f"too_long({len(text)}chars>100)"
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if chinese_chars == 0:
        return False, "no_chinese"
    return True, "ok"


def is_valid_hyde(text: str) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "empty"
    text = text.strip()
    if len(text) < 10:
        return False, f"too_short({len(text)}chars<10)"
    if len(text) > 1500:
        return False, f"too_long({len(text)}chars>1500)"
    return True, "ok"


# ── Cache helpers ────────────────────────────────────────

def _load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            result[obj["qid"]] = obj
    return result


def _append_jsonl(path: Path, entry: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Query loading ────────────────────────────────────────

def load_train_queries(sample: int = 0) -> list[tuple[str, str]]:
    queries = []
    with open(QUERIES_TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                queries.append((parts[0].strip(), parts[1].strip()))
    if sample > 0 and sample < len(queries):
        queries = queries[:sample]
    logger.info(f"Loaded {len(queries)} training queries" + (" (sampled)" if sample > 0 else ""))
    return queries


# ── LLM generation ───────────────────────────────────────

def _get_llm(llm_url: str, max_tokens: int):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="default",
        api_key="EMPTY",
        base_url=llm_url,
        temperature=0,
        max_tokens=max_tokens,
    )


def generate_rewrites(
    queries: list[tuple[str, str]],
    llm_url: str,
    batch_size: int,
    output_path: Path = REWRITE_OUTPUT,
) -> dict[str, str]:
    from langchain_core.prompts import ChatPromptTemplate

    cache = _load_jsonl(output_path)
    cached_qids = set(cache.keys())
    missing = [(qid, text) for qid, text in queries if qid not in cached_qids]

    if not missing:
        logger.info(f"All {len(queries)} rewrite queries cached")
        return {qid: entry.get("rewritten", entry.get("output", entry["original"]))
                for qid, entry in cache.items()}

    logger.info(f"Rewrite: {len(missing)}/{len(queries)} need generation "
                f"({len(cached_qids)} cached)")

    llm = _get_llm(llm_url, max_tokens=128)
    chain = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM),
        ("human", REWRITE_HUMAN),
    ]) | llm

    success = 0
    filtered = 0
    failed = 0
    t_start = time.time()

    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        inputs = [{"query": text} for _, text in chunk]
        results = chain.batch(inputs, config={"max_concurrency": batch_size}, return_exceptions=True)

        for (qid, original), result in zip(chunk, results):
            try:
                if isinstance(result, Exception):
                    raise result
                raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
                rewritten = raw.replace("\n", " ").strip()
            except Exception as e:
                logger.warning(f"Rewrite LLM failed for qid={qid}: {e}")
                rewritten = original
                failed += 1
                entry = {"qid": qid, "original": original, "rewritten": rewritten}
                cache[qid] = entry
                _append_jsonl(output_path, entry)
                continue

            valid, reason = is_valid_rewrite(rewritten)
            if not valid:
                logger.debug(f"Rewrite filtered qid={qid}: {reason}")
                rewritten = original
                filtered += 1
            else:
                success += 1

            entry = {"qid": qid, "original": original, "rewritten": rewritten}
            cache[qid] = entry
            _append_jsonl(output_path, entry)

        processed = i + len(chunk)
        if processed % (batch_size * 10) == 0 or processed >= len(missing):
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 0.1)
            eta = (len(missing) - processed) / max(rate, 0.01) / 60
            logger.info(f"  Rewrite: {processed}/{len(missing)} | "
                        f"ok={success} filt={filtered} fail={failed} | "
                        f"{rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"Rewrite done in {elapsed/60:.1f}min: success={success} filtered={filtered} failed={failed}")
    return {qid: entry["rewritten"] for qid, entry in cache.items()}


def generate_hyde(
    queries: list[tuple[str, str]],
    llm_url: str,
    batch_size: int,
    output_path: Path = HYDE_OUTPUT,
) -> dict[str, str]:
    from langchain_core.prompts import ChatPromptTemplate

    cache = _load_jsonl(output_path)
    cached_qids = set(cache.keys())
    missing = [(qid, text) for qid, text in queries if qid not in cached_qids]

    if not missing:
        logger.info(f"All {len(queries)} HyDE answers cached")
        return {qid: entry.get("hyde", entry.get("output", entry["original"]))
                for qid, entry in cache.items()}

    logger.info(f"HyDE: {len(missing)}/{len(queries)} need generation "
                f"({len(cached_qids)} cached)")

    llm = _get_llm(llm_url, max_tokens=512)
    chain = ChatPromptTemplate.from_messages([("system", HYDE_SYSTEM)]) | llm

    success = 0
    filtered = 0
    failed = 0
    t_start = time.time()

    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        inputs = [{"query": text} for _, text in chunk]
        results = chain.batch(inputs, config={"max_concurrency": batch_size}, return_exceptions=True)

        for (qid, original), result in zip(chunk, results):
            try:
                if isinstance(result, Exception):
                    raise result
                raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
                hyde_text = raw.replace("\n", " ").strip()
            except Exception as e:
                logger.warning(f"HyDE LLM failed for qid={qid}: {e}")
                hyde_text = original
                failed += 1
                entry = {"qid": qid, "original": original, "hyde": hyde_text}
                cache[qid] = entry
                _append_jsonl(output_path, entry)
                continue

            valid, reason = is_valid_hyde(hyde_text)
            if not valid:
                logger.debug(f"HyDE filtered qid={qid}: {reason}")
                hyde_text = original
                filtered += 1
            else:
                success += 1

            entry = {"qid": qid, "original": original, "hyde": hyde_text}
            cache[qid] = entry
            _append_jsonl(output_path, entry)

        processed = i + len(chunk)
        if processed % (batch_size * 10) == 0 or processed >= len(missing):
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 0.1)
            eta = (len(missing) - processed) / max(rate, 0.01) / 60
            logger.info(f"  HyDE:    {processed}/{len(missing)} | "
                        f"ok={success} filt={filtered} fail={failed} | "
                        f"{rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"HyDE done in {elapsed/60:.1f}min: success={success} filtered={filtered} failed={failed}")
    return {qid: entry["hyde"] for qid, entry in cache.items()}


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate rewritten queries and HyDE pseudo-answers"
    )
    parser.add_argument(
        "--task", choices=["rewrite", "hyde", "both"], default="both",
        help="Which generation task to run (default: both)",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help="Sample N queries (0 = all). Use for dry-run testing.",
    )
    parser.add_argument(
        "--llm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible endpoint (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for LLM inference (default: 32)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  PHASE 2: Training Data Augmentation")
    logger.info("  Tasks:    " + args.task)
    logger.info("  LLM URL:  " + args.llm_url)
    logger.info("  Sample:   " + (str(args.sample) if args.sample else "all (258K)"))
    logger.info("=" * 60)

    queries = load_train_queries(sample=args.sample)
    total_queries = len(queries)

    if not queries:
        logger.error("No queries loaded")
        return 1

    if args.task in ("rewrite", "both"):
        logger.info("-" * 40)
        logger.info(f"  Generating rewritten queries for {total_queries} queries...")
        _ = generate_rewrites(queries, args.llm_url, args.batch_size)
        logger.info(f"  Output: {REWRITE_OUTPUT}")

    if args.task in ("hyde", "both"):
        logger.info("-" * 40)
        logger.info(f"  Generating HyDE answers for {total_queries} queries...")
        _ = generate_hyde(queries, args.llm_url, args.batch_size)
        logger.info(f"  Output: {HYDE_OUTPUT}")

    logger.info("=" * 60)
    if args.task in ("rewrite", "both"):
        n = sum(1 for _ in open(REWRITE_OUTPUT, "r", encoding="utf-8"))
        logger.info(f"  Rewrite cache: {n} entries → {REWRITE_OUTPUT}")
    if args.task in ("hyde", "both"):
        n = sum(1 for _ in open(HYDE_OUTPUT, "r", encoding="utf-8"))
        logger.info(f"  HyDE cache:    {n} entries → {HYDE_OUTPUT}")
    logger.info("=" * 60)
    logger.info("  Next: python scripts/exp007/merge_training_data_phase2.py")
    logger.info("        python scripts/exp007/train_embedding_phase2.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
