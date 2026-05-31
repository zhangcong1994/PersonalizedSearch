"""
Exp-009 Step 1: Query Augmentation (改写 + HyDE).

读抽样后的 query JSONL → 调用 LLM → 输出改写/HyDE JSONL。

复用 exp-007 中验证过的 P2 改写 prompt + H2 HyDE prompt。

输入格式:
  {"qid": "50000", "query": "如何...", "stratum": "T1", ...}

输出格式:
  改写: {"qid": "50000", "original": "如何...", "rewritten": "改写后文本"}
  HyDE: {"qid": "50000", "original": "如何...", "hyde": "HyDE 伪答案文本"}

缓存: 已生成的 qid 自动跳过（支持断点续跑）。

用法:
  # 本地 vLLM (Qwen3-4B)
  python scripts/exp009/run_query_augment.py \
      --backend vllm --llm-url http://localhost:8000/v1 \
      --input data/processed/exp009_sampled_queries.jsonl \
      --output-rw data/processed/exp009_rewritten_queries.jsonl \
      --output-hy data/processed/exp009_hyde_answers.jsonl

  # DeepSeek API (vLLM 部署失败时的降级方案)
  python scripts/exp009/run_query_augment.py \
      --backend deepseek \
      --input data/processed/exp009_sampled_queries.jsonl \
      --output-rw data/processed/exp009_rewritten_queries.jsonl \
      --output-hy data/processed/exp009_hyde_answers.jsonl
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Prompts（复用 exp-007 E2a-P2 / E2c-H2，已验证最优）──

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

HYDE_SYSTEM = """你是一个知识渊博的助手。请根据用户的问题，生成一段 100-150 字的回答。不需要保证完全准确，但应包含与该问题相关的关键概念、术语和背景信息。

要求：
1. 使用正式、信息丰富的语言
2. 包含与问题相关的专业术语和关键概念
3. 回答长度为 100-150 字
4. 就像你在写一个百科全书条目

用户问题: {query}
回答:"""


# ── I/O helpers ──────────────────────────────────────────

def load_queries_jsonl(path: Path) -> list[tuple[str, str]]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries.append((obj["qid"], obj["query"]))
    logger.info(f"Loaded {len(queries)} queries from {path.name}")
    return queries


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["qid"]] = obj
    return cache


def append_cache(path: Path, entry: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Quality filters ──────────────────────────────────────

def is_valid_rewrite(text: str) -> bool:
    text = text.strip()
    if not text or len(text) > 100:
        return False
    if sum(1 for c in text if '\u4e00' <= c <= '\u9fff') == 0:
        return False
    return True


def is_valid_hyde(text: str) -> bool:
    text = text.strip()
    if not text or len(text) < 10 or len(text) > 1500:
        return False
    return True


# ── LLM 调用（vLLM / DeepSeek API 双后端）──────────────

def _get_llm(backend: str, max_tokens: int, llm_url: str | None = None):
    from langchain_openai import ChatOpenAI

    if backend == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=0.1,
            max_tokens=max_tokens,
        )
    else:
        return ChatOpenAI(
            model="default",
            api_key="EMPTY",
            base_url=llm_url or "http://localhost:8000/v1",
            temperature=0,
            max_tokens=max_tokens,
        )


# ── 生成函数 ─────────────────────────────────────────────

def generate_rewrites(
    queries: list[tuple[str, str]],
    backend: str,
    llm_url: str | None,
    batch_size: int,
    output_path: Path,
):
    from langchain_core.prompts import ChatPromptTemplate

    cache = load_cache(output_path)
    cached_qids = set(cache.keys())
    missing = [(qid, text) for qid, text in queries if qid not in cached_qids]

    if not missing:
        logger.info(f"All {len(queries)} rewrite queries cached (skipping)")
        return

    logger.info(f"Rewrite: {len(missing)}/{len(queries)} to generate ({len(cached_qids)} cached)")

    llm = _get_llm(backend, max_tokens=128, llm_url=llm_url)
    chain = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM),
        ("human", REWRITE_HUMAN),
    ]) | llm

    ok = filt = fail = 0
    t_start = time.time()

    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        inputs = [{"query": text} for _, text in chunk]
        results = chain.batch(inputs, config={"max_concurrency": batch_size},
                              return_exceptions=True)

        for (qid, original), result in zip(chunk, results):
            try:
                if isinstance(result, Exception):
                    raise result
                raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
                rewritten = raw.replace("\n", " ").strip()
            except Exception as e:
                logger.warning(f"Rewrite failed for qid={qid}: {e}")
                rewritten = original
                fail += 1
                entry = {"qid": qid, "original": original, "rewritten": rewritten}
                cache[qid] = entry
                append_cache(output_path, entry)
                continue

            if is_valid_rewrite(rewritten):
                ok += 1
            else:
                rewritten = original
                filt += 1

            entry = {"qid": qid, "original": original, "rewritten": rewritten}
            cache[qid] = entry
            append_cache(output_path, entry)

        processed = i + len(chunk)
        if processed % (batch_size * 10) == 0 or processed >= len(missing):
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 0.1)
            eta = (len(missing) - processed) / max(rate, 0.01) / 60
            logger.info(f"  Rewrite: {processed}/{len(missing)} | "
                        f"ok={ok} filt={filt} fail={fail} | "
                        f"{rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"Rewrite done in {elapsed/60:.1f}min: ok={ok} filtered={filt} failed={fail}")


def generate_hyde(
    queries: list[tuple[str, str]],
    backend: str,
    llm_url: str | None,
    batch_size: int,
    output_path: Path,
):
    from langchain_core.prompts import ChatPromptTemplate

    cache = load_cache(output_path)
    cached_qids = set(cache.keys())
    missing = [(qid, text) for qid, text in queries if qid not in cached_qids]

    if not missing:
        logger.info(f"All {len(queries)} HyDE answers cached (skipping)")
        return

    logger.info(f"HyDE: {len(missing)}/{len(queries)} to generate ({len(cached_qids)} cached)")

    llm = _get_llm(backend, max_tokens=512, llm_url=llm_url)
    chain = ChatPromptTemplate.from_messages([("system", HYDE_SYSTEM)]) | llm

    ok = filt = fail = 0
    t_start = time.time()

    for i in range(0, len(missing), batch_size):
        chunk = missing[i:i + batch_size]
        inputs = [{"query": text} for _, text in chunk]
        results = chain.batch(inputs, config={"max_concurrency": batch_size},
                              return_exceptions=True)

        for (qid, original), result in zip(chunk, results):
            try:
                if isinstance(result, Exception):
                    raise result
                raw = result.content.strip() if hasattr(result, "content") else str(result).strip()
                hyde_text = raw.replace("\n", " ").strip()
            except Exception as e:
                logger.warning(f"HyDE failed for qid={qid}: {e}")
                hyde_text = original
                fail += 1
                entry = {"qid": qid, "original": original, "hyde": hyde_text}
                cache[qid] = entry
                append_cache(output_path, entry)
                continue

            if is_valid_hyde(hyde_text):
                ok += 1
            else:
                hyde_text = original
                filt += 1

            entry = {"qid": qid, "original": original, "hyde": hyde_text}
            cache[qid] = entry
            append_cache(output_path, entry)

        processed = i + len(chunk)
        if processed % (batch_size * 10) == 0 or processed >= len(missing):
            elapsed = time.time() - t_start
            rate = processed / max(elapsed, 0.1)
            eta = (len(missing) - processed) / max(rate, 0.01) / 60
            logger.info(f"  HyDE:    {processed}/{len(missing)} | "
                        f"ok={ok} filt={filt} fail={fail} | "
                        f"{rate:.1f}q/s | ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"HyDE done in {elapsed/60:.1f}min: ok={ok} filtered={filt} failed={fail}")


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 Step 1: Query augmentation (rewrite + HyDE)"
    )
    parser.add_argument("--backend", choices=["vllm", "deepseek"], default="vllm",
                        help="LLM backend: vllm (local Qwen3-4B) or deepseek (API)")
    parser.add_argument("--input", required=True,
                        help="Sampled queries JSONL (qid, query, stratum, ...)")
    parser.add_argument("--output-rw", required=True,
                        help="Output path for rewritten queries JSONL")
    parser.add_argument("--output-hy", required=True,
                        help="Output path for HyDE answers JSONL")
    parser.add_argument("--llm-url", default=None,
                        help="vLLM OpenAI-compatible endpoint (required when --backend vllm)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--task", choices=["rewrite", "hyde", "both"], default="both",
                        help="Which task to run (default: both)")
    args = parser.parse_args()

    if args.backend == "vllm" and not args.llm_url:
        parser.error("--llm-url is required when --backend vllm")

    input_path = Path(args.input)
    output_rw = Path(args.output_rw)
    output_hy = Path(args.output_hy)

    from src.utils.config import DATA_ROOT
    if not input_path.is_absolute():
        input_path = DATA_ROOT / input_path
    if not output_rw.is_absolute():
        output_rw = DATA_ROOT / output_rw
    if not output_hy.is_absolute():
        output_hy = DATA_ROOT / output_hy

    logger.info("=" * 60)
    logger.info("  Exp-009 Step 1: Query Augmentation")
    logger.info(f"  Backend:  {args.backend}")
    logger.info(f"  Input:    {input_path}")
    logger.info(f"  OutputRW: {output_rw}")
    logger.info(f"  OutputHY: {output_hy}")
    if args.backend == "vllm":
        logger.info(f"  LLM URL:  {args.llm_url}")
    logger.info(f"  Task:     {args.task}")
    logger.info("=" * 60)

    queries = load_queries_jsonl(input_path)
    if not queries:
        logger.error("No queries loaded")
        return 1

    if args.task in ("rewrite", "both"):
        logger.info("-" * 40)
        generate_rewrites(queries, args.backend, args.llm_url, args.batch_size, output_rw)
        rw_lines = sum(1 for _ in open(output_rw, "r", encoding="utf-8")) if output_rw.exists() else 0
        logger.info(f"Rewrite cache: {rw_lines} entries → {output_rw}")

    if args.task in ("hyde", "both"):
        logger.info("-" * 40)
        generate_hyde(queries, args.backend, args.llm_url, args.batch_size, output_hy)
        hy_lines = sum(1 for _ in open(output_hy, "r", encoding="utf-8")) if output_hy.exists() else 0
        logger.info(f"HyDE cache:    {hy_lines} entries → {output_hy}")

    logger.info("=" * 60)
    logger.info("  Step 1 complete")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
