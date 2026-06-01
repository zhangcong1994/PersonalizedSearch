"""
Exp-009 阶段六：类别构造（4 个子阶段）。

流程:
  python scripts/exp009/construct_categories.py select                     # 6.1 筛选标准+拒答
  python scripts/exp009/construct_categories.py detect-contradictions      # 6.2 矛盾检测
  python scripts/exp009/construct_categories.py rewrite --workers 8        # 6.3 deepseek-chat 并发改写
  python scripts/exp009/construct_categories.py assemble                   # 6.4 组装切分

5 类 SFT 数据：标准搜索问答(55%) + 引文强调(12%) + 信息不足/拒答(17%) + 噪声干扰(8%) + 矛盾处理(8%)
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT, RAW_DATA_DIR
from src.generation.prompts import SYSTEM_PROMPT, FEW_SHOT, CONTEXT_TEMPLATE, QUESTION_TEMPLATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────

T2RANKING_DIR = RAW_DATA_DIR / "t2ranking"
QRELS_RETRIEVAL_FILE = T2RANKING_DIR / "qrels.retrieval.train.tsv"
COLLECTION_FILE = T2RANKING_DIR / "collection.tsv"

INPUT_FILE = DATA_ROOT / "data" / "processed" / "exp009_filtered_bucketed.jsonl"
STATE_FILE = DATA_ROOT / "data" / "processed" / "exp009_category_state.json"
STAGE61_DIR = DATA_ROOT / "data" / "processed" / "exp009_stage61"

REWRITE_OUTPUT = DATA_ROOT / "data" / "processed" / "exp009_rewritten.jsonl"

TRAIN_OUTPUT = DATA_ROOT / "data" / "processed" / "exp009_sft_train.jsonl"
VAL_OUTPUT = DATA_ROOT / "data" / "processed" / "exp009_sft_val.jsonl"

# 6.1 分配比例（基于实际可用数据 3589 条：A=1698 B=679 C=1212）
# 2020 target (standard+val+citation) > 1698 available A → 按余量调整
N_STANDARD = 1200
N_VAL = 200
N_REFUSAL = 460
N_CITATION = 300
N_NOISE = 220
N_CONTRADICTION = 220

VAL_SEED = 999
SAMPLING_SEED = 42

# ── 拒答关键词（同 filter_and_bucket.py）──────────────────

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


# ── 数据加载 (shared) ────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


def is_refusal_kw(answer: str) -> bool:
    return any(kw in answer for kw in REFUSAL_KW)


def get_top10_pids(entry: dict) -> list[str]:
    return [p["pid"] for p in entry.get("passages", [])]


def count_relevant_hits(entry: dict, retrieval_qrels: dict[str, set[str]]) -> int:
    qid = entry["qid"]
    relevant = retrieval_qrels.get(qid, set())
    top10 = get_top10_pids(entry)
    return len([pid for pid in top10 if pid in relevant])


# ── Prompt builders (shared) ──────────────────────────────

SYSTEM_BASE = SYSTEM_PROMPT + FEW_SHOT


def format_passage(idx: int, pid: str, text: str) -> str:
    return f"[{idx}] 来源: {pid}\n{text}"


def build_user_message(query: str, passages: list[dict]) -> str:
    lines = [format_passage(i + 1, p["pid"], p["text"]) for i, p in enumerate(passages)]
    return CONTEXT_TEMPLATE.format(context="\n\n".join(lines)) + "\n\n" + QUESTION_TEMPLATE.format(question=query)




# ═══════════════════════════════════════════════════════════
# 6.1 — 筛选分类
# ═══════════════════════════════════════════════════════════

CITATION_SYSTEM = (
    "你是一个 AI 搜索助手。你会收到用户的提问和若干篇相关的参考资料（从网络来源检索）。"
    "请根据这些资料生成一个高质量的搜索答案。\n"
    "\n"
    "核心原则：\n"
    "1. 忠于资料 — 只在资料支持的情况下给出回答\n"
    "2. 综合多篇资料 — 融合多篇资料的信息为一个连贯的整体\n"
    "3. 严格引用来源 — 每一个关键事实性陈述后面都必须标注来源编号，如 [来源: 3]。"
    "同一句话引用多个来源时标注 [来源: 1, 5]。不得遗漏任何一个来源引用\n"
    "4. 诚实面对不足 — 如果所有资料都无法回答用户问题，明确说明\n"
    "\n"
    "回答结构：\n"
    "- 先给出核心答案或结论（1-3 句话说清）\n"
    "- 再根据需要展开。每个事实陈述都要带来源标注\n"
    "- 语言简洁直接\n"
    "\n"
    + FEW_SHOT
)

NOISE_SYSTEM = (
    "你是一个 AI 搜索助手。你会收到用户的提问和若干篇参考资料（从网络来源检索）。\n"
    "\n"
    "重要提醒：部分参考资料可能与用户问题完全无关。请仔细判断每篇资料的相关性。\n"
    "\n"
    "核心原则：\n"
    "1. 忠于相关资料 — 只在相关且可靠资料支持的情况下给出回答。资料里没有的信息不要编造\n"
    "2. 忽略无关信息 — 对与用户问题无关的资料，直接忽略，不要受其干扰\n"
    "3. 综合多篇资料 — 将相关信息融合成一个连贯的整体\n"
    "4. 引用来源 — 关键事实性陈述后面标注来源编号，如 [来源: 3]\n"
    "5. 诚实面对不足 — 如果所有资料都无法回答用户问题，明确说明\n"
    "\n"
    "回答结构：\n"
    "- 先给出核心答案或结论\n"
    "- 再根据需要展开\n"
    "- 语言简洁直接\n"
    "\n"
    + FEW_SHOT
)

CONTRADICTION_SYSTEM = (
    "你是一个 AI 搜索助手。你会收到用户的提问和若干篇相关的参考资料（从网络来源检索）。\n"
    "\n"
    "核心原则：\n"
    "1. 忠于资料 — 只在资料支持的情况下给出回答\n"
    "2. 对比分析差异 — 如果多篇资料存在分歧或矛盾，请明确指出各方的说法，分析可能的原因\n"
    "3. 给出综合判断 — 在对比分析后，给出最可靠的结论或建议，并说明你的判断依据\n"
    "4. 引用来源 — 关键事实性陈述后面标注来源编号，如 [来源: 3]\n"
    "5. 诚实面对不足 — 如果资料不足以判断哪方正确，请诚实说明\n"
    "\n"
    "回答结构：\n"
    "- 先给出综合结论\n"
    "- 再列出各方说法（标注来源和分歧点）\n"
    "- 最后给出你的分析判断\n"
    "- 语言简洁直接\n"
    "\n"
    + FEW_SHOT
)


def cmd_select(args):
    logger.info("=" * 60)
    logger.info("  6.1: Select standard + refusal entries")
    logger.info("=" * 60)

    logger.info("Loading filtered data...")
    entries = load_jsonl(INPUT_FILE)
    logger.info(f"  Total: {len(entries):,} entries")

    logger.info("Loading retrieval qrels...")
    retrieval_qrels = load_retrieval_qrels(QRELS_RETRIEVAL_FILE)

    # 按 bucket 分组
    by_bucket: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    for e in entries:
        b = e.get("bucket", "C")
        by_bucket[b].append(e)

    logger.info(f"  Bucket A: {len(by_bucket['A']):,}")
    logger.info(f"  Bucket B: {len(by_bucket['B']):,}")
    logger.info(f"  Bucket C: {len(by_bucket['C']):,}")

    # 桶 A 中 hallu=PASS 的作为标准类的候选
    bucket_a_pass = [e for e in by_bucket["A"] if e.get("_hallu_verdict", "PASS") == "PASS"]
    logger.info(f"  Bucket A + hallu=PASS: {len(bucket_a_pass):,}")

    random.seed(SAMPLING_SEED)
    random.shuffle(bucket_a_pass)

    # 标准类：从桶 A 采 N_STANDARD + N_VAL
    n_avail = min(len(bucket_a_pass), N_STANDARD + N_VAL)
    standard_entries = bucket_a_pass[:n_avail]
    citation_pool = [e for e in bucket_a_pass[n_avail:] if e.get("_hallu_verdict", "PASS") == "PASS"]

    logger.info(f"  Standard (inc. val): {len(standard_entries):,}")
    logger.info(f"  Citation pool:       {len(citation_pool):,}")

    # 拒答类：桶 C 中"该拒答的"——含拒答关键词 + 检索确实无相关文档
    refusal_entries = []
    for e in by_bucket["C"]:
        answer = e.get("teacher_answer", "")
        hits = count_relevant_hits(e, retrieval_qrels)
        if is_refusal_kw(answer) and hits == 0:
            refusal_entries.append(e)

    if len(refusal_entries) > N_REFUSAL:
        random.seed(SAMPLING_SEED)
        refusal_entries = random.sample(refusal_entries, N_REFUSAL)

    logger.info(f"  Refusal (correct):   {len(refusal_entries):,}")

    # 噪声类候选：桶 B
    noise_pool = by_bucket["B"]
    logger.info(f"  Noise pool (bucket B): {len(noise_pool):,}")

    # 矛盾检测候选：桶 A 剩余（citation_pool）+ 桶 B（noise_pool）
    contradiction_pool = citation_pool + noise_pool
    logger.info(f"  Contradiction pool:   {len(contradiction_pool):,}")

    # 保存
    STAGE61_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(standard_entries, STAGE61_DIR / "standard.jsonl")
    save_jsonl(refusal_entries, STAGE61_DIR / "refusal.jsonl")
    save_jsonl(citation_pool, STAGE61_DIR / "citation_pool.jsonl")
    save_jsonl(noise_pool, STAGE61_DIR / "noise_pool.jsonl")
    save_jsonl(contradiction_pool, STAGE61_DIR / "contradiction_pool.jsonl")

    # 状态
    state = {
        "stage": "select_done",
        "counts": {
            "standard": len(standard_entries),
            "refusal": len(refusal_entries),
            "citation_pool": len(citation_pool),
            "noise_pool": len(noise_pool),
            "contradiction_pool": len(contradiction_pool),
        },
        "targets": {
            "standard": N_STANDARD,
            "refusal": N_REFUSAL,
            "citation": N_CITATION,
            "noise": N_NOISE,
            "contradiction": N_CONTRADICTION,
            "val": N_VAL,
        },
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info("-" * 60)
    logger.info("  SELECT SUMMARY")
    logger.info(f"  Standard (train+val): {len(standard_entries):,}")
    logger.info(f"  Refusal:              {len(refusal_entries):,}")
    logger.info(f"  Citation pool:        {len(citation_pool):,}")
    logger.info(f"  Noise pool:           {len(noise_pool):,}")
    logger.info(f"  Contradiction pool:   {len(contradiction_pool):,}")
    logger.info(f"  State → {STATE_FILE}")
    logger.info("=" * 60)
    logger.info("  Next: python scripts/exp009/construct_categories.py detect-contradictions")
    logger.info("=" * 60)
    return 0


# ═══════════════════════════════════════════════════════════
# 6.2 — 矛盾检测
# ═══════════════════════════════════════════════════════════

CONTRADICTION_DETECT_SYSTEM = """你是一个文本分析专家。你的任务是判断 AI 生成的搜索回答是否包含了对多文档间矛盾的识别和处理。

请检查回答是否做了以下任何一件事：
1. 明确指出了不同来源对同一问题的不同说法
2. 对比了不同来源的数据/观点差异
3. 解释了信息冲突的可能原因
4. 在矛盾信息中给出了综合判断或建议

输出格式（严格 JSON）：
{"has_contradiction": true/false, "contradiction_type": "数值矛盾"|"观点对立"|"信息冲突"|"无", "reason": "一句话说明"}
"""

CONTRADICTION_DETECT_USER = """请判断以下 AI 回答是否包含了对多文档矛盾的识别和处理。

【用户问题】
{query}

【AI 回答】
{answer}
"""


def check_contradiction(entry: dict) -> tuple[bool, str, str]:
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
        max_tokens=128,
    )

    user_msg = CONTRADICTION_DETECT_USER.format(
        query=entry["query"],
        answer=entry["teacher_answer"],
    )

    response = llm.invoke([
        SystemMessage(content=CONTRADICTION_DETECT_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    raw = response.content.strip()

    try:
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        result = json.loads(raw)
        has = result.get("has_contradiction", False)
        ctype = result.get("contradiction_type", "无")
        reason = result.get("reason", raw)
        return has, ctype, reason
    except json.JSONDecodeError:
        has = "true" in raw.lower() and "has_contradiction" in raw.lower()
        return has, "?" if has else "无", raw


def run_contradiction_detection(entries: list[dict], workers: int = 8) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"  Detecting contradictions ({workers} concurrent)...")

    found = []
    processed = 0
    t_start = time.time()

    def process_one(idx: int, entry: dict):
        try:
            has, ctype, reason = check_contradiction(entry)
        except Exception as e:
            has, ctype, reason = False, "ERROR", str(e)
        return idx, entry, has, ctype, reason

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, i, e): i for i, e in enumerate(entries)}

        for fut in as_completed(futures):
            idx, entry, has, ctype, reason = fut.result()
            processed += 1

            entry["_has_contradiction"] = has
            entry["_contradiction_type"] = ctype
            entry["_contradiction_reason"] = reason

            if has:
                found.append(entry)

            if processed % 200 == 0:
                elapsed = time.time() - t_start
                rate = processed / max(elapsed, 0.1)
                eta = (len(entries) - processed) / max(rate, 0.01) / 60
                logger.info(f"  Progress: {processed}/{len(entries)}  found={len(found)}  "
                            f"{rate:.1f}q/s  ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"  Detection done in {elapsed/60:.1f}min: found {len(found):,} contradictions "
                f"out of {len(entries):,} ({len(found)/max(len(entries),1)*100:.1f}%)")
    return found


def cmd_detect_contradictions(args):
    logger.info("=" * 60)
    logger.info("  6.2: Detect contradictions")
    logger.info("=" * 60)

    pool_file = STAGE61_DIR / "contradiction_pool.jsonl"
    if not pool_file.exists():
        logger.error(f"Contradiction pool not found: {pool_file}. Run 'select' first.")
        return 1

    entries = load_jsonl(pool_file)
    logger.info(f"  Pool size: {len(entries):,} entries")

    found = run_contradiction_detection(entries, workers=args.workers)

    # 如果找到的超过 N_CONTRADICTION，随机采样
    random.seed(SAMPLING_SEED)
    if len(found) > N_CONTRADICTION:
        found = random.sample(found, N_CONTRADICTION)
        logger.info(f"  Downsampled to {N_CONTRADICTION:,}")

    save_jsonl(found, STAGE61_DIR / "contradictions.jsonl")

    # 更新状态
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        logger.error("State file not found. Run 'select' first.")
        return 1

    state["stage"] = "contradiction_done"
    state["counts"]["contradiction"] = len(found)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    logger.info("-" * 60)
    logger.info(f"  Contradictions found: {len(found):,}")
    logger.info(f"  State → {STATE_FILE}")
    logger.info("=" * 60)
    logger.info("  Next: python scripts/exp009/construct_categories.py rewrite --workers 8")
    logger.info("=" * 60)
    return 0


# ═══════════════════════════════════════════════════════════
# 6.3 — 改写生成（Batch API）
# ═══════════════════════════════════════════════════════════

def _load_collection_samples(path: Path, n: int, exclude_pids: set[str], pool_size: int = 50000) -> list[dict]:
    """从 collection 中随机采样 n 条，排除 exclude_pids 中的 pid."""
    pool = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid = parts[0]
            text = parts[1]
            if pid in exclude_pids:
                continue
            pool.append({"pid": pid, "text": text})
            if len(pool) >= pool_size:
                break

    random.seed(SAMPLING_SEED)
    return random.sample(pool, min(n, len(pool)))


def _call_deepseek(system_text: str, user_text: str) -> str:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
        max_tokens=1024,
    )

    response = llm.invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=user_text),
    ])
    return response.content


def cmd_rewrite(args):
    logger.info("=" * 60)
    logger.info(f"  6.3: Rewrite via deepseek-chat ({args.workers} concurrent)")
    logger.info("=" * 60)

    citation_file = STAGE61_DIR / "citation_pool.jsonl"
    noise_file = STAGE61_DIR / "noise_pool.jsonl"
    contradiction_file = STAGE61_DIR / "contradictions.jsonl"

    missing = []
    for f in [citation_file, noise_file, contradiction_file]:
        if not f.exists():
            missing.append(f.name)
    if missing:
        logger.error(f"Missing files: {missing}. Run previous subcommands first.")
        return 1

    citation_pool = load_jsonl(citation_file)
    noise_pool = load_jsonl(noise_file)
    contradictions = load_jsonl(contradiction_file)

    random.seed(SAMPLING_SEED)
    citation_sample = random.sample(citation_pool, min(N_CITATION, len(citation_pool)))

    used_qids = {e["qid"] for e in contradictions}
    noise_candidates = [e for e in noise_pool if e["qid"] not in used_qids]
    random.seed(SAMPLING_SEED)
    random.shuffle(noise_candidates)
    noise_sample = noise_candidates[:N_NOISE]

    contradiction_sample = contradictions

    logger.info(f"  Citation emphasis:  {len(citation_sample):,} entries")
    logger.info(f"  Noise injection:    {len(noise_sample):,} entries")
    logger.info(f"  Contradiction:      {len(contradiction_sample):,} entries")
    logger.info(f"  Total rewrites:     {len(citation_sample)+len(noise_sample)+len(contradiction_sample):,}")

    n_noise_passages = 0
    irrelevant = []
    if noise_sample:
        all_positive_pids = set()
        for e in noise_sample:
            for p in e.get("passages", []):
                all_positive_pids.add(p["pid"])
        n_noise_passages = len(noise_sample) * 3
        logger.info(f"  Sampling {n_noise_passages:,} irrelevant passages from collection...")
        irrelevant = _load_collection_samples(
            COLLECTION_FILE, n_noise_passages, all_positive_pids, args.collection_sample_pool
        )
        logger.info(f"  Sampled {len(irrelevant):,} irrelevant passages")

    # 构建所有请求
    tasks: list[dict] = []
    for entry in citation_sample:
        tasks.append({
            "custom_id": f"citation_{entry['qid']}",
            "entry": entry,
            "system": CITATION_SYSTEM,
            "passages": entry["passages"],
            "category": "citation",
        })

    irr_idx = 0
    for entry in noise_sample:
        n_inject = random.randint(2, 3)
        injected = list(entry["passages"])
        for j in range(n_inject):
            if irr_idx < len(irrelevant):
                injected.append({"pid": irrelevant[irr_idx]["pid"], "text": irrelevant[irr_idx]["text"]})
                irr_idx += 1
        tasks.append({
            "custom_id": f"noise_{entry['qid']}",
            "entry": entry,
            "system": NOISE_SYSTEM,
            "passages": injected,
            "category": "noise",
        })

    logger.info(f"  Injected: {irr_idx} irrelevant passages across {len(noise_sample)} noise entries")

    for entry in contradiction_sample:
        tasks.append({
            "custom_id": f"contradiction_{entry['qid']}",
            "entry": entry,
            "system": CONTRADICTION_SYSTEM,
            "passages": entry["passages"],
            "category": "contradiction",
        })

    # 并发调用
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[dict] = []
    done = 0
    failed = 0
    t_start = time.time()

    def process_one(task: dict):
        user_msg = build_user_message(task["entry"]["query"], task["passages"])
        try:
            answer = _call_deepseek(task["system"], user_msg)
        except Exception as e:
            answer = ""
            logger.warning(f"  Error {task['custom_id']}: {e}")
        return task["custom_id"], task["entry"]["qid"], task["category"], answer

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}

        for fut in as_completed(futures):
            custom_id, qid, cat, answer = fut.result()
            done += 1
            if not answer:
                failed += 1

            results.append({"custom_id": custom_id, "qid": qid, "category": cat, "answer": answer})

            if done % 100 == 0:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 0.1)
                eta = (len(tasks) - done) / max(rate, 0.01) / 60
                logger.info(f"  Progress: {done}/{len(tasks)}  failed={failed}  "
                            f"{rate:.1f}q/s  ETA {eta:.0f}min")

    elapsed = time.time() - t_start
    logger.info(f"  Rewrite done in {elapsed/60:.1f}min: {done:,} generated, {failed} failed")

    REWRITE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REWRITE_OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cat_counts: dict[str, int] = {}
    for r in results:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  {cat}: {count:,} rewritten")

    logger.info(f"  Output: {REWRITE_OUTPUT}")
    logger.info("=" * 60)
    logger.info("  Next: python scripts/exp009/construct_categories.py assemble")
    logger.info("=" * 60)
    return 0


# ═══════════════════════════════════════════════════════════
# 6.4 — 组装切分
# ═══════════════════════════════════════════════════════════

def build_sft_entry(entry: dict, category: str, answer: str | None = None) -> dict:
    return {
        "qid": entry["qid"],
        "query": entry["query"],
        "passages": entry.get("passages", []),
        "answer": answer if answer is not None else entry.get("teacher_answer", ""),
        "category": category,
        "bucket": entry.get("bucket", "?"),
        "teacher_model": entry.get("model", "qwen3-max"),
    }


def cmd_assemble(args):
    logger.info("=" * 60)
    logger.info("  6.4: Assemble SFT train/val split")
    logger.info("=" * 60)

    # 加载 6.1 产物
    standard_file = STAGE61_DIR / "standard.jsonl"
    refusal_file = STAGE61_DIR / "refusal.jsonl"

    for f in [standard_file, refusal_file]:
        if not f.exists():
            logger.error(f"Missing: {f.name}. Run 'select' first.")
            return 1

    standard_entries = load_jsonl(standard_file)
    refusal_entries = load_jsonl(refusal_file)

    # 加载 6.3 产物
    rewritten: dict[str, dict] = {}
    if REWRITE_OUTPUT.exists():
        with open(REWRITE_OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rewritten[r["custom_id"]] = r
    else:
        logger.warning(f"Rewrite output not found: {REWRITE_OUTPUT}. "
                       "Run rewrite first. Proceeding without rewritten entries.")

    # 5 类：
    # 1. 标准：从 standard_entries 拆 train + val
    random.seed(VAL_SEED)
    val_indices = set(random.sample(range(len(standard_entries)), min(N_VAL, len(standard_entries))))
    val_entries = [e for i, e in enumerate(standard_entries) if i in val_indices]

    sft_train = []
    for i, e in enumerate(standard_entries):
        if i not in val_indices:
            sft_train.append(build_sft_entry(e, "standard"))

    # 2. 拒答
    for e in refusal_entries:
        sft_train.append(build_sft_entry(e, "refusal"))

    # 3. 引文强调
    citation_count = 0
    for custom_id, rw in rewritten.items():
        if not custom_id.startswith("citation_"):
            continue
        qid = custom_id.split("_", 1)[1]
        match = None
        for e in load_jsonl(STAGE61_DIR / "citation_pool.jsonl"):
            if e["qid"] == qid:
                match = e
                break
        if match:
            sft_train.append(build_sft_entry(match, "citation_emphasis", rw["answer"]))
            citation_count += 1

    # 4. 噪声
    noise_count = 0
    for custom_id, rw in rewritten.items():
        if not custom_id.startswith("noise_"):
            continue
        qid = custom_id.split("_", 1)[1]
        match = None
        for e in load_jsonl(STAGE61_DIR / "noise_pool.jsonl"):
            if e["qid"] == qid:
                match = e
                break
        if match:
            sft_train.append(build_sft_entry(match, "noise", rw["answer"]))
            noise_count += 1

    # 5. 矛盾
    contradiction_count = 0
    for custom_id, rw in rewritten.items():
        if not custom_id.startswith("contradiction_"):
            continue
        qid = custom_id.split("_", 1)[1]
        ct_file = STAGE61_DIR / "contradictions.jsonl"
        match = None
        if ct_file.exists():
            for e in load_jsonl(ct_file):
                if e["qid"] == qid:
                    match = e
                    break
        if match:
            sft_train.append(build_sft_entry(match, "contradiction", rw["answer"]))
            contradiction_count += 1

    # 拆 train/val
    val_qids = {e["qid"] for e in val_entries}
    train_sft = [e for e in sft_train if e["qid"] not in val_qids]
    val_sft = [build_sft_entry(e, "standard") for e in val_entries]

    # 统计
    cat_counts: dict[str, int] = {}
    for e in train_sft:
        cat_counts[e["category"]] = cat_counts.get(e["category"], 0) + 1

    logger.info("-" * 60)
    logger.info("  ASSEMBLY SUMMARY")
    logger.info(f"  Standard:      {cat_counts.get('standard', 0):,}")
    logger.info(f"  Citation:      {cat_counts.get('citation_emphasis', 0):,}")
    logger.info(f"  Refusal:       {cat_counts.get('refusal', 0):,}")
    logger.info(f"  Noise:         {cat_counts.get('noise', 0):,}")
    logger.info(f"  Contradiction: {cat_counts.get('contradiction', 0):,}")
    logger.info(f"  ─────────────────────────")
    logger.info(f"  Train total:   {len(train_sft):,}")
    logger.info(f"  Val total:     {len(val_sft):,}")

    save_jsonl(train_sft, TRAIN_OUTPUT)
    save_jsonl(val_sft, VAL_OUTPUT)

    logger.info(f"  Train → {TRAIN_OUTPUT}")
    logger.info(f"  Val   → {VAL_OUTPUT}")
    logger.info("=" * 60)
    logger.info("  Stage 6 complete!")
    logger.info("=" * 60)
    return 0


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 阶段六：类别构造（4 个子阶段）"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_select = sub.add_parser("select", help="6.1 筛选标准问答 + 拒答（无 API）")

    p_detect = sub.add_parser("detect-contradictions", help="6.2 矛盾检测（deepseek-chat）")
    p_detect.add_argument("--workers", type=int, default=8,
                          help="并发数 (default: 8)")

    p_rewrite = sub.add_parser("rewrite", help="6.3 deepseek-chat 并发改写")
    p_rewrite.add_argument("--workers", type=int, default=8,
                           help="并发数 (default: 8)")
    p_rewrite.add_argument("--collection-sample-pool", type=int, default=50000,
                           help="从 collection 中读取多少行用于噪声采样 (default: 50000)")

    p_assemble = sub.add_parser("assemble", help="6.4 组装 train/val 切分")

    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return 1

    cmds = {
        "select": cmd_select,
        "detect-contradictions": cmd_detect_contradictions,
        "rewrite": cmd_rewrite,
        "assemble": cmd_assemble,
    }

    return cmds[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
