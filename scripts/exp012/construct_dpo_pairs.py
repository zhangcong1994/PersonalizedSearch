"""
Exp-012 Phase 0: 从 exp-011 多样本 + Judge 评分数据构造自对比式 DPO 训练对。

数据流：
  generation JSONL (含 answer/passages/system_prompt)  ←→  Judge JSONL (含 scores)
       │                                                          │
       └──────────── join on query_id ────────────────────────────┘
                              │
                    group by original_query_id
                              │
              per-query: best vs worst / best vs median
                              │
                    reconstruct full prompt
                              │
                   filter by gap threshold
                              │
                  output DPO training JSONL

用法:
  python scripts/exp012/construct_dpo_pairs.py \
      --generation results/exp011/generation/qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl \
      --judge results/exp011/judge_scores/qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl \
      --output-dir data/processed/exp012/

  或使用默认路径:
  python scripts/exp012/construct_dpo_pairs.py
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 默认路径 ────────────────────────────────────────────────

DEFAULT_GENERATION = (
    DATA_ROOT / "results" / "exp011" / "generation"
    / "qwen3-8b-nothink_v1-full_t0.8_n5_s42.jsonl"
)
DEFAULT_JUDGE = (
    DATA_ROOT / "results" / "exp011" / "judge_scores"
    / "qwen3-8b-nothink_v1-full_t0.8_n5_s42_judged.jsonl"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "data" / "processed" / "exp012"

# ── 配对策略 ────────────────────────────────────────────────

PAIRING_STRATEGIES = {
    "best_vs_worst": {
        "description": "最高分 chosen vs 最低分 rejected",
        "chosen_selector": lambda scores: max(scores, key=lambda x: x[1]),
        "rejected_selector": lambda scores: min(scores, key=lambda x: x[1]),
    },
    "best_vs_median": {
        "description": "最高分 chosen vs 中位数分 rejected",
        "chosen_selector": lambda scores: max(scores, key=lambda x: x[1]),
        "rejected_selector": lambda scores: sorted(scores, key=lambda x: x[1])[len(scores) // 2],
    },
    "best_vs_first_below": {
        "description": "最高分 chosen vs 第一个低够 min_gap 的 rejected（需 min_gap 参数）",
        "chosen_selector": lambda scores: max(scores, key=lambda x: x[1]),
        "rejected_selector": None,  # 特殊处理：在 construct_pairs() 中实现
    },
}


# ── 工具函数 ────────────────────────────────────────────────

def load_jsonl(filepath: Path) -> list[dict]:
    """加载 JSONL 文件。"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {filepath.name}")
    return records


def load_gen_indexed(generation_file: Path) -> dict[str, dict]:
    """加载 generation 文件，按 query_id 建立索引。"""
    records = load_jsonl(generation_file)
    index = {}
    for r in records:
        qid = r["query_id"]
        if qid in index:
            logger.warning(f"Duplicate query_id in generation: {qid}")
        index[qid] = r
    return index


def load_judge_indexed(judge_file: Path) -> dict[str, float]:
    """加载 Judge 评分文件，提取 query_id → total_score。"""
    records = load_jsonl(judge_file)
    index = {}
    for r in records:
        qid = r["query_id"]
        score = r.get("aggregation", {}).get("total_score")
        if score is None:
            logger.warning(f"No total_score for {qid}, skipping")
            continue
        index[qid] = score
    return index


def extract_original_qid(composite_qid: str) -> str:
    """从复合 query_id（如 '1869_s0'）还原原始 query_id。"""
    if "_s" in composite_qid:
        parts = composite_qid.rsplit("_s", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
    return composite_qid


def clean_answer(text: str) -> str:
    """移除答案中回显的 prompt 模板片段。

    exp-011 部分生成会把输入 prompt 结构吐回到答案开头：
      【参考资料】
      [1] 来源: 123  ...
      【用户问题】
      原始问题
      【回答】
      <实际答案内容>

    清洗规则：
    1. 【参考资料】 开头 → 找后面第一个 【回答】 或 【核心结论】/【核心答案】，前面砍掉
    2. 【核心结论】/【核心答案】 → 保留，这是 v1-full prompt 指示的正当答案格式
    """
    text = text.strip()
    if not text:
        return text

    # 匹配 "【参考资料】" 开头（含全角半角空格）
    stripped = text.lstrip()
    if stripped.startswith("【参考资料】"):
        # 在文本中寻找切分标记，优先级：【回答】 > 【核心结论】 > 【核心答案】
        for marker in ["\n【回答】", "【回答】"]:
            idx = stripped.find(marker)
            if idx >= 0:
                after = stripped[idx + len(marker):].strip().lstrip("\n").lstrip()
                if after:
                    return after

        # 没有 【回答】 标记，尝试找 【核心结论】 或 【核心答案】
        for marker in ["\n【核心结论】", "【核心结论】", "\n【核心答案】", "【核心答案】"]:
            idx = stripped.find(marker)
            if idx > 10:  # 确保不是刚开头的个别字符
                after = stripped[idx:].strip()
                if after:
                    return after

        # 兜底：如果实在找不到切分标记，返回原文本
        # （避免误删没有格式标签的正常答案）
        return text

    return text


def build_full_prompt(system_prompt: str, passages: list[dict], query_text: str) -> str:
    """重建喂给 LLM 的完整 prompt 文本。

    与 generate_multi_sample.py 的 build_prompt() 完全一致：
    - passages 按 [{rank}] 来源: {pid} 格式化，文本截断至 800 字符
    - user_prompt = "参考资料: ...\n\n用户问题: ...\n\n请根据以上资料回答问题："
    - full_prompt = system_prompt + "\n\n" + user_prompt
    """
    context_parts = []
    for p in passages:
        pid = p.get("pid", "unknown")
        rank = p.get("rank", 1)
        text = p.get("text", "")
        context_parts.append(f"[{rank}] 来源: {pid}\n{text[:800]}")

    context = "\n\n".join(context_parts)
    user_prompt = f"参考资料:\n{context}\n\n用户问题: {query_text}\n\n请根据以上参考资料回答问题："

    return f"{system_prompt}\n\n{user_prompt}"


# ── 主逻辑 ──────────────────────────────────────────────────

def construct_pairs(
    gen_index: dict[str, dict],
    judge_scores: dict[str, float],
    strategy_name: str,
    min_gap: float,
) -> list[dict]:
    """构造 DPO 训练对。

    返回: list of {"query_id", "prompt", "chosen", "rejected", "chosen_score", "rejected_score", "gap"}
    """

    strategy = PAIRING_STRATEGIES[strategy_name]
    chosen_selector = strategy["chosen_selector"]
    rejected_selector = strategy["rejected_selector"]

    # 1. 按原始 query_id 分组
    groups: dict[str, list[tuple[str, float, dict]]] = defaultdict(list)
    for qid, score in judge_scores.items():
        orig_qid = extract_original_qid(qid)
        gen_record = gen_index.get(qid)
        if gen_record is None:
            logger.warning(f"Judge entry {qid} has no matching generation record, skipping")
            continue
        groups[orig_qid].append((qid, score, gen_record))

    # 2. 每 query 构造 DPO 对
    pairs = []
    skipped_gap = 0
    skipped_same_text = 0
    skipped_one_sample = 0
    skipped_no_gen = 0

    for orig_qid, samples in sorted(groups.items()):
        if len(samples) < 2:
            skipped_one_sample += 1
            continue

        # ── best_vs_first_below: 特殊逻辑 ──
        if strategy_name == "best_vs_first_below":
            # 按分数降序排列
            sorted_samples = sorted(samples, key=lambda x: x[1], reverse=True)
            chosen_score, chosen_rec = sorted_samples[0][1], sorted_samples[0][2]

            # 从第二名往下找，第一个比 chosen 低够 min_gap 的
            rejected_entry = None
            for _, score, rec in sorted_samples[1:]:
                if chosen_score - score >= min_gap:
                    rejected_entry = (score, rec)
                    break

            if rejected_entry is None:
                skipped_gap += 1
                continue

            rejected_score, rejected_rec = rejected_entry
        else:
            # ── 普通策略: lambda selector ──
            chosen_entry = chosen_selector(samples)
            rejected_entry = rejected_selector(samples)
            _, chosen_score, chosen_rec = chosen_entry
            _, rejected_score, rejected_rec = rejected_entry

        # 文本相同检查（尽管分数不同，Judge 噪声）
        chosen_text = clean_answer(chosen_rec.get("answer", ""))
        rejected_text = clean_answer(rejected_rec.get("answer", ""))
        if not chosen_text or not rejected_text:
            continue
        if chosen_text == rejected_text:
            skipped_same_text += 1
            continue

        # 对比度过滤（best_vs_first_below 已内置 min_gap 检查，此处为兜底）
        gap = chosen_score - rejected_score
        if gap < min_gap:
            skipped_gap += 1
            continue

        # 重建 prompt（两个 record 的 system_prompt + passages + query_text 应相同）
        prompt = build_full_prompt(
            chosen_rec["system_prompt"],
            chosen_rec["passages"],
            chosen_rec.get("query_text", ""),
        )

        pairs.append({
            "query_id": orig_qid,
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "chosen_score": round(chosen_score, 1),
            "rejected_score": round(rejected_score, 1),
            "gap": round(gap, 1),
        })

    logger.info(
        f"  Strategy={strategy_name}, gap>={min_gap}: "
        f"produced {len(pairs)} pairs"
        f" (skipped: {skipped_gap} gap < {min_gap}, "
        f"{skipped_same_text} same-text, "
        f"{skipped_one_sample} single-sample, "
        f"{skipped_no_gen} no-gen)"
    )

    return pairs


def print_quality_report(all_pairs: dict[str, list[dict]]) -> None:
    """打印质量检查报告。"""
    print("\n" + "=" * 70)
    print("  DPO 数据质量报告")
    print("=" * 70)

    for label, pairs in all_pairs.items():
        print(f"\n--- {label} ({len(pairs)} pairs) ---")

        if not pairs:
            print("  (empty)")
            continue

        gaps = [p["gap"] for p in pairs]
        chosen_scores = [p["chosen_score"] for p in pairs]
        rejected_scores = [p["rejected_score"] for p in pairs]
        chosen_lens = [len(p["chosen"]) for p in pairs]
        rejected_lens = [len(p["rejected"]) for p in pairs]

        print(f"  gap:       mean={sum(gaps)/len(gaps):.1f}, "
              f"min={min(gaps):.1f}, max={max(gaps):.1f}")
        print(f"  chosen:    mean={sum(chosen_scores)/len(chosen_scores):.1f}, "
              f"min={min(chosen_scores):.1f}, max={max(chosen_scores):.1f}")
        print(f"  rejected:  mean={sum(rejected_scores)/len(rejected_scores):.1f}, "
              f"min={min(rejected_scores):.1f}, max={max(rejected_scores):.1f}")
        print(f"  chosen len:   mean={sum(chosen_lens)/len(chosen_lens):.0f}, "
              f"min={min(chosen_lens)}, max={max(chosen_lens)}")
        print(f"  rejected len: mean={sum(rejected_lens)/len(rejected_lens):.0f}, "
              f"min={min(rejected_lens)}, max={max(rejected_lens)}")
        print(f"  len ratio (chosen/rejected): {sum(chosen_lens)/len(chosen_lens)/max(1,sum(rejected_lens)/len(rejected_lens)):.2f}")

    # 共享 query 统计（所有策略共用同一批 query，只是配对方式不同）
    first_label = list(all_pairs.keys())[0]
    first_pairs = all_pairs[first_label]
    if first_pairs:
        print(f"\n--- Top 5 最大对比度对 ({first_label}) ---")
        sorted_by_gap = sorted(first_pairs, key=lambda p: p["gap"], reverse=True)
        for p in sorted_by_gap[:5]:
            chosen_preview = p["chosen"][:80].replace("\n", "\\n")
            rejected_preview = p["rejected"][:80].replace("\n", "\\n")
            print(f"  qid={p['query_id']:8s}  gap={p['gap']:5.1f}  "
                  f"chosen={p['chosen_score']:5.1f}  rejected={p['rejected_score']:5.1f}")
            print(f"    chosen:   {chosen_preview}...")
            print(f"    rejected: {rejected_preview}...")

        print(f"\n--- Bottom 5 最小对比度对 ({first_label}) ---")
        for p in sorted_by_gap[-5:]:
            chosen_preview = p["chosen"][:80].replace("\n", "\\n")
            rejected_preview = p["rejected"][:80].replace("\n", "\\n")
            print(f"  qid={p['query_id']:8s}  gap={p['gap']:5.1f}  "
                  f"chosen={p['chosen_score']:5.1f}  rejected={p['rejected_score']:5.1f}")
            print(f"    chosen:   {chosen_preview}...")
            print(f"    rejected: {rejected_preview}...")

    print("\n" + "=" * 70)
    print(f"  总计: {sum(len(v) for v in all_pairs.values())} pairs 来自 {len(all_pairs)} 个策略配置")
    print("=" * 70 + "\n")


def save_pairs(pairs: list[dict], output_path: Path) -> None:
    """保存 DPO 对到 JSONL 文件。"""
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {output_path}")


# ── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp-012: 从 exp-011 多样本数据构造自对比式 DPO 训练对"
    )
    parser.add_argument(
        "--generation", type=Path, default=DEFAULT_GENERATION,
        help="exp-011 多样本生成 JSONL 路径",
    )
    parser.add_argument(
        "--judge", type=Path, default=DEFAULT_JUDGE,
        help="exp-011 Judge 评分 JSONL 路径",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="输出目录",
    )
    parser.add_argument(
        "--strategies", type=str, default="best_vs_worst,best_vs_median",
        help="配对策略，逗号分隔。可选: best_vs_worst, best_vs_median",
    )
    parser.add_argument(
        "--min-gaps", type=str, default="5,10",
        help="对比度阈值列表，逗号分隔。如 '5,10'",
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="跳过打印质量报告",
    )
    args = parser.parse_args()

    # 验证输入文件
    if not args.generation.exists():
        logger.error(f"Generation file not found: {args.generation}")
        return 1
    if not args.judge.exists():
        logger.error(f"Judge file not found: {args.judge}")
        return 1

    # 加载数据
    logger.info(f"Loading generation: {args.generation}")
    gen_index = load_gen_indexed(args.generation)
    logger.info(f"Loading judge scores: {args.judge}")
    judge_scores = load_judge_indexed(args.judge)

    # 交叉验证
    gen_only = set(gen_index.keys()) - set(judge_scores.keys())
    judge_only = set(judge_scores.keys()) - set(gen_index.keys())
    common = set(gen_index.keys()) & set(judge_scores.keys())
    logger.info(f"  Common query_ids: {len(common)}, gen-only: {len(gen_only)}, judge-only: {len(judge_only)}")

    if gen_only:
        logger.warning(f"  Generation entries without judge scores: {sorted(gen_only)[:10]}...")
    if judge_only:
        logger.warning(f"  Judge entries without generation: {sorted(judge_only)[:10]}...")

    # 解析策略和阈值
    strategy_names = [s.strip() for s in args.strategies.split(",")]
    min_gaps = [float(g.strip()) for g in args.min_gaps.split(",")]

    for sn in strategy_names:
        if sn not in PAIRING_STRATEGIES:
            logger.error(f"Unknown strategy: {sn}. Available: {list(PAIRING_STRATEGIES.keys())}")
            return 1

    # 构造 DPO 对
    all_pairs: dict[str, list[dict]] = {}

    for strategy_name in strategy_names:
        for min_gap in min_gaps:
            label = f"{strategy_name}_gap{int(min_gap)}"
            pairs = construct_pairs(gen_index, judge_scores, strategy_name, min_gap)
            all_pairs[label] = pairs

            # 保存
            output_file = args.output_dir / f"exp012_dpo_{label}.jsonl"
            save_pairs(pairs, output_file)

    # 打印质量报告
    if not args.no_report:
        print_quality_report(all_pairs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
