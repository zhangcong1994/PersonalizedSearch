"""
Exp-005 双 Judge 校准脚本。

工作流程：
  1. 从已有答案文件中采样，拆分为人类校准集 + 双 Judge 验证集
  2. 双 Judge 评分：Judge A (deepseek-reasoner, thinking) + Judge B (glm-4.7, thinking)
  3. 计算 Judge-Judge 一致性 + Judge-Human 一致性（人工标注后）
  4. 生成人类标注 CSV 模板

输出目录结构:
  {results_dir}/exp005/calibration/
    sampled_manifest.json       # 采样清单（校准/验证集 qid）
    judge_a_scores.jsonl        # Judge A 评分结果
    judge_b_scores.jsonl        # Judge B 评分结果
    judge_agreement_report.json # Judge-Judge 一致性报告
    human_annotation_template.csv  # 人类校准集的标注模板

用法:
  python scripts/exp005/dual_judge_calibration.py \
    --answers-file {已有生成结果的.jsonl} \
    --calibration-size 50 \
    --verification-size 100 \
    --judge-a deepseek-reasoner \
    --judge-b glm-4.7 \
    --seed 42

  # 仅执行某个步骤
  python scripts/exp005/dual_judge_calibration.py ... --skip-sample --skip-judge
"""

import os
import sys
import json
import csv
import time
import random
import logging
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from collections import Counter

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.intent.api_client import APIClientFactory
from src.evaluation.judge_prompts import (
    get_batch_system_prompt,
    build_gen_stage_judge_input,
    parse_judge_response,
    ALL_CORE_DIMS,
)
from src.evaluation.aggregation import (
    aggregate_core6_scores,
    CORE6_DIM_LABELS,
    CORE6_WEIGHTS,
    CORE6_GATE_DIMS,
)
from src.utils.config import DATA_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005" / "calibration"
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2

HUMAN_DIM_LABELS = {
    "veracity": "信息准确性 (1-4)",
    "synthesis_quality": "信息整合质量 (1-4)",
    "citation_quality": "引文质量 (1-4)",
    "relevance": "相关性 (1-4)",
    "user_experience": "用户体验 (1-4)",
    "safety": "安全合规 (1-4)",
    "overall": "综合质量 (1-4)",
}

QUERY_TYPE_KEYWORDS = {
    "factoid": ["什么时候", "多少", "谁", "哪里", "哪一年", "日期", "年龄",
                "多高", "多长", "多大", "什么时间", "何时"],
    "comparison": ["区别", "对比", "不同", "差异", " vs ", "比较", "哪个更好",
                   "哪个更", "优缺点", "优劣"],
    "concept": ["什么是", "定义", "概念", "解释", "什么意思", "原理", "指的是",
                "是什么", "含义"],
    "how_to": ["如何", "怎么", "怎样", "步骤", "方法", "做法", "教程", "指南",
               "操作", "配置", "设置"],
    "open_ended": ["影响", "未来", "趋势", "前景", "发展", "历史", "原因",
                   "为什么", "为何", "作用", "意义", "重要性"],
}


def classify_query(query_text: str) -> str:
    """基于关键词对查询类型进行粗分类。"""
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        if any(kw in query_text for kw in keywords):
            return qtype
    if query_text.endswith("?") or query_text.endswith("？"):
        return "open_ended"
    return "factoid"


# ===========================================================================
# Step 1: 查询采样
# ===========================================================================

def sample_queries(
    answers_file: Path,
    calibration_size: int = 50,
    verification_size: int = 100,
    seed: int = 42,
) -> dict:
    """
    从已有答案文件中简单随机采样，拆分为校准集和验证集。

    Returns:
        {
            "calibration": [{"qid": str, "query_text": str, "type": str}, ...],
            "verification": [...],
            "all_qids": [str, ...],
        }
    """
    logger.info(f"Loading answers from {answers_file}")
    answers = load_generations(answers_file)
    logger.info(f"Loaded {len(answers)} answers")

    entries = []
    for item in answers:
        qid = item.get("query_id", "")
        if not qid:
            continue
        entries.append({
            "qid": qid,
            "query_text": item.get("query_text", item.get("query", "")),
            "type": item.get("query_type", classify_query(item.get("query_text", item.get("query", "")))),
        })

    total_needed = calibration_size + verification_size
    if len(entries) < total_needed:
        logger.warning(f"Only {len(entries)} answers, need {total_needed}. Using all.")
        total_needed = len(entries)
        calibration_size = min(calibration_size, total_needed)

    random.seed(seed)
    sampled = random.sample(entries, total_needed)

    calibration = sampled[:calibration_size]
    verification = sampled[calibration_size:]

    all_qids = [item["qid"] for item in sampled]

    logger.info(
        f"Sampled: calibration={len(calibration)}, "
        f"verification={len(verification)}, total={len(calibration) + len(verification)}"
    )

    return {"calibration": calibration, "verification": verification, "all_qids": all_qids}


def load_generations(filepath: Path) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ===========================================================================
# Step 2: 双 Judge 评分
# ===========================================================================

JUDGE_CONFIGS = {
    "deepseek-reasoner": {
        "client_type": "deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "thinking": True,
        "max_tokens": 4096,
        "temperature": 1.0,
    },
    "glm-4.7": {
        "client_type": "zhipu",
        "env_key": "ZHIPU_API_KEY",
        "thinking": True,
        "max_tokens": 4096,
        "temperature": 1.0,
    },
    "glm-4-flash": {
        "client_type": "zhipu",
        "env_key": "ZHIPU_API_KEY",
        "thinking": False,
        "max_tokens": 2048,
        "temperature": 0.0,
    },
    "deepseek-chat": {
        "client_type": "deepseek",
        "env_key": "DEEPSEEK_API_KEY",
        "thinking": False,
        "max_tokens": 2048,
        "temperature": 0.0,
    },
}


def _create_judge_client(judge_model: str):
    config = JUDGE_CONFIGS.get(judge_model)
    if config is None:
        raise ValueError(
            f"Unknown judge model: {judge_model}. "
            f"Available: {list(JUDGE_CONFIGS.keys())}"
        )

    return APIClientFactory.create(
        config["client_type"],
        model=judge_model,
        api_key=os.getenv(config["env_key"]),
        max_tokens=config["max_tokens"],
        temperature=config["temperature"],
        thinking=config["thinking"],
    )


def judge_single_batch(client, query: str, passages: list[dict], answer: str, batch: int) -> dict | None:
    system_prompt = get_batch_system_prompt(batch)
    user_message = build_gen_stage_judge_input(query, passages, answer, batch=batch)
    full_prompt = f"{system_prompt}\n\n{user_message}"

    for attempt in range(MAX_RETRIES):
        try:
            result = client.generate_with_reasoning(full_prompt)
            raw_response = result["content"]
            parsed = parse_judge_response(raw_response)

            if parsed is None:
                logger.warning(
                    f"Batch{batch} parse failed (attempt {attempt + 1}). "
                    f"Preview: {raw_response[:200]}..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_BASE ** (attempt + 1))
                continue

            return {
                "scores": parsed,
                "raw_response": raw_response,
                "reasoning_content": result["reasoning_content"],
            }

        except Exception as e:
            logger.warning(f"Batch{batch} API error (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_BASE ** (attempt + 1))

    logger.error(f"Batch{batch} failed after {MAX_RETRIES} attempts")
    return None


def _judge_one_item(
    item: dict,
    judge_model: str,
    stagger_delay: float = 0.5,
) -> dict | None:
    """对单条 query 执行双批评分，返回输出 dict，失败返回 None。"""
    import os as _os
    import time as _time
    import random as _random

    if stagger_delay > 0:
        _time.sleep(_random.uniform(0, stagger_delay))

    client = _create_judge_client(judge_model)

    qid = item.get("query_id", "")
    query_text = item.get("query_text", item.get("query", ""))
    passages = item.get("passages", [])
    answer = item.get("answer", "")

    if isinstance(passages, list) and passages and not isinstance(passages[0], dict):
        passages = [
            {"pid": f"doc-{j}", "text": str(p), "rank": j + 1}
            for j, p in enumerate(passages)
        ]

    result1 = judge_single_batch(client, query_text, passages, answer, batch=1)
    result2 = judge_single_batch(client, query_text, passages, answer, batch=2)

    if not result1 or not result2:
        return None

    all_scores = {}
    all_scores.update(result1.get("scores", {}))
    all_scores.update(result2.get("scores", {}))
    aggregation = aggregate_core6_scores(all_scores)

    return {
        "query_id": qid,
        "query_text": query_text,
        "judge_model": judge_model,
        "scores": all_scores,
        "aggregation": aggregation,
        "batch1_raw": result1.get("raw_response", ""),
        "batch1_reasoning": result1.get("reasoning_content", ""),
        "batch2_raw": result2.get("raw_response", ""),
        "batch2_reasoning": result2.get("reasoning_content", ""),
    }


def run_single_judge(
    judge_model: str,
    judge_label: str,
    answers_file: Path,
    output_file: Path,
    force: bool = False,
    qid_filter: set[str] | None = None,
    concurrency: int = 3,
    stagger_delay: float = 0.5,
):
    """运行单个 Judge 对答案评分。如提供 qid_filter 则仅评分指定 qid。"""
    answers = load_generations(answers_file)
    if qid_filter is not None:
        answers = [a for a in answers if a.get("query_id", "") in qid_filter]
    logger.info(f"Running Judge {judge_label} ({judge_model}) on {len(answers)} answers (concurrency={concurrency}, stagger={stagger_delay}s)")

    existing = {}
    if output_file.exists() and not force:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing[r.get("query_id", "")] = r

    pending = [a for a in answers if a.get("query_id", "") not in existing]
    if not pending:
        logger.info(f"  All already cached in {output_file.name}")
        return

    os.makedirs(output_file.parent, exist_ok=True)

    lock = threading.Lock()
    processed = 0
    failed = 0
    total = len(pending)

    with open(output_file, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(_judge_one_item, item, judge_model, stagger_delay): item for item in pending}

            for future in as_completed(futures):
                result = future.result()
                with lock:
                    if result is None:
                        failed += 1
                    else:
                        result["judge_label"] = judge_label
                        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out_f.flush()
                    processed += 1
                    if processed % 10 == 0:
                        logger.info(f"  {judge_label}: {processed}/{total} (failed {failed})")

    logger.info(f"Judge {judge_label} done: {processed} processed, {failed} failed")


def run_dual_judge(
    judge_a_model: str,
    judge_b_model: str,
    answers_file: Path,
    judge_a_output: Path,
    judge_b_output: Path,
    force: bool = False,
    qid_filter: set[str] | None = None,
    concurrency: int = 3,
    stagger_delay: float = 0.5,
):
    """运行双 Judge 评分（Judge A 和 Judge B 并行执行）。"""
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_single_judge, judge_a_model, "Judge-A", answers_file, judge_a_output, force, qid_filter, concurrency, stagger_delay),
            executor.submit(run_single_judge, judge_b_model, "Judge-B", answers_file, judge_b_output, force, qid_filter, concurrency, stagger_delay),
        ]
        for future in as_completed(futures):
            future.result()


# ===========================================================================
# Step 3: 一致性计算
# ===========================================================================

def _extract_scores(judge_results: list[dict], dims: list[str] | None = None) -> dict[str, dict[str, int]]:
    if dims is None:
        dims = ALL_CORE_DIMS
    extracted = {}
    for r in judge_results:
        qid = r.get("query_id", "")
        scores = r.get("scores", {})
        parsed = {}
        for d in dims:
            val = scores.get(d)
            if val is None:
                continue
            if isinstance(val, dict):
                parsed[d] = int(val.get("score", 0))
            else:
                parsed[d] = int(val)
        if parsed:
            extracted[qid] = parsed
    return extracted


def compute_cohens_kappa(human_scores: list[int | None], judge_scores: list[int | None]) -> dict:
    valid = [(h, j) for h, j in zip(human_scores, judge_scores)
             if h is not None and j is not None]
    n = len(valid)
    if n < 5:
        return {"kappa": float("nan"), "n_valid": n}

    observed = Counter()
    for h, j in valid:
        observed[(h, j)] += 1

    total = sum(observed.values())
    p_o = sum(observed[(k, k)] for k in range(1, 5)) / total

    human_marginal = Counter(h for h, _ in valid)
    judge_marginal = Counter(j for _, j in valid)
    p_e = sum(
        (human_marginal.get(k, 0) / total) * (judge_marginal.get(k, 0) / total)
        for k in range(1, 5)
    )

    if p_e == 1.0:
        return {"kappa": 1.0 if p_o == 1.0 else 0.0, "n_valid": n}

    kappa = (p_o - p_e) / (1 - p_e)
    return {"kappa": round(kappa, 4), "n_valid": n}


def compute_exact_agreement(scores_a: list[int | None], scores_b: list[int | None]) -> dict:
    valid = [(a, b) for a, b in zip(scores_a, scores_b)
             if a is not None and b is not None]
    if not valid:
        return {"rate": float("nan"), "n_valid": 0}
    exact = sum(1 for a, b in valid if a == b)
    return {"rate": round(exact / len(valid), 4), "n_valid": len(valid)}


def compute_soft_agreement(scores_a: list[int | None], scores_b: list[int | None]) -> dict:
    """±1 档宽松一致率"""
    valid = [(a, b) for a, b in zip(scores_a, scores_b)
             if a is not None and b is not None]
    if not valid:
        return {"rate": float("nan"), "n_valid": 0}
    soft = sum(1 for a, b in valid if abs(a - b) <= 1)
    return {"rate": round(soft / len(valid), 4), "n_valid": len(valid)}


def compute_spearman(scores_a: list[int | None], scores_b: list[int | None]) -> dict:
    try:
        from scipy.stats import spearmanr as _spearmanr
    except ImportError:
        return _compute_spearman_fallback(scores_a, scores_b)

    valid_a, valid_b = [], []
    for a, b in zip(scores_a, scores_b):
        if a is not None and b is not None:
            valid_a.append(a)
            valid_b.append(b)
    if len(valid_a) < 5:
        return {"rho": float("nan"), "p_value": float("nan"), "n_valid": len(valid_a)}
    rho, p = _spearmanr(valid_a, valid_b)
    return {"rho": round(rho, 4), "p_value": round(p, 6), "n_valid": len(valid_a)}


def _compute_spearman_fallback(scores_a: list[int | None], scores_b: list[int | None]) -> dict:
    """scipy 不可用时使用手写 Spearman's ρ（适用于存在 tie 的小数据集）。"""
    valid_pairs = [(a, b) for a, b in zip(scores_a, scores_b)
                   if a is not None and b is not None]
    n = len(valid_pairs)
    if n < 5:
        return {"rho": float("nan"), "p_value": float("nan"), "n_valid": n}

    def rank(values):
        sorted_unique = sorted(set(values))
        rank_map = {v: i + 1 for i, v in enumerate(sorted_unique)}
        return [rank_map[v] for v in values]

    a_vals = [p[0] for p in valid_pairs]
    b_vals = [p[1] for p in valid_pairs]
    rank_a = rank(a_vals)
    rank_b = rank(b_vals)

    d2_sum = sum((ra - rb) ** 2 for ra, rb in zip(rank_a, rank_b))
    rho = 1 - (6 * d2_sum) / (n * (n ** 2 - 1))
    return {"rho": round(max(-1, min(1, rho)), 4), "p_value": float("nan"), "n_valid": n}


def _load_human_annotations(annotation_file: Path) -> dict[str, dict] | None:
    """加载人类标注 CSV，返回 {qid: {dim: score}} 或 None。"""
    try:
        annotations = {}
        with open(annotation_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            dims = list(HUMAN_DIM_LABELS.keys())
            for row in reader:
                qid = row.get("query_id", "").strip()
                if not qid:
                    continue
                ann = {}
                for dim in dims:
                    val = row.get(dim, "").strip()
                    if val:
                        try:
                            ann[dim] = int(float(val))
                        except ValueError:
                            ann[dim] = None
                    else:
                        ann[dim] = None
                ann["notes"] = row.get("notes", "").strip()
                annotations[qid] = ann
        return annotations if annotations else None
    except Exception as e:
        logger.error(f"Failed to load annotations from {annotation_file}: {e}")
        return None


def compute_mean_diff(scores_a: list[int | None], scores_b: list[int | None]) -> dict:
    """计算 Judge A - Judge B 的评分平均差异（正数 = A 偏高）"""
    diffs = [a - b for a, b in zip(scores_a, scores_b)
             if a is not None and b is not None]
    if not diffs:
        return {"mean_diff": float("nan"), "n_valid": 0}
    return {"mean_diff": round(sum(diffs) / len(diffs), 3), "n_valid": len(diffs)}


def compute_judge_agreement(
    judge_a_file: Path,
    judge_b_file: Path,
    output_file: Path,
    human_file: Path | None = None,
    dims: list[str] | None = None,
):
    """计算双 Judge 一致性（如有 human 标注则同时计算 Judge-Human 一致性）。"""
    if dims is None:
        dims = ALL_CORE_DIMS

    judge_a = load_generations(judge_a_file)
    judge_b = load_generations(judge_b_file)

    scores_a = _extract_scores(judge_a, dims)
    scores_b = _extract_scores(judge_b, dims)

    common_qids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    logger.info(f"Common qids for judge agreement: {len(common_qids)}")

    human_scores = None
    if human_file and human_file.exists():
        human_scores = _load_human_annotations(human_file)
        if human_scores:
            logger.info(f"Loaded {len(human_scores)} human annotations")
        else:
            logger.warning("Failed to load human annotations")

    report = {
        "judge_a_model": judge_a[0].get("judge_model", "unknown") if judge_a else "unknown",
        "judge_b_model": judge_b[0].get("judge_model", "unknown") if judge_b else "unknown",
        "common_samples": len(common_qids),
        "per_dimension": {},
        "aggregate_scores": {},
        "human_calibration": {},
    }

    for dim in dims:
        a_vals = [scores_a.get(qid, {}).get(dim) for qid in common_qids]
        b_vals = [scores_b.get(qid, {}).get(dim) for qid in common_qids]

        exact = compute_exact_agreement(a_vals, b_vals)
        soft = compute_soft_agreement(a_vals, b_vals)
        kappa = compute_cohens_kappa(a_vals, b_vals)
        rho = compute_spearman(a_vals, b_vals)
        mdiff = compute_mean_diff(a_vals, b_vals)

        label = CORE6_DIM_LABELS.get(dim, dim)
        report["per_dimension"][dim] = {
            "label": label,
            "exact_agreement": exact["rate"],
            "soft_agreement_pm1": soft["rate"],
            "cohens_kappa": kappa["kappa"],
            "spearman_rho": rho["rho"],
            "mean_diff_judge_a_minus_b": mdiff["mean_diff"],
            "n_valid": exact["n_valid"],
            "judge_a_mean": round(sum(v for v in a_vals if v is not None) / max(1, sum(1 for v in a_vals if v is not None)), 3),
            "judge_b_mean": round(sum(v for v in b_vals if v is not None) / max(1, sum(1 for v in b_vals if v is not None)), 3),
            "judge_a_distribution": dict(Counter(v for v in a_vals if v is not None)),
            "judge_b_distribution": dict(Counter(v for v in b_vals if v is not None)),
        }

    a_total = [scores_a.get(qid, {}).get("veracity") for qid in common_qids]
    b_total = [scores_b.get(qid, {}).get("veracity") for qid in common_qids]

    report["aggregate_scores"]["exact_agreement"] = compute_exact_agreement(a_total, b_total)["rate"]
    report["aggregate_scores"]["soft_agreement_pm1"] = compute_soft_agreement(a_total, b_total)["rate"]
    report["aggregate_scores"]["cohens_kappa"] = compute_cohens_kappa(a_total, b_total)["kappa"]
    report["aggregate_scores"]["spearman_rho"] = compute_spearman(a_total, b_total)["rho"]

    if human_scores:
        human_common = [q for q in common_qids if q in human_scores]
        logger.info(f"Common qids for human calibration: {len(human_common)}")
        for dim in dims:
            h_vals = [human_scores.get(qid, {}).get(dim) for qid in human_common]
            a_vals = [scores_a.get(qid, {}).get(dim) for qid in human_common]
            b_vals = [scores_b.get(qid, {}).get(dim) for qid in human_common]

            label = CORE6_DIM_LABELS.get(dim, dim)
            report["human_calibration"][dim] = {
                "label": label,
                "judge_a_vs_human": {
                    "exact_agreement": compute_exact_agreement(h_vals, a_vals)["rate"],
                    "cohens_kappa": compute_cohens_kappa(h_vals, a_vals)["kappa"],
                    "spearman_rho": compute_spearman(h_vals, a_vals)["rho"],
                    "n_valid": len(human_common),
                },
                "judge_b_vs_human": {
                    "exact_agreement": compute_exact_agreement(h_vals, b_vals)["rate"],
                    "cohens_kappa": compute_cohens_kappa(h_vals, b_vals)["kappa"],
                    "spearman_rho": compute_spearman(h_vals, b_vals)["rho"],
                    "n_valid": len(human_common),
                },
            }

    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Agreement report saved to {output_file}")

    _print_agreement_summary(report)


def _print_agreement_summary(report: dict):
    print()
    print("=" * 70)
    print("  DUAL JUDGE AGREEMENT REPORT")
    print("=" * 70)
    print(f"  Judge A: {report['judge_a_model']}")
    print(f"  Judge B: {report['judge_b_model']}")
    print(f"  Common samples: {report['common_samples']}")
    print()
    print("  Per-Dimension Judge-Judge Agreement:")
    print(f"  {'Dimension':<22s} {'Exact':>6s} {'±1':>6s} {'Kappa':>7s} {'ρ':>7s} {'Δ(A-B)':>7s}")
    print("  " + "-" * 60)
    for dim, d in report.get("per_dimension", {}).items():
        print(
            f"  {d['label']:<22s} "
            f"{d['exact_agreement']:>6.3f} "
            f"{d['soft_agreement_pm1']:>6.3f} "
            f"{d['cohens_kappa']:>7.3f} "
            f"{d['spearman_rho']:>7.3f} "
            f"{d['mean_diff_judge_a_minus_b']:>+7.3f}"
        )

    if report.get("human_calibration"):
        print()
        print("  Per-Dimension Judge-Human Agreement:")
        print(f"  {'Dimension':<22s} {'A-H Kappa':>10s} {'B-H Kappa':>10s} {'A-H ρ':>8s} {'B-H ρ':>8s}")
        print("  " + "-" * 50)
        for dim, d in report["human_calibration"].items():
            print(
                f"  {d['label']:<22s} "
                f"{d['judge_a_vs_human']['cohens_kappa']:>10.3f} "
                f"{d['judge_b_vs_human']['cohens_kappa']:>10.3f} "
                f"{d['judge_a_vs_human']['spearman_rho']:>8.3f} "
                f"{d['judge_b_vs_human']['spearman_rho']:>8.3f}"
            )

    print("=" * 70)


# ===========================================================================
# Step 4: 人类标注模板生成
# ===========================================================================

def generate_annotation_template(
    answers_file: Path,
    calibration_qids: set[str],
    output_file: Path,
):
    """生成人类标注 CSV 模板（仅 calibration 集的答案）。"""
    answers = load_generations(answers_file)
    calibration_answers = [a for a in answers if a.get("query_id", "") in calibration_qids]

    if not calibration_answers:
        logger.warning("No calibration answers found!")
        return

    fieldnames = (
        ["query_id", "query_text", "answer", "passages_summary"]
        + list(HUMAN_DIM_LABELS.keys())
        + ["notes"]
    )

    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in calibration_answers:
            passages = item.get("passages", [])
            passages_summary = " | ".join(
                p.get("text", str(p))[:100] + "..."
                for p in passages[:5]
            )

            row = {
                "query_id": item.get("query_id", ""),
                "query_text": item.get("query_text", item.get("query", "")),
                "answer": item.get("answer", ""),
                "passages_summary": passages_summary,
                "notes": "",
            }
            for dim in HUMAN_DIM_LABELS:
                if dim != "overall" and dim != "notes":
                    row[dim] = ""
            row["overall"] = ""

            writer.writerow(row)

    logger.info(f"Annotation template saved to {output_file} ({len(calibration_answers)} samples)")


# ===========================================================================
# 主流程
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-005 Dual Judge Calibration")

    parser.add_argument("--answers-file", type=str, required=True,
                        help="已有的生成结果 JSONL 文件（如 results/exp005/generations/deepseek-chat.jsonl）")
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR),
                        help="Output directory for calibration results")
    parser.add_argument("--calibration-size", type=int, default=50,
                        help="Number of queries for human calibration set")
    parser.add_argument("--verification-size", type=int, default=100,
                        help="Number of queries for dual-judge verification set")
    parser.add_argument("--judge-a", type=str, default="deepseek-reasoner",
                        help="Primary Judge model")
    parser.add_argument("--judge-b", type=str, default="glm-4.7",
                        help="Fallback Judge model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="每个 Judge 内部并发请求数（每个 Judge 独立创建 client）")
    parser.add_argument("--stagger-delay", type=float, default=0.5,
                        help="并发请求的随机错峰延迟上限（秒），避免触发 API 限流")

    parser.add_argument("--skip-sample", action="store_true",
                        help="Skip sampling step (use cached manifest)")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judge evaluation step")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps (ignore cache)")

    args = parser.parse_args()

    answers_file = Path(args.answers_file)
    if not answers_file.exists():
        logger.error(f"Answers file not found: {answers_file}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    manifest_file = Path(args.output_dir) / "sampled_manifest.json"
    judge_a_cal_file = Path(args.output_dir) / "judge_a_scores.jsonl"
    judge_b_cal_file = Path(args.output_dir) / "judge_b_scores.jsonl"
    agreement_file = Path(args.output_dir) / "judge_agreement_report.json"
    annotation_file = Path(args.output_dir) / "human_annotation_template.csv"

    # Step 1: Sample
    if args.skip_sample and manifest_file.exists():
        logger.info(f"Loading cached manifest from {manifest_file}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = sample_queries(
            answers_file,
            args.calibration_size,
            args.verification_size,
            args.seed,
        )
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"Manifest saved to {manifest_file}")

    cal_qids = set(item["qid"] for item in manifest["calibration"])
    all_qids = set(manifest["all_qids"])
    logger.info(f"Total queries: "
                f"cal={len(manifest['calibration'])}, ver={len(manifest['verification'])}")

    # Step 2: Dual judge
    if not args.skip_judge:
        logger.info(f"Running dual judge: A={args.judge_a}, B={args.judge_b}")
        run_dual_judge(
            args.judge_a, args.judge_b,
            answers_file, judge_a_cal_file, judge_b_cal_file,
            args.force, all_qids, args.concurrency, args.stagger_delay,
        )

    # Step 3: Compute agreement
    if judge_a_cal_file.exists() and judge_b_cal_file.exists():
        compute_judge_agreement(judge_a_cal_file, judge_b_cal_file, agreement_file)

    # Step 4: Generate human annotation template
    if answers_file.exists():
        generate_annotation_template(answers_file, cal_qids, annotation_file)

    logger.info("Calibration pipeline complete.")
    logger.info(f"Results in: {args.output_dir}")
