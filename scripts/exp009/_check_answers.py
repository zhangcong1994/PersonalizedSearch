"""临时脚本：教师答案质量分析 + 意外拒答交叉检查"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import RAW_DATA_DIR

QRELS_RETRIEVAL_FILE = RAW_DATA_DIR / "t2ranking" / "qrels.retrieval.train.tsv"

# 精确拒答关键词（否定/无法作答语义，误判风险低）
STRICT_REFUSAL_KW = [
    "无法确定", "无法回答", "没有提供", "无法提供",
    "资料中未", "资料中没有", "没有提及",
    "未提及", "无相关信息", "没有相关信息",
    "没能找到", "没有找到", "未找到",
    "参考资料中未",
    "未涉及", "没有涉及",
    "无法判断", "无法确认",
    "没有直接提供",
]

def load_qrels(path: Path) -> dict[str, set[str]]:
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

def has_refusal_kw(answer: str) -> bool:
    """纯关键词检测（不含长度阈值）"""
    return any(kw in answer for kw in STRICT_REFUSAL_KW)

def is_refusal(answer: str) -> bool:
    """长度+关键词双重检测"""
    if len(answer) > 400:
        return False
    return has_refusal_kw(answer)

def main():
    # ── 加载 ──
    print("Loading qrels...")
    qrels = load_qrels(QRELS_RETRIEVAL_FILE)
    print(f"  Qrels: {len(qrels):,} queries with relevance labels")

    print("Loading teacher answers...")
    with open("data/processed/exp009_teacher_answers.jsonl", "r", encoding="utf-8") as f:
        answers = [json.loads(line) for line in f if line.strip()]
    print(f"  Answers: {len(answers):,} entries")

    # ── 分类 ──
    refusal_list = []
    normal = []
    rescued_by_len = []  # 有关键词但 >400 chars，被长度阈值排除

    for e in answers:
        a = e["teacher_answer"]
        has_kw = has_refusal_kw(a)
        is_ref = is_refusal(a)
        top10_pids = [p["pid"] for p in e.get("passages", [])]
        relevant = qrels.get(e["qid"], set())
        hits = [pid for pid in top10_pids if pid in relevant]
        entry = {
            "qid": e["qid"],
            "query": e["query"],
            "answer": a,
            "top10_pids": top10_pids,
            "relevant_pids": relevant,
            "hits": hits,
            "num_hits": len(hits),
        }
        if is_ref:
            refusal_list.append(entry)
        else:
            normal.append(e)
            if has_kw and len(a) > 400:
                rescued_by_len.append(entry)

    # ── 统计 ──
    accidental = [r for r in refusal_list if r["num_hits"] > 0]
    justified = [r for r in refusal_list if r["num_hits"] == 0]
    no_cite_normal = [e for e in normal if "[来源" not in e["teacher_answer"]]

    # 意外拒答中真正短的（len<=400 且 top-10 有 relevant）
    unresolved_accidental = [r for r in refusal_list if r["num_hits"] > 0]

    print()
    print("=" * 70)
    print("  SUMMARY (len>400 = not refusal)")
    print("=" * 70)
    print(f"  Total answers:              {len(answers):,}")
    print(f"  Refusal (strict + len<=400):{len(refusal_list):,} ({len(refusal_list)/len(answers)*100:.1f}%)")
    print(f"    Accidental (has relevant): {len(unresolved_accidental):,} ({len(unresolved_accidental)/max(1,len(refusal_list))*100:.1f}% of refusals)")
    print(f"    Justified (no relevant):   {len(justified):,} ({len(justified)/max(1,len(refusal_list))*100:.1f}% of refusals)")
    print(f"  Normal answers:             {len(normal):,}")
    print(f"    No citation:              {len(no_cite_normal):,}")
    rescued_w_relevant = [r for r in rescued_by_len if r["num_hits"] > 0]
    print(f"  Rescued by len>400:         {len(rescued_by_len):,} (had keywords but answer too long to be refusal)")
    print(f"    of which w/ relevant:     {len(rescued_w_relevant):,} (were false-positive accidental refusals)")
    print(f"  No qrels for query:         {sum(1 for r in refusal_list if r['qid'] not in qrels):,}")

    # ── 长度阈值救回来的样例 ──
    print()
    print("=" * 70)
    print(f"  RESCUED BY LENGTH ({len(rescued_by_len)} total, {len(rescued_w_relevant)} w/ relevant hits)")
    print("  Had refusal keywords but answer > 400 chars, now classified as normal")
    print("=" * 70)
    rescued_w_relevant.sort(key=lambda x: x["num_hits"], reverse=True)
    for i, r in enumerate(rescued_w_relevant[:5]):
        print(f"\n--- Rescued [{i+1}] qid={r['qid']} ---")
        print(f"  Query:    {r['query'][:60]}")
        print(f"  Length:   {len(r['answer'])} chars")
        print(f"  Relevant in top-10: {r['num_hits']} / {len(r['top10_pids'])}")
        for j in range(0, min(len(r['answer']), 300), 100):
            chunk = r["answer"][j:j+100]
            print(f"    {chunk}")
        print(f"    ...({len(r['answer'])} chars total)")

    # ── 真·意外拒答详情 ──
    print()
    print("=" * 70)
    print(f"  UNRESOLVED ACCIDENTAL REFUSALS ({len(unresolved_accidental)} samples)")
    print("  Short refusal + top-10 has relevant passages — THESE WILL BE DISCARDED")
    print("=" * 70)

    unresolved_accidental.sort(key=lambda x: x["num_hits"], reverse=True)

    n_show = min(8, len(unresolved_accidental))
    for i, r in enumerate(unresolved_accidental[:n_show]):
        print(f"\n--- Accidental [{i+1}] qid={r['qid']} ---")
        print(f"  Query:    {r['query']}")
        print(f"  Length:   {len(r['answer'])} chars")
        print(f"  Relevant in top-10: {r['num_hits']} / {len(r['top10_pids'])}")
        print(f"    PIDs:   {r['hits']}")
        print(f"  Answer:")
        ans = r["answer"]
        for j in range(0, min(len(ans), 300), 100):
            chunk = ans[j:j+100]
            print(f"    {chunk}")
        if len(ans) > 300:
            print(f"    ...({len(ans)} chars total)")

    # ── 合理拒答样例 ──
    print()
    print("=" * 70)
    print(f"  JUSTIFIED REFUSALS ({len(justified)} samples)")
    print("  Teacher correctly refused — no relevant passages in top-10")
    print("=" * 70)

    random.seed(42)
    for i, r in enumerate(random.sample(justified, min(5, len(justified)))):
        print(f"\n--- Justified [{i+1}] qid={r['qid']} ---")
        print(f"  Query:    {r['query']}")
        all_rel = r["relevant_pids"]
        print(f"  All relevant pids in qrels: {len(all_rel)}")
        print(f"  Relevant in top-10: {r['num_hits']}")
        print(f"  Answer ({len(r['answer'])} chars):")
        ans = r["answer"]
        for j in range(0, min(len(ans), 300), 100):
            chunk = ans[j:j+100]
            print(f"    {chunk}")
        if len(ans) > 300:
            print(f"    ...({len(ans)} chars total)")

    # ── 无引用正常答案 ──
    print()
    print("=" * 70)
    print(f"  NO-CITATION NORMAL ({len(no_cite_normal)} samples)")
    print("=" * 70)
    for i, e in enumerate(no_cite_normal[:5]):
        a = e["teacher_answer"]
        top10_pids = [p["pid"] for p in e.get("passages", [])]
        relevant = qrels.get(e["qid"], set())
        hits = [pid for pid in top10_pids if pid in relevant]
        print(f"\n--- [{i+1}] qid={e['qid']} ---")
        print(f"  Query:    {e['query'][:60]}")
        print(f"  Relevant in top-10: {len(hits)}/{len(top10_pids)}")
        print(f"  Answer ({len(a)} chars): {a[:250]}...")

    print()

if __name__ == "__main__":
    main()
