"""将 DashScope 批量调用结果对齐为与本地生成结果相同的 JSONL 格式。

输入：
  - prompts_for_batch.txt: 包含 QUERY N | qid=XXXX 映射
  - batch output JSONL: DashScope 批量调用返回的结果
  - 参考 JSONL: 已有的本地生成结果（如 qwen3-4b-nothink.jsonl），提供 query_text 和 passages

输出：
  - results/exp005/generation/qwen3-max.jsonl: 对齐后的结果

用法：
  python scripts/exp005/align_batch_result.py
"""

import json
import re
import sys
from pathlib import Path


RESULTS_DIR = Path("results/exp005")
BATCH_PROMPTS_FILE = RESULTS_DIR / "prompts_for_batch.txt"
BATCH_OUTPUT_FILE = RESULTS_DIR / "generation" / "78c92c51-76a8-4863-99ac-7cbdc3ffb594_1779970904759_success.jsonl"
REFERENCE_FILE = RESULTS_DIR / "generation" / "qwen3-4b-nothink.jsonl"
OUTPUT_FILE = RESULTS_DIR / "generation" / "qwen3-max.jsonl"
MODEL_ID = "qwen3-max"


def parse_prompts_mapping(filepath: Path) -> dict[str, str]:
    """从 prompts_for_batch.txt 中提取 custom_id -> qid 映射。"""
    mapping = {}
    pattern = re.compile(r"^===== QUERY (\d+) \| qid=(\d+) =====")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                custom_id = m.group(1)
                qid = m.group(2)
                mapping[custom_id] = qid
    print(f"Parsed {len(mapping)} custom_id -> qid mappings from {filepath.name}")
    return mapping


def parse_batch_output(filepath: Path) -> dict[str, str]:
    """从批量调用输出中提取 custom_id -> answer 映射。"""
    answers = {}
    errors = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            custom_id = record.get("custom_id", "")
            if record.get("error"):
                errors += 1
                answers[custom_id] = f"[BATCH ERROR: {record['error']}]"
                continue
            try:
                content = record["response"]["body"]["choices"][0]["message"]["content"]
                answers[custom_id] = content
            except (KeyError, IndexError) as e:
                errors += 1
                answers[custom_id] = f"[PARSE ERROR: {e}]"
    print(f"Parsed {len(answers)} answers from {filepath.name} ({errors} errors)")
    return answers


def load_reference_info(filepath: Path) -> dict[str, dict]:
    """从已有结果文件中加载 qid -> {query_text, passages} 映射。"""
    ref = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["query_id"]
            ref[qid] = {
                "query_text": record["query_text"],
                "passages": record["passages"],
            }
    print(f"Loaded {len(ref)} reference records from {filepath.name}")
    return ref


def main():
    if not BATCH_PROMPTS_FILE.exists():
        print(f"ERROR: Prompts file not found: {BATCH_PROMPTS_FILE}")
        sys.exit(1)
    if not BATCH_OUTPUT_FILE.exists():
        print(f"ERROR: Batch output file not found: {BATCH_OUTPUT_FILE}")
        sys.exit(1)
    if not REFERENCE_FILE.exists():
        print(f"ERROR: Reference file not found: {REFERENCE_FILE}")
        sys.exit(1)

    custom_to_qid = parse_prompts_mapping(BATCH_PROMPTS_FILE)
    custom_to_answer = parse_batch_output(BATCH_OUTPUT_FILE)
    qid_to_ref = load_reference_info(REFERENCE_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    matched = 0
    missing_qid = 0
    missing_answer = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for custom_id, qid in custom_to_qid.items():
            if custom_id not in custom_to_answer:
                missing_answer += 1
                print(f"WARNING: custom_id={custom_id} (qid={qid}) has no answer in batch output")
                continue

            answer = custom_to_answer[custom_id]

            if qid not in qid_to_ref:
                missing_qid += 1
                print(f"WARNING: qid={qid} not found in reference file")
                continue

            ref = qid_to_ref[qid]

            result = {
                "query_id": qid,
                "query_text": ref["query_text"],
                "model_id": MODEL_ID,
                "answer": answer,
                "passages": ref["passages"],
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            matched += 1

    print(f"\nResults:")
    print(f"  Total queries in prompts:  {len(custom_to_qid)}")
    print(f"  Total answers in batch:    {len(custom_to_answer)}")
    print(f"  Matched and written:       {matched}")
    print(f"  Missing answer:            {missing_answer}")
    print(f"  Missing reference qid:     {missing_qid}")
    print(f"\nOutput: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
