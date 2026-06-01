"""
Exp-009 阶段三：教师生成（qwen3-max Batch API）。

使用阿里云百炼 Batch File API（OpenAI 兼容），将 5000 条 query 的检索结果
拼成 prompt，批量调用 qwen3-max 生成参考答案。Batch 模式费用仅为实时调用的 50%。

流程:
  python scripts/exp009/generate_teacher_answers.py test     # (推荐) 免费全链路测试，¥0
  python scripts/exp009/generate_teacher_answers.py prepare  # 准备 batch 输入 JSONL
  python scripts/exp009/generate_teacher_answers.py submit   # 上传文件 + 创建任务 + 轮询
  python scripts/exp009/generate_teacher_answers.py align    # 下载结果 + 对齐输出

费用估算: 5000 × ~2000 input tokens × ¥0.003/K × 50% = ~¥15

参考: https://help.aliyun.com/zh/model-studio/batch-interfaces-compatible-with-openai/
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from src.generation.prompts import SYSTEM_PROMPT, FEW_SHOT, CONTEXT_TEMPLATE, QUESTION_TEMPLATE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────────

MODEL_ID = "qwen3-max"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BATCH_ENDPOINT = "/v1/chat/completions"

DEFAULT_INPUT = DATA_ROOT / "data" / "processed" / "exp009_reranked_top10.jsonl"
DEFAULT_BATCH_INPUT = DATA_ROOT / "data" / "processed" / "exp009_batch_input.jsonl"
DEFAULT_OUTPUT = DATA_ROOT / "data" / "processed" / "exp009_teacher_answers.jsonl"
DEFAULT_BATCH_RESULT = DATA_ROOT / "data" / "processed" / "exp009_batch_result.jsonl"
STATE_FILE = DATA_ROOT / "data" / "processed" / "exp009_batch_state.json"

# 测试模型（阿里云百炼提供的免费测试端点）
TEST_MODEL_ID = "batch-test-model"
TEST_ENDPOINT = "/v1/chat/ds-test"
TEST_STATE_FILE = DATA_ROOT / "data" / "processed" / "exp009_batch_test_state.json"

SYSTEM_TEXT = SYSTEM_PROMPT + FEW_SHOT


# ── 工具函数 ──────────────────────────────────────────────

def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return DATA_ROOT / p if not p.is_absolute() else p


def load_reranked(path: Path) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data):,} entries from {path.name}")
    return data


def format_passage(idx: int, pid: str, text: str) -> str:
    return f"[{idx}] 来源: {pid}\n{text}"


def build_user_message(query: str, passages: list[dict]) -> str:
    passage_lines = [format_passage(i + 1, p["pid"], p["text"]) for i, p in enumerate(passages)]
    context = "\n\n".join(passage_lines)
    return CONTEXT_TEMPLATE.format(context=context) + "\n\n" + QUESTION_TEMPLATE.format(question=query)


def build_messages(query: str, passages: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user", "content": build_user_message(query, passages)},
    ]


def build_batch_line(qid: str, query: str, passages: list[dict]) -> dict:
    return {
        "custom_id": qid,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": {
            "model": MODEL_ID,
            "messages": build_messages(query, passages),
            "temperature": 0.3,
            "max_tokens": 1024,
            "enable_thinking": False,
        },
    }


# ── Step 1: prepare batch input JSONL ────────────────────

def cmd_prepare(args):
    inp = _resolve(args.input)
    outp = _resolve(args.batch_input)

    logger.info("=" * 60)
    logger.info("  Step 3a: Prepare batch input JSONL")
    logger.info(f"  Input:  {inp}")
    logger.info(f"  Output: {outp}")
    logger.info(f"  Model:  {MODEL_ID}")
    logger.info("=" * 60)

    entries = load_reranked(inp)
    if not entries:
        logger.error("No entries loaded")
        return 1

    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        for entry in entries:
            batch_line = build_batch_line(entry["qid"], entry["query"], entry["results"])
            f.write(json.dumps(batch_line, ensure_ascii=False) + "\n")

    file_size_mb = outp.stat().st_size / 1024 / 1024
    logger.info(f"Wrote {len(entries):,} batch requests → {outp} ({file_size_mb:.1f} MB)")
    logger.info(f"Estimated cost: {len(entries):,} × ~2000 tokens × ¥0.003/K × 50% ≈ ¥{len(entries)*2000/1000*0.003*0.5:.0f}")
    logger.info("=" * 60)
    logger.info("  Next: python scripts/exp009/generate_teacher_answers.py submit")
    logger.info("=" * 60)
    return 0


# ── Step 2: submit batch job ─────────────────────────────

def _get_client():
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")

    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _upload_file(client, file_path: Path) -> str:
    logger.info(f"Uploading batch input file: {file_path.name}...")
    with open(file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"  File ID: {file_obj.id}")
    return file_obj.id


def _create_batch(client, file_id: str) -> str:
    logger.info(f"Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window="24h",
    )
    logger.info(f"  Batch ID: {batch.id}")
    return batch.id


def _poll_batch(client, batch_id: str) -> str:
    logger.info(f"Waiting for batch job {batch_id} to complete...")
    status = ""
    poll_count = 0
    while status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(30)
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        poll_count += 1
        progress = ""
        if hasattr(batch, "request_counts") and batch.request_counts:
            rc = batch.request_counts
            progress = f" | completed={rc.completed} failed={rc.failed} total={rc.total}"
        logger.info(f"  [{poll_count}] status={status}{progress}")
    return status


def _save_state(state: dict):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    logger.info(f"State saved to {STATE_FILE}")


def cmd_submit(args):
    inp = _resolve(args.batch_input)

    logger.info("=" * 60)
    logger.info("  Step 3b: Submit batch job")
    logger.info(f"  Input:  {inp}")
    logger.info("=" * 60)

    if not inp.exists():
        logger.error(f"File not found: {inp}. Run 'prepare' first.")
        return 1

    client = _get_client()

    # 断点续跑：如果 STATE_FILE 存在，跳过已完成的步骤
    state = {}
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info(f"Found saved state: {json.dumps(state, indent=2)}")

    file_id = state.get("file_id")
    if not file_id:
        file_id = _upload_file(client, inp)
        state["file_id"] = file_id
        _save_state(state)

    batch_id = state.get("batch_id")
    if not batch_id:
        batch_id = _create_batch(client, file_id)
        state["batch_id"] = batch_id
        _save_state(state)

    status = state.get("status", "")
    if status not in ("completed", "failed", "expired", "cancelled"):
        status = _poll_batch(client, batch_id)
        state["status"] = status

        if status == "completed":
            batch = client.batches.retrieve(batch_id)
            state["output_file_id"] = batch.output_file_id
            if hasattr(batch, "error_file_id") and batch.error_file_id:
                state["error_file_id"] = batch.error_file_id
        elif status == "failed":
            batch = client.batches.retrieve(batch_id)
            if hasattr(batch, "errors") and batch.errors:
                state["errors"] = str(batch.errors)
        _save_state(state)

    if status == "completed":
        logger.info("Batch job completed!")
        logger.info(f"  Output file ID: {state.get('output_file_id')}")
        logger.info("=" * 60)
        logger.info("  Next: python scripts/exp009/generate_teacher_answers.py align")
        logger.info("=" * 60)
    elif status == "failed":
        logger.error(f"Batch job FAILED: {state.get('errors', 'unknown')}")
        return 1
    else:
        logger.warning(f"Batch job status: {status}. Run 'submit' again if it's still processing.")
        return 1

    return 0


# ── Step 3: align results ────────────────────────────────

def _download_results(client, file_id: str, output_path: Path):
    logger.info(f"Downloading results (file_id={file_id})...")
    content = client.files.content(file_id)
    content.write_to_file(str(output_path))
    lines = sum(1 for _ in open(output_path, "r", encoding="utf-8"))
    logger.info(f"  Downloaded {lines:,} lines → {output_path}")


def cmd_align(args):
    if not STATE_FILE.exists():
        logger.error(f"State file not found: {STATE_FILE}. Run 'submit' first.")
        return 1

    with open(STATE_FILE, "r", encoding="utf-8") as f:
        state = json.load(f)

    output_file_id = state.get("output_file_id")
    if not output_file_id:
        logger.error("No output_file_id in state. Did the batch job complete?")
        return 1

    inp = _resolve(args.input)
    batch_result_path = _resolve(args.batch_result)
    outp = _resolve(args.output)

    logger.info("=" * 60)
    logger.info("  Step 3c: Align batch results")
    logger.info(f"  Reference: {inp}")
    logger.info(f"  Batch out: {batch_result_path}")
    logger.info(f"  Final:     {outp}")
    logger.info("=" * 60)

    if not batch_result_path.exists():
        client = _get_client()
        _download_results(client, output_file_id, batch_result_path)

    reranked = load_reranked(inp)
    qid_to_entry: dict[str, dict] = {e["qid"]: e for e in reranked}

    answers: dict[str, str] = {}
    errors: list[str] = []

    with open(batch_result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record.get("custom_id", "")
            response = record.get("response", {})
            status_code = response.get("status_code", 0)

            if status_code == 200:
                body = response.get("body", {})
                choices = body.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    content = msg.get("content", "")
                    answers[qid] = content
                else:
                    errors.append(qid)
                    answers[qid] = ""
            else:
                errors.append(qid)
                answers[qid] = ""

    outp.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with open(outp, "w", encoding="utf-8") as f:
        for entry in reranked:
            qid = entry["qid"]
            answer = answers.get(qid, "")
            out_entry = {
                "qid": qid,
                "query": entry["query"],
                "passages": entry["results"],
                "teacher_answer": answer,
                "model": MODEL_ID,
                "temperature": 0.3,
            }
            if answer:
                f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
                written += 1
            else:
                skipped += 1

    logger.info(f"Aligned: {written:,} answers written, {skipped:,} empty/errored")
    if errors:
        logger.warning(f"  {len(errors):,} requests had errors or empty responses")
    logger.info(f"Output: {outp}")
    logger.info("=" * 60)
    logger.info("  Step 3 complete")
    logger.info("=" * 60)
    return 0


# ── Step 0: test with batch-test-model ───────────────────

def cmd_test(args):
    inp = _resolve(args.batch_input) if args.batch_input else DEFAULT_BATCH_INPUT

    logger.info("=" * 60)
    logger.info("  Step 0: Batch API smoke test")
    logger.info(f"  Source:      {inp}")
    logger.info(f"  Test lines:  {args.n}")
    logger.info(f"  Model:       {TEST_MODEL_ID} (test, ¥0)")
    logger.info("=" * 60)

    if not inp.exists():
        logger.error(f"Batch input not found: {inp}. Run 'prepare' first.")
        return 1

    # 从正式 batch 输入文件中取前 N 条，替换为测试模型
    test_path = _resolve(args.output)
    with open(inp, "r", encoding="utf-8") as fin, open(test_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= args.n:
                break
            record = json.loads(line.strip())
            record["url"] = TEST_ENDPOINT
            record["body"]["model"] = TEST_MODEL_ID
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_size_kb = test_path.stat().st_size / 1024
    logger.info(f"Test file: {test_path.name} ({args.n} requests, {file_size_kb:.1f} KB)")

    client = _get_client()

    # 上传
    logger.info(f"Uploading test file...")
    with open(test_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"  File ID: {file_obj.id}")

    # 创建 batch 任务
    logger.info(f"Creating test batch job...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=TEST_ENDPOINT,
        completion_window="24h",
    )
    batch_id = batch.id
    logger.info(f"  Batch ID: {batch_id}")

    # 轮询
    logger.info(f"Waiting for test job {batch_id}...")
    status = ""
    poll_count = 0
    while status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(10)
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        poll_count += 1
        logger.info(f"  [{poll_count}] status={status}")

    if status != "completed":
        batch = client.batches.retrieve(batch_id)
        err_info = str(batch.errors) if hasattr(batch, "errors") and batch.errors else "unknown"
        logger.error(f"Test batch FAILED: {err_info}")
        logger.error(f"See: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return 1

    # 下载输出
    output_file_id = batch.output_file_id
    logger.info(f"Test completed! Output file ID: {output_file_id}")

    test_result_path = _resolve(args.result)
    content = client.files.content(output_file_id)
    content.write_to_file(str(test_result_path))

    lines = sum(1 for _ in open(test_result_path, "r", encoding="utf-8"))
    logger.info(f"Downloaded {lines} lines → {test_result_path}")

    # 快速查验
    logger.info("-" * 40)
    logger.info("Checking first 3 results:")
    with open(test_result_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            record = json.loads(line.strip())
            qid = record.get("custom_id", "?")
            sc = record.get("response", {}).get("status_code", 0)
            choices = record.get("response", {}).get("body", {}).get("choices", [])
            msg = choices[0].get("message", {}).get("content", "") if choices else "?"
            preview = msg[:80].replace("\n", " ") if msg else "?"
            logger.info(f"  [{qid}] status={sc} content={preview}...")

    if lines == args.n:
        logger.info(f"All {args.n} requests returned results — batch API pipeline is working correctly!")
    else:
        logger.warning(f"Expected {args.n} lines, got {lines}")

    logger.info("=" * 60)
    logger.info("  Test passed! Pipeline is ready for real submission.")
    logger.info(f"  Next: python scripts/exp009/generate_teacher_answers.py submit")
    logger.info("=" * 60)
    return 0


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp-009 阶段三：教师生成（qwen3-max Batch API）"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_test = sub.add_parser("test", help="先用 batch-test-model 做免费全链路测试（¥0）")
    p_test.add_argument("-n", type=int, default=50,
                        help="测试条数（default: 50, 不超过 100）")
    p_test.add_argument("--batch-input", default=None,
                        help="正式 batch input JSONL，默认从中取前N条")
    p_test.add_argument("--output", default=str(DATA_ROOT / "data" / "processed" / "exp009_batch_test_input.jsonl"),
                        help="测试输入文件路径")
    p_test.add_argument("--result", default=str(DATA_ROOT / "data" / "processed" / "exp009_batch_test_result.jsonl"),
                        help="测试结果文件路径")

    p_prepare = sub.add_parser("prepare", help="准备 batch 输入 JSONL")
    p_prepare.add_argument("--input", default=str(DEFAULT_INPUT),
                           help=f"Reranked top-10 JSONL (default: {DEFAULT_INPUT})")
    p_prepare.add_argument("--batch-input", default=str(DEFAULT_BATCH_INPUT),
                           help=f"Output batch input JSONL (default: {DEFAULT_BATCH_INPUT})")

    p_submit = sub.add_parser("submit", help="上传文件 + 创建 batch 任务 + 轮询")
    p_submit.add_argument("--batch-input", default=str(DEFAULT_BATCH_INPUT),
                          help=f"Batch input JSONL (default: {DEFAULT_BATCH_INPUT})")

    p_align = sub.add_parser("align", help="下载 batch 结果 + 对齐输出")
    p_align.add_argument("--input", default=str(DEFAULT_INPUT),
                         help=f"Reranked top-10 JSONL for reference (default: {DEFAULT_INPUT})")
    p_align.add_argument("--batch-result", default=str(DEFAULT_BATCH_RESULT),
                         help=f"Batch result JSONL path (default: {DEFAULT_BATCH_RESULT})")
    p_align.add_argument("--output", default=str(DEFAULT_OUTPUT),
                         help=f"Final teacher answers JSONL (default: {DEFAULT_OUTPUT})")

    args = parser.parse_args()

    if args.cmd == "test":
        return cmd_test(args)
    elif args.cmd == "prepare":
        return cmd_prepare(args)
    elif args.cmd == "submit":
        return cmd_submit(args)
    elif args.cmd == "align":
        return cmd_align(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
