"""
Exp-010 生成脚本 —— 支持多 Prompt 版本的批量生成。

与 exp-005 的 generate_exp005.py 功能对应，但：
  - 使用 src/generation/prompts_v2.py（PromptV2Manager）管理 prompt 版本
  - 输入侧统一用 [文档 N] 格式（而非旧版 [N] 来源: pid）
  - 输出默认写到 results/exp010/generations/

用法：
  # 基线（v0 prompt）
  python scripts/exp010/generate_exp010.py --model qwen3-4b-nothink \\
      --input data/exp005_queries.jsonl --prompt-version v0

  # Phase 1（v1-full prompt）
  python scripts/exp010/generate_exp010.py --model qwen3-4b-nothink \\
      --input data/exp005_queries.jsonl --prompt-version v1-full

  # 消融变体
  python scripts/exp010/generate_exp010.py --model qwen3-4b-nothink \\
      --input results/exp010/queries_50.jsonl --prompt-version abl-no-cot
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.config import DATA_ROOT
from src.generation.prompts_v2 import PromptV2Manager, list_versions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp010"
GENERATIONS_DIR = RESULTS_DIR / "generations"

MODEL_CONFIGS = {
    "qwen3-4b-nothink": {
        "hf_id": "Qwen/Qwen3-4B",
        "backend": "local",
        "params": "4B",
        "max_tokens": 1024,
        "temperature": 0.3,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen3-8b-nothink": {
        "hf_id": "Qwen/Qwen3-8B",
        "backend": "local",
        "params": "8B",
        "max_tokens": 1024,
        "temperature": 0.3,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "qwen3-8b-thinking": {
        "hf_id": "Qwen/Qwen3-8B",
        "backend": "local",
        "params": "8B",
        "max_tokens": 1024,
        "temperature": 0.3,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    },
    "qwen3-max": {
        "hf_id": None,
        "backend": "api",
        "provider": "dashscope",
        "model": "qwen3-max",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
}


def load_input_queries(filepath: Path) -> list[dict]:
    queries = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    logger.info(f"Loaded {len(queries)} input queries from {filepath.name}")
    return queries


def build_prompt(
    query_text: str,
    passages: list[dict],
    prompt_manager: PromptV2Manager,
) -> tuple[str, str]:
    """构造 system prompt 和 user prompt。输入侧统一用 [文档 N] 格式。"""
    system_prompt = prompt_manager.get_system_prompt()

    context_parts = []
    for i, p in enumerate(passages):
        pid = p.get("pid", f"doc-{i}")
        rank = p.get("rank", i + 1)
        text = p.get("text", "")
        text_truncated = text[:800]
        # 与旧版关键差异：用 [文档 N] 而非 [N] 来源: pid
        context_parts.append(f"[文档 {rank}]\n{text_truncated}")

    context = "\n\n".join(context_parts)
    user_prompt = prompt_manager.build_user_prompt(query_text, context)

    return system_prompt, user_prompt


def generate_local_vllm(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    vllm_url: str,
):
    """使用 vLLM OpenAI-compatible API 生成答案。"""
    from langchain_openai import ChatOpenAI
    from src.intent.api_client import LangChainLLMClient

    llm_kwargs = dict(
        model=model_config["hf_id"],
        api_key="not-needed",
        base_url=vllm_url,
        max_tokens=model_config.get("max_tokens", 1024),
        temperature=model_config.get("temperature", 0.3),
    )
    extra_body = model_config.get("extra_body")
    if extra_body:
        llm_kwargs["model_kwargs"] = {"extra_body": extra_body}

    llm = ChatOpenAI(**llm_kwargs)
    client = LangChainLLMClient(llm)
    _run_generation(client, model_config, query_data, prompt_manager, output_file, "vllm")


def generate_api(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    api_key: Optional[str] = None,
):
    """使用 API 模型生成答案。"""
    from langchain_openai import ChatOpenAI
    from src.intent.api_client import LangChainLLMClient

    provider = model_config["provider"]
    model_name = model_config["model"]

    if provider == "dashscope":
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        llm = ChatOpenAI(
            model=model_name,
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.3),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = LangChainLLMClient(llm)
    _run_generation(client, model_config, query_data, prompt_manager, output_file, "api")


def generate_local_hf(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
):
    """使用 HuggingFace Transformers 直接生成（备选方案）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id = model_config.get("alt_hf_id", model_config["hf_id"])
    logger.info(f"Loading model from {hf_id}...")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=True,
        cache_dir=str(DATA_ROOT / "models"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=str(DATA_ROOT / "models"),
    )
    model.eval()

    os.makedirs(output_file.parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            system_prompt, user_prompt = build_prompt(query_text, passages, prompt_manager)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            start_time = time.time()
            inputs = tokenizer([text], return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=model_config.get("max_tokens", 1024),
                    temperature=model_config.get("temperature", 0.3),
                    do_sample=True,
                )

            output_ids = generated_ids[0][len(inputs.input_ids[0]):]
            answer = tokenizer.decode(output_ids, skip_special_tokens=True)
            elapsed_ms = int((time.time() - start_time) * 1000)

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": model_config.get("id", hf_id),
                "answer": answer.strip(),
                "passages": passages,
                "system_prompt": system_prompt,
                "generation_time_ms": elapsed_ms,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{len(query_data)}")

    logger.info(f"Saved {len(query_data)} generations to {output_file}")


def _run_generation(
    client,
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptV2Manager,
    output_file: Path,
    backend: str,
):
    """通用生成循环。"""
    from langchain_core.messages import HumanMessage

    os.makedirs(output_file.parent, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(query_data):
            qid = item.get("query_id", f"q-{i}")
            query_text = item.get("query_text", item.get("query", ""))
            passages = item.get("passages", [])

            system_prompt, user_prompt = build_prompt(query_text, passages, prompt_manager)

            # 纯文本拼接（与评估格式一致，不使用 chat_template）
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            start_time = time.time()
            try:
                answer = client.generate(full_prompt)
            except Exception as e:
                logger.warning(f"  Error on query {qid}: {e}")
                answer = f"[GENERATION ERROR: {e}]"
            elapsed_ms = int((time.time() - start_time) * 1000)

            result = {
                "query_id": qid,
                "query_text": query_text,
                "model_id": model_config.get("id", model_config.get("model", "unknown")),
                "answer": answer.strip(),
                "passages": passages,
                "system_prompt": system_prompt,
                "generation_time_ms": elapsed_ms,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % 10 == 0:
                logger.info(
                    f"  Generated {i + 1}/{len(query_data)} "
                    f"({elapsed_ms}ms avg on last query)"
                )

    logger.info(f"Saved {len(query_data)} generations to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-010 Multi-Version Generation Runner")
    parser.add_argument("--model", type=str, required=True,
                        help="Model ID (e.g., qwen3-4b-nothink)")
    parser.add_argument("--prompt-version", type=str, default="v0",
                        choices=list_versions(),
                        help="Prompt 版本 (v0=基线, v1-full=Phase1, v2=Phase2, v3=Phase3, abl-*=消融)")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL file with queries and passages")
    parser.add_argument("--output-dir", type=str, default=str(GENERATIONS_DIR),
                        help="Output directory for generation results")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "api", "vllm", "hf"],
                        help="Inference backend (auto-detect from config)")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1",
                        help="vLLM server URL")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API Key (default: from env)")
    parser.add_argument("--max", type=int, default=0,
                        help="Max queries to process (0=all)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-generation (ignore cache)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if args.model not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    query_data = load_input_queries(input_path)
    if args.max > 0:
        query_data = query_data[:args.max]

    prompt_manager = PromptV2Manager(version=args.prompt_version)
    model_config = MODEL_CONFIGS[args.model]
    model_config["id"] = args.model

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 文件名含 prompt 版本
    output_file = output_dir / f"{args.model}-{args.prompt_version}.jsonl"

    if output_file.exists() and not args.force:
        logger.info(f"Skipping {args.model}-{args.prompt_version} (cached at {output_file})")
        sys.exit(0)

    backend = args.backend
    if backend == "auto":
        backend = model_config.get("backend", "api")

    logger.info(
        f"Generating: model={args.model} prompt={args.prompt_version} "
        f"backend={backend} queries={len(query_data)}"
    )

    if backend == "api":
        generate_api(model_config, query_data, prompt_manager, output_file, args.api_key)
    elif backend == "vllm":
        generate_local_vllm(model_config, query_data, prompt_manager, output_file, args.vllm_url)
    elif backend == "hf":
        generate_local_hf(model_config, query_data, prompt_manager, output_file)
    else:
        logger.error(f"Unknown backend: {backend}")
        sys.exit(1)

    logger.info(f"Done: {output_file}")
