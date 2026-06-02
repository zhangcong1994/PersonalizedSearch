"""
Exp-005 多模型批量生成脚本。

支持两类模型：
  - 本地模型（vLLM OpenAI-compatible server 或 HuggingFace Transformers）
  - API 模型（DeepSeek / OpenAI，复用 src/intent/api_client.py）

输入：查询 + 精排后的 top-K passages（exp-004 输出或理想排序）
输出：results/exp005/generations/{model_id}.jsonl

用法：
  # API 模型
  python scripts/generate_exp005.py --model deepseek-chat --input data/exp005_queries.jsonl

  # 本地 vLLM 模型
  python scripts/generate_exp005.py --model qwen3-4b --local --vllm-url http://localhost:8000/v1

  # 批量全部模型
  python scripts/generate_exp005.py --all --input data/exp005_queries.jsonl
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import DATA_ROOT, PROJECT_ROOT
from src.generation.prompts import PromptManager, get_default_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = DATA_ROOT / "results" / "exp005"
GENERATIONS_DIR = RESULTS_DIR / "generations"

MODEL_CONFIGS = {
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "backend": "local",
        "params": "1.5B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "backend": "local",
        "params": "3B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "qwen3-4b": {
        "hf_id": "Qwen/Qwen3-4B",
        "backend": "local",
        "params": "4B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
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
    "qwen3-8b": {
        "hf_id": "Qwen/Qwen3-8B",
        "backend": "local",
        "params": "8B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "backend": "local",
        "params": "7B",
        "max_tokens": 1024,
        "temperature": 0.3,
        "quantization": "int4",
        "alt_hf_id": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    },
    "glm4-9b": {
        "hf_id": "THUDM/glm-4-9b-chat",
        "backend": "local",
        "params": "9B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "local",
        "params": "8B",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "deepseek-chat": {
        "hf_id": None,
        "backend": "api",
        "provider": "deepseek",
        "model": "deepseek-chat",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
    "gpt-4o-mini": {
        "hf_id": None,
        "backend": "api",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_tokens": 1024,
        "temperature": 0.3,
    },
}


def load_input_queries(filepath: Path) -> list[dict]:
    """加载输入查询 + passages 的 JSONL 文件。"""
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
    prompt_manager: PromptManager,
) -> tuple[str, str]:
    system_prompt = prompt_manager.get_system_prompt()

    context_parts = []
    for i, p in enumerate(passages):
        pid = p.get("pid", f"doc-{i}")
        rank = p.get("rank", i + 1)
        text = p.get("text", "")
        text_truncated = text[:800]
        context_parts.append(f"[{rank}] 来源: {pid}\n{text_truncated}")

    context = "\n\n".join(context_parts)
    user_prompt = prompt_manager.build_user_prompt(query_text, context)

    return system_prompt, user_prompt


def generate_api(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptManager,
    output_file: Path,
    api_key: Optional[str] = None,
):
    """使用 API 模型生成答案。"""
    from src.intent.api_client import APIClientFactory, LangChainLLMClient
    from langchain_openai import ChatOpenAI

    provider = model_config["provider"]
    model_name = model_config["model"]

    if provider == "deepseek":
        deepseek_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        llm = ChatOpenAI(
            model=model_name,
            api_key=deepseek_key,
            base_url="https://api.deepseek.com/v1",
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.3),
        )
    elif provider == "openai":
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(
            model=model_name,
            api_key=openai_key,
            max_tokens=model_config.get("max_tokens", 1024),
            temperature=model_config.get("temperature", 0.3),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    client = LangChainLLMClient(llm)
    _run_generation(client, model_config, query_data, prompt_manager, output_file, "api")


def generate_local_vllm(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptManager,
    output_file: Path,
    vllm_url: str,
):
    """使用 vLLM OpenAI-compatible API 生成答案。"""
    from src.intent.api_client import LangChainLLMClient
    from langchain_openai import ChatOpenAI

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


def generate_local_hf(
    model_config: dict,
    query_data: list[dict],
    prompt_manager: PromptManager,
    output_file: Path,
):
    """使用 HuggingFace Transformers 直接生成答案（备选方案）。"""
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
    prompt_manager: PromptManager,
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
                logger.info(f"  Generated {i + 1}/{len(query_data)} "
                            f"({elapsed_ms}ms avg on last query)")

    logger.info(f"Saved {len(query_data)} generations to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp-005 Multi-Model Generation Runner")
    parser.add_argument("--model", type=str, help="Model ID (e.g., deepseek-chat, qwen3-4b)")
    parser.add_argument("--all", action="store_true", help="Run all models")
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

    query_data = load_input_queries(input_path)
    if args.max > 0:
        query_data = query_data[:args.max]

    prompt_manager = get_default_prompts()

    models_to_run = []
    if args.all:
        models_to_run = list(MODEL_CONFIGS.keys())
    elif args.model:
        if args.model not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}")
            sys.exit(1)
        models_to_run = [args.model]
    else:
        logger.error("Specify --model or --all")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for model_id in models_to_run:
        config = MODEL_CONFIGS[model_id]
        config["id"] = model_id

        output_file = output_dir / f"{model_id}.jsonl"

        if output_file.exists() and not args.force:
            logger.info(f"Skipping {model_id} (cached at {output_file})")
            continue

        backend = args.backend
        if backend == "auto":
            backend = config.get("backend", "api")

        logger.info(f"Generating with {model_id} ({backend})...")

        if backend == "api":
            generate_api(config, query_data, prompt_manager, output_file, args.api_key)
        elif backend == "vllm":
            generate_local_vllm(config, query_data, prompt_manager, output_file, args.vllm_url)
        elif backend == "hf":
            generate_local_hf(config, query_data, prompt_manager, output_file)
        else:
            logger.error(f"Unknown backend: {backend}")
            sys.exit(1)

    logger.info("All generations complete")
