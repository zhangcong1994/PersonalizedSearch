"""
Exp-012: 将 DPO LoRA adapter 合并到基座模型，产出可直接 vLLM 加载的完整模型。

用法:
  python scripts/exp012/merge_adapter.py

  python scripts/exp012/merge_adapter.py \
      --adapter /root/autodl-tmp/models/exp012-dpo-pilot \
      --output /root/autodl-tmp/models/exp012-dpo-pilot-merged
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.config import DATA_ROOT, MODEL_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_ADAPTER = DATA_ROOT / "models" / "exp012-dpo-pilot"
DEFAULT_OUTPUT = DATA_ROOT / "models" / "exp012-dpo-pilot-merged"


def main():
    parser = argparse.ArgumentParser(description="Merge DPO LoRA adapter into base model")
    parser.add_argument("--adapter", type=str, default=str(DEFAULT_ADAPTER),
                        help="LoRA adapter 目录")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B",
                        help="基座模型 HF ID")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="合并后的输出目录")
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error(f"Adapter not found: {adapter_path}")
        return 1

    output_path = Path(args.output)
    if output_path.exists():
        logger.warning(f"Output already exists: {output_path}")
        logger.warning("Delete it first or use --output to specify a new path")
        return 1

    logger.info("=" * 60)
    logger.info("  Merge DPO LoRA Adapter")
    logger.info("=" * 60)
    logger.info(f"  Adapter:   {adapter_path}")
    logger.info(f"  Base:      {args.base_model}")
    logger.info(f"  Output:    {output_path}")
    logger.info("=" * 60)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    # Step 1: 找本地已下载的模型
    base_path = args.base_model
    hf_cache_name = f"models--{args.base_model.replace('/', '--')}"
    hf_cache_path = MODEL_CACHE_DIR / hf_cache_name
    if hf_cache_path.exists():
        logger.info(f"Using HF cache: {hf_cache_path}")
    else:
        local_name = args.base_model.split("/")[-1]
        local_path = MODEL_CACHE_DIR / local_name
        if local_path.exists() and (local_path / "config.json").exists():
            base_path = str(local_path)
            logger.info(f"Using local model: {base_path}")

    # 关键：必须用 bf16 全精度加载，不能用 4-bit。
    # 4-bit 下 merge_and_unload() 产出的是 4-bit 量化 tensor，保存后 vLLM 无法正确解析。
    logger.info("Loading base model (bf16 full precision, ~16 GB)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=str(MODEL_CACHE_DIR),
    )

    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging adapter into base model...")
    merged = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}...")
    merged.save_pretrained(str(output_path), safe_serialization=True)
    merged.config.save_pretrained(str(output_path))

    # 不保存 tokenizer 文件 —— Qwen3 的 save_pretrained 会产生重复 chat_template
    # 条目导致 vLLM 启动时 Pydantic 校验失败。vLLM 通过 --tokenizer 指定即可。
    logger.info("  Skipping tokenizer save (use --tokenizer Qwen/Qwen3-8B in vLLM serve)")

    logger.info("=" * 60)
    logger.info("  Merge complete!")
    logger.info(f"  Merged model: {output_path}")
    logger.info("")
    logger.info("  启动 vLLM:")
    logger.info(f"  vllm serve {output_path} --tokenizer Qwen/Qwen3-8B --host 0.0.0.0 --port 8000")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
