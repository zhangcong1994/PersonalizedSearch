"""
Download all 5 reranker models for exp-004.

Models:
  - BAAI/bge-reranker-base              (278M, FlagEmbedding)
  - Alibaba-NLP/gte-multilingual-reranker-base (306M, sentence-transformers)
  - BAAI/bge-reranker-v2-m3             (568M, FlagEmbedding)
  - Qwen/Qwen3-Reranker-0.6B            (0.6B, transformers)
  - mixedbread-ai/mxbai-rerank-base-v2   (0.5B, transformers)

All models are downloaded to MODEL_CACHE_DIR (respects PERSONALIZEDSEARCH_DATA_ROOT).

Usage:
  python scripts/download_rerankers.py
  python scripts/download_rerankers.py --model bge-v2-m3   # single model
  python scripts/download_rerankers.py --dry-run            # check only
"""

import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import MODEL_CACHE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RERANKER_MODELS = {
    "bge-base": {
        "hf_id": "BAAI/bge-reranker-base",
        "params": "278M",
        "disk": "~280MB",
    },
    "gte-mul": {
        "hf_id": "Alibaba-NLP/gte-multilingual-reranker-base",
        "params": "306M",
        "disk": "~0.6GB",
    },
    "bge-v2-m3": {
        "hf_id": "BAAI/bge-reranker-v2-m3",
        "params": "568M",
        "disk": "~560MB",
    },
    "qwen3-rerank": {
        "hf_id": "Qwen/Qwen3-Reranker-0.6B",
        "params": "0.6B",
        "disk": "~1.2GB",
    },
    "mxbai-v2": {
        "hf_id": "mixedbread-ai/mxbai-rerank-base-v2",
        "params": "0.5B",
        "disk": "~1GB",
    },
}


def download_model(model_id: str, model_info: dict, cache_dir: str):
    hf_id = model_info["hf_id"]
    local_dir = os.path.join(cache_dir, hf_id.replace("/", "--"))

    if os.path.isdir(local_dir) and any(
        f.endswith((".bin", ".safetensors", ".json")) for f in os.listdir(local_dir)
    ):
        logger.info(f"[{model_id}] Already downloaded: {local_dir}")
        return

    logger.info(f"[{model_id}] Downloading {hf_id} ({model_info['params']}, {model_info['disk']}) ...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(f"[{model_id}] Downloaded to: {local_dir}")
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download exp-004 reranker models")
    parser.add_argument(
        "--model",
        choices=list(RERANKER_MODELS.keys()),
        default=None,
        help="Download a single model (default: all 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )
    args = parser.parse_args()

    print(f"Model cache directory: {MODEL_CACHE_DIR}")
    print()

    model_ids = [args.model] if args.model else list(RERANKER_MODELS.keys())

    total_params = 0
    for model_id in model_ids:
        info = RERANKER_MODELS[model_id]
        total_params += float(info["params"].replace("B", "").replace("M", "")) * (
            1000 if "B" in info["params"] else 1
        )
        status = "  [DRY RUN]" if args.dry_run else ""
        logger.info(f"Model: {model_id:<14} {info['hf_id']:<50} {info['params']:>6}  {info['disk']}{status}")

    print()
    if args.dry_run:
        total_disk = "~3.5GB"
        logger.info(f"Dry run complete. Would download {len(model_ids)} model(s), ~{total_params:.0f}M params, {total_disk} total")
    else:
        for model_id in model_ids:
            download_model(model_id, RERANKER_MODELS[model_id], str(MODEL_CACHE_DIR))

        logger.info(f"All {len(model_ids)} model(s) downloaded to {MODEL_CACHE_DIR}")


if __name__ == "__main__":
    main()
