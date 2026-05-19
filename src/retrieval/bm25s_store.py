"""
Sharded BM25S query wrapper.

Usage:
    bm25 = ShardedBM25S.load(data/bm25s_index/t2ranking)
    scores = bm25.get_scores(["机器学习", "算法", "模型"])
"""
import json
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ShardedBM25S:
    def __init__(self, shards: list, offsets: list[int], n_docs: int):
        self.shards = shards
        self.offsets = offsets
        self.n_docs = n_docs

    @staticmethod
    def load(index_dir):
        index_dir = Path(index_dir)
        manifest_path = index_dir / "shards.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"shards.json not found in {index_dir}. "
                "Run scripts/build_bm25s_index.py first."
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        import bm25s

        shards = []
        for i in range(manifest["n_shards"]):
            shard_path = index_dir / f"shard_{i:04d}"
            logger.debug(f"Loading shard {i}: {shard_path}")
            t0 = time.time()
            shard = bm25s.BM25.load(str(shard_path), load_corpus=False)
            logger.debug(f"  Shard {i} loaded in {time.time()-t0:.1f}s")
            shards.append(shard)

        return ShardedBM25S(
            shards=shards,
            offsets=manifest["shard_offsets"],
            n_docs=manifest["n_docs"],
        )

    def get_scores(self, query_tokens: list[str]) -> "np.ndarray":
        scores = np.empty(self.n_docs, dtype=np.float32)
        for i, shard in enumerate(self.shards):
            start = self.offsets[i]
            end = self.offsets[i + 1]
            scores[start:end] = shard.get_scores(query_tokens)
        return scores

    def __len__(self):
        return self.n_docs
