"""
LLM output cache for query rewriting experiments.

Cache key: (experiment_id, qid)
Cache file: {RESULTS_DIR}/rewrite_cache/{experiment_id}.jsonl
Each line: {"qid": "1", "original": "...", "output": ...}

Usage:
    from src.retrieval.rewrite_cache import RewriteCache
    cache = RewriteCache(results_dir)
    cached = cache.get("E2a-B1", "42")
    cache.put("E2a-B1", "42", original="查询文本", output="改写文本")
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class RewriteCache:
    def __init__(self, results_dir: Path):
        self._cache_dir = results_dir / "rewrite_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded = {}  # {experiment_id: {qid: output}}

    def _file_path(self, experiment_id: str) -> Path:
        return self._cache_dir / f"{experiment_id}.jsonl"

    def _load(self, experiment_id: str) -> dict[str, str | list[str]]:
        if experiment_id in self._loaded:
            return self._loaded[experiment_id]

        result = {}
        fp = self._file_path(experiment_id)
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    result[obj["qid"]] = obj["output"]
            logger.info(f"Loaded {len(result)} cached entries for {experiment_id}")
        self._loaded[experiment_id] = result
        return result

    def get(self, experiment_id: str, qid: str):
        data = self._load(experiment_id)
        return data.get(qid)

    def get_batch(self, experiment_id: str, qids: list[str]) -> dict[str, str | list[str]]:
        data = self._load(experiment_id)
        return {qid: data[qid] for qid in qids if qid in data}

    def put(self, experiment_id: str, qid: str, original: str, output: str | list[str]):
        if experiment_id not in self._loaded:
            self._load(experiment_id)
        self._loaded[experiment_id][qid] = output

        fp = self._file_path(experiment_id)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "qid": qid,
                "original": original,
                "output": output,
            }, ensure_ascii=False) + "\n")

    def put_batch(self, experiment_id: str, entries: list[tuple[str, str, str | list[str]]]):
        if not entries:
            return
        if experiment_id not in self._loaded:
            self._load(experiment_id)
        for qid, original, output in entries:
            self._loaded[experiment_id][qid] = output

        fp = self._file_path(experiment_id)
        with open(fp, "a", encoding="utf-8") as f:
            for qid, original, output in entries:
                f.write(json.dumps({
                    "qid": qid,
                    "original": original,
                    "output": output,
                }, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(entries)} entries to {self._file_path(experiment_id)}")
