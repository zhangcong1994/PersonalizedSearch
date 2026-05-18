import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_results(results: list[dict], path: str, meta: dict = None):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if meta:
        lines.append(json.dumps({"__meta__": meta}, ensure_ascii=False))
    for r in results:
        line = {
            "qid": r["qid"],
            "query": r["query"],
            "relevant_pids": sorted(r["relevant_pids"]),
        }
        for key, pids in r.get("retrievals", {}).items():
            line[key] = pids
        lines.append(json.dumps(line, ensure_ascii=False))
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved {len(results)} results to {p}")


def load_results(path: str) -> tuple[list[dict], dict]:
    results = []
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "__meta__" in obj:
                meta = obj["__meta__"]
                continue
            relevant = set(obj.pop("relevant_pids", []))
            retrievals = {}
            qid = obj.pop("qid")
            query = obj.pop("query", "")
            for key in list(obj.keys()):
                retrievals[key] = obj.pop(key)
            results.append({
                "qid": qid,
                "query": query,
                "relevant_pids": relevant,
                "retrievals": retrievals,
            })
    if meta:
        logger.info(f"Loaded {len(results)} results from {path} (meta: {json.dumps(meta, ensure_ascii=False)})")
    else:
        logger.info(f"Loaded {len(results)} results from {path}")
    return results, meta
