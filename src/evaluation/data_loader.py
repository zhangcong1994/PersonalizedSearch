import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HTML_RE = re.compile(r"<[^>]*>")
TRUNCATE_LEN = 2000
MIN_TEXT_LEN = 10


def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = text.strip()
    return text


def load_queries(path: Path) -> list[tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    logger.info(f"Loaded {len(pairs)} queries from {path.name}")
    return pairs


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                qid, pid = parts[0], parts[1]
                qrels.setdefault(qid, set()).add(pid)
    logger.info(f"Loaded qrels: {len(qrels)} queries, {sum(len(v) for v in qrels.values())} pairs")
    return qrels


def load_passages(path: Path, max_passages: int = 0) -> tuple[list[str], list[str]]:
    pids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            pid, text = parts[0], parts[1]
            text = clean_text(text)
            if len(text) < MIN_TEXT_LEN:
                continue
            if len(text) > TRUNCATE_LEN:
                text = text[:TRUNCATE_LEN]
            pids.append(pid)
            texts.append(text)
            if max_passages > 0 and len(pids) >= max_passages:
                break
    logger.info(f"Loaded {len(pids)} passages")
    return pids, texts
