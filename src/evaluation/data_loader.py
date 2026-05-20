import re
import html
import logging
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)

HTML_RE = re.compile(r"<[^>]*>")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PUA_RE = re.compile(r"[\uE000-\uF8FF\u200E\u200F\u202A-\u202E\uFEFF]+")
TRUNCATE_LEN = 2000
MIN_TEXT_LEN = 10


def clean_text(text: str) -> str:
    text = HTML_RE.sub("", text)
    text = html.unescape(text)
    text = URL_RE.sub("", text)
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = CONTROL_RE.sub("", text)
    text = PUA_RE.sub("", text)
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


def load_qrels_graded(path: Path) -> dict[str, dict[str, int]]:
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 3:
                qid, pid, label = parts[0], parts[1], int(parts[2])
            elif len(parts) == 4:
                qid, _, pid, label = parts[0], parts[1], parts[2], int(parts[3])
            else:
                continue
            if label > 0:
                qrels.setdefault(qid, {})[pid] = label
    logger.info(f"Loaded graded qrels: {len(qrels)} queries, "
                f"{sum(len(v) for v in qrels.values())} pairs")
    return qrels


def load_passages(path: Path, max_passages: int = 0, show_progress: bool = False) -> tuple[list[str], list[str]]:
    pids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()
        iterator = f
        if show_progress and tqdm is not None:
            iterator = tqdm(f, desc="Loading passages", unit="lines", mininterval=2)
        for line in iterator:
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
