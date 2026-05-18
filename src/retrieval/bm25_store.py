import os
import pickle
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_data_root_raw = os.environ.get("PERSONALIZEDSEARCH_DATA_ROOT")
if _data_root_raw:
    DEFAULT_STORE_DIR = Path(_data_root_raw).resolve() / "data" / "bm25_index"
else:
    DEFAULT_STORE_DIR = Path(__file__).resolve().parents[2] / "data" / "bm25_index"


def _tokenize(text: str, stopwords: Optional[set] = None) -> list[str]:
    import jieba

    tokens = jieba.lcut(text)
    if stopwords:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1 and w not in stopwords]
    else:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1]
    return tokens


def _load_stopwords() -> set[str]:
    import jieba.analyse

    try:
        default_path = os.path.join(jieba.analyse.STOP_WORDS)
    except Exception:
        return _default_stopwords()

    stopwords = set()
    if os.path.exists(default_path):
        with open(default_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    stopwords.add(line)

    if not stopwords:
        stopwords = _default_stopwords()

    return stopwords


def _default_stopwords() -> set[str]:
    return {
        "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
        "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
        "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
        "所", "为", "所以", "因为", "但是", "然而", "而且", "或", "或者",
        "可以", "还是", "只是", "如果", "虽然", "并", "但", "却", "从", "以",
        "之", "将", "把", "被", "让", "给", "跟", "与", "同", "及", "及至",
        "及其", "以及", "及其", "关于", "对于", "按照", "根据", "经过", "通过",
        "哦", "啊", "嗯", "呢", "吧", "么", "吗", "呀",
    }


def build(
    texts: list[str],
    store_dir: Optional[Path] = None,
    name: str = "t2ranking",
) -> Path:
    import jieba
    from rank_bm25 import BM25Okapi

    if store_dir is None:
        store_dir = DEFAULT_STORE_DIR
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    stopwords = _load_stopwords()
    logger.info(f"Tokenizing {len(texts)} passages (jieba, stopwords={len(stopwords)})...")

    tokenized = [_tokenize(t, stopwords) for t in texts]
    n_tokens = sum(len(tk) for tk in tokenized)
    logger.info(f"Tokenized: {n_tokens:,} total tokens, avg {n_tokens // max(len(tokenized), 1)} tokens/doc")

    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    pkl_path = store_dir / f"{name}.pkl"
    data = {
        "bm25": bm25,
        "tokenized": tokenized,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = pkl_path.stat().st_size / (1024 * 1024)
    logger.info(f"BM25 index saved: {pkl_path} ({size_mb:.1f} MB)")
    return pkl_path


def load(store_dir: Optional[Path] = None, name: str = "t2ranking") -> tuple:
    if store_dir is None:
        store_dir = DEFAULT_STORE_DIR
    store_dir = Path(store_dir)

    pkl_path = store_dir / f"{name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {pkl_path}. "
            f"Run 'python scripts/build_bm25_index.py' first."
        )

    logger.info(f"Loading BM25 index from {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    bm25 = data["bm25"]
    tokenized = data["tokenized"]
    logger.info(f"BM25 index loaded: {len(tokenized)} docs, {bm25.corpus_size} terms")
    return bm25, tokenized


def tokenize_query(query: str) -> list[str]:
    return _tokenize(query, _load_stopwords())
