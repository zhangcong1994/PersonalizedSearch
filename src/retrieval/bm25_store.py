import os
import pickle
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

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


_worker_stopwords: Optional[set] = None


def _init_worker(stopwords_set: set) -> None:
    global _worker_stopwords
    _worker_stopwords = stopwords_set
    import jieba
    jieba.lcut("")


def _tokenize_worker(text: str) -> list[str]:
    global _worker_stopwords
    import jieba

    try:
        tokens = jieba.lcut(text)
    except Exception:
        return []

    sw = _worker_stopwords
    if sw:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1 and w not in sw]
    else:
        tokens = [w.strip() for w in tokens if w.strip() and len(w) > 1]
    return tokens


def build(
    texts: list[str],
    store_dir: Optional[Path] = None,
    name: str = "t2ranking",
    n_jobs: int = 1,
) -> Path:
    import jieba
    from rank_bm25 import BM25Okapi

    if store_dir is None:
        store_dir = DEFAULT_STORE_DIR
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(texts)
    if n_jobs == 0:
        n_jobs = 1
    elif n_jobs < 0:
        n_jobs = mp.cpu_count()

    max_safe_jobs = max(1, min(mp.cpu_count(), 8))
    if n_jobs > max_safe_jobs:
        logger.warning(
            f"Limiting workers from {n_jobs} to {max_safe_jobs} "
            f"(to avoid memory pressure from loading jieba per worker)"
        )
        n_jobs = max_safe_jobs

    stopwords = _load_stopwords()

    use_multiprocessing = n_jobs > 1 and n_total > 1000

    if use_multiprocessing:
        import math

        logger.info(
            f"Tokenizing {n_total:,} passages (jieba, stopwords={len(stopwords)}, workers={n_jobs})..."
        )
        chunksize = max(500, math.ceil(n_total / (n_jobs * 20)))
        with mp.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(stopwords,),
            maxtasksperchild=max(1, n_total // (n_jobs * 5)),
        ) as pool:
            iterator = pool.imap_unordered(_tokenize_worker, texts, chunksize=chunksize)
            if tqdm is not None:
                iterator = tqdm(iterator, total=n_total, desc="Tokenizing", unit="docs", mininterval=2)
            tokenized = list(iterator)
    else:
        logger.info(f"Tokenizing {n_total:,} passages (jieba, stopwords={len(stopwords)})...")
        iterator = texts
        if tqdm is not None:
            iterator = tqdm(texts, desc="Tokenizing", unit="docs", mininterval=2)
        tokenized = [_tokenize(t, stopwords) for t in iterator]

    n_tokens = sum(len(tk) for tk in tokenized)
    logger.info(f"Tokenized: {n_tokens:,} total tokens, avg {n_tokens // max(len(tokenized), 1)} tokens/doc")

    logger.info("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)

    pkl_path = store_dir / f"{name}.pkl"
    data = {
        "bm25": bm25,
        "tokenized": tokenized,
    }
    logger.info(f"Saving index to {pkl_path}...")
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
