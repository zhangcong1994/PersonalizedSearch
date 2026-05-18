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


def _compute_global_stats(
    tokenized: list,
    show_progress: bool = False,
) -> tuple:
    import math

    nd = {}
    total_len = 0
    n_docs = len(tokenized)

    iterator = tokenized
    if show_progress and tqdm is not None:
        iterator = tqdm(tokenized, desc="Computing global IDF", unit="docs", mininterval=2)

    for doc in iterator:
        total_len += len(doc)
        seen = set()
        for word in doc:
            if word not in seen:
                nd[word] = nd.get(word, 0) + 1
                seen.add(word)

    avgdl = total_len / n_docs if n_docs > 0 else 1.0
    idf = {
        word: math.log((n_docs - doc_count + 0.5) / (doc_count + 0.5) + 1)
        for word, doc_count in nd.items()
    }
    logger.info(f"Global stats: {len(idf):,} unique terms, avgdl={avgdl:.1f}")
    return idf, avgdl, n_docs


class ShardedBM25:
    def __init__(self, shards: list, shard_offsets: list):
        self._shards = shards
        self._offsets = shard_offsets
        self.corpus_size = sum(s.corpus_size for s in shards)
        self.doc_len = []
        for s in shards:
            self.doc_len.extend(s.doc_len)

    def get_scores(self, query: list[str]) -> "np.ndarray":
        import numpy as np

        all_scores = np.empty(sum(len(s.doc_len) for s in self._shards), dtype=np.float64)
        start = 0
        for s in self._shards:
            n = len(s.doc_len)
            all_scores[start:start + n] = s.get_scores(query)
            start += n
        return all_scores


def build(
    texts: list[str],
    store_dir: Optional[Path] = None,
    name: str = "t2ranking",
    n_jobs: int = 1,
    n_shards: int = 4,
) -> list[Path]:
    import math
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

    if n_shards < 1:
        n_shards = max(1, n_total // 500000)
    n_shards = min(n_shards, max(1, n_total // 50000))
    shard_size = math.ceil(n_total / n_shards)

    # ── tokenization ──────────────────────────────────────
    stopwords = _load_stopwords()
    use_multiprocessing = n_jobs > 1 and n_total > 1000

    if use_multiprocessing:
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
    logger.info(f"Tokenized: {n_tokens:,} total tokens, avg {n_tokens // max(n_total, 1)} tokens/doc")

    # ── compute global IDF ────────────────────────────────
    global_idf, global_avgdl, n_docs = _compute_global_stats(tokenized, show_progress=True)

    # ── build shards ──────────────────────────────────────
    logger.info(
        f"Building {n_shards} shard(s), {shard_size:,} docs each "
        f"({global_avgdl:.0f} avg tokens/doc, {len(global_idf):,} terms)"
    )

    saved_paths = []
    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, n_total)
        shard_tokenized = tokenized[start:end]

        logger.info(f"  Shard {i+1}/{n_shards}: building BM25 for docs {start:,}–{end:,}...")
        bm25 = BM25Okapi(shard_tokenized)

        shard_terms = set()
        for doc in shard_tokenized:
            shard_terms.update(doc)
        bm25.idf = {k: v for k, v in global_idf.items() if k in shard_terms}
        bm25.avgdl = global_avgdl

        del shard_tokenized, shard_terms

        pkl_path = store_dir / f"{name}_shard_{i:04d}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"bm25": bm25}, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = pkl_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Shard {i+1}/{n_shards} saved: {pkl_path.name} ({size_mb:.1f} MB)")
        saved_paths.append(pkl_path)

        del bm25

    del tokenized

    # ── save metadata ─────────────────────────────────────
    meta_path = store_dir / f"{name}_meta.pkl"
    meta = {
        "version": 2,
        "name": name,
        "n_shards": n_shards,
        "n_docs": n_total,
        "n_terms": len(global_idf),
        "global_avgdl": global_avgdl,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Metadata saved: {meta_path.name}")

    return saved_paths


def load(store_dir: Optional[Path] = None, name: str = "t2ranking") -> tuple:
    if store_dir is None:
        store_dir = DEFAULT_STORE_DIR
    store_dir = Path(store_dir)

    meta_path = store_dir / f"{name}_meta.pkl"
    single_path = store_dir / f"{name}.pkl"

    if meta_path.exists():
        # ── v2 sharded format ──────────────────────────
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        n_shards = meta.get("n_shards", 0)
        if n_shards < 1:
            raise RuntimeError(f"Invalid shard metadata in {meta_path}")

        logger.info(f"Loading sharded BM25 index: {n_shards} shard(s), {meta['n_docs']:,} docs")
        shards = []
        for i in range(n_shards):
            shard_path = store_dir / f"{name}_shard_{i:04d}.pkl"
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard: {shard_path}")
            with open(shard_path, "rb") as f:
                data = pickle.load(f)
            shards.append(data["bm25"])

        offsets = [0]
        for s in shards:
            offsets.append(offsets[-1] + len(s.doc_len))

        bm25 = ShardedBM25(shards, offsets)
        logger.info(
            f"BM25 index loaded: {meta['n_docs']:,} docs across {n_shards} shard(s), "
            f"{bm25.corpus_size} terms"
        )
        return bm25, None

    elif single_path.exists():
        # ── v1 single file format ──────────────────────
        logger.info(f"Loading BM25 index from {single_path}...")
        with open(single_path, "rb") as f:
            data = pickle.load(f)

        bm25 = data["bm25"]
        tokenized = data.get("tokenized")
        if tokenized is None:
            tokenized = bm25.corpus
        logger.info(f"BM25 index loaded: {len(tokenized)} docs, {bm25.corpus_size} terms")
        return bm25, tokenized

    else:
        raise FileNotFoundError(
            f"BM25 index not found in {store_dir}. "
            f"Run 'python scripts/build_bm25_index.py' first."
        )


def tokenize_query(query: str) -> list[str]:
    return _tokenize(query, _load_stopwords())
