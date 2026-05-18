"""
PRF (Pseudo-Relevance Feedback) retriever.

No LLM call required. Steps:
1. First-pass retrieval with original query → top-K results
2. TF-IDF term extraction from top results → N expansion terms
3. Expand original query with extracted terms
4. Second-pass retrieval with expanded query

Usage:
    from src.retrieval.prf import PRFRetriever
    prf = PRFRetriever(dense_retriever)
    pids = prf.retrieve(query="查询文本", texts=passages, pids=passage_pids, top_k=10)
"""

import logging
import math

logger = logging.getLogger(__name__)


def _compute_tf(texts: list[str], term: str) -> dict[int, float]:
    scores = {}
    for i, text in enumerate(texts):
        count = text.count(term)
        if count > 0:
            scores[i] = count / len(text)
    return scores


def _compute_idf(term: str, texts: list[str]) -> float:
    n = len(texts)
    df = sum(1 for t in texts if term in t)
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _tokenize(text: str) -> list[str]:
    return [w for w in text.replace("\n", " ").split() if len(w) >= 2]


def extract_prf_terms(
    query: str,
    texts: list[str],
    top_k: int = 20,
    num_terms: int = 5,
    weighted: bool = False,
) -> list[str]:
    query_terms = set(_tokenize(query))
    tfidf_scores = {}

    for text in texts:
        for term in _tokenize(text):
            if term in query_terms or len(term) < 2:
                continue
            if term not in tfidf_scores:
                idf = _compute_idf(term, texts)
                tfidf_scores[term] = {"tf": {}, "idf": idf}

    for term, data in tfidf_scores.items():
        data["tf"] = _compute_tf(texts, term)

    if weighted:
        term_scores = {}
        for term, data in tfidf_scores.items():
            score = sum(tf_val for tf_val in data["tf"].values()) * data["idf"]
            term_scores[term] = score
    else:
        term_scores = {}
        for term, data in tfidf_scores.items():
            term_scores[term] = len(data["tf"]) * data["idf"]

    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    return [term for term, _ in sorted_terms[:num_terms]]


class PRFRetriever:
    def __init__(self, dense_retriever, prf_top_k: int = 20, num_terms: int = 5, weighted: bool = False):
        self._retriever = dense_retriever
        self._prf_top_k = prf_top_k
        self._num_terms = num_terms
        self._weighted = weighted

    def retrieve(
        self,
        query: str,
        passages: list[str],
        pids: list[str],
        top_k: int = 10,
    ) -> list[str]:
        docs = self._retriever.invoke(query)
        first_pass_pids = [doc.metadata.get("pid", "?") for doc in docs[:self._prf_top_k]]

        pid_to_idx = {p: i for i, p in enumerate(pids)}
        top_indices = [pid_to_idx[pid] for pid in first_pass_pids if pid in pid_to_idx]
        top_texts = [passages[i] for i in top_indices]

        expansion_terms = extract_prf_terms(
            query=query,
            texts=top_texts,
            top_k=self._prf_top_k,
            num_terms=self._num_terms,
            weighted=self._weighted,
        )

        if expansion_terms:
            expanded_query = query + " " + " ".join(expansion_terms)
            logger.debug(f"PRF expanded query: {expanded_query[:100]}...")
        else:
            expanded_query = query

        docs2 = self._retriever.invoke(expanded_query)
        return [doc.metadata.get("pid", "?") for doc in docs2[:top_k]]
