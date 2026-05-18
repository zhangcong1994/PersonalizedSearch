"""
Multi-Query RRF (Reciprocal Rank Fusion) retriever.

Used by E2b experiments: generate N sub-queries via LLM, retrieve from each,
merge results with RRF.

Usage:
    from src.retrieval.multi_query import MultiQueryRetriever

    mqr = MultiQueryRetriever(dense_retriever, pids, rrf_k=60)
    merged = mqr.retrieve(sub_queries=["术语版", "上下文版"], top_k=10)
    # merged: list[str] of pids in ranked order
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def rrf_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    c: int = 60,
    top_k: int = 10,
) -> list[str]:
    scores = defaultdict(float)
    for ranked in ranked_lists:
        for rank, pid in enumerate(ranked):
            scores[pid] += 1.0 / (k + rank + 1)

    sorted_pids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in sorted_pids[:top_k]]


class MultiQueryRetriever:
    def __init__(self, dense_retriever, pids: list[str], rrf_k: int = 60):
        self._retriever = dense_retriever
        self._pids = pids
        self._rrf_k = rrf_k

    def retrieve(
        self,
        sub_queries: list[str],
        top_k: int = 10,
        original_query: str = None,
    ) -> list[str]:
        if not sub_queries:
            return []

        ranked_lists = []
        for sq in sub_queries:
            docs = self._retriever.invoke(sq)
            pid_list = [doc.metadata.get("pid", "?") for doc in docs[:top_k]]
            ranked_lists.append(pid_list)

        return rrf_fusion(ranked_lists, k=self._rrf_k, c=self._rrf_k, top_k=top_k)
