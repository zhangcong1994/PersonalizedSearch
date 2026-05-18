"""
HyDE (Hypothetical Document Embedding) retriever.

Generates a fake answer via LLM, then uses that answer as the retrieval query.
HyDE-only mode: retrieve with fake answer only.
HyDE+query mode: retrieve with both original query and fake answer, merge via RRF.

Usage:
    from src.retrieval.hyde import HyDERetriever
    hyde = HyDERetriever(dense_retriever)
    pids = hyde.retrieve(fake_answer="...", top_k=10)
"""

import logging

from .multi_query import rrf_fusion

logger = logging.getLogger(__name__)


class HyDERetriever:
    def __init__(self, dense_retriever, rrf_k: int = 60):
        self._retriever = dense_retriever
        self._rrf_k = rrf_k

    def retrieve_hyde_only(self, fake_answer: str, top_k: int = 10) -> list[str]:
        docs = self._retriever.invoke(fake_answer)
        return [doc.metadata.get("pid", "?") for doc in docs[:top_k]]

    def retrieve_hyde_with_query(
        self,
        original_query: str,
        fake_answer: str,
        top_k: int = 10,
    ) -> list[str]:
        docs_query = self._retriever.invoke(original_query)
        pids_query = [doc.metadata.get("pid", "?") for doc in docs_query[:top_k]]

        docs_hyde = self._retriever.invoke(fake_answer)
        pids_hyde = [doc.metadata.get("pid", "?") for doc in docs_hyde[:top_k]]

        return rrf_fusion(
            [pids_query, pids_hyde],
            k=self._rrf_k,
            c=self._rrf_k,
            top_k=top_k,
        )
