"""
src/rag/reranker.py
────────────────────
Cross-encoder re-ranking of retrieved chunks.

Why re-ranking?
  The bi-encoder (sentence-transformers) used in Phase 1 is fast but approximate —
  it embeds query and document independently then compares vectors. A cross-encoder
  reads the query AND document TOGETHER, giving much more accurate relevance scores
  at the cost of speed. We run it only on the small set of candidates already
  retrieved, so it stays fast overall.

  Typical improvement: +8-15% on top-4 precision vs dense-only retrieval.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Free, ~80MB, runs fully offline after first download
  - Trained on Microsoft MARCO passage ranking
  - Returns a raw logit score (higher = more relevant)

Usage:
    reranker = CrossEncoderReranker()
    top_chunks = reranker.rerank(query="What is Q4 revenue?", documents=candidates)
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logger import get_logger, log_metrics

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Re-ranks a list of candidate Documents using a cross-encoder model.
    Returns the top-N documents sorted by true relevance to the query.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        top_n:      Optional[int] = None,
        device:     Optional[str] = None,
    ):
        cfg = settings.reranker
        self.model_name = model_name or (cfg.model_name if cfg else "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.top_n      = top_n      or (cfg.top_n      if cfg else 4)
        self.device     = device     or (cfg.device     if cfg else "cpu")

        logger.info(f"Loading cross-encoder: {self.model_name} on {self.device}")
        t0 = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            load_time   = time.perf_counter() - t0
            log_metrics({"reranker_load_seconds": load_time})
            logger.info(f"Cross-encoder ready in {load_time:.1f}s")
        except ImportError:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )

    def rerank(
        self,
        query:     str,
        documents: List[Document],
        top_n:     Optional[int] = None,
    ) -> List[Document]:
        """
        Score each document against the query and return top_n sorted by score.

        Args:
            query:     The user question.
            documents: Candidate chunks from Phase 1 retrieval.
            top_n:     Override default top_n from settings.

        Returns:
            Sorted list of Documents (most relevant first), length ≤ top_n.
            Each document gets a 'rerank_score' field added to its metadata.
        """
        n = top_n or self.top_n

        if not documents:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        t0     = time.perf_counter()
        scores = self._model.predict(pairs)
        elapsed = time.perf_counter() - t0

        log_metrics({
            "reranker_candidates": len(documents),
            "reranker_seconds":    elapsed,
        })

        # Attach score to metadata and sort descending
        scored: List[Tuple[Document, float]] = []
        for doc, score in zip(documents, scores):
            enriched_doc = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 4)},
            )
            scored.append((enriched_doc, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored[:n]]

        logger.info(
            f"Reranking: {len(documents)} candidates → top {len(top_docs)} "
            f"in {elapsed*1000:.0f}ms | "
            f"top score={scored[0][1]:.3f}"
        )
        return top_docs

    def rerank_with_scores(
        self,
        query:     str,
        documents: List[Document],
        top_n:     Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """Same as rerank() but returns (Document, score) tuples."""
        reranked = self.rerank(query, documents, top_n)
        return [
            (doc, doc.metadata.get("rerank_score", 0.0))
            for doc in reranked
        ]