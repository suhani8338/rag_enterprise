"""
src/vectorstore/chroma_store.py
────────────────────────────────
ChromaDB vector store (replaces Pinecone/Weaviate).

Features:
  • Dense vector search (cosine similarity, sentence-transformers)
  • Sparse BM25 search (rank_bm25) — replaces Pinecone's sparse index
  • Hybrid fusion with configurable dense/sparse weights
  • MMR (Maximal Marginal Relevance) for result diversity
  • Persistent on-disk storage — survives restarts

Usage:
    store = ChromaVectorStore(embedder.as_langchain_embeddings())
    store.add_documents(chunks)
    results = store.hybrid_search("What is our Q4 revenue?", k=4)
    diverse = store.mmr_search("What is our Q4 revenue?", k=4)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.utils.config import settings
from src.utils.logger import get_logger, log_metrics, timed

logger = get_logger(__name__)


class ChromaVectorStore:
    """
    Manages a persistent ChromaDB collection with hybrid search.
    """

    def __init__(
        self,
        embedding_function,                        # HuggingFaceEmbeddings instance
        persist_directory: Optional[Path] = None,
        collection_name:   Optional[str]  = None,
    ):
        cfg = settings.vectorstore
        self.persist_dir      = persist_directory or settings.paths.chroma_db
        self.collection_name  = collection_name   or cfg.collection_name
        self.dense_weight     = cfg.dense_weight
        self.sparse_weight    = cfg.sparse_weight
        self.embedding_fn     = embedding_function

        # Persistent Chroma client
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # LangChain Chroma wrapper (handles add/query nicely)
        self._store = Chroma(
            client             = self._client,
            collection_name    = self.collection_name,
            embedding_function = self.embedding_fn,
        )

        # BM25 index — rebuilt from stored docs whenever needed
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []

        logger.info(
            f"ChromaVectorStore ready | collection='{self.collection_name}' "
            f"persist='{self.persist_dir}'"
        )

    # ── Ingest ─────────────────────────────────────────────────────────────────

    @timed
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ) -> List[str]:
        """
        Add a list of Documents to ChromaDB in batches.
        Returns list of Chroma document IDs.
        """
        if not documents:
            logger.warning("add_documents called with empty list — skipping")
            return []

        all_ids: List[str] = []
        t0 = time.perf_counter()

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids   = self._store.add_documents(batch)
            all_ids.extend(ids)
            logger.info(
                f"  Indexed batch {i // batch_size + 1}: "
                f"{len(batch)} chunks → Chroma"
            )

        elapsed = time.perf_counter() - t0
        log_metrics({
            "chroma_docs_added": len(all_ids),
            "chroma_index_seconds": elapsed,
        })
        logger.info(f"Indexed {len(all_ids)} chunks in {elapsed:.2f}s")

        # Rebuild BM25 index with the new docs
        self._rebuild_bm25()
        return all_ids

    # ── BM25 (sparse) ──────────────────────────────────────────────────────────

    def _rebuild_bm25(self) -> None:
        """(Re)build the BM25 index from all stored Chroma documents."""
        try:
            result = self._store.get(include=["documents", "metadatas"])
            texts  = result.get("documents") or []
            metas  = result.get("metadatas") or []
            ids    = result.get("ids") or []

            self._bm25_docs = [
                Document(page_content=t, metadata={**m, "chroma_id": i})
                for t, m, i in zip(texts, metas, ids)
            ]
            tokenized = [doc.page_content.lower().split() for doc in self._bm25_docs]
            self._bm25 = BM25Okapi(tokenized) if tokenized else None
            logger.debug(f"BM25 index rebuilt with {len(self._bm25_docs)} docs")
        except Exception as e:
            logger.warning(f"BM25 rebuild failed: {e}")
            self._bm25 = None

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """Return (Document, normalised_score) pairs via BM25."""
        if self._bm25 is None or not self._bm25_docs:
            return []
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # Normalise to [0, 1]
        max_score = scores.max() if scores.max() > 0 else 1.0
        normed    = scores / max_score
        top_idxs  = np.argsort(normed)[::-1][:top_k]
        return [(self._bm25_docs[i], float(normed[i])) for i in top_idxs]

    # ── Search methods ─────────────────────────────────────────────────────────

    def dense_search(
        self,
        query: str,
        k:     int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Cosine-similarity vector search. Returns (doc, score) pairs."""
        k = k or settings.retrieval.top_k
        results = self._store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter
        )
        return results  # [(Document, float)]

    def hybrid_search(
        self,
        query:  str,
        k:      int  = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Weighted fusion of dense (cosine) + sparse (BM25) rankings.

        Score = dense_weight * dense_score + sparse_weight * bm25_score
        Returns top-k Documents sorted by fused score.
        """
        k = k or settings.retrieval.top_k
        dense_results  = self.dense_search(query, k=k, filter=filter)
        sparse_results = self._bm25_search(query, top_k=k)

        # Build score maps keyed by page_content fingerprint
        dense_map:  Dict[str, Tuple[Document, float]] = {
            doc.page_content: (doc, score) for doc, score in dense_results
        }
        sparse_map: Dict[str, Tuple[Document, float]] = {
            doc.page_content: (doc, score) for doc, score in sparse_results
        }

        all_texts = set(dense_map) | set(sparse_map)
        fused: List[Tuple[Document, float]] = []

        for text in all_texts:
            d_score = dense_map[text][1]  if text in dense_map  else 0.0
            s_score = sparse_map[text][1] if text in sparse_map else 0.0
            doc     = (dense_map.get(text) or sparse_map.get(text))[0]
            fused.append((doc, self.dense_weight * d_score + self.sparse_weight * s_score))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in fused[:k]]

    def mmr_search(
        self,
        query:      str,
        k:          int   = None,
        lambda_val: float = None,
        filter:     Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Maximal Marginal Relevance — balances relevance with diversity.

        lambda_val:  0.0 = max diversity, 1.0 = max relevance (default 0.5)
        """
        k          = k          or settings.retrieval.final_k
        lambda_val = lambda_val or settings.retrieval.mmr_lambda
        fetch_k    = settings.retrieval.top_k  # candidates before MMR pruning

        results = self._store.max_marginal_relevance_search(
            query,
            k         = k,
            fetch_k   = fetch_k,
            lambda_mult = lambda_val,
            filter    = filter,
        )
        logger.debug(f"MMR returned {len(results)} diverse chunks")
        return results

    # ── Utility ────────────────────────────────────────────────────────────────

    def collection_count(self) -> int:
        """Return number of documents stored in the Chroma collection."""
        return self._store._collection.count()

    def delete_collection(self) -> None:
        """⚠️  Permanently delete the collection (for testing/resets)."""
        self._client.delete_collection(self.collection_name)
        self._bm25 = None
        self._bm25_docs = []
        logger.warning(f"Collection '{self.collection_name}' deleted")

    def load_bm25_from_existing(self) -> None:
        """
        Call this on startup if ChromaDB already has data
        (e.g., from a previous run) to rebuild the BM25 index.
        """
        self._rebuild_bm25()
        logger.info(
            f"BM25 loaded from existing ChromaDB "
            f"({len(self._bm25_docs)} docs)"
        )