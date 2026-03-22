"""
src/embedding/embedder.py
──────────────────────────
Free, fully local embeddings using sentence-transformers.

Model: all-MiniLM-L6-v2
  - 384 dimensions
  - ~90 MB download (cached after first run)
  - ~14k tokens/sec on CPU
  - Comparable quality to OpenAI ada-002 on most retrieval benchmarks

Usage:
    embedder = LocalEmbedder()
    vectors = embedder.embed_texts(["hello world", "enterprise RAG"])
    lc_embeddings = embedder.as_langchain_embeddings()  # for ChromaDB
"""

from __future__ import annotations

import time
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config import settings
from src.utils.logger import get_logger, log_metrics, timed

logger = get_logger(__name__)


class LocalEmbedder:
    """
    Wraps HuggingFaceEmbeddings with batch processing, progress logging,
    and MLflow metric capture.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device:     Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        cfg = settings.embedding
        self.model_name = model_name or cfg.model_name
        self.device     = device     or cfg.device
        self.batch_size = batch_size or cfg.batch_size

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
        t0 = time.perf_counter()

        self._hf_embeddings = HuggingFaceEmbeddings(
            model_name       = self.model_name,
            model_kwargs     = {"device": self.device},
            encode_kwargs    = {
                "normalize_embeddings": True,   # cosine similarity works out of box
                "batch_size":           self.batch_size,
                "show_progress_bar":    False,
            },
        )
        load_time = time.perf_counter() - t0
        log_metrics({"embedding_model_load_seconds": load_time})
        logger.info(f"Embedding model ready in {load_time:.1f}s")

    # ── Core embed ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings. Returns list of float vectors.
        Automatically batches to respect GPU/CPU memory limits.
        """
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} text(s)...")
        t0 = time.perf_counter()

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs  = self._hf_embeddings.embed_documents(batch)
            all_vectors.extend(vecs)
            logger.debug(f"  Batch {i // self.batch_size + 1}: {len(batch)} texts")

        elapsed = time.perf_counter() - t0
        throughput = len(texts) / elapsed if elapsed > 0 else 0
        log_metrics({
            "embedding_count":       len(texts),
            "embedding_seconds":     elapsed,
            "embedding_throughput":  throughput,
        })
        logger.info(
            f"Embedded {len(texts)} texts in {elapsed:.2f}s "
            f"({throughput:.0f} texts/sec)"
        )
        return all_vectors

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string (uses query-optimised pooling)."""
        return self._hf_embeddings.embed_query(query)

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed a list of LangChain Document objects."""
        texts = [doc.page_content for doc in documents]
        return self.embed_texts(texts)

    # ── LangChain-compatible interface ─────────────────────────────────────────

    def as_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return the underlying HuggingFaceEmbeddings object.
        Pass this directly to ChromaDB / LangChain vectorstore constructors.
        """
        return self._hf_embeddings

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimension (needed when creating Chroma collection)."""
        test_vec = self.embed_query("dimension check")
        return len(test_vec)