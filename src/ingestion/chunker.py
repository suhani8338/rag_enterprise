"""
src/ingestion/chunker.py
─────────────────────────
Splits LangChain Documents into smaller chunks while preserving metadata.

Strategy:
  1. RecursiveCharacterTextSplitter for natural sentence/paragraph splits.
  2. Each chunk inherits all parent metadata + adds chunk_index and char_count.
  3. Returns a list of enriched Document objects ready for embedding.

Usage:
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(docs)
    stats  = chunker.get_stats(chunks)
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logger import get_logger, timed

logger = get_logger(__name__)


class DocumentChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter with metadata enrichment.
    """

    def __init__(
        self,
        chunk_size:    int = None,
        chunk_overlap: int = None,
        separators:    List[str] = None,
    ):
        cfg = settings.chunking
        self.chunk_size    = chunk_size    or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.separators    = separators    or cfg.separators

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size         = self.chunk_size,
            chunk_overlap      = self.chunk_overlap,
            separators         = self.separators,
            length_function    = len,
            is_separator_regex = False,
        )
        logger.info(
            f"Chunker ready | size={self.chunk_size} overlap={self.chunk_overlap}"
        )

    # ── Core split ─────────────────────────────────────────────────────────────

    @timed
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split each Document into chunks, enriching metadata.

        Adds per-chunk metadata:
          chunk_index   — position within the parent document
          char_count    — characters in this chunk
          token_estimate — rough token count (char_count // 4)
          parent_source — source path of the originating document

        Returns:
          Flat list of chunk Documents, numbered globally and per-source.
        """
        all_chunks: List[Document] = []
        global_idx = 0

        for doc_idx, doc in enumerate(documents):
            raw_chunks = self._splitter.split_text(doc.page_content)
            source_name = doc.metadata.get("file_name", f"doc_{doc_idx}")

            for chunk_idx, chunk_text in enumerate(raw_chunks):
                # Deep-copy metadata so mutations don't bleed back
                meta = copy.deepcopy(doc.metadata)
                meta.update({
                    "chunk_index":    chunk_idx,
                    "global_index":   global_idx,
                    "char_count":     len(chunk_text),
                    "token_estimate": len(chunk_text) // 4,
                    "parent_source":  doc.metadata.get("source", ""),
                })
                all_chunks.append(Document(page_content=chunk_text, metadata=meta))
                global_idx += 1

            if len(raw_chunks) > 0:
                logger.debug(
                    f"  {source_name} → {len(raw_chunks)} chunk(s)"
                )

        logger.info(
            f"Chunking complete | {len(documents)} doc(s) → {len(all_chunks)} chunk(s)"
        )
        return all_chunks

    # ── Stats ──────────────────────────────────────────────────────────────────

    def get_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Return summary statistics for a list of chunks."""
        if not chunks:
            return {"total_chunks": 0}

        char_counts = [len(c.page_content) for c in chunks]
        token_estimates = [c.metadata.get("token_estimate", 0) for c in chunks]

        sources = {}
        for c in chunks:
            src = c.metadata.get("file_name", "unknown")
            sources[src] = sources.get(src, 0) + 1

        return {
            "total_chunks":        len(chunks),
            "avg_chars":           round(sum(char_counts) / len(char_counts), 1),
            "min_chars":           min(char_counts),
            "max_chars":           max(char_counts),
            "avg_token_estimate":  round(sum(token_estimates) / len(token_estimates), 1),
            "sources":             sources,
        }