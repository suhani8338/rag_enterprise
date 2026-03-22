"""
tests/test_phase1.py
─────────────────────
Unit & integration tests for Phase 1 components.

Run:  pytest tests/test_phase1.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_loader import DocumentLoader
from src.utils.config import settings
from src.utils.metadata_store import MetadataStore


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_raw_dir(tmp_path):
    """Create a temporary directory with sample files."""
    txt = tmp_path / "sample.txt"
    txt.write_text("Hello world. " * 200, encoding="utf-8")

    csv = tmp_path / "data.csv"
    csv.write_text("name,age,city\nAlice,30,Hyderabad\nBob,25,Mumbai\n")

    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nSome content here.\n" * 50)

    return tmp_path


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog. " * 30,
            metadata={"source": "test.txt", "file_name": "test.txt", "file_type": "txt"},
        ),
        Document(
            page_content="Enterprise RAG systems use vector databases for retrieval. " * 20,
            metadata={"source": "rag.txt", "file_name": "rag.txt", "file_type": "txt"},
        ),
    ]


@pytest.fixture
def meta_store(tmp_path):
    return MetadataStore(db_path=tmp_path / "test.db")


# ── DocumentLoader tests ───────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_txt(self, tmp_raw_dir):
        loader = DocumentLoader()
        docs = loader.load_file(tmp_raw_dir / "sample.txt")
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "txt"
        assert len(docs[0].page_content) > 0

    def test_load_csv(self, tmp_raw_dir):
        loader = DocumentLoader()
        docs = loader.load_file(tmp_raw_dir / "data.csv")
        assert len(docs) == 2  # 2 data rows
        assert "Alice" in docs[0].page_content or "Alice" in docs[1].page_content

    def test_load_markdown(self, tmp_raw_dir):
        loader = DocumentLoader()
        docs = loader.load_file(tmp_raw_dir / "notes.md")
        assert len(docs) == 1
        assert docs[0].metadata["file_type"] == "md"

    def test_load_directory(self, tmp_raw_dir):
        loader = DocumentLoader()
        docs = loader.load_directory(tmp_raw_dir)
        # txt (1) + csv (2 rows) + md (1) = 4
        assert len(docs) >= 3

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("content")
        loader = DocumentLoader()
        docs = loader.load_file(f)
        assert docs == []


# ── DocumentChunker tests ──────────────────────────────────────────────────────

class TestDocumentChunker:
    def test_chunk_count(self, sample_docs):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_docs)
        assert len(chunks) > len(sample_docs)  # must produce more chunks than docs

    def test_chunk_metadata_preserved(self, sample_docs):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(sample_docs)
        for chunk in chunks:
            assert "chunk_index"    in chunk.metadata
            assert "char_count"     in chunk.metadata
            assert "token_estimate" in chunk.metadata
            assert "source"         in chunk.metadata

    def test_chunk_size_respected(self, sample_docs):
        size = 150
        chunker = DocumentChunker(chunk_size=size, chunk_overlap=0)
        chunks = chunker.chunk_documents(sample_docs)
        oversized = [c for c in chunks if len(c.page_content) > size * 1.1]
        # Allow 10% tolerance for splitter edge cases
        assert len(oversized) == 0, f"{len(oversized)} chunks exceed size limit"

    def test_stats_shape(self, sample_docs):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks  = chunker.chunk_documents(sample_docs)
        stats   = chunker.get_stats(chunks)
        assert "total_chunks"        in stats
        assert "avg_chars"           in stats
        assert "avg_token_estimate"  in stats
        assert stats["total_chunks"] == len(chunks)

    def test_empty_input(self):
        chunker = DocumentChunker()
        chunks  = chunker.chunk_documents([])
        assert chunks == []


# ── MetadataStore tests ────────────────────────────────────────────────────────

class TestMetadataStore:
    def test_insert_source(self, meta_store, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("hello")
        source_id = meta_store.insert_source(f, "pdf", 5)
        assert len(source_id) == 32  # MD5 hex

    def test_insert_chunks(self, meta_store, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        source_id = meta_store.insert_source(f, "txt", 5)
        chunks = [
            {"source_id": source_id, "chunk_index": i, "char_count": 100}
            for i in range(5)
        ]
        meta_store.insert_chunks(chunks)
        df = meta_store.query("SELECT COUNT(*) AS n FROM fact_chunk")
        assert df["n"].iloc[0] == 5

    def test_source_summary_schema(self, meta_store, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        meta_store.insert_source(f, "txt", 1)
        df = meta_store.get_source_summary()
        assert "file_name" in df.columns
        assert "status"    in df.columns

    def test_log_event(self, meta_store):
        meta_store.log_event("test_event", "details here")
        df = meta_store.query("SELECT * FROM audit_log")
        assert len(df) == 1
        assert df["event_type"].iloc[0] == "test_event"


# ── Config tests ───────────────────────────────────────────────────────────────

class TestConfig:
    def test_settings_loaded(self):
        assert settings.project_name is not None
        assert settings.paths.raw_data.exists()
        assert settings.chunking.chunk_size > 0
        assert settings.embedding.model_name != ""

    def test_paths_exist(self):
        assert settings.paths.raw_data.is_dir()
        assert settings.paths.processed_data.is_dir()