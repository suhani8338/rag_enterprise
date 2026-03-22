"""
src/ingestion/document_loader.py
──────────────────────────────────
Unified document loader.

Supports: PDF, CSV, TXT, HTML, Markdown.
Returns a list of LangChain Document objects with rich metadata.

Usage:
    loader = DocumentLoader()
    docs = loader.load_file(Path("data/raw/report.pdf"))
    all_docs = loader.load_directory(Path("data/raw"))
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_core.documents import Document

from src.utils.config import settings
from src.utils.logger import get_logger, timed

logger = get_logger(__name__)


# ── Individual loaders ─────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> List[Document]:
    """Extract text from PDF, one Document per page."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            # Best-effort: grab first non-empty line as section header
            first_line = text.split("\n")[0][:120]
            docs.append(Document(
                page_content=text,
                metadata={
                    "source":       str(path),
                    "file_name":    path.name,
                    "file_type":    "pdf",
                    "page_number":  page_num,
                    "total_pages":  len(pdf),
                    "section_header": first_line,
                },
            ))
    logger.info(f"  PDF: {path.name} → {len(docs)} pages")
    return docs


def _load_csv(path: Path) -> List[Document]:
    """
    Load CSV — each row becomes a Document.
    Row text = JSON-like string of column→value pairs.
    Useful for structured data that agents can query semantically.
    """
    docs = []
    df = pd.read_csv(path, dtype=str).fillna("")
    for idx, row in df.iterrows():
        text = "\n".join(f"{col}: {val}" for col, val in row.items())
        docs.append(Document(
            page_content=text,
            metadata={
                "source":    str(path),
                "file_name": path.name,
                "file_type": "csv",
                "row_index": int(idx),  # type: ignore[arg-type]
                "columns":   ", ".join(df.columns.tolist()),
            },
        ))
    logger.info(f"  CSV: {path.name} → {len(docs)} rows")
    return docs


def _load_text(path: Path) -> List[Document]:
    """Load plain text or Markdown as a single Document."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    file_type = "md" if path.suffix == ".md" else "txt"
    doc = Document(
        page_content=text,
        metadata={
            "source":    str(path),
            "file_name": path.name,
            "file_type": file_type,
        },
    )
    logger.info(f"  TXT/MD: {path.name} → 1 document")
    return [doc]


def _load_html(path: Path) -> List[Document]:
    """Parse HTML and extract visible text (strips scripts/styles)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "head"]):
        tag.decompose()

    text = soup.get_text(separator="\n").strip()
    title = soup.title.string if soup.title else path.stem

    doc = Document(
        page_content=text,
        metadata={
            "source":           str(path),
            "file_name":        path.name,
            "file_type":        "html",
            "section_header":   str(title)[:120],
        },
    )
    logger.info(f"  HTML: {path.name} → 1 document")
    return [doc]


# ── Dispatcher ─────────────────────────────────────────────────────────────────

_LOADER_MAP = {
    ".pdf":  _load_pdf,
    ".csv":  _load_csv,
    ".txt":  _load_text,
    ".md":   _load_text,
    ".html": _load_html,
    ".htm":  _load_html,
}


class DocumentLoader:
    """
    Orchestrates multi-format document ingestion.

    Attributes:
        supported_extensions  — from settings.yaml
    """

    def __init__(self):
        self.supported_extensions = settings.ingestion.supported_extensions

    def load_file(self, path: Path) -> List[Document]:
        """Load a single file. Returns [] for unsupported types."""
        suffix = path.suffix.lower()
        loader_fn = _LOADER_MAP.get(suffix)
        if loader_fn is None:
            logger.warning(f"Skipping unsupported file type: {path.name}")
            return []
        try:
            return loader_fn(path)
        except Exception as e:
            logger.error(f"Failed to load {path.name}: {e}")
            return []

    @timed
    def load_directory(
        self,
        directory:  Path,
        recursive:  bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Recursively scan a directory and load all supported files.

        Args:
            directory:  Root folder to scan.
            recursive:  If True, descend into sub-folders.
            extensions: Override the default supported_extensions list.

        Returns:
            Flat list of LangChain Documents.
        """
        exts = extensions or self.supported_extensions
        glob_pattern = "**/*" if recursive else "*"

        all_docs: List[Document] = []
        files = [p for p in directory.glob(glob_pattern)
                 if p.is_file() and p.suffix.lower() in exts]

        logger.info(f"Found {len(files)} file(s) in {directory}")

        for file_path in sorted(files):
            docs = self.load_file(file_path)
            all_docs.extend(docs)

        logger.info(f"Loaded {len(all_docs)} document(s) total")
        return all_docs

    def load_files(self, paths: List[Path]) -> List[Document]:
        """Load a specific list of files."""
        all_docs: List[Document] = []
        for p in paths:
            all_docs.extend(self.load_file(p))
        return all_docs