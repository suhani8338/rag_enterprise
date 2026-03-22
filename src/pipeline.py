"""
src/pipeline.py
────────────────
Phase 1 orchestrator: Ingestion → Chunking → Embedding → Vector Index.

Run with:
    python -m src.pipeline                          # index data/raw/
    python -m src.pipeline --dir path/to/docs       # custom directory
    python -m src.pipeline --file path/to/doc.pdf   # single file
    python -m src.pipeline --query "your question"  # test retrieval
    python -m src.pipeline --reset                  # wipe & re-index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import rich
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.embedding.embedder import LocalEmbedder
from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_loader import DocumentLoader
from src.utils.config import settings
from src.utils.logger import get_logger, mlflow_run, log_metrics, log_params
from src.utils.metadata_store import MetadataStore
from src.vectorstore.chroma_store import ChromaVectorStore

logger  = get_logger(__name__)
console = Console()


class Phase1Pipeline:
    """
    End-to-end Phase 1 pipeline:
      DocumentLoader → DocumentChunker → LocalEmbedder → ChromaVectorStore
    with metadata tracked in SQLite.
    """

    def __init__(self):
        console.print(Panel.fit(
            f"[bold cyan]{settings.project_name}[/] — Phase 1: Ingestion & Indexing",
            border_style="cyan",
        ))

        self.loader   = DocumentLoader()
        self.chunker  = DocumentChunker()
        self.embedder = LocalEmbedder()
        self.store    = ChromaVectorStore(self.embedder.as_langchain_embeddings())
        self.meta_db  = MetadataStore()

        # Load BM25 from any pre-existing Chroma data
        if self.store.collection_count() > 0:
            console.print(
                f"[dim]Found {self.store.collection_count()} existing chunk(s) "
                "in ChromaDB — loading BM25 index...[/]"
            )
            self.store.load_bm25_from_existing()

    # ── Ingest ─────────────────────────────────────────────────────────────────

    def run_ingestion(
        self,
        directory: Optional[Path] = None,
        file_path: Optional[Path] = None,
    ) -> int:
        """
        Full ingest: load → chunk → embed → index → metadata.
        Returns number of chunks indexed.
        """
        with mlflow_run(run_name="phase1_ingestion"):
            log_params({
                "chunk_size":    settings.chunking.chunk_size,
                "chunk_overlap": settings.chunking.chunk_overlap,
                "embedding_model": settings.embedding.model_name,
                "dense_weight":  settings.vectorstore.dense_weight,
            })

            # ── 1. Load ───────────────────────────────────────────────────────
            console.rule("[bold]Step 1 · Load documents")
            if file_path:
                raw_docs = self.loader.load_file(file_path)
            else:
                src_dir  = directory or settings.paths.raw_data
                raw_docs = self.loader.load_directory(src_dir)

            if not raw_docs:
                console.print("[yellow]⚠  No documents loaded. Add files to data/raw/[/]")
                return 0

            console.print(f"[green]✓[/] Loaded [bold]{len(raw_docs)}[/] document(s)")

            # ── 2. Persist source metadata ────────────────────────────────────
            for doc in raw_docs:
                src_path = Path(doc.metadata.get("source", "unknown"))
                if src_path.exists():
                    self.meta_db.insert_source(
                        file_path = src_path,
                        file_type = doc.metadata.get("file_type", "unknown"),
                        file_size = src_path.stat().st_size,
                    )

            # ── 3. Chunk ──────────────────────────────────────────────────────
            console.rule("[bold]Step 2 · Chunk documents")
            chunks = self.chunker.chunk_documents(raw_docs)
            stats  = self.chunker.get_stats(chunks)
            log_metrics({
                "total_chunks":       stats["total_chunks"],
                "avg_chars_per_chunk": stats["avg_chars"],
            })
            _print_chunk_stats(console, stats)

            # ── 4. Store chunk metadata ───────────────────────────────────────
            from src.utils.metadata_store import MetadataStore
            chunk_rows = []
            for chunk in chunks:
                src_path  = Path(chunk.metadata.get("source", ""))
                source_id = MetadataStore._make_source_id(src_path) if src_path.exists() else "unknown"
                chunk_rows.append({
                    "source_id":      source_id,
                    "chunk_index":    chunk.metadata.get("chunk_index", 0),
                    "char_count":     chunk.metadata.get("char_count", 0),
                    "page_number":    chunk.metadata.get("page_number"),
                    "section_header": chunk.metadata.get("section_header"),
                })
            self.meta_db.insert_chunks(chunk_rows)

            # ── 5. Embed & Index ──────────────────────────────────────────────
            console.rule("[bold]Step 3 · Embed & index")
            console.print(
                f"[dim]Embedding {len(chunks)} chunks with "
                f"[italic]{settings.embedding.model_name}[/]...[/]"
            )
            chroma_ids = self.store.add_documents(chunks)
            log_metrics({"chroma_collection_size": self.store.collection_count()})

            # ── 6. Mark embedded ──────────────────────────────────────────────
            self.meta_db.log_event(
                "ingestion_complete",
                f"{len(chunks)} chunks indexed from {len(raw_docs)} docs",
            )

            console.print(
                f"\n[bold green]✓ Phase 1 complete![/] "
                f"{len(chunks)} chunks in ChromaDB | "
                f"collection size: {self.store.collection_count()}"
            )
            return len(chunks)

    # ── Query ──────────────────────────────────────────────────────────────────

    def run_query(self, query: str, mode: str = "hybrid") -> None:
        """
        Test retrieval against the indexed collection.

        mode: "dense" | "sparse" | "hybrid" | "mmr"
        """
        if self.store.collection_count() == 0:
            console.print("[red]No documents indexed yet. Run ingestion first.[/]")
            return

        console.rule(f"[bold]Query · mode={mode}")
        console.print(f"[italic]{query}[/]\n")

        t0 = time.perf_counter()

        if mode == "dense":
            results = [doc for doc, _ in self.store.dense_search(query)]
        elif mode == "mmr":
            results = self.store.mmr_search(query)
        else:  # hybrid (default)
            results = self.store.hybrid_search(query)

        elapsed = time.perf_counter() - t0

        _print_results(console, results, elapsed)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def print_status(self) -> None:
        """Print current index and metadata status."""
        console.rule("[bold]Index Status")
        console.print(f"[bold]ChromaDB chunks:[/] {self.store.collection_count()}")

        src_df = self.meta_db.get_source_summary()
        if not src_df.empty:
            table = Table(title="Ingested Sources", show_lines=True)
            for col in src_df.columns:
                table.add_column(col)
            for _, row in src_df.iterrows():
                table.add_row(*[str(v) for v in row])
            console.print(table)

        chunk_df = self.meta_db.get_chunk_stats()
        if not chunk_df.empty:
            table2 = Table(title="Chunk Statistics by File Type", show_lines=True)
            for col in chunk_df.columns:
                table2.add_column(col)
            for _, row in chunk_df.iterrows():
                table2.add_row(*[str(v) for v in row])
            console.print(table2)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe ChromaDB collection (for development resets)."""
        console.print("[yellow]Resetting ChromaDB collection...[/]")
        self.store.delete_collection()
        console.print("[green]✓ Collection reset[/]")


# ── Pretty-print helpers ───────────────────────────────────────────────────────

def _print_chunk_stats(console: Console, stats: dict) -> None:
    table = Table(title="Chunk Statistics", show_lines=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Total chunks",       str(stats["total_chunks"]))
    table.add_row("Avg chars / chunk",  str(stats["avg_chars"]))
    table.add_row("Min chars",          str(stats["min_chars"]))
    table.add_row("Max chars",          str(stats["max_chars"]))
    table.add_row("Avg token estimate", str(stats["avg_token_estimate"]))
    console.print(table)

    src_table = Table(title="Chunks per Source", show_lines=True)
    src_table.add_column("File")
    src_table.add_column("Chunks", justify="right")
    for fname, count in stats.get("sources", {}).items():
        src_table.add_row(fname, str(count))
    console.print(src_table)


def _print_results(console: Console, results, elapsed: float) -> None:
    console.print(
        f"[dim]Retrieved {len(results)} chunk(s) in {elapsed*1000:.0f}ms[/]\n"
    )
    for i, doc in enumerate(results, 1):
        meta   = doc.metadata
        source = meta.get("file_name", "?")
        page   = meta.get("page_number", "")
        page_s = f" · page {page}" if page else ""
        console.print(
            f"[bold cyan]─── Result {i}[/] "
            f"[dim]{source}{page_s}[/]"
        )
        # Print first 400 characters of the chunk
        preview = doc.page_content[:400].replace("\n", " ")
        console.print(f"  {preview}[dim]...[/]\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 RAG pipeline")
    parser.add_argument("--dir",    type=Path, help="Directory to ingest")
    parser.add_argument("--file",   type=Path, help="Single file to ingest")
    parser.add_argument("--query",  type=str,  help="Run a test query")
    parser.add_argument("--mode",   type=str,  default="hybrid",
                        choices=["dense", "hybrid", "mmr"],
                        help="Search mode for --query")
    parser.add_argument("--status", action="store_true", help="Print index status")
    parser.add_argument("--reset",  action="store_true", help="Reset ChromaDB collection")
    args = parser.parse_args()

    pipeline = Phase1Pipeline()

    if args.reset:
        pipeline.reset()
        return

    if args.status:
        pipeline.print_status()
        return

    if args.query:
        # Query only — no ingestion
        pipeline.run_query(args.query, mode=args.mode)
        return

    # Default: ingest + (optionally) query
    pipeline.run_ingestion(directory=args.dir, file_path=args.file)
    pipeline.print_status()

    # Demo query if data was indexed
    if pipeline.store.collection_count() > 0:
        console.rule("\n[bold]Demo Query")
        pipeline.run_query(
            "Summarise the key findings in the documents.",
            mode="hybrid",
        )


if __name__ == "__main__":
    main()