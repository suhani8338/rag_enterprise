"""
src/scheduler/refresh_scheduler.py
────────────────────────────────────
APScheduler-based automated pipeline:

  Every N minutes:
    1. Scan data/raw/ for files modified in the last 24 hours
    2. Re-ingest changed files (load → chunk → embed)
    3. Update ChromaDB index (delete old chunks for that file, add new ones)
    4. Update SQLite metadata
    5. Rebuild BM25 index
    6. Log a refresh run to MLflow

This mirrors the Airflow DAG pattern from the job description but runs
entirely locally without any infrastructure.

Usage:
  # Run the scheduler in the foreground (blocks):
  python -m src.scheduler.refresh_scheduler

  # Run once immediately (useful for testing / CI):
  python -m src.scheduler.refresh_scheduler --once

  # Change interval:
  python -m src.scheduler.refresh_scheduler --interval 30
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import mlflow
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from rich.console import Console

from src.embedding.embedder import LocalEmbedder
from src.ingestion.chunker import DocumentChunker
from src.ingestion.document_loader import DocumentLoader
from src.utils.config import settings
from src.utils.logger import get_logger
from src.utils.metadata_store import MetadataStore
from src.vectorstore.chroma_store import ChromaVectorStore

logger  = get_logger(__name__)
console = Console()


class RefreshScheduler:
    """
    Monitors data/raw/ and keeps the ChromaDB index in sync with the filesystem.

    Tracks file modification times so only genuinely changed files are
    re-processed — avoids redundant embedding on every run.
    """

    def __init__(
        self,
        interval_minutes:    Optional[int]  = None,
        changed_files_hours: Optional[int]  = None,
        timezone:            Optional[str]  = None,
    ):
        cfg = settings.scheduler
        self.interval_minutes    = interval_minutes    or (cfg.refresh_interval_minutes if cfg else 60)
        self.changed_files_hours = changed_files_hours or (cfg.changed_files_hours      if cfg else 24)
        self.tz                  = timezone            or (cfg.timezone                 if cfg else "UTC")

        # Lazy-init heavy components on first refresh (not at scheduler creation)
        self._embedder:  Optional[LocalEmbedder]      = None
        self._store:     Optional[ChromaVectorStore]  = None
        self._loader:    DocumentLoader               = DocumentLoader()
        self._chunker:   DocumentChunker              = DocumentChunker()
        self._meta_db:   MetadataStore                = MetadataStore()

        logger.info(
            f"RefreshScheduler created | "
            f"interval={self.interval_minutes}min "
            f"window={self.changed_files_hours}h"
        )

    # ── Lazy component init ────────────────────────────────────────────────────

    def _ensure_components(self) -> None:
        if self._embedder is None:
            self._embedder = LocalEmbedder()
            self._store    = ChromaVectorStore(self._embedder.as_langchain_embeddings())

    # ── Core refresh logic ────────────────────────────────────────────────────

    def refresh_once(self) -> dict:
        """
        One complete refresh cycle.
        Returns a summary dict with counts for MLflow logging.
        """
        self._ensure_components()
        console.rule(
            f"[bold]Refresh run — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        )

        # Step 1: Find changed files
        changed = self._find_changed_files()
        if not changed:
            console.print("[dim]No changed files detected — index is up to date.[/]")
            return {"changed_files": 0, "new_chunks": 0}

        console.print(f"[green]Found {len(changed)} changed file(s):[/]")
        for f in changed:
            console.print(f"  • {f.name}")

        total_new_chunks = 0

        for file_path in changed:
            n = self._refresh_file(file_path)
            total_new_chunks += n

        # Step 5: Rebuild BM25 after all updates
        self._store.load_bm25_from_existing()
        console.print(
            f"[bold green]✓ Refresh complete[/] | "
            f"{len(changed)} file(s) · {total_new_chunks} new chunks"
        )

        summary = {
            "changed_files": len(changed),
            "new_chunks":    total_new_chunks,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        }
        self._log_to_mlflow(summary)
        return summary

    def _refresh_file(self, file_path: Path) -> int:
        """Re-ingest a single file: delete old chunks, add new ones."""
        logger.info(f"Refreshing {file_path.name}")

        # Step 2: Load the file
        docs = self._loader.load_file(file_path)
        if not docs:
            logger.warning(f"  {file_path.name} returned no documents — skipping")
            return 0

        # Step 3: Delete existing chunks for this file from ChromaDB
        self._delete_file_chunks(file_path)

        # Step 4: Chunk + embed + index
        chunks = self._chunker.chunk_documents(docs)
        if not chunks:
            return 0

        self._store.add_documents(chunks)

        # Update SQLite metadata
        source_id = MetadataStore._make_source_id(file_path)
        self._meta_db.insert_source(
            file_path = file_path,
            file_type = file_path.suffix.lstrip("."),
            file_size = file_path.stat().st_size,
        )
        chunk_rows = [
            {
                "source_id":   source_id,
                "chunk_index": c.metadata.get("chunk_index", i),
                "char_count":  c.metadata.get("char_count", 0),
                "page_number": c.metadata.get("page_number"),
            }
            for i, c in enumerate(chunks)
        ]
        self._meta_db.insert_chunks(chunk_rows)
        self._meta_db.update_source_status(source_id, "done", len(chunks))
        self._meta_db.log_event(
            "refresh", f"{file_path.name} → {len(chunks)} chunks re-indexed"
        )

        console.print(f"  [green]✓[/] {file_path.name} → {len(chunks)} chunks")
        return len(chunks)

    def _delete_file_chunks(self, file_path: Path) -> None:
        """
        Remove all ChromaDB documents belonging to this file.
        ChromaDB supports metadata-filtered deletes.
        """
        try:
            collection = self._store._store._collection
            # Query IDs matching this source file
            results = collection.get(
                where={"source": str(file_path)},
                include=[],
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                logger.info(f"  Deleted {len(ids)} old chunks for {file_path.name}")
        except Exception as e:
            logger.warning(f"  Could not delete old chunks for {file_path.name}: {e}")

    # ── File change detection ──────────────────────────────────────────────────

    def _find_changed_files(self) -> List[Path]:
        """
        Return files in data/raw/ modified within the last `changed_files_hours`.
        Respects the supported_extensions list from settings.
        """
        raw_dir     = settings.paths.raw_data
        exts        = settings.ingestion.supported_extensions
        cutoff_secs = self.changed_files_hours * 3600
        now         = time.time()

        changed = []
        for path in raw_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in exts:
                continue
            age = now - path.stat().st_mtime
            if age <= cutoff_secs:
                changed.append(path)

        return sorted(changed)

    # ── MLflow ────────────────────────────────────────────────────────────────

    def _log_to_mlflow(self, summary: dict) -> None:
        mlflow.set_tracking_uri(f"file://{settings.paths.mlflow_uri}")
        mlflow.set_experiment("scheduled_refresh")
        with mlflow.start_run(run_name="refresh"):
            mlflow.log_params({
                "interval_minutes":    self.interval_minutes,
                "changed_files_hours": self.changed_files_hours,
            })
            mlflow.log_metrics({
                "changed_files": summary["changed_files"],
                "new_chunks":    summary["new_chunks"],
            })

    # ── Scheduler start ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the blocking APScheduler loop."""
        scheduler = BlockingScheduler(timezone=self.tz)
        scheduler.add_job(
            self.refresh_once,
            trigger  = IntervalTrigger(minutes=self.interval_minutes),
            id       = "data_refresh",
            name     = "Data Refresh Pipeline",
            max_instances = 1,     # prevent overlap if a run takes longer than interval
        )

        console.print(
            f"\n[bold cyan]Refresh Scheduler started[/]\n"
            f"[dim]Checking every {self.interval_minutes} minute(s) "
            f"for files changed in the last {self.changed_files_hours} hour(s)[/]\n"
            f"[dim]Press Ctrl+C to stop[/]\n"
        )

        # Run once immediately on startup
        self.refresh_once()

        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            console.print("\n[dim]Scheduler stopped.[/]")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data refresh scheduler")
    parser.add_argument("--once",     action="store_true",
                        help="Run one refresh cycle and exit (no loop)")
    parser.add_argument("--interval", type=int, default=None,
                        help="Override refresh interval in minutes")
    parser.add_argument("--hours",    type=int, default=None,
                        help="Override changed-files window in hours")
    args = parser.parse_args()

    sched = RefreshScheduler(
        interval_minutes    = args.interval,
        changed_files_hours = args.hours,
    )

    if args.once:
        summary = sched.refresh_once()
        console.print(f"\n[bold]Summary:[/] {summary}")
    else:
        sched.start()


if __name__ == "__main__":
    main()