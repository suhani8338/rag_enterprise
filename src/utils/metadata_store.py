"""
src/utils/metadata_store.py
────────────────────────────
SQLite-backed metadata store (replaces Snowflake).

Schema (dimensional model):
  dim_source   — one row per source file
  fact_chunk   — one row per text chunk with FK to dim_source

Usage:
    store = MetadataStore()
    source_id = store.insert_source("my_doc.pdf", "pdf", 1024)
    store.insert_chunks([{"chunk_id": "...", "source_id": source_id, ...}])
    df = store.query("SELECT * FROM fact_chunk LIMIT 10")
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MetadataStore:
    """Thin wrapper around SQLite that mirrors a Snowflake dimensional model."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.paths.sqlite_db
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._create_schema()
        logger.info(f"MetadataStore initialised at {self.db_path}")

    # ── Schema ─────────────────────────────────────────────────────────────────

    def _create_schema(self) -> None:
        ddl = """
        -- Dimension: one row per ingested source file
        CREATE TABLE IF NOT EXISTS dim_source (
            source_id       TEXT PRIMARY KEY,   -- MD5 of absolute path
            file_name       TEXT NOT NULL,
            file_type       TEXT NOT NULL,      -- pdf | csv | txt | html | md
            file_size_bytes INTEGER,
            ingested_at     REAL NOT NULL,      -- Unix timestamp
            num_chunks      INTEGER DEFAULT 0,
            status          TEXT DEFAULT 'pending'  -- pending | done | error
        );

        -- Fact: one row per text chunk
        CREATE TABLE IF NOT EXISTS fact_chunk (
            chunk_id        TEXT PRIMARY KEY,   -- MD5(source_id + chunk_index)
            source_id       TEXT NOT NULL REFERENCES dim_source(source_id),
            chunk_index     INTEGER NOT NULL,
            char_count      INTEGER,
            token_estimate  INTEGER,            -- char_count // 4
            page_number     INTEGER,            -- PDF only
            section_header  TEXT,               -- nearest heading (best-effort)
            embedded_at     REAL,               -- Unix timestamp (filled later)
            chroma_id       TEXT                -- ChromaDB document id
        );

        -- Simple audit log
        CREATE TABLE IF NOT EXISTS audit_log (
            event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time  REAL NOT NULL,
            event_type  TEXT NOT NULL,
            detail      TEXT
        );
        """
        self._conn.executescript(ddl)
        self._conn.commit()

    # ── Write helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_source_id(file_path: Path) -> str:
        return hashlib.md5(str(file_path.resolve()).encode()).hexdigest()

    @staticmethod
    def _make_chunk_id(source_id: str, chunk_index: int) -> str:
        return hashlib.md5(f"{source_id}_{chunk_index}".encode()).hexdigest()

    def insert_source(
        self,
        file_path:  Path,
        file_type:  str,
        file_size:  int,
    ) -> str:
        """Upsert a dim_source row; return source_id."""
        source_id = self._make_source_id(file_path)
        self._conn.execute(
            """
            INSERT OR IGNORE INTO dim_source
                (source_id, file_name, file_type, file_size_bytes, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_id, file_path.name, file_type, file_size, time.time()),
        )
        self._conn.commit()
        return source_id

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Bulk-insert fact_chunk rows.
        Each dict must have: source_id, chunk_index, char_count.
        Optional: page_number, section_header, chroma_id.
        """
        rows = []
        for c in chunks:
            chunk_id = self._make_chunk_id(c["source_id"], c["chunk_index"])
            rows.append((
                chunk_id,
                c["source_id"],
                c["chunk_index"],
                c.get("char_count", 0),
                c.get("char_count", 0) // 4,
                c.get("page_number"),
                c.get("section_header"),
                c.get("chroma_id"),
            ))
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO fact_chunk
                (chunk_id, source_id, chunk_index, char_count,
                 token_estimate, page_number, section_header, chroma_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def update_source_status(self, source_id: str, status: str, num_chunks: int = 0) -> None:
        self._conn.execute(
            "UPDATE dim_source SET status=?, num_chunks=? WHERE source_id=?",
            (status, num_chunks, source_id),
        )
        self._conn.commit()

    def mark_chunks_embedded(self, chunk_ids: List[str]) -> None:
        ts = time.time()
        self._conn.executemany(
            "UPDATE fact_chunk SET embedded_at=? WHERE chunk_id=?",
            [(ts, cid) for cid in chunk_ids],
        )
        self._conn.commit()

    def log_event(self, event_type: str, detail: str = "") -> None:
        self._conn.execute(
            "INSERT INTO audit_log (event_time, event_type, detail) VALUES (?,?,?)",
            (time.time(), event_type, detail),
        )
        self._conn.commit()

    # ── Read helpers ───────────────────────────────────────────────────────────

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Run arbitrary SQL and return a DataFrame (read-only queries)."""
        return pd.read_sql_query(sql, self._conn, params=params)

    def get_source_summary(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT
                s.file_name,
                s.file_type,
                s.status,
                s.num_chunks,
                ROUND(s.file_size_bytes / 1024.0, 1) AS size_kb,
                datetime(s.ingested_at, 'unixepoch') AS ingested_at
            FROM dim_source s
            ORDER BY s.ingested_at DESC
            """
        )

    def get_chunk_stats(self) -> pd.DataFrame:
        return self.query(
            """
            SELECT
                s.file_type,
                COUNT(c.chunk_id)           AS total_chunks,
                AVG(c.char_count)           AS avg_chars,
                AVG(c.token_estimate)       AS avg_tokens,
                SUM(CASE WHEN c.embedded_at IS NOT NULL THEN 1 ELSE 0 END) AS embedded
            FROM fact_chunk c
            JOIN dim_source s ON s.source_id = c.source_id
            GROUP BY s.file_type
            """
        )

    def close(self) -> None:
        self._conn.close()