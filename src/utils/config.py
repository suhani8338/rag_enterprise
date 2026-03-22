"""
src/utils/config.py
───────────────────
Loads settings.yaml and exposes a typed Settings dataclass.
Import this everywhere instead of hardcoding paths/params.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

# ── Locate project root (two levels above this file) ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "settings.yaml"


# ── Dataclasses (typed, IDE-friendly) ─────────────────────────────────────────

@dataclass
class PathsConfig:
    raw_data:       Path
    processed_data: Path
    chroma_db:      Path
    sqlite_db:      Path
    mlflow_uri:     str

@dataclass
class IngestionConfig:
    supported_extensions: List[str]
    batch_size: int

@dataclass
class ChunkingConfig:
    chunk_size:    int
    chunk_overlap: int
    separators:    List[str]

@dataclass
class EmbeddingConfig:
    model_name: str
    device:     str
    batch_size: int

@dataclass
class VectorstoreConfig:
    collection_name: str
    distance_metric: str
    dense_weight:    float
    sparse_weight:   float

@dataclass
class RetrievalConfig:
    top_k:      int
    mmr_lambda: float
    final_k:    int

@dataclass
class Settings:
    project_name:    str
    project_version: str
    paths:           PathsConfig
    ingestion:       IngestionConfig
    chunking:        ChunkingConfig
    embedding:       EmbeddingConfig
    vectorstore:     VectorstoreConfig
    retrieval:       RetrievalConfig
    project_root:    Path = field(default_factory=lambda: PROJECT_ROOT)


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
    """Parse settings.yaml and return a fully resolved Settings object."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    root = PROJECT_ROOT

    paths = PathsConfig(
        raw_data       = root / raw["paths"]["raw_data"],
        processed_data = root / raw["paths"]["processed_data"],
        chroma_db      = root / raw["paths"]["chroma_db"],
        sqlite_db      = root / raw["paths"]["sqlite_db"],
        mlflow_uri     = str(root / raw["paths"]["mlflow_uri"]),
    )

    # Create directories on first load
    for attr in ("raw_data", "processed_data", "chroma_db"):
        getattr(paths, attr).mkdir(parents=True, exist_ok=True)
    paths.sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    ing   = raw["ingestion"]
    chk   = raw["chunking"]
    emb   = raw["embedding"]
    vs    = raw["vectorstore"]
    ret   = raw["retrieval"]
    log   = raw["logging"]

    # Configure root logger from yaml
    logging.basicConfig(
        level   = getattr(logging, log["level"]),
        format  = log["format"],
    )

    return Settings(
        project_name    = raw["project"]["name"],
        project_version = raw["project"]["version"],
        paths           = paths,
        ingestion       = IngestionConfig(
            supported_extensions = ing["supported_extensions"],
            batch_size           = ing["batch_size"],
        ),
        chunking = ChunkingConfig(
            chunk_size    = chk["chunk_size"],
            chunk_overlap = chk["chunk_overlap"],
            separators    = chk["separators"],
        ),
        embedding = EmbeddingConfig(
            model_name = emb["model_name"],
            device     = emb["device"],
            batch_size = emb["batch_size"],
        ),
        vectorstore = VectorstoreConfig(
            collection_name = vs["collection_name"],
            distance_metric = vs["distance_metric"],
            dense_weight    = vs["dense_weight"],
            sparse_weight   = vs["sparse_weight"],
        ),
        retrieval = RetrievalConfig(
            top_k      = ret["top_k"],
            mmr_lambda = ret["mmr_lambda"],
            final_k    = ret["final_k"],
        ),
    )


# Singleton — import `settings` directly from other modules
settings = load_settings()