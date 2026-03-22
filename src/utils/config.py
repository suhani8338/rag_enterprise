"""
src/utils/config.py
───────────────────
Loads settings.yaml and exposes a typed Settings dataclass.
Import this everywhere instead of hardcoding paths/params.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH  = PROJECT_ROOT / "config" / "settings.yaml"


# ── Phase 1 dataclasses ───────────────────────────────────────────────────────

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

# ── Phase 2 dataclasses ───────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider:    str
    model:       str
    base_url:    str
    temperature: float
    max_tokens:  int

@dataclass
class RerankerConfig:
    model_name: str
    top_n:      int
    device:     str

@dataclass
class QueryRewritingConfig:
    enabled:      bool
    num_variants: int

@dataclass
class RAGConfig:
    default_persona:   str
    max_context_chars: int
    cite_sources:      bool

# ── Phase 3 dataclasses ───────────────────────────────────────────────────────

@dataclass
class AgentsConfig:
    sql_keywords:    List[str]
    sql_max_rows:    int
    memory_window:   int
    synthesis_mode:  str


# ── Master Settings ───────────────────────────────────────────────────────────

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
    # Phase 2 (Optional)
    llm:             Optional[LLMConfig]            = field(default=None)
    reranker:        Optional[RerankerConfig]       = field(default=None)
    query_rewriting: Optional[QueryRewritingConfig] = field(default=None)
    rag:             Optional[RAGConfig]            = field(default=None)
    # Phase 3 (Optional)
    agents:          Optional[AgentsConfig]         = field(default=None)
    project_root:    Path = field(default_factory=lambda: PROJECT_ROOT)


def load_settings(config_path: Path = CONFIG_PATH) -> Settings:
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
    for attr in ("raw_data", "processed_data", "chroma_db"):
        getattr(paths, attr).mkdir(parents=True, exist_ok=True)
    paths.sqlite_db.parent.mkdir(parents=True, exist_ok=True)

    ing = raw["ingestion"]
    chk = raw["chunking"]
    emb = raw["embedding"]
    vs  = raw["vectorstore"]
    ret = raw["retrieval"]
    log = raw["logging"]

    logging.basicConfig(
        level  = getattr(logging, log["level"]),
        format = log["format"],
    )

    # Phase 2
    llm_cfg = reranker_cfg = qr_cfg = rag_cfg = None
    if "llm" in raw:
        l = raw["llm"]
        llm_cfg = LLMConfig(
            provider=l["provider"], model=l["model"],
            base_url=l["base_url"], temperature=l["temperature"],
            max_tokens=l["max_tokens"],
        )
    if "reranker" in raw:
        r = raw["reranker"]
        reranker_cfg = RerankerConfig(
            model_name=r["model_name"], top_n=r["top_n"], device=r["device"],
        )
    if "query_rewriting" in raw:
        q = raw["query_rewriting"]
        qr_cfg = QueryRewritingConfig(enabled=q["enabled"], num_variants=q["num_variants"])
    if "rag" in raw:
        rc = raw["rag"]
        rag_cfg = RAGConfig(
            default_persona=rc["default_persona"],
            max_context_chars=rc["max_context_chars"],
            cite_sources=rc["cite_sources"],
        )

    # Phase 3
    agents_cfg = None
    if "agents" in raw:
        a = raw["agents"]
        agents_cfg = AgentsConfig(
            sql_keywords   = a.get("sql_keywords", []),
            sql_max_rows   = a.get("sql_max_rows", 20),
            memory_window  = a.get("memory_window", 6),
            synthesis_mode = a.get("synthesis_mode", "weighted"),
        )

    return Settings(
        project_name    = raw["project"]["name"],
        project_version = raw["project"]["version"],
        paths           = paths,
        ingestion       = IngestionConfig(
            supported_extensions=ing["supported_extensions"],
            batch_size=ing["batch_size"],
        ),
        chunking = ChunkingConfig(
            chunk_size=chk["chunk_size"],
            chunk_overlap=chk["chunk_overlap"],
            separators=chk["separators"],
        ),
        embedding = EmbeddingConfig(
            model_name=emb["model_name"],
            device=emb["device"],
            batch_size=emb["batch_size"],
        ),
        vectorstore = VectorstoreConfig(
            collection_name=vs["collection_name"],
            distance_metric=vs["distance_metric"],
            dense_weight=vs["dense_weight"],
            sparse_weight=vs["sparse_weight"],
        ),
        retrieval = RetrievalConfig(
            top_k=ret["top_k"],
            mmr_lambda=ret["mmr_lambda"],
            final_k=ret["final_k"],
        ),
        llm             = llm_cfg,
        reranker        = reranker_cfg,
        query_rewriting = qr_cfg,
        rag             = rag_cfg,
        agents          = agents_cfg,
    )


settings = load_settings()