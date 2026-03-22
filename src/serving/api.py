"""
src/serving/api.py
───────────────────
FastAPI REST endpoint wrapping the Phase 3 AgentSystem.

Endpoints:
  POST /ask              — single question, returns full JSON response
  POST /ask/stream       — Server-Sent Events stream, one event per agent node
  GET  /health           — liveness check (returns status + ChromaDB chunk count)
  GET  /history          — return current session chat history
  POST /history/reset    — clear chat history
  GET  /status           — index stats (sources, chunks, last ingestion time)

The AgentSystem is initialised once at startup and shared across all requests
(singleton pattern via FastAPI lifespan). This avoids reloading the embedding
model and LangGraph graph on every request.

Session management:
  Each client passes an optional session_id header. Different sessions get
  independent chat_history. Without a session_id, a single shared history
  is used (fine for single-user local development).

Run:
  uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.agents.graph import AgentSystem
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Session store (in-memory, per session_id) ─────────────────────────────────

_sessions: Dict[str, AgentSystem] = {}
_DEFAULT_SESSION = "default"


def _get_system(session_id: str) -> AgentSystem:
    """Return (or create) the AgentSystem for this session."""
    if session_id not in _sessions:
        logger.info(f"Creating new AgentSystem for session '{session_id}'")
        _sessions[session_id] = AgentSystem()
    return _sessions[session_id]


# ── Lifespan: lazy-load AgentSystem on first request ──────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API startup — lazy-loading AgentSystem on first request...")
    logger.info("API ready")
    yield
    logger.info("API shutdown")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = settings.project_name,
    description = "Multi-Agent RAG System — Phase 4 REST API",
    version     = settings.project_version,
    lifespan    = lifespan,
)

cfg = settings.serving
app.add_middleware(
    CORSMiddleware,
    allow_origins     = cfg.cors_origins if cfg else ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    persona:  str = Field(default="analyst",
                          pattern="^(analyst|executive|engineer|hr)$")

class AskResponse(BaseModel):
    question:      str
    final_answer:  str
    final_sources: List[str]
    intent:        str
    agent_route:   List[str]
    rag_chunks:    Optional[int]
    sql_query:     Optional[str]
    latency_ms:    float
    session_id:    str

class HealthResponse(BaseModel):
    status:       str
    chroma_chunks: int
    sessions:     int
    version:      str

class StatusResponse(BaseModel):
    chroma_chunks: int
    sources:       List[dict]
    chunk_stats:   List[dict]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Liveness check — always returns 200 if the server is up."""
    from src.embedding.embedder import LocalEmbedder
    from src.vectorstore.chroma_store import ChromaVectorStore
    try:
        embedder = LocalEmbedder()
        store    = ChromaVectorStore(embedder.as_langchain_embeddings())
        chunks   = store.collection_count()
    except Exception:
        chunks = -1

    return HealthResponse(
        status        = "ok",
        chroma_chunks = chunks,
        sessions      = len(_sessions),
        version       = settings.project_version,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(
    request:    AskRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Ask a question. The AgentSystem auto-routes to the correct agent(s).
    Pass X-Session-Id header to maintain conversation history across calls.
    """
    session_id = x_session_id or _DEFAULT_SESSION
    system     = _get_system(session_id)

    t0 = time.perf_counter()
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: system.ask(request.question, persona=request.persona)
        )
    except Exception as e:
        logger.error(f"ask() failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    return AskResponse(
        question      = request.question,
        final_answer  = result.get("final_answer", ""),
        final_sources = result.get("final_sources", []),
        intent        = result.get("intent", ""),
        agent_route   = result.get("agent_route", []),
        rag_chunks    = result.get("rag_chunks"),
        sql_query     = result.get("sql_query"),
        latency_ms    = round(latency_ms, 1),
        session_id    = session_id,
    )


@app.post("/ask/stream", tags=["RAG"])
async def ask_stream(
    request:      AskRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Server-Sent Events stream. Each agent node emits one event as it completes,
    then a final 'done' event with the complete answer.

    Event format:  data: {"node": "supervisor", "intent": "rag", ...}
    """
    session_id = x_session_id or _DEFAULT_SESSION
    system     = _get_system(session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()

        def _stream():
            return list(system.stream(request.question, persona=request.persona))

        try:
            events = await loop.run_in_executor(None, _stream)
            for node_name, partial_state in events:
                payload = {"node": node_name}
                # Add the most useful field from each node
                if node_name == "supervisor":
                    payload["intent"]      = partial_state.get("intent", "")
                    payload["agent_route"] = partial_state.get("agent_route", [])
                elif node_name == "retriever":
                    payload["rag_chunks"] = partial_state.get("rag_chunks", 0)
                    preview = partial_state.get("rag_answer", "")[:200]
                    payload["preview"] = preview
                elif node_name == "sql":
                    payload["sql_query"]  = partial_state.get("sql_query", "")
                    payload["sql_answer"] = partial_state.get("sql_answer", "")[:200]
                elif node_name == "synthesizer":
                    payload["final_answer"]  = partial_state.get("final_answer", "")
                    payload["final_sources"] = partial_state.get("final_sources", [])

                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(0)   # yield control to event loop

            yield f"data: {json.dumps({'node': 'done'})}\n\n"

        except Exception as e:
            logger.error(f"stream failed: {e}")
            yield f"data: {json.dumps({'node': 'error', 'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history", tags=["Session"])
async def get_history(x_session_id: Optional[str] = Header(default=None)):
    """Return the chat history for this session."""
    session_id = x_session_id or _DEFAULT_SESSION
    if session_id not in _sessions:
        return {"session_id": session_id, "history": []}
    system = _sessions[session_id]
    return {"session_id": session_id, "history": system.history}


@app.post("/history/reset", tags=["Session"])
async def reset_history(x_session_id: Optional[str] = Header(default=None)):
    """Clear conversation memory for this session."""
    session_id = x_session_id or _DEFAULT_SESSION
    if session_id in _sessions:
        _sessions[session_id].reset_memory()
    return {"session_id": session_id, "status": "reset"}


@app.get("/status", response_model=StatusResponse, tags=["Meta"])
async def status():
    """Return index statistics from SQLite."""
    from src.utils.metadata_store import MetadataStore
    try:
        meta = MetadataStore()
        src_df   = meta.get_source_summary()
        chunk_df = meta.get_chunk_stats()

        from src.embedding.embedder import LocalEmbedder
        from src.vectorstore.chroma_store import ChromaVectorStore
        embedder = LocalEmbedder()
        store    = ChromaVectorStore(embedder.as_langchain_embeddings())

        return StatusResponse(
            chroma_chunks = store.collection_count(),
            sources       = src_df.to_dict("records"),
            chunk_stats   = chunk_df.to_dict("records"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))