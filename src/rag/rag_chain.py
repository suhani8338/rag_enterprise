"""
src/rag/rag_chain.py
─────────────────────
The full Phase 2 RAG pipeline:

  User question
      ↓  QueryRewriter        → 3 search variants
      ↓  ChromaVectorStore    → hybrid search on all variants → deduplicated candidates
      ↓  CrossEncoderReranker → re-score and keep top N
      ↓  format_context()     → assemble context string with citations
      ↓  PromptTemplate       → inject persona + context + question
      ↓  Ollama / LLM         → generate grounded answer
      ↓  RAGResponse          → structured result with sources

Usage:
    chain = RAGChain()
    result = chain.ask("What is our parental leave policy?")
    print(result.answer)
    print(result.sources)

    # Change persona mid-session
    result = chain.ask("Summarise Q4 revenue", persona="executive")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.embedding.embedder import LocalEmbedder
from src.rag.llm_factory import get_llm
from src.rag.prompt_templates import (
    AVAILABLE_PERSONAS,
    format_context,
    get_compare_prompt,
    get_rag_prompt,
    get_summarise_prompt,
)
from src.rag.query_rewriter import QueryRewriter
from src.rag.reranker import CrossEncoderReranker
from src.utils.config import settings
from src.utils.logger import get_logger, log_metrics, mlflow_run
from src.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """Structured result from the RAG chain."""
    question:        str
    answer:          str
    sources:         List[str]          # ["filename.pdf page 3", ...]
    persona:         str
    retrieved_chunks: int
    reranked_chunks:  int
    latency_ms:      float
    query_variants:  List[str] = field(default_factory=list)

    def pretty_print(self) -> None:
        """Print a formatted answer with sources to stdout."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        console.print(
            Panel(
                f"[bold]{self.answer}[/]",
                title=f"[cyan]Answer[/] [dim]({self.persona} · {self.latency_ms:.0f}ms)[/]",
                border_style="cyan",
            )
        )
        if self.sources:
            console.print("[dim]Sources:[/]")
            for src in self.sources:
                console.print(f"  [dim]•[/] {src}")

        console.print(
            f"\n[dim]Retrieved: {self.retrieved_chunks} chunks "
            f"→ reranked to {self.reranked_chunks}[/]"
        )


# ── RAGChain ───────────────────────────────────────────────────────────────────

class RAGChain:
    """
    Orchestrates all Phase 2 components into a single .ask() call.

    Components initialised once and reused across queries:
      - LocalEmbedder     (embedding model)
      - ChromaVectorStore (persistent index from Phase 1)
      - QueryRewriter     (LLM-based query expansion)
      - CrossEncoderReranker (re-scoring)
      - Ollama LLM        (answer generation)
    """

    def __init__(self):
        from rich.console import Console
        from rich.panel import Panel
        console = Console()
        console.print(Panel.fit(
            f"[bold cyan]{settings.project_name}[/] — Phase 2: RAG Pipeline",
            border_style="cyan",
        ))

        # Phase 1 components (re-used)
        self.embedder = LocalEmbedder()
        self.store    = ChromaVectorStore(self.embedder.as_langchain_embeddings())

        if self.store.collection_count() == 0:
            raise RuntimeError(
                "ChromaDB is empty. Run Phase 1 ingestion first:\n"
                "  python -m src.pipeline"
            )
        self.store.load_bm25_from_existing()

        # Phase 2 components
        self.llm      = get_llm()
        self.rewriter = QueryRewriter(self.llm)
        self.reranker = CrossEncoderReranker()

        self._default_persona = settings.rag.default_persona if settings.rag else "analyst"
        logger.info("RAGChain ready")

    # ── Core ask ───────────────────────────────────────────────────────────────

    def ask(
        self,
        question:       str,
        persona:        Optional[str] = None,
        mode:           str = "hybrid",   # "dense" | "hybrid" | "mmr"
        skip_rewrite:   bool = False,
    ) -> RAGResponse:
        """
        Full RAG pipeline: question → grounded cited answer.

        Args:
            question:     User's natural language question.
            persona:      Override default persona ("analyst","executive","engineer","hr").
            mode:         Retrieval mode passed to ChromaVectorStore.
            skip_rewrite: Skip query rewriting (faster, lower recall).

        Returns:
            RAGResponse with answer, sources, and performance metrics.
        """
        persona = persona or self._default_persona
        if persona not in AVAILABLE_PERSONAS:
            logger.warning(f"Unknown persona '{persona}', using 'analyst'")
            persona = "analyst"

        t0 = time.perf_counter()

        with mlflow_run(run_name="rag_query"):
            log_metrics({"query_length": len(question)})

            # ── Step 1: Query rewriting ────────────────────────────────────────
            if skip_rewrite:
                query_variants = [question]
            else:
                query_variants = self.rewriter.rewrite(question)

            # ── Step 2: Multi-query retrieval ─────────────────────────────────
            all_chunks: List[Document] = []
            seen_content: set = set()

            for variant in query_variants:
                if mode == "dense":
                    results = [doc for doc, _ in self.store.dense_search(variant)]
                elif mode == "mmr":
                    results = self.store.mmr_search(variant)
                else:
                    results = self.store.hybrid_search(variant)

                for doc in results:
                    # Deduplicate by content fingerprint
                    key = doc.page_content[:100]
                    if key not in seen_content:
                        seen_content.add(key)
                        all_chunks.append(doc)

            logger.info(
                f"Retrieved {len(all_chunks)} unique chunks "
                f"from {len(query_variants)} query variants"
            )
            log_metrics({"retrieved_chunks": len(all_chunks)})

            # ── Step 3: Re-rank ───────────────────────────────────────────────
            reranked = self.reranker.rerank(question, all_chunks)
            log_metrics({"reranked_chunks": len(reranked)})

            # ── Step 4: Build context ─────────────────────────────────────────
            max_chars = settings.rag.max_context_chars if settings.rag else 6000
            context   = format_context(reranked, max_chars=max_chars)

            # ── Step 5: Generate answer ───────────────────────────────────────
            prompt_template = get_rag_prompt(persona=persona)
            chain           = prompt_template | self.llm | StrOutputParser()
            answer          = chain.invoke({
                "context":  context,
                "question": question,
            })

            # ── Step 6: Extract sources ───────────────────────────────────────
            sources = _extract_sources(reranked)

            latency_ms = (time.perf_counter() - t0) * 1000
            log_metrics({"rag_latency_ms": latency_ms})

            return RAGResponse(
                question         = question,
                answer           = answer.strip(),
                sources          = sources,
                persona          = persona,
                retrieved_chunks = len(all_chunks),
                reranked_chunks  = len(reranked),
                latency_ms       = latency_ms,
                query_variants   = query_variants,
            )

    # ── Summarise ──────────────────────────────────────────────────────────────

    def summarise(self, persona: str = None) -> RAGResponse:
        """
        Summarise all documents in the index.
        Retrieves a broad sample via MMR for maximum coverage.
        """
        persona = persona or self._default_persona
        t0      = time.perf_counter()

        # Broad MMR retrieval (more chunks for summarisation)
        chunks  = self.store.mmr_search("overview summary key topics", k=8, lambda_val=0.3)
        context = format_context(chunks, max_chars=8000)

        chain  = get_summarise_prompt(persona) | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context})

        return RAGResponse(
            question         = "Summarise all documents",
            answer           = answer.strip(),
            sources          = _extract_sources(chunks),
            persona          = persona,
            retrieved_chunks = len(chunks),
            reranked_chunks  = len(chunks),
            latency_ms       = (time.perf_counter() - t0) * 1000,
        )

    # ── Compare ────────────────────────────────────────────────────────────────

    def compare(self, topic_a: str, topic_b: str, persona: str = None) -> RAGResponse:
        """
        Compare two topics using retrieved context from both.
        """
        persona = persona or self._default_persona
        t0      = time.perf_counter()

        # Retrieve context for both topics
        chunks_a = self.store.hybrid_search(topic_a, k=5)
        chunks_b = self.store.hybrid_search(topic_b, k=5)
        combined = chunks_a + chunks_b

        reranked = self.reranker.rerank(f"{topic_a} vs {topic_b}", combined)
        context  = format_context(reranked)

        chain  = get_compare_prompt(persona) | self.llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "topic_a": topic_a,
            "topic_b": topic_b,
        })

        return RAGResponse(
            question         = f"Compare {topic_a} vs {topic_b}",
            answer           = answer.strip(),
            sources          = _extract_sources(reranked),
            persona          = persona,
            retrieved_chunks = len(combined),
            reranked_chunks  = len(reranked),
            latency_ms       = (time.perf_counter() - t0) * 1000,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_sources(documents: List[Document]) -> List[str]:
    """Build a deduplicated list of source citation strings."""
    seen    = set()
    sources = []
    for doc in documents:
        meta   = doc.metadata
        fname  = meta.get("file_name", "unknown")
        page   = meta.get("page_number", "")
        score  = meta.get("rerank_score", "")
        label  = f"{fname}"
        if page:
            label += f", page {page}"
        if score:
            label += f" (relevance={score})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources