"""
src/agents/retriever_agent.py
──────────────────────────────
Retriever Agent — LangGraph node that wraps the full Phase 2 RAG pipeline.

What it does:
  Reads the question and persona from AgentState, runs the complete Phase 2
  pipeline (query rewriting → hybrid retrieval → cross-encoder reranking →
  LLM generation), and writes the answer + sources back into state.

Why it's a separate agent (not just calling RAGChain directly):
  In the multi-agent graph, the RetrieverAgent can be skipped entirely when
  the Supervisor routes to sql-only. It also participates in the shared
  memory (chat_history) and can be swapped for a different retrieval
  strategy without touching the graph structure.

Memory integration:
  The agent injects the last N turns from chat_history into the question
  context so follow-up questions work correctly:
    Turn 1: "What is AcmeMesh?"
    Turn 2: "How much does it cost?"  ← needs prior context to resolve "it"
"""

from __future__ import annotations

from typing import Any, Dict

from src.agents.state import AgentState
from src.rag.rag_chain import RAGChain
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverAgent:
    """
    Wraps RAGChain as a LangGraph node.
    Initialised once; __call__ is invoked by LangGraph per turn.
    """

    def __init__(self, rag_chain: RAGChain):
        self._chain        = rag_chain
        cfg                = settings.agents
        self._memory_window = cfg.memory_window if cfg else 6
        logger.info("RetrieverAgent ready")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node — returns partial state update."""
        question = state.get("question", "")
        persona  = state.get("persona", "analyst")
        history  = state.get("chat_history", [])

        # Skip if supervisor didn't route here
        route = state.get("agent_route", [])
        if "retriever" not in route:
            logger.debug("RetrieverAgent skipped (not in route)")
            return {}

        # Inject recent chat history into question for follow-up resolution
        augmented_question = _augment_with_history(question, history, self._memory_window)

        logger.info(f"RetrieverAgent answering: '{question[:80]}'")

        try:
            response = self._chain.ask(
                augmented_question,
                persona      = persona,
                mode         = "hybrid",
                skip_rewrite = False,
            )
            return {
                "rag_answer":  response.answer,
                "rag_sources": response.sources,
                "rag_chunks":  response.retrieved_chunks,
            }
        except Exception as e:
            logger.error(f"RetrieverAgent failed: {e}")
            return {
                "rag_answer":  f"[Retriever error: {e}]",
                "rag_sources": [],
                "rag_chunks":  0,
                "error":       str(e),
            }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _augment_with_history(question: str, history: list, window: int) -> str:
    """
    Prepend the last `window` turns of chat history to the question.
    This lets the LLM resolve pronouns and follow-up references correctly.

    Example output:
        [Previous conversation]
        User: What is AcmeMesh?
        Assistant: AcmeMesh is a service mesh product...

        [Current question]
        How much does it cost?
    """
    if not history:
        return question

    recent = history[-(window):]
    lines  = ["[Previous conversation]"]
    for turn in recent:
        role    = turn.get("role", "user").capitalize()
        content = turn.get("content", "")[:300]   # truncate long turns
        lines.append(f"{role}: {content}")

    lines.append("\n[Current question]")
    lines.append(question)
    return "\n".join(lines)