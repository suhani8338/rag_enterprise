"""
src/agents/supervisor_agent.py
───────────────────────────────
Supervisor Agent — the router at the top of the LangGraph graph.

Responsibilities:
  1. Classify the user's intent into one of four categories:
       "rag"      — question answered from documents (ChromaDB)
       "sql"      — question answered from structured data (SQLite)
       "both"     — question needs both document context AND structured data
       "chitchat" — greeting / meta question, no retrieval needed
  2. Set agent_route so downstream nodes know which agents to activate.
  3. Write a short plan string explaining its reasoning (useful for debugging
     and for showing interviewers that you understand agent planning loops).

Two-stage classification:
  Stage 1 — Keyword heuristic (fast, no LLM call needed):
    If the question contains SQL-like keywords ("how many", "list all",
    "which products", "average price", etc.) → lean toward "sql".
  Stage 2 — LLM classification (only fires when heuristic is uncertain):
    Asks the LLM to pick one of the four intents and explain why.

This hybrid approach keeps latency low for clear-cut queries while
handling edge cases correctly.
"""

from __future__ import annotations

import re
from typing import Any, Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.agents.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Intent classification prompt ───────────────────────────────────────────────

_CLASSIFY_TEMPLATE = """\
You are a router for an enterprise knowledge system. Classify the user's question
into EXACTLY ONE of these intents and respond with only the intent label:

  rag      — needs information from text documents (reports, policies, guides)
  sql      — needs counts, lists, or aggregations from structured/tabular data
  both     — needs both document text AND structured data to answer fully
  chitchat — greeting, small talk, or a meta question about the system itself

Examples:
  "What is our remote work policy?"          → rag
  "How many products are in the cloud tier?" → sql
  "What does AcmeMesh do and what does it cost?" → both
  "Hello, how are you?"                      → chitchat

User question: {question}

Intent (one word only):"""

_CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_CLASSIFY_TEMPLATE,
)

_VALID_INTENTS = {"rag", "sql", "both", "chitchat"}

# ── Agent ─────────────────────────────────────────────────────────────────────

class SupervisorAgent:
    """
    Classifies intent and sets agent_route in AgentState.
    Used as a LangGraph node function.
    """

    def __init__(self, llm):
        self.llm       = llm
        self._chain    = _CLASSIFY_PROMPT | llm | StrOutputParser()
        cfg            = settings.agents
        self._sql_kws  = [kw.lower() for kw in (cfg.sql_keywords if cfg else [])]
        logger.info("SupervisorAgent ready")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node — returns partial state update."""
        question = state.get("question", "")
        logger.info(f"Supervisor classifying: '{question[:80]}'")

        intent = self._classify(question)
        route  = _intent_to_route(intent)
        plan   = f"Intent={intent} | Route={route} | Q='{question[:60]}'"

        logger.info(f"Supervisor → intent={intent} route={route}")
        return {
            "intent":      intent,
            "agent_route": route,
            "plan":        plan,
        }

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify(self, question: str) -> str:
        q_lower = question.lower()

        # Stage 1: keyword heuristic
        sql_hits = sum(1 for kw in self._sql_kws if kw in q_lower)
        has_doc_words = any(w in q_lower for w in [
            "policy", "explain", "describe", "what is", "how does",
            "overview", "guide", "tell me about", "summarise", "summarize",
        ])

        # Chitchat shortcut
        if re.match(r"^(hi|hello|hey|thanks|thank you|bye|good\s+\w+)[!\s.,]*$",
                    q_lower.strip()):
            return "chitchat"

        if sql_hits >= 2 and not has_doc_words:
            return "sql"
        if sql_hits >= 1 and has_doc_words:
            return "both"
        if sql_hits == 0 and has_doc_words:
            return "rag"

        # Stage 2: LLM classification for ambiguous cases
        try:
            raw = self._chain.invoke({"question": question}).strip().lower()
            # Extract first word only — guard against verbose LLM output
            intent = raw.split()[0] if raw else "rag"
            return intent if intent in _VALID_INTENTS else "rag"
        except Exception as e:
            logger.warning(f"LLM classification failed ({e}), defaulting to 'rag'")
            return "rag"


# ── Route mapping ──────────────────────────────────────────────────────────────

def _intent_to_route(intent: str) -> list:
    return {
        "rag":      ["retriever"],
        "sql":      ["sql"],
        "both":     ["retriever", "sql"],
        "chitchat": [],
    }.get(intent, ["retriever"])