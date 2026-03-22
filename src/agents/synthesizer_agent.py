"""
src/agents/synthesizer_agent.py
─────────────────────────────────
Synthesizer Agent — the final node in the LangGraph graph.

Responsibilities:
  1. Collect answers from whichever agents ran (RetrieverAgent, SQLAgent, or both).
  2. Merge them into a single coherent, cited response.
  3. Handle edge cases: only one agent ran, both failed, chitchat.

Three synthesis modes (set in settings.yaml):
  "weighted"  — LLM merges both answers with citation attribution (default)
  "concat"    — simple concatenation with headers, no extra LLM call
  "llm_merge" — single LLM call synthesises everything from scratch

Memory update:
  After synthesis, appends (user question, final answer) to chat_history
  so subsequent turns have full conversational context.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.agents.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Synthesis prompt (weighted mode) ──────────────────────────────────────────

_SYNTHESIZE_TEMPLATE = """\
{persona_preamble}

You have received answers from two specialised agents. Synthesize them into
a single, clear, well-cited response to the user's question.

Rules:
- Integrate both answers naturally — do not just concatenate them
- Cite sources inline as [Doc: filename] or [Data: table/query]
- If one answer contradicts the other, note the discrepancy
- If one answer adds no new information, omit it gracefully
- Match the persona's preferred tone and format

Original question: {question}

─── Answer from Document Retrieval ────────────────────────────────
{rag_answer}
Sources: {rag_sources}
───────────────────────────────────────────────────────────────────

─── Answer from Structured Data (SQL) ─────────────────────────────
{sql_answer}
Query used: {sql_query}
───────────────────────────────────────────────────────────────────

Synthesized answer:"""

_SYNTHESIZE_PROMPT = PromptTemplate(
    input_variables=[
        "persona_preamble", "question",
        "rag_answer", "rag_sources",
        "sql_answer", "sql_query",
    ],
    template=_SYNTHESIZE_TEMPLATE,
)

# ── Chitchat prompt ────────────────────────────────────────────────────────────

_CHITCHAT_TEMPLATE = """\
You are a helpful enterprise assistant. Respond naturally and briefly to
the user's message. If they ask what you can do, tell them you can answer
questions about company documents and structured data.

User: {question}
Assistant:"""

_CHITCHAT_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_CHITCHAT_TEMPLATE,
)


class SynthesizerAgent:
    """
    Final LangGraph node — merges all agent outputs into one final answer.
    """

    def __init__(self, llm):
        self.llm              = llm
        self._synth_chain     = _SYNTHESIZE_PROMPT | llm | StrOutputParser()
        self._chitchat_chain  = _CHITCHAT_PROMPT   | llm | StrOutputParser()
        cfg                   = settings.agents
        self._mode            = cfg.synthesis_mode if cfg else "weighted"
        logger.info(f"SynthesizerAgent ready | mode={self._mode}")

    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """LangGraph node — produces final_answer and updates chat_history."""
        question = state.get("question", "")
        intent   = state.get("intent", "rag")
        persona  = state.get("persona", "analyst")
        history  = state.get("chat_history", [])

        # ── Chitchat: no retrieval needed ─────────────────────────────────────
        if intent == "chitchat":
            try:
                answer = self._chitchat_chain.invoke({"question": question})
            except Exception:
                answer = "Hello! I can answer questions about company documents and data."
            return self._build_result(answer, [], question, history)

        rag_answer  = state.get("rag_answer", "")
        rag_sources = state.get("rag_sources", [])
        sql_answer  = state.get("sql_answer", "")
        sql_query   = state.get("sql_query", "")

        # ── Single-agent: only one ran ────────────────────────────────────────
        route = state.get("agent_route", [])

        if route == ["retriever"] or (rag_answer and not sql_answer):
            final = rag_answer or "[No answer from retriever]"
            return self._build_result(final, rag_sources, question, history)

        if route == ["sql"] or (sql_answer and not rag_answer):
            final = sql_answer or "[No answer from SQL agent]"
            sources = [f"SQL: {sql_query[:80]}"] if sql_query else []
            return self._build_result(final, sources, question, history)

        # ── Both agents ran — synthesize ──────────────────────────────────────
        final, sources = self._synthesize(
            question, persona, rag_answer, rag_sources, sql_answer, sql_query
        )
        return self._build_result(final, sources, question, history)

    # ── Synthesis strategies ───────────────────────────────────────────────────

    def _synthesize(
        self,
        question:    str,
        persona:     str,
        rag_answer:  str,
        rag_sources: List[str],
        sql_answer:  str,
        sql_query:   str,
    ):
        if self._mode == "concat":
            return _concat_merge(rag_answer, rag_sources, sql_answer, sql_query)

        # weighted or llm_merge — use the LLM
        from src.rag.prompt_templates import PERSONA_PREAMBLES
        preamble = PERSONA_PREAMBLES.get(persona, PERSONA_PREAMBLES["analyst"])
        try:
            answer = self._synth_chain.invoke({
                "persona_preamble": preamble,
                "question":         question,
                "rag_answer":       rag_answer or "No document answer available.",
                "rag_sources":      ", ".join(rag_sources) if rag_sources else "none",
                "sql_answer":       sql_answer or "No structured data answer available.",
                "sql_query":        sql_query  or "none",
            })
            sources = list(rag_sources)
            if sql_query:
                sources.append(f"SQL: {sql_query[:80]}")
            return answer.strip(), sources
        except Exception as e:
            logger.warning(f"LLM synthesis failed ({e}), falling back to concat")
            return _concat_merge(rag_answer, rag_sources, sql_answer, sql_query)

    # ── State builder ──────────────────────────────────────────────────────────

    def _build_result(
        self,
        answer:   str,
        sources:  List[str],
        question: str,
        history:  List[Dict],
    ) -> Dict[str, Any]:
        # Update memory
        updated_history = list(history) + [
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer[:500]},  # cap stored length
        ]
        return {
            "final_answer":  answer.strip(),
            "final_sources": sources,
            "chat_history":  updated_history,
        }


# ── Concat fallback ────────────────────────────────────────────────────────────

def _concat_merge(rag_answer, rag_sources, sql_answer, sql_query):
    parts = []
    if rag_answer:
        parts.append(f"**From documents:**\n{rag_answer}")
    if sql_answer:
        parts.append(f"**From structured data:**\n{sql_answer}")
    combined = "\n\n".join(parts) or "[No answer available]"
    sources  = list(rag_sources)
    if sql_query:
        sources.append(f"SQL: {sql_query[:80]}")
    return combined, sources