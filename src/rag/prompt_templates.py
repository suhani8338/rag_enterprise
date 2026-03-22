"""
src/rag/prompt_templates.py
────────────────────────────
All LLM prompt templates for the RAG pipeline.

Personas:
  Each persona changes the tone, focus, and format of the answer.
  The same underlying context is presented differently depending on who is asking.

  • analyst   — detailed, data-driven, bullet points, numerical precision
  • executive — concise, strategic, key takeaways only
  • engineer  — technical depth, implementation details, code/config awareness
  • hr        — policy-focused, plain language, compliance emphasis

Domain prompt templates:
  • default_rag   — standard grounded Q&A with citations
  • summarise     — multi-document synthesis
  • compare       — side-by-side comparison of two topics

Usage:
    template = get_rag_prompt(persona="analyst")
    chain = template | llm | StrOutputParser()
    answer = chain.invoke({"context": "...", "question": "..."})
"""

from __future__ import annotations

from typing import Dict

from langchain_core.prompts import PromptTemplate

# ── Persona system messages ────────────────────────────────────────────────────

PERSONA_PREAMBLES: Dict[str, str] = {
    "analyst": (
        "You are a meticulous business analyst. "
        "Provide detailed, data-driven answers with specific numbers and metrics where available. "
        "Use bullet points for key findings. Flag any gaps or uncertainties explicitly."
    ),
    "executive": (
        "You are a concise executive assistant briefing a senior leader. "
        "Lead with the single most important insight. "
        "Limit your response to 3-5 bullet points maximum. Omit operational detail."
    ),
    "engineer": (
        "You are a senior software engineer. "
        "Prioritise technical accuracy, implementation details, and system constraints. "
        "Reference specific versions, configs, or code patterns where mentioned in the context."
    ),
    "hr": (
        "You are an HR specialist. "
        "Answer in plain, accessible language. "
        "Emphasise policy, compliance requirements, and employee rights. "
        "If a question touches on legal matters, recommend consulting legal counsel."
    ),
}

# ── Core RAG template ──────────────────────────────────────────────────────────

_RAG_TEMPLATE = """\
{persona_preamble}

You must answer ONLY using the context provided below.
If the context does not contain enough information to answer the question,
say: "I could not find sufficient information in the provided documents."
Do NOT use any prior knowledge outside the context.

─── CONTEXT ───────────────────────────────────────────────
{context}
───────────────────────────────────────────────────────────

Question: {question}

Answer (cite sources as [source: filename, page N] where possible):"""

# ── Summarise template ─────────────────────────────────────────────────────────

_SUMMARISE_TEMPLATE = """\
{persona_preamble}

Summarise the following document excerpts into a coherent overview.
Cover all major topics mentioned. Use the persona's preferred style.
Cite sources as [source: filename] inline.

─── DOCUMENT EXCERPTS ─────────────────────────────────────
{context}
───────────────────────────────────────────────────────────

Comprehensive summary:"""

# ── Compare template ───────────────────────────────────────────────────────────

_COMPARE_TEMPLATE = """\
{persona_preamble}

Using ONLY the context below, compare and contrast: {topic_a} vs {topic_b}.
Present a structured comparison with clear criteria.
If information on either topic is missing from the context, say so explicitly.

─── CONTEXT ───────────────────────────────────────────────
{context}
───────────────────────────────────────────────────────────

Comparison:"""


# ── Public API ─────────────────────────────────────────────────────────────────

def get_rag_prompt(persona: str = "analyst") -> PromptTemplate:
    """
    Return the standard RAG prompt injected with the given persona.

    Args:
        persona: One of "analyst", "executive", "engineer", "hr".

    Returns:
        PromptTemplate with variables: context, question.
        (persona_preamble is pre-filled)
    """
    preamble = PERSONA_PREAMBLES.get(persona, PERSONA_PREAMBLES["analyst"])
    template = _RAG_TEMPLATE.replace("{persona_preamble}", preamble)
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )


def get_summarise_prompt(persona: str = "analyst") -> PromptTemplate:
    """Return the summarisation prompt for the given persona."""
    preamble = PERSONA_PREAMBLES.get(persona, PERSONA_PREAMBLES["analyst"])
    template = _SUMMARISE_TEMPLATE.replace("{persona_preamble}", preamble)
    return PromptTemplate(
        input_variables=["context"],
        template=template,
    )


def get_compare_prompt(persona: str = "analyst") -> PromptTemplate:
    """Return the compare prompt for the given persona."""
    preamble = PERSONA_PREAMBLES.get(persona, PERSONA_PREAMBLES["analyst"])
    template = _COMPARE_TEMPLATE.replace("{persona_preamble}", preamble)
    return PromptTemplate(
        input_variables=["context", "topic_a", "topic_b"],
        template=template,
    )


def format_context(documents, max_chars: int = 6000) -> str:
    """
    Concatenate retrieved Document chunks into a single context string
    with source citations and character budget enforcement.

    Args:
        documents:  List of LangChain Documents (post-reranking).
        max_chars:  Hard limit on total context length (keep under LLM window).

    Returns:
        Formatted context string ready to inject into a prompt.
    """
    parts   = []
    total   = 0

    for i, doc in enumerate(documents, 1):
        meta     = doc.metadata
        source   = meta.get("file_name", "unknown")
        page     = meta.get("page_number", "")
        score    = meta.get("rerank_score", "")
        page_str = f", page {page}" if page else ""
        score_str = f" | relevance={score}" if score else ""

        header  = f"[{i}] {source}{page_str}{score_str}"
        content = doc.page_content.strip()
        chunk   = f"{header}\n{content}\n"

        if total + len(chunk) > max_chars:
            # Trim to fit
            remaining = max_chars - total - len(header) - 5
            if remaining > 100:
                chunk = f"{header}\n{content[:remaining]}...\n"
                parts.append(chunk)
            break

        parts.append(chunk)
        total += len(chunk)

    return "\n".join(parts)


AVAILABLE_PERSONAS = list(PERSONA_PREAMBLES.keys())