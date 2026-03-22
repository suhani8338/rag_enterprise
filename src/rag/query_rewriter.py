"""
src/rag/query_rewriter.py
──────────────────────────
Rewrites the user's raw question into multiple search-optimised variants
before hitting ChromaDB. This dramatically improves recall because a single
phrasing often misses relevant chunks that a synonym or restructured sentence
would find.

Strategy:
  1. Send the original query to the LLM with a prompt asking for N variants.
  2. Parse the numbered list response.
  3. Return original + variants — the retriever runs all of them and deduplicates.

Why this matters on a resume:
  Query rewriting is a standard production RAG technique. It directly addresses
  the vocabulary mismatch problem between how users ask questions and how
  documents are written.

Usage:
    rewriter = QueryRewriter(llm)
    variants = rewriter.rewrite("What is our remote work policy?")
    # → ["What is our remote work policy?",
    #    "remote working guidelines employees",
    #    "work from home rules and allowances",
    #    "hybrid office attendance policy"]
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

_REWRITE_TEMPLATE = """\
You are an expert at reformulating search queries to maximise document retrieval coverage.

Given the original question below, generate {num_variants} alternative search queries
that capture the same intent using different vocabulary, structure, or focus.

Rules:
- Each variant must be on its own numbered line: 1. ... 2. ... 3. ...
- Variants should use synonyms, rephrasings, and related terms
- Keep each variant under 20 words
- Do NOT add explanation — output the numbered list only

Original question: {question}

Alternative queries:"""

_REWRITE_PROMPT = PromptTemplate(
    input_variables=["question", "num_variants"],
    template=_REWRITE_TEMPLATE,
)


class QueryRewriter:
    """
    Uses the local LLM to generate N search-optimised query variants.
    Falls back gracefully (returns original query only) if LLM is unavailable.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm          = llm
        self.num_variants = settings.query_rewriting.num_variants if settings.query_rewriting else 3
        self.enabled      = settings.query_rewriting.enabled      if settings.query_rewriting else True
        self._chain       = _REWRITE_PROMPT | llm | StrOutputParser()
        logger.info(f"QueryRewriter ready | variants={self.num_variants} enabled={self.enabled}")

    def rewrite(self, question: str) -> List[str]:
        """
        Return original query + N rewritten variants.
        Always includes the original as the first element.

        Args:
            question: Raw user query string.

        Returns:
            List of query strings (original first, then variants).
        """
        if not self.enabled:
            return [question]

        try:
            raw_output = self._chain.invoke({
                "question":     question,
                "num_variants": self.num_variants,
            })
            variants = _parse_numbered_list(raw_output)

            if not variants:
                logger.warning("Query rewriting returned no variants — using original")
                return [question]

            # Deduplicate while preserving order; original always first
            seen    = {question.lower().strip()}
            results = [question]
            for v in variants:
                v_clean = v.strip()
                if v_clean.lower() not in seen and v_clean:
                    seen.add(v_clean.lower())
                    results.append(v_clean)

            logger.info(
                f"Query rewriting: '{question[:60]}' → {len(results)} variants"
            )
            return results

        except Exception as e:
            logger.warning(f"Query rewriting failed ({e}) — using original query")
            return [question]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_numbered_list(text: str) -> List[str]:
    """Extract lines from a numbered list like '1. foo\n2. bar\n3. baz'."""
    lines = []
    for line in text.strip().splitlines():
        # Match "1. text", "1) text", "- text"
        match = re.match(r"^[\d]+[.)]\s*(.+)$", line.strip())
        if match:
            lines.append(match.group(1).strip())
        elif line.strip().startswith("- "):
            lines.append(line.strip()[2:].strip())
    return lines