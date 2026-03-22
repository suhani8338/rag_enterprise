"""
tests/test_phase2.py
─────────────────────
Unit tests for Phase 2 components.
These tests mock the LLM and ChromaDB so they run without Ollama or a live index.

Run:  pytest tests/test_phase2.py -v
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag.prompt_templates import (
    AVAILABLE_PERSONAS,
    format_context,
    get_compare_prompt,
    get_rag_prompt,
    get_summarise_prompt,
)
from src.rag.query_rewriter import QueryRewriter, _parse_numbered_list
from src.rag.reranker import CrossEncoderReranker


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs() -> List[Document]:
    return [
        Document(
            page_content="Acme Corporation Q4 revenue was $1.2B, up 18% YoY.",
            metadata={"file_name": "annual_report.txt", "page_number": 3},
        ),
        Document(
            page_content="Remote work policy: employees must be in office 2 days per week.",
            metadata={"file_name": "employee_handbook.txt", "page_number": 12},
        ),
        Document(
            page_content="AcmeCloud compute supports Kubernetes with auto-scaling.",
            metadata={"file_name": "tech_overview.md"},
        ),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.invoke.return_value = "1. revenue growth Q4\n2. quarterly financial results\n3. annual income report"
    # Make LLM chainable (supports | operator)
    chain_mock = MagicMock()
    chain_mock.invoke.return_value = "1. revenue growth Q4\n2. quarterly financial results"
    llm.__or__ = MagicMock(return_value=chain_mock)
    return llm


# ── Prompt template tests ──────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_all_personas_available(self):
        assert "analyst"   in AVAILABLE_PERSONAS
        assert "executive" in AVAILABLE_PERSONAS
        assert "engineer"  in AVAILABLE_PERSONAS
        assert "hr"        in AVAILABLE_PERSONAS

    def test_rag_prompt_has_required_variables(self):
        prompt = get_rag_prompt(persona="analyst")
        assert "context"  in prompt.input_variables
        assert "question" in prompt.input_variables

    def test_rag_prompt_contains_persona(self):
        analyst_prompt   = get_rag_prompt(persona="analyst")
        executive_prompt = get_rag_prompt(persona="executive")
        # Prompts should differ (persona injected)
        assert analyst_prompt.template != executive_prompt.template

    def test_summarise_prompt_variables(self):
        prompt = get_summarise_prompt()
        assert "context" in prompt.input_variables

    def test_compare_prompt_variables(self):
        prompt = get_compare_prompt()
        assert "context"  in prompt.input_variables
        assert "topic_a"  in prompt.input_variables
        assert "topic_b"  in prompt.input_variables

    def test_format_context_basic(self, sample_docs):
        ctx = format_context(sample_docs)
        assert "annual_report.txt"    in ctx
        assert "employee_handbook.txt" in ctx
        assert "Q4 revenue"           in ctx

    def test_format_context_respects_max_chars(self, sample_docs):
        ctx = format_context(sample_docs, max_chars=100)
        assert len(ctx) <= 200  # allow some header overhead

    def test_format_context_empty(self):
        ctx = format_context([])
        assert ctx == ""

    def test_format_context_includes_page_numbers(self, sample_docs):
        ctx = format_context(sample_docs)
        assert "page 3"  in ctx
        assert "page 12" in ctx


# ── QueryRewriter tests ────────────────────────────────────────────────────────

class TestQueryRewriter:
    def test_parse_numbered_list(self):
        raw = "1. first variant\n2. second variant\n3. third variant"
        results = _parse_numbered_list(raw)
        assert len(results) == 3
        assert results[0] == "first variant"
        assert results[2] == "third variant"

    def test_parse_numbered_list_with_parens(self):
        raw = "1) first\n2) second"
        results = _parse_numbered_list(raw)
        assert len(results) == 2

    def test_parse_empty_returns_empty(self):
        assert _parse_numbered_list("") == []

    def test_rewrite_always_includes_original(self, mock_llm):
        rewriter = QueryRewriter(mock_llm)
        question = "What is our Q4 revenue?"
        variants = rewriter.rewrite(question)
        assert variants[0] == question

    def test_rewrite_deduplicates(self, mock_llm):
        rewriter = QueryRewriter(mock_llm)
        variants = rewriter.rewrite("What is Q4 revenue?")
        # No duplicates
        assert len(variants) == len(set(v.lower() for v in variants))

    def test_rewrite_disabled_returns_original_only(self, mock_llm):
        rewriter = QueryRewriter(mock_llm)
        rewriter.enabled = False
        variants = rewriter.rewrite("test question")
        assert variants == ["test question"]

    def test_rewrite_fallback_on_llm_error(self):
        bad_llm = MagicMock()
        bad_llm.__or__ = MagicMock(side_effect=Exception("LLM down"))
        rewriter = QueryRewriter(bad_llm)
        variants = rewriter.rewrite("test question")
        assert variants == ["test question"]


# ── CrossEncoderReranker tests ─────────────────────────────────────────────────

class TestCrossEncoderReranker:
    def test_rerank_returns_correct_count(self, sample_docs):
        reranker = CrossEncoderReranker(top_n=2)
        results  = reranker.rerank("Q4 revenue", sample_docs)
        assert len(results) == 2

    def test_rerank_adds_score_to_metadata(self, sample_docs):
        reranker = CrossEncoderReranker(top_n=3)
        results  = reranker.rerank("What is the remote work policy?", sample_docs)
        for doc in results:
            assert "rerank_score" in doc.metadata
            assert isinstance(doc.metadata["rerank_score"], float)

    def test_rerank_sorts_by_relevance(self, sample_docs):
        reranker = CrossEncoderReranker(top_n=3)
        results  = reranker.rerank("remote work office policy", sample_docs)
        scores   = [doc.metadata["rerank_score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_empty_input(self):
        reranker = CrossEncoderReranker()
        results  = reranker.rerank("any query", [])
        assert results == []

    def test_rerank_with_scores_returns_tuples(self, sample_docs):
        reranker = CrossEncoderReranker(top_n=2)
        results  = reranker.rerank_with_scores("Q4", sample_docs)
        assert len(results) == 2
        for doc, score in results:
            assert isinstance(score, float)
            assert isinstance(doc, Document)

    def test_top_n_capped_at_input_size(self, sample_docs):
        reranker = CrossEncoderReranker(top_n=100)  # more than available
        results  = reranker.rerank("query", sample_docs)
        assert len(results) == len(sample_docs)