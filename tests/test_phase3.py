"""
tests/test_phase3.py
─────────────────────
Unit + integration tests for Phase 3 multi-agent components.
LLM and ChromaDB are mocked — no Ollama needed to run these.

Run: pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.state import AgentState
from src.agents.supervisor_agent import SupervisorAgent, _intent_to_route
from src.agents.sql_agent import SQLAgent, _clean_sql, _guard_select_only
from src.agents.synthesizer_agent import SynthesizerAgent, _concat_merge


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """LLM that returns a controllable string."""
    llm = MagicMock()
    chain = MagicMock()
    chain.invoke = MagicMock(return_value="rag")
    llm.__or__ = MagicMock(return_value=chain)
    llm.invoke  = MagicMock(return_value="rag")
    return llm, chain


@pytest.fixture
def tmp_db(tmp_path):
    """Temporary SQLite database with a products table."""
    db = tmp_path / "test.db"
    conn = sqlite3.connect(str(db))
    conn.execute("""
        CREATE TABLE products (
            product_id TEXT, product_name TEXT, category TEXT,
            price_usd TEXT, margin_pct TEXT, launched TEXT,
            region TEXT, status TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?,?,?)",
        [
            ("P001","AcmeCloud Compute","Cloud","499","72","2021-03","Global","Active"),
            ("P002","AcmeDB Managed",   "Cloud","299","68","2022-06","Global","Active"),
            ("P003","Acme ERP v12",     "Software","1200","55","2018-09","NA/EU","Mature"),
        ]
    )
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def sql_agent(mock_llm, tmp_db):
    llm, chain = mock_llm
    # Make chain return valid SQL for text2sql call
    chain.invoke = MagicMock(return_value="SELECT * FROM products LIMIT 5")
    with patch("src.agents.sql_agent.settings") as mock_settings:
        mock_settings.paths.sqlite_db = tmp_db
        mock_settings.paths.raw_data  = Path("/nonexistent")
        mock_settings.agents          = MagicMock(sql_max_rows=20)
        agent = SQLAgent(llm, db_path=tmp_db)
    return agent, chain


@pytest.fixture
def base_state() -> AgentState:
    return {
        "question":     "How many cloud products do we have?",
        "persona":      "analyst",
        "chat_history": [],
        "agent_route":  ["sql"],
    }


# ── SupervisorAgent tests ──────────────────────────────────────────────────────

class TestSupervisorAgent:
    def test_chitchat_detection(self, mock_llm):
        llm, _ = mock_llm
        with patch("src.agents.supervisor_agent.settings") as ms:
            ms.agents = MagicMock(sql_keywords=["how many","count","total"])
            sup = SupervisorAgent(llm)
        result = sup({"question": "Hello!", "agent_route": [], "chat_history": []})
        assert result["intent"] == "chitchat"
        assert result["agent_route"] == []

    def test_sql_keyword_routing(self, mock_llm):
        llm, _ = mock_llm
        with patch("src.agents.supervisor_agent.settings") as ms:
            ms.agents = MagicMock(sql_keywords=["how many","count","total","price"])
            sup = SupervisorAgent(llm)
        result = sup({"question": "How many products have a price under 500?",
                      "agent_route": [], "chat_history": []})
        assert result["intent"] in ("sql", "both")

    def test_rag_routing_for_policy_question(self, mock_llm):
        llm, chain = mock_llm
        chain.invoke = MagicMock(return_value="rag")
        with patch("src.agents.supervisor_agent.settings") as ms:
            ms.agents = MagicMock(sql_keywords=["how many","count"])
            sup = SupervisorAgent(llm)
        result = sup({"question": "What is our remote work policy?",
                      "agent_route": [], "chat_history": []})
        assert result["intent"] == "rag"
        assert "retriever" in result["agent_route"]

    def test_intent_to_route_mapping(self):
        assert _intent_to_route("rag")      == ["retriever"]
        assert _intent_to_route("sql")      == ["sql"]
        assert _intent_to_route("both")     == ["retriever", "sql"]
        assert _intent_to_route("chitchat") == []
        assert _intent_to_route("unknown")  == ["retriever"]   # fallback

    def test_plan_is_set(self, mock_llm):
        llm, _ = mock_llm
        with patch("src.agents.supervisor_agent.settings") as ms:
            ms.agents = MagicMock(sql_keywords=[])
            sup = SupervisorAgent(llm)
        result = sup({"question": "Hello!", "agent_route": [], "chat_history": []})
        assert "plan" in result
        assert len(result["plan"]) > 0


# ── SQLAgent tests ─────────────────────────────────────────────────────────────

class TestSQLAgent:
    def test_query_products(self, sql_agent):
        agent, _ = sql_agent
        df = agent.query("SELECT * FROM products")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_select_only_guard_allows_select(self):
        _guard_select_only("SELECT * FROM products WHERE price < 500")  # no exception

    def test_select_only_guard_blocks_insert(self):
        with pytest.raises(ValueError, match="INSERT"):
            _guard_select_only("INSERT INTO products VALUES ('X','Y','Z')")

    def test_select_only_guard_blocks_drop(self):
        with pytest.raises(ValueError, match="DROP"):
            _guard_select_only("DROP TABLE products")

    def test_select_only_guard_blocks_delete(self):
        with pytest.raises(ValueError, match="DELETE"):
            _guard_select_only("DELETE FROM products")

    def test_clean_sql_strips_markdown(self):
        raw = "```sql\nSELECT * FROM products\n```"
        assert _clean_sql(raw) == "SELECT * FROM products"

    def test_clean_sql_strips_plain_backticks(self):
        raw = "```\nSELECT 1\n```"
        assert _clean_sql(raw) == "SELECT 1"

    def test_clean_sql_passthrough(self):
        raw = "SELECT id FROM dim_source LIMIT 5"
        assert _clean_sql(raw) == raw

    def test_skipped_when_not_in_route(self, sql_agent, base_state):
        agent, _ = sql_agent
        state = {**base_state, "agent_route": ["retriever"]}
        result = agent(state)
        assert result == {}

    def test_sql_agent_node_returns_sql_fields(self, sql_agent, base_state):
        agent, chain = sql_agent
        # chain returns SQL for t2s, answer for s2a
        chain.invoke = MagicMock(side_effect=[
            "SELECT COUNT(*) as n FROM products WHERE category='Cloud'",
            "There are 2 cloud products.",
        ])
        result = agent(base_state)
        assert "sql_query" in result
        assert "sql_answer" in result


# ── SynthesizerAgent tests ─────────────────────────────────────────────────────

class TestSynthesizerAgent:
    def test_rag_only_passthrough(self, mock_llm):
        llm, _ = mock_llm
        synth = SynthesizerAgent(llm)
        state: AgentState = {
            "question":     "What is the leave policy?",
            "intent":       "rag",
            "persona":      "analyst",
            "agent_route":  ["retriever"],
            "rag_answer":   "Employees get 16 weeks parental leave.",
            "rag_sources":  ["employee_handbook.txt, page 5"],
            "sql_answer":   "",
            "sql_query":    "",
            "chat_history": [],
        }
        result = synth(state)
        assert result["final_answer"] == "Employees get 16 weeks parental leave."
        assert "employee_handbook.txt" in result["final_sources"][0]

    def test_sql_only_passthrough(self, mock_llm):
        llm, _ = mock_llm
        synth = SynthesizerAgent(llm)
        state: AgentState = {
            "question":     "How many products?",
            "intent":       "sql",
            "persona":      "analyst",
            "agent_route":  ["sql"],
            "rag_answer":   "",
            "rag_sources":  [],
            "sql_answer":   "There are 10 products.",
            "sql_query":    "SELECT COUNT(*) FROM products",
            "chat_history": [],
        }
        result = synth(state)
        assert result["final_answer"] == "There are 10 products."

    def test_chitchat_returns_answer(self, mock_llm):
        llm, chain = mock_llm
        chain.invoke = MagicMock(return_value="Hello! I can help with documents and data.")
        synth = SynthesizerAgent(llm)
        state: AgentState = {
            "question": "Hi!", "intent": "chitchat",
            "persona": "analyst", "agent_route": [],
            "chat_history": [],
        }
        result = synth(state)
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_chat_history_updated(self, mock_llm):
        llm, _ = mock_llm
        synth = SynthesizerAgent(llm)
        state: AgentState = {
            "question": "What is the policy?", "intent": "rag",
            "persona": "analyst", "agent_route": ["retriever"],
            "rag_answer": "Policy is X.", "rag_sources": [],
            "sql_answer": "", "sql_query": "", "chat_history": [],
        }
        result = synth(state)
        history = result["chat_history"]
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_concat_merge_both_present(self):
        answer, sources = _concat_merge(
            "Doc answer.", ["doc.pdf"],
            "SQL answer.", "SELECT * FROM t"
        )
        assert "Doc answer." in answer
        assert "SQL answer." in answer
        assert any("SQL:" in s for s in sources)

    def test_concat_merge_one_missing(self):
        answer, sources = _concat_merge("Doc answer.", ["doc.pdf"], "", "")
        assert "Doc answer." in answer

    def test_empty_inputs_handled(self, mock_llm):
        llm, _ = mock_llm
        synth  = SynthesizerAgent(llm)
        state: AgentState = {
            "question": "Test?", "intent": "rag", "persona": "analyst",
            "agent_route": ["retriever"],
            "rag_answer": "", "rag_sources": [],
            "sql_answer": "", "sql_query": "", "chat_history": [],
        }
        result = synth(state)
        assert "final_answer" in result   # should not crash


# ── State schema tests ─────────────────────────────────────────────────────────

class TestAgentState:
    def test_state_accepts_all_fields(self):
        state: AgentState = {
            "question": "Q", "persona": "analyst", "chat_history": [],
            "intent": "rag", "agent_route": ["retriever"], "plan": "test",
            "rag_answer": "A", "rag_sources": [], "rag_chunks": 3,
            "sql_query": "SELECT 1", "sql_result": None, "sql_answer": "B",
            "final_answer": "Final", "final_sources": [], "error": None,
        }
        assert state["intent"] == "rag"
        assert state["rag_chunks"] == 3