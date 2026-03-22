"""
tests/test_phase4.py
─────────────────────
Unit tests for Phase 4 components.
FastAPI tested with TestClient — no running server needed.
RAGAS evaluator tested with mocked RAGChain — no Ollama needed.
Scheduler tested with temp filesystem — no timing dependencies.

Run: pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.evaluation.ragas_eval import RAGASEvaluator, _get_contexts, DEFAULT_TEST_QUESTIONS
from src.scheduler.refresh_scheduler import RefreshScheduler


# ── FastAPI tests ──────────────────────────────────────────────────────────────

class TestAPI:
    """Test FastAPI endpoints using TestClient (no server needed)."""

    @pytest.fixture
    def mock_agent_system(self):
        system = MagicMock()
        system.ask.return_value = {
            "final_answer":  "Employees get 16 weeks parental leave.",
            "final_sources": ["employee_handbook.txt, page 5"],
            "intent":        "rag",
            "agent_route":   ["retriever"],
            "rag_chunks":    4,
            "sql_query":     None,
            "chat_history":  [],
        }
        system.history = []
        return system

    @pytest.fixture
    def client(self, mock_agent_system):
        # Patch AgentSystem before importing app
        with patch("src.serving.api._get_system", return_value=mock_agent_system):
            with patch("src.serving.api._sessions", {"default": mock_agent_system}):
                from src.serving.api import app
                return TestClient(app, raise_server_exceptions=False)

    def test_health_endpoint_returns_200(self, client):
        with patch("src.serving.api.LocalEmbedder"), \
             patch("src.serving.api.ChromaVectorStore") as mock_store:
            mock_store.return_value.collection_count.return_value = 42
            r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_ask_endpoint_returns_answer(self, client, mock_agent_system):
        with patch("src.serving.api._get_system", return_value=mock_agent_system):
            r = client.post("/ask", json={
                "question": "What is the parental leave policy?",
                "persona":  "analyst",
            })
        assert r.status_code == 200
        data = r.json()
        assert "final_answer" in data
        assert "intent"       in data
        assert "latency_ms"   in data

    def test_ask_validates_persona(self, client):
        r = client.post("/ask", json={
            "question": "test",
            "persona":  "invalid_persona",
        })
        assert r.status_code == 422   # Pydantic validation error

    def test_ask_validates_empty_question(self, client):
        r = client.post("/ask", json={"question": "", "persona": "analyst"})
        assert r.status_code == 422

    def test_history_endpoint(self, client, mock_agent_system):
        with patch("src.serving.api._get_system", return_value=mock_agent_system), \
             patch("src.serving.api._sessions", {"default": mock_agent_system}):
            r = client.get("/history")
        assert r.status_code == 200
        assert "history" in r.json()

    def test_history_reset_endpoint(self, client, mock_agent_system):
        with patch("src.serving.api._get_system", return_value=mock_agent_system), \
             patch("src.serving.api._sessions", {"default": mock_agent_system}):
            r = client.post("/history/reset")
        assert r.status_code == 200
        assert r.json()["status"] == "reset"

    def test_session_id_header_creates_new_session(self, client, mock_agent_system):
        with patch("src.serving.api._get_system", return_value=mock_agent_system):
            r = client.post(
                "/ask",
                json    = {"question": "test?", "persona": "analyst"},
                headers = {"X-Session-Id": "test-session-123"},
            )
        assert r.status_code == 200


# ── RAGAS evaluator tests ──────────────────────────────────────────────────────

class TestRAGASEvaluator:

    @pytest.fixture
    def mock_rag_chain(self):
        from src.rag.rag_chain import RAGResponse
        chain = MagicMock()
        chain.ask.return_value = RAGResponse(
            question         = "test question",
            answer           = "test answer",
            sources          = ["doc.txt, page 1"],
            persona          = "analyst",
            retrieved_chunks = 4,
            reranked_chunks  = 4,
            latency_ms       = 250.0,
        )
        return chain

    @pytest.fixture
    def evaluator(self, mock_rag_chain):
        with patch("src.evaluation.ragas_eval.settings") as ms:
            ms.evaluation    = MagicMock(
                test_set_path   = "data/eval_test_set.json",
                metrics         = ["faithfulness", "answer_relevancy"],
                experiment_name = "test_eval",
            )
            ms.project_root  = Path("/tmp")
            ms.paths         = MagicMock(mlflow_uri="/tmp/mlruns")
            ms.llm           = MagicMock(model="mistral", base_url="http://localhost:11434")
            ms.embedding     = MagicMock(model_name="sentence-transformers/all-MiniLM-L6-v2")
            ev = RAGASEvaluator(rag_chain=mock_rag_chain)
            ev._test_set_path = Path("/tmp/nonexistent_test_set.json")
        return ev, mock_rag_chain

    def test_load_test_set_returns_defaults_when_no_file(self, evaluator):
        ev, _ = evaluator
        questions = ev.load_test_set()
        assert len(questions) == len(DEFAULT_TEST_QUESTIONS)
        assert "question"     in questions[0]
        assert "ground_truth" in questions[0]

    def test_save_and_load_test_set(self, tmp_path):
        from src.evaluation.ragas_eval import RAGASEvaluator
        ev = RAGASEvaluator.__new__(RAGASEvaluator)
        ev._test_set_path = tmp_path / "test_set.json"
        ev._metrics = ["faithfulness"]
        ev._experiment_name = "test"

        custom = [{"question": "Q1?", "ground_truth": "A1."}]
        ev.save_test_set(custom)

        loaded = ev.load_test_set()
        assert loaded[0]["question"] == "Q1?"

    def test_get_contexts_from_response(self, mock_rag_chain):
        response = mock_rag_chain.ask.return_value
        contexts = _get_contexts(response)
        assert isinstance(contexts, list)
        assert len(contexts) > 0

    def test_default_test_questions_have_required_fields(self):
        for item in DEFAULT_TEST_QUESTIONS:
            assert "question"     in item, f"Missing 'question' in: {item}"
            assert "ground_truth" in item, f"Missing 'ground_truth' in: {item}"
            assert len(item["question"]) > 5


# ── Scheduler tests ────────────────────────────────────────────────────────────

class TestRefreshScheduler:

    @pytest.fixture
    def tmp_raw_dir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        # Create a recently modified file
        f = raw / "recent.txt"
        f.write_text("recent content " * 50)
        # Create an old file (modify mtime to 48h ago)
        old = raw / "old.txt"
        old.write_text("old content " * 50)
        old_time = time.time() - (48 * 3600)
        import os
        os.utime(old, (old_time, old_time))
        return raw, tmp_path

    def test_find_changed_files_returns_recent_only(self, tmp_raw_dir):
        raw_dir, tmp_path = tmp_raw_dir
        with patch("src.scheduler.refresh_scheduler.settings") as ms:
            ms.paths     = MagicMock(raw_data=raw_dir, sqlite_db=tmp_path / "meta.db")
            ms.ingestion = MagicMock(supported_extensions=[".txt"])
            ms.scheduler = MagicMock(
                refresh_interval_minutes=60,
                changed_files_hours=24,
                timezone="UTC",
            )
            sched = RefreshScheduler(changed_files_hours=24)
            sched._loader  = MagicMock()
            sched._chunker = MagicMock()

            with patch.object(sched, "_find_changed_files",
                              wraps=sched._find_changed_files):
                changed = sched._find_changed_files()

        # Only the recently modified file should appear
        names = [f.name for f in changed]
        assert "recent.txt" in names
        assert "old.txt"    not in names

    def test_refresh_once_no_changes_returns_zero(self):
        sched = RefreshScheduler.__new__(RefreshScheduler)
        sched.interval_minutes    = 60
        sched.changed_files_hours = 24
        sched.tz                  = "UTC"
        sched._embedder  = None
        sched._store     = MagicMock()
        sched._loader    = MagicMock()
        sched._chunker   = MagicMock()
        sched._meta_db   = MagicMock()

        with patch.object(sched, "_ensure_components"), \
             patch.object(sched, "_find_changed_files", return_value=[]), \
             patch.object(sched, "_log_to_mlflow"):
            result = sched.refresh_once()

        assert result["changed_files"] == 0
        assert result["new_chunks"]    == 0

    def test_scheduler_config_from_settings(self):
        with patch("src.scheduler.refresh_scheduler.settings") as ms:
            ms.scheduler = MagicMock(
                refresh_interval_minutes=30,
                changed_files_hours=12,
                timezone="UTC",
            )
            sched = RefreshScheduler()
        assert sched.interval_minutes    == 30
        assert sched.changed_files_hours == 12

    def test_scheduler_override_params(self):
        sched = RefreshScheduler(interval_minutes=15, changed_files_hours=6)
        assert sched.interval_minutes    == 15
        assert sched.changed_files_hours == 6