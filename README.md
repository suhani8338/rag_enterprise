# Enterprise RAG System — All Phases

**Multi-Agent RAG System** | Production-grade, fully local, zero API costs

---

## Full Pipeline Architecture

```
Raw Documents (PDF / CSV / TXT / HTML / MD)
        ↓  DocumentLoader
LangChain Documents + rich metadata
        ↓  DocumentChunker  (RecursiveCharacterTextSplitter)
        ↓  LocalEmbedder  (all-MiniLM-L6-v2, free, offline)
        ↓  ChromaVectorStore  (ChromaDB + BM25 hybrid + MMR)
                                                      ← PHASE 1 COMPLETE
──────────────────────────────────────────────────────────────────────────
        ↓  QueryRewriter  (3 search variants)
        ↓  Hybrid retrieval + CrossEncoderReranker
        ↓  PromptTemplate (persona) + Ollama + Mistral 7B
Grounded, cited answer (RAGResponse)
                                                      ← PHASE 2 COMPLETE
──────────────────────────────────────────────────────────────────────────
User question
        ↓  SupervisorAgent  → intent + agent_route
        ├── rag   → RetrieverAgent  (Phase 2 RAGChain)    ─┐
        ├── sql   → SQLAgent  (Text→SQL→SQLite)            ├─► SynthesizerAgent
        └── both  → RetrieverAgent + SQLAgent (parallel)  ─┘
                                                      ← PHASE 3 COMPLETE
──────────────────────────────────────────────────────────────────────────
        ↓  FastAPI  (REST: /ask, /ask/stream, /health, /status)
        ↓  Streamlit UI  (chat + agent trace panel + streaming SSE)
        ↓  RAGAS Evaluation  (faithfulness, relevancy, context precision)
        ↓  APScheduler  (auto data refresh → re-embed → index update)
        ↓  MLflow  (all experiments, metrics, artifacts across all phases)
                                                      ← PHASE 4 COMPLETE
```

---

## Quick Start

```bash
# 1. Virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Ollama (install from https://ollama.com, then:)
ollama pull mistral

# 4. Sample data + index
python scripts/generate_sample_data.py
python -m src.pipeline

# 5. Test each phase
python -m src.rag_pipeline --question "What is our parental leave policy?"
python -m src.agent_pipeline --question "How many cloud products do we have?"

# 6. Start the API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# 7. Start the Streamlit UI (in a second terminal)
streamlit run src/serving/streamlit_app.py
```

---

## Phase 1 — Ingestion & Indexing

```bash
python -m src.pipeline                                    # index data/raw/
python -m src.pipeline --file path/to/report.pdf         # single file
python -m src.pipeline --query "Q4 revenue" --mode mmr   # test retrieval
python -m src.pipeline --status                          # index stats
python -m src.pipeline --reset                           # wipe & re-index
pytest tests/test_phase1.py -v
```

---

## Phase 2 — RAG Pipeline

```bash
python -m src.rag_pipeline --question "What is our parental leave policy?"
python -m src.rag_pipeline --question "Summarise Q4" --persona executive
python -m src.rag_pipeline --question "How does AcmeMesh work?" --persona engineer
python -m src.rag_pipeline --compare "cloud" "software"
python -m src.rag_pipeline --summarise
python -m src.rag_pipeline --interactive
pytest tests/test_phase2.py -v
```

Interactive commands: `/persona <n>` · `/mode <dense|hybrid|mmr>` · `/summarise` · `/quit`

---

## Phase 3 — Multi-Agent System

```bash
python -m src.agent_pipeline --question "How many cloud products do we have?"
python -m src.agent_pipeline --question "What is our remote work policy?"
python -m src.agent_pipeline --question "What does AcmeMesh cost and how does it work?"
python -m src.agent_pipeline --trace "Which products launched in 2024?"
python -m src.agent_pipeline --interactive
pytest tests/test_phase3.py -v
```

Interactive commands: `/persona <n>` · `/trace` · `/memory` · `/reset` · `/quit`

---

## Phase 4 — Serving, Evaluation & Scheduling

### FastAPI server

```bash
# Start
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# Interactive API docs
open http://localhost:8000/docs

# Test endpoints directly
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What is our parental leave policy?", "persona": "analyst"}'

# Multi-turn with session
curl -X POST http://localhost:8000/ask \
     -H "X-Session-Id: my-session" \
     -H "Content-Type: application/json" \
     -d '{"question": "How many cloud products?", "persona": "analyst"}'

curl -X POST http://localhost:8000/history/reset \
     -H "X-Session-Id: my-session"
```

**API endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Single question → full JSON response |
| POST | `/ask/stream` | SSE stream, one event per agent node |
| GET | `/health` | Liveness check + chunk count |
| GET | `/status` | Index stats from SQLite |
| GET | `/history` | Session chat history |
| POST | `/history/reset` | Clear session memory |

### Streamlit UI

```bash
# Requires the FastAPI server to be running
streamlit run src/serving/streamlit_app.py
# Opens at http://localhost:8501
```

Features: streaming chat · agent trace panel · persona selector ·
index stats · example questions · session memory with clear button.

### RAGAS Evaluation

```bash
# Generate the default test set file (8 questions)
python -m src.evaluation.ragas_eval --generate
# Edit data/eval_test_set.json to add your own questions

# Run full evaluation (scores all questions, logs to MLflow)
python -m src.evaluation.ragas_eval

# Limit to 3 questions (faster, for testing)
python -m src.evaluation.ragas_eval --questions 3

# Change persona
python -m src.evaluation.ragas_eval --persona executive
```

Results saved to `data/eval_results.json` and logged to MLflow
experiment `ragas_evaluation`.

### APScheduler (data refresh)

```bash
# Run the scheduler loop (checks every 60 min by default)
python -m src.scheduler.refresh_scheduler

# Run one refresh cycle and exit
python -m src.scheduler.refresh_scheduler --once

# Override interval and window
python -m src.scheduler.refresh_scheduler --interval 30 --hours 6
```

The scheduler scans `data/raw/` for files modified within the last 24 hours,
re-ingests them, deletes their old chunks from ChromaDB, adds the new ones,
rebuilds the BM25 index, and logs the run to MLflow.

### MLflow (all phases)

```bash
mlflow ui
# Open http://localhost:5000
```

Experiments logged:
- `phase1_ingestion` — chunk counts, embedding throughput
- `rag_query`        — retrieval counts, reranker latency, RAG latency
- `ragas_evaluation` — faithfulness, answer_relevancy, context_precision
- `scheduled_refresh` — changed files, new chunks per run

### Run all tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
rag_enterprise/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── config/
│   └── settings.yaml                  # All tunable parameters (all phases)
│
├── data/
│   ├── raw/                           # Drop your documents here
│   ├── processed/
│   ├── chroma_db/                     # Persistent vector index (auto-created)
│   ├── metadata.db                    # SQLite: dim_source, fact_chunk, products
│   ├── eval_test_set.json             # RAGAS test questions (auto-generated)
│   └── eval_results.json              # RAGAS results (auto-generated)
│
├── mlruns/                            # MLflow experiment logs (auto-created)
│
├── scripts/
│   └── generate_sample_data.py
│
├── src/
│   ├── ingestion/                     # ── PHASE 1 ──
│   │   ├── document_loader.py
│   │   └── chunker.py
│   ├── embedding/
│   │   └── embedder.py
│   ├── vectorstore/
│   │   └── chroma_store.py
│   │
│   ├── rag/                           # ── PHASE 2 ──
│   │   ├── llm_factory.py
│   │   ├── query_rewriter.py
│   │   ├── reranker.py
│   │   ├── prompt_templates.py
│   │   └── rag_chain.py
│   │
│   ├── agents/                        # ── PHASE 3 ──
│   │   ├── state.py
│   │   ├── supervisor_agent.py
│   │   ├── retriever_agent.py
│   │   ├── sql_agent.py
│   │   ├── synthesizer_agent.py
│   │   └── graph.py
│   │
│   ├── serving/                       # ── PHASE 4 ──
│   │   ├── api.py                     # FastAPI app
│   │   └── streamlit_app.py           # Streamlit chat UI
│   ├── evaluation/
│   │   └── ragas_eval.py              # RAGAS scoring + MLflow logging
│   ├── scheduler/
│   │   └── refresh_scheduler.py       # APScheduler refresh pipeline
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── metadata_store.py
│   │
│   ├── pipeline.py                    # Phase 1 CLI
│   ├── rag_pipeline.py                # Phase 2 CLI
│   └── agent_pipeline.py              # Phase 3 CLI
│
└── tests/
    ├── test_phase1.py                 # 15 tests
    ├── test_phase2.py                 # 18 tests
    ├── test_phase3.py                 # 22 tests
    └── test_phase4.py                 # 20 tests
```

---

## Tuning Parameters (settings.yaml)

### Phase 1
| Parameter | Default | Effect |
|---|---|---|
| `chunking.chunk_size` | 512 | Larger = more context per chunk |
| `chunking.chunk_overlap` | 64 | Larger = less info loss at boundaries |
| `vectorstore.dense_weight` | 0.7 | Cosine similarity weight |
| `vectorstore.sparse_weight` | 0.3 | BM25 keyword weight |
| `retrieval.top_k` | 10 | Candidates before re-ranking |
| `retrieval.final_k` | 4 | Chunks after MMR |
| `embedding.device` | cpu | Change to `cuda` if GPU available |

### Phase 2
| Parameter | Default | Effect |
|---|---|---|
| `llm.model` | mistral | Any Ollama model (llama3, phi3, etc.) |
| `llm.temperature` | 0.1 | Lower = more deterministic |
| `reranker.top_n` | 4 | Final chunks to LLM |
| `query_rewriting.num_variants` | 3 | More = higher recall, slower |
| `rag.default_persona` | analyst | Default tone |
| `rag.max_context_chars` | 6000 | Context window budget |

### Phase 3
| Parameter | Default | Effect |
|---|---|---|
| `agents.sql_max_rows` | 20 | Max SQL result rows |
| `agents.memory_window` | 6 | Chat turns kept in context |
| `agents.synthesis_mode` | weighted | weighted / concat / llm_merge |

### Phase 4
| Parameter | Default | Effect |
|---|---|---|
| `serving.port` | 8000 | FastAPI port |
| `serving.cors_origins` | localhost:8501 | Allowed origins |
| `evaluation.metrics` | faithfulness, answer_relevancy, context_precision | RAGAS metrics |
| `scheduler.refresh_interval_minutes` | 60 | How often to scan for changes |
| `scheduler.changed_files_hours` | 24 | Files modified within this window are re-indexed |

---

## Free Local Stack

| Production tool | Local replacement |
|---|---|
| AWS S3 | `data/` directory |
| Snowflake | SQLite + DuckDB |
| Pinecone / Weaviate | ChromaDB |
| OpenAI embeddings | `all-MiniLM-L6-v2` |
| GPT-4 | Ollama + Mistral 7B |
| Cohere Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| AutoGen / CrewAI | LangGraph |
| AWS ECS / Docker | FastAPI + uvicorn |
| Apache Airflow | APScheduler |
| AWS SageMaker | Local uvicorn process |