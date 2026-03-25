# Enterprise Multi-Agent RAG System

**Production-grade, fully local, zero API costs**

A complete multi-agent Retrieval-Augmented Generation system that ingests enterprise documents, indexes them in a vector database, and uses coordinated AI agents to answer complex business queries with grounded, cited responses. Everything runs locally — no cloud accounts, no API keys, no usage fees.

---

## What it does

You drop documents into `data/raw/` and the system handles everything else. It loads PDFs, CSVs, text files, HTML, and Markdown; splits them into overlapping chunks; embeds them with a free local model; and stores them in a persistent vector index with both dense and sparse search. When you ask a question, a supervisor agent classifies your intent and routes to the right specialist — a retriever agent for document questions, a SQL agent for structured data queries, or both in parallel for questions that need both. A synthesizer agent merges the answers into one cited response. The whole thing is served via a FastAPI REST endpoint and a Streamlit chat UI with real-time streaming.

---

## Architecture

```
Documents (PDF / CSV / TXT / HTML / MD)
        ↓  DocumentLoader  (PyMuPDF · pandas · BeautifulSoup)
        ↓  DocumentChunker  (RecursiveCharacterTextSplitter · size=512 overlap=64)
        ↓  LocalEmbedder  (all-MiniLM-L6-v2 · 384-dim · free · offline)
        ↓  ChromaVectorStore  (ChromaDB + BM25 hybrid + MMR)
        ↓  SQLite  (dim_source + fact_chunk metadata · products table)
                                        ↑ indexed once, queried on every turn
────────────────────────────────────────────────────────────────────────────────
User question
        ↓  SupervisorAgent
              keyword heuristic → Qwen 1.7B fallback
              → intent: rag | sql | both | chitchat
              → agent_route

        ├── rag   ──► RetrieverAgent
        │               QueryRewriter (Qwen 1.7B → 3 variants)
        │               hybrid retrieval → CrossEncoderReranker (ms-marco-MiniLM)
        │               PromptTemplate (persona) → Qwen 1.7B → grounded answer
        │
        ├── sql   ──► SQLAgent
        │               Qwen 1.7B text→SQL → SQLite → natural language answer
        │
        └── both  ──► RetrieverAgent + SQLAgent  (parallel)
                            ↓
                      SynthesizerAgent
                        merge · cite · update chat_history
                            ↓
                      final_answer + sources
────────────────────────────────────────────────────────────────────────────────
FastAPI  (/ask · /ask/stream · /health · /status · /history)
Streamlit UI  (chat · agent trace panel · SSE streaming · persona selector)
RAGAS Evaluation  (faithfulness · answer relevancy · context precision)
APScheduler  (scan data/raw/ → re-embed changed files → update ChromaDB)
```

---

## Quick Start

```powershell
# 1. Virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Dependencies
pip install -r requirements.txt

# 3. Install Ollama from https://ollama.com, then pull the model
ollama pull qwen2:1.7b

# 4. Generate sample documents and index them
python scripts/generate_sample_data.py
python -m src.pipeline

# 5. Ask a question via the agent system
python -m src.agent_pipeline --question "How many cloud products do we have?"
python -m src.agent_pipeline --question "What is our parental leave policy?"

# 6. Start the API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000

# 7. Start the Streamlit UI (new terminal)
streamlit run src/serving/streamlit_app.py
```

---

## CLI Reference

### Document ingestion

```powershell
python -m src.pipeline                                   # index data/raw/
python -m src.pipeline --file path\to\report.pdf        # single file
python -m src.pipeline --query "Q4 revenue" --mode mmr  # test retrieval
python -m src.pipeline --status                         # index stats
python -m src.pipeline --reset                          # wipe and re-index
```

### RAG pipeline (direct, no agents)

```powershell
python -m src.rag_pipeline --question "What is our parental leave policy?"
python -m src.rag_pipeline --question "Summarise Q4" --persona executive
python -m src.rag_pipeline --question "How does AcmeMesh work?" --persona engineer
python -m src.rag_pipeline --compare "cloud" "software"
python -m src.rag_pipeline --summarise
python -m src.rag_pipeline --interactive
```

Interactive commands: `/persona <n>` · `/mode <dense|hybrid|mmr>` · `/summarise` · `/quit`

### Multi-agent system

```powershell
python -m src.agent_pipeline --question "How many cloud products do we have?"
python -m src.agent_pipeline --question "What is our remote work policy?"
python -m src.agent_pipeline --question "What does AcmeMesh cost and how does it work?"
python -m src.agent_pipeline --trace "Which products launched in 2024?"
python -m src.agent_pipeline --interactive
```

Interactive commands: `/persona <n>` · `/trace` · `/memory` · `/reset` · `/quit`

### FastAPI server

```powershell
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Single question → full JSON response |
| POST | `/ask/stream` | SSE stream, one event per agent node |
| GET | `/health` | Liveness check + chunk count |
| GET | `/status` | Index stats from SQLite |
| GET | `/history` | Session chat history |
| POST | `/history/reset` | Clear session memory |

```powershell
# Test from PowerShell
Invoke-RestMethod -Uri http://localhost:8000/health

Invoke-RestMethod -Uri http://localhost:8000/ask `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"question": "What is our parental leave policy?", "persona": "analyst"}'
```

### Streamlit UI

```powershell
streamlit run src/serving/streamlit_app.py
# Opens at http://localhost:8501
```

Features: streaming chat · agent trace panel (intent, route, SQL query per turn) ·
persona selector · index stats · example questions · session memory with clear button.

### RAGAS evaluation

```powershell
python -m src.evaluation.ragas_eval --generate   # write default test set
python -m src.evaluation.ragas_eval              # run full evaluation
python -m src.evaluation.ragas_eval --questions 3  # faster — 3 questions only
```

Results saved to `data/eval_results.json`. Scores faithfulness, answer relevancy,
and context precision using Qwen 1.7B as the judge.

### Data refresh scheduler

```powershell
python -m src.scheduler.refresh_scheduler --once      # single cycle
python -m src.scheduler.refresh_scheduler             # continuous loop
python -m src.scheduler.refresh_scheduler --interval 30 --hours 6
```

Scans `data/raw/` for recently modified files, re-ingests them, removes stale
ChromaDB chunks, adds new ones, and rebuilds the BM25 index.

### Run all tests

```powershell
pytest tests\ -v
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
│   └── settings.yaml                  # All tunable parameters
│
├── data/
│   ├── raw/                           # Drop your documents here
│   ├── processed/
│   ├── chroma_db/                     # Persistent vector index (auto-created)
│   ├── metadata.db                    # SQLite: dim_source, fact_chunk, products
│   ├── eval_test_set.json             # RAGAS test questions (auto-generated)
│   └── eval_results.json              # RAGAS results (auto-generated)
│
├── scripts/
│   └── generate_sample_data.py        # Creates 4 realistic sample documents
│
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py         # PDF / CSV / TXT / HTML / MD loaders
│   │   └── chunker.py                 # RecursiveCharacterTextSplitter + metadata
│   ├── embedding/
│   │   └── embedder.py                # all-MiniLM-L6-v2 · free · offline
│   ├── vectorstore/
│   │   └── chroma_store.py            # ChromaDB + BM25 hybrid search + MMR
│   │
│   ├── rag/
│   │   ├── llm_factory.py             # Builds Qwen 1.7B via Ollama (stub fallback)
│   │   ├── query_rewriter.py          # LLM-based query expansion (3 variants)
│   │   ├── reranker.py                # cross-encoder/ms-marco re-ranking
│   │   ├── prompt_templates.py        # Persona prompts + context formatter
│   │   └── rag_chain.py               # Full RAG pipeline → RAGResponse
│   │
│   ├── agents/
│   │   ├── state.py                   # AgentState TypedDict (shared graph state)
│   │   ├── supervisor_agent.py        # Intent classification + routing
│   │   ├── retriever_agent.py         # Wraps RAGChain as a LangGraph node
│   │   ├── sql_agent.py               # Text→SQL→SQLite→natural-language answer
│   │   ├── synthesizer_agent.py       # Merges answers + updates memory
│   │   └── graph.py                   # LangGraph state machine + AgentSystem
│   │
│   ├── serving/
│   │   ├── api.py                     # FastAPI app
│   │   └── streamlit_app.py           # Streamlit chat UI
│   ├── evaluation/
│   │   └── ragas_eval.py              # RAGAS scoring
│   ├── scheduler/
│   │   └── refresh_scheduler.py       # APScheduler refresh pipeline
│   │
│   ├── utils/
│   │   ├── config.py                  # Typed settings loaded from YAML
│   │   ├── logger.py                  # Logging helpers
│   │   └── metadata_store.py          # SQLite schema
│   │
│   ├── pipeline.py                    # Ingestion CLI
│   ├── rag_pipeline.py                # RAG CLI
│   └── agent_pipeline.py              # Agent CLI
│
└── tests/
    ├── test_phase1.py                 # 15 tests
    ├── test_phase2.py                 # 18 tests
    ├── test_phase3.py                 # 22 tests
    └── test_phase4.py                 # 20 tests
```

---

## Tuning Parameters

Edit `config/settings.yaml` to change any of these without touching code:

### Ingestion & indexing
| Parameter | Default | Effect |
|---|---|---|
| `chunking.chunk_size` | 512 | Larger = more context per chunk |
| `chunking.chunk_overlap` | 64 | Larger = less info loss at boundaries |
| `vectorstore.dense_weight` | 0.7 | Cosine similarity weight in hybrid search |
| `vectorstore.sparse_weight` | 0.3 | BM25 keyword weight in hybrid search |
| `retrieval.top_k` | 10 | Candidates before re-ranking |
| `retrieval.final_k` | 4 | Chunks passed to LLM after MMR |
| `embedding.device` | cpu | Change to `cuda` if GPU available |

### LLM & RAG
| Parameter | Default | Effect |
|---|---|---|
| `llm.model` | qwen2:1.7b | Any model pulled via Ollama |
| `llm.temperature` | 0.1 | Lower = more deterministic answers |
| `reranker.top_n` | 4 | Final chunks after cross-encoder scoring |
| `query_rewriting.num_variants` | 3 | More = higher recall, slower |
| `rag.default_persona` | analyst | analyst / executive / engineer / hr |
| `rag.max_context_chars` | 6000 | Hard limit on context fed to LLM |

### Agents
| Parameter | Default | Effect |
|---|---|---|
| `agents.sql_max_rows` | 20 | Max rows returned from SQLite |
| `agents.memory_window` | 6 | Chat turns kept in context |
| `agents.synthesis_mode` | weighted | weighted / concat / llm_merge |

### Serving & evaluation
| Parameter | Default | Effect |
|---|---|---|
| `serving.port` | 8000 | FastAPI port |
| `evaluation.metrics` | faithfulness, answer_relevancy, context_precision | RAGAS metrics |
| `scheduler.refresh_interval_minutes` | 60 | How often to scan for changes |
| `scheduler.changed_files_hours` | 24 | Re-index files modified within this window |

---

## Free Local Stack

| Production equivalent | What's used here |
|---|---|
| AWS S3 | `data/` directory |
| Snowflake | SQLite + DuckDB |
| Pinecone / Weaviate | ChromaDB |
| OpenAI embeddings | `all-MiniLM-L6-v2` |
| GPT-4 | Qwen 1.7B via Ollama |
| Cohere Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| AutoGen / CrewAI | LangGraph |
| AWS ECS / Docker | FastAPI + uvicorn |
| Apache Airflow | APScheduler |
| AWS SageMaker | Local uvicorn process |