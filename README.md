# Enterprise RAG System — Phases 1, 2 & 3

**Multi-Agent RAG System** | Production-grade, fully local, zero API costs

---

## Full Pipeline Architecture

```
Raw Documents (PDF / CSV / TXT / HTML / MD)
        ↓  DocumentLoader
LangChain Documents + rich metadata
        ↓  DocumentChunker  (RecursiveCharacterTextSplitter)
Text chunks with inherited metadata
        ↓  LocalEmbedder  (all-MiniLM-L6-v2, free, offline)
384-dim vectors
        ↓  ChromaVectorStore
Persistent ChromaDB + BM25 sparse index
                                                      ← PHASE 1 COMPLETE
──────────────────────────────────────────────────────────────────────────
        ↓  QueryRewriter  (Ollama LLM → 3 search variants)
Multiple query variants
        ↓  ChromaVectorStore  (hybrid search on all variants)
~10 deduplicated candidate chunks
        ↓  CrossEncoderReranker  (ms-marco-MiniLM, free, offline)
Top 4 chunks scored by true relevance
        ↓  PromptTemplate  (persona injection)
        ↓  Ollama + Mistral 7B  (local, free)
Grounded, cited answer
                                                      ← PHASE 2 COMPLETE
──────────────────────────────────────────────────────────────────────────
User question
        ↓  SupervisorAgent   → intent classification → agent_route
        │
        ├── intent="rag"  ──► RetrieverAgent  (Phase 2 RAGChain)
        │                          ↓
        ├── intent="sql"  ──► SQLAgent  (Text→SQL→SQLite→Answer)
        │                          ↓
        └── intent="both" ──► RetrieverAgent ─┐
                             SQLAgent        ─┴──► SynthesizerAgent
                                                        ↓
                                              final_answer + sources
                                              chat_history updated
                                                      ← PHASE 3 COMPLETE
```

Metadata tracked in SQLite (dim_source + fact_chunk + products).
All runs logged to local MLflow.

---

## Quick Start (Fresh Setup)

### 1. Create virtual environment
```bash
cd rag_enterprise
python -m venv .venv

# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama + pull model
```bash
# Download from https://ollama.com and install, then:
ollama pull mistral
# Ollama starts automatically after install (or run: ollama serve)
```

### 4. Generate sample documents
```bash
python scripts/generate_sample_data.py
```
Creates 4 realistic files in `data/raw/`:
- `annual_report.txt` — company financials and strategy
- `products.csv` — structured product catalogue (also seeded into SQLite)
- `tech_overview.md` — cloud architecture documentation
- `employee_handbook.txt` — HR policies and benefits

### 5. Run Phase 1 — index documents
```bash
python -m src.pipeline
```

### 6. Run Phase 2 — RAG questions
```bash
python -m src.rag_pipeline --question "What is our parental leave policy?"
```

### 7. Run Phase 3 — multi-agent system
```bash
python -m src.agent_pipeline --question "How many cloud products do we have?"
```

---

## Phase 1 — Ingestion & Indexing

### What it does
Loads documents → chunks → embeds with a free local model → stores in ChromaDB
with hybrid BM25 + dense search. Tracks all metadata in SQLite.

### CLI
```bash
# Index all files in data/raw/
python -m src.pipeline

# Index a specific directory
python -m src.pipeline --dir path/to/your/docs

# Index a single file
python -m src.pipeline --file path/to/report.pdf

# Test retrieval (no ingestion)
python -m src.pipeline --query "What is our Q4 revenue?"
python -m src.pipeline --query "Key products" --mode mmr
python -m src.pipeline --query "Key products" --mode dense

# Check index status
python -m src.pipeline --status

# Wipe ChromaDB and re-index
python -m src.pipeline --reset
```

### Tests
```bash
pytest tests/test_phase1.py -v
```

---

## Phase 2 — RAG Pipeline

### What it does
Adds query rewriting, cross-encoder re-ranking, persona injection, and
grounded LLM answers on top of the Phase 1 index.

### CLI
```bash
# Ask a question (default persona: analyst)
python -m src.rag_pipeline --question "What is our parental leave policy?"

# Change persona
python -m src.rag_pipeline --question "Summarise Q4 results" --persona executive
python -m src.rag_pipeline --question "How does AcmeMesh work?" --persona engineer
python -m src.rag_pipeline --question "What is the remote work policy?" --persona hr

# Change retrieval mode
python -m src.rag_pipeline --question "Key findings" --mode mmr
python -m src.rag_pipeline --question "Key findings" --mode dense

# Skip query rewriting (faster)
python -m src.rag_pipeline --question "Q4 revenue" --no-rewrite

# Summarise all indexed documents
python -m src.rag_pipeline --summarise --persona executive

# Compare two topics
python -m src.rag_pipeline --compare "cloud" "software"

# Interactive chat loop
python -m src.rag_pipeline --interactive
```

### Interactive commands (Phase 2)
```
/persona executive    switch persona
/mode mmr             switch retrieval mode
/summarise            summarise all documents
/quit                 exit
```

### Python API
```python
from src.rag.rag_chain import RAGChain

chain = RAGChain()
result = chain.ask("What is our Q4 revenue?")
print(result.answer)
print(result.sources)
result.pretty_print()
```

### Tests
```bash
pytest tests/test_phase2.py -v
```

---

## Phase 3 — Multi-Agent Orchestration

### What it does
Adds a LangGraph state machine with four specialised agents:

| Agent | Role |
|---|---|
| **SupervisorAgent** | Classifies intent (rag/sql/both/chitchat) and sets routing |
| **RetrieverAgent** | Runs the full Phase 2 RAGChain against ChromaDB |
| **SQLAgent** | Converts natural language to SQL, queries SQLite, returns plain-English answer |
| **SynthesizerAgent** | Merges answers from all agents into one cited response, updates memory |

Intent classification uses a two-stage approach: fast keyword heuristics first,
LLM fallback only for ambiguous cases.

The SQL Agent automatically seeds a `products` table from `products.csv` on
first run, giving it richer data to query alongside the metadata tables.

Memory persists across turns — follow-up questions work correctly without
re-stating context.

### CLI
```bash
# Ask a single question (auto-routes to correct agent)
python -m src.agent_pipeline --question "How many cloud products do we have?"
python -m src.agent_pipeline --question "What is our remote work policy?"
python -m src.agent_pipeline --question "What does AcmeMesh cost and how does it work?"

# Show full agent trace (intent, route, each agent's output)
python -m src.agent_pipeline --trace "Which products launched in 2024?"

# Change persona
python -m src.agent_pipeline --question "Top products by margin" --persona executive

# Interactive multi-turn chat with memory
python -m src.agent_pipeline --interactive
```

### Interactive commands (Phase 3)
```
/persona executive    switch persona
/trace                toggle agent trace on/off
/memory               show last 6 turns of chat history
/reset                clear conversation memory
/quit                 exit
```

### Python API
```python
from src.agents.graph import AgentSystem

system = AgentSystem()

# Single question — auto-routed
result = system.ask("How many cloud products do we have?")
print(result["final_answer"])
print(result["intent"])        # "sql"
print(result["agent_route"])   # ["sql"]

# Follow-up — memory preserved automatically
result = system.ask("Which one has the highest margin?")
print(result["final_answer"])  # knows "one" refers to cloud products

# Show agent trace
print(result["plan"])          # supervisor's reasoning
print(result["sql_query"])     # SQL generated
print(result["rag_answer"])    # RAG answer (if retriever ran)

# Reset memory between sessions
system.reset_memory()

# Stream execution (see each agent complete in real time)
for node_name, partial_state in system.stream("What is AcmeMesh?"):
    print(f"✓ {node_name} completed")
```

### Tests
```bash
pytest tests/test_phase3.py -v
```

### Run all tests
```bash
pytest tests/ -v
```

---

## View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

Every pipeline run logs: chunk counts, embedding throughput, reranker latency,
RAG latency, and retrieval metrics.

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
│   ├── processed/                     # Reserved for Phase 4
│   ├── chroma_db/                     # Persistent vector index (auto-created)
│   └── metadata.db                    # SQLite: dim_source, fact_chunk, products
│
├── mlruns/                            # MLflow experiment logs (auto-created)
│
├── scripts/
│   └── generate_sample_data.py        # Creates 4 realistic test documents
│
├── src/
│   ├── ingestion/                     # ── PHASE 1 ──
│   │   ├── document_loader.py         # PDF/CSV/TXT/HTML/MD loaders
│   │   └── chunker.py                 # RecursiveCharacterTextSplitter + metadata
│   │
│   ├── embedding/                     # ── PHASE 1 ──
│   │   └── embedder.py                # all-MiniLM-L6-v2, free & offline
│   │
│   ├── vectorstore/                   # ── PHASE 1 ──
│   │   └── chroma_store.py            # ChromaDB + BM25 hybrid + MMR
│   │
│   ├── rag/                           # ── PHASE 2 ──
│   │   ├── llm_factory.py             # Builds Ollama LLM (stub fallback)
│   │   ├── query_rewriter.py          # LLM-based query expansion (3 variants)
│   │   ├── reranker.py                # cross-encoder/ms-marco re-ranking
│   │   ├── prompt_templates.py        # Persona prompts + context formatter
│   │   └── rag_chain.py               # Full pipeline: question → RAGResponse
│   │
│   ├── agents/                        # ── PHASE 3 ──
│   │   ├── state.py                   # AgentState TypedDict (shared graph state)
│   │   ├── supervisor_agent.py        # Intent classification + routing
│   │   ├── retriever_agent.py         # Wraps RAGChain as a LangGraph node
│   │   ├── sql_agent.py               # Text→SQL→SQLite→natural-language answer
│   │   ├── synthesizer_agent.py       # Merges answers + updates memory
│   │   └── graph.py                   # LangGraph state machine + AgentSystem
│   │
│   ├── utils/
│   │   ├── config.py                  # Typed settings for all phases
│   │   ├── logger.py                  # Logging + MLflow helpers
│   │   └── metadata_store.py          # SQLite dim_source + fact_chunk schema
│   │
│   ├── pipeline.py                    # Phase 1 CLI entry point
│   ├── rag_pipeline.py                # Phase 2 CLI entry point
│   └── agent_pipeline.py              # Phase 3 CLI entry point
│
└── tests/
    ├── test_phase1.py                 # 15 tests: ingestion, chunking, vectorstore
    ├── test_phase2.py                 # 18 tests: reranker, rewriter, prompts
    └── test_phase3.py                 # 22 tests: agents, routing, SQL safety
```

---

## Tuning Parameters

Edit `config/settings.yaml`:

### Phase 1

| Parameter | Default | Effect |
|---|---|---|
| `chunking.chunk_size` | 512 | Larger = more context per chunk |
| `chunking.chunk_overlap` | 64 | Larger = less info loss at boundaries |
| `vectorstore.dense_weight` | 0.7 | Higher = trust cosine similarity more |
| `vectorstore.sparse_weight` | 0.3 | Higher = trust BM25 keyword match more |
| `retrieval.top_k` | 10 | Candidates before re-ranking |
| `retrieval.final_k` | 4 | Final chunks after MMR |
| `retrieval.mmr_lambda` | 0.5 | 0 = max diversity, 1 = max relevance |
| `embedding.device` | cpu | Change to `cuda` if GPU available |

### Phase 2

| Parameter | Default | Effect |
|---|---|---|
| `llm.model` | mistral | Any Ollama model (e.g. llama3, phi3) |
| `llm.temperature` | 0.1 | Lower = more deterministic answers |
| `reranker.top_n` | 4 | Final chunks passed to LLM |
| `query_rewriting.num_variants` | 3 | More = higher recall, slower |
| `query_rewriting.enabled` | true | Set false to skip rewriting |
| `rag.default_persona` | analyst | Default persona for all queries |
| `rag.max_context_chars` | 6000 | Hard limit on context fed to LLM |

### Phase 3

| Parameter | Default | Effect |
|---|---|---|
| `agents.sql_keywords` | (list) | Keywords that trigger SQL routing |
| `agents.sql_max_rows` | 20 | Max rows returned from SQLite |
| `agents.memory_window` | 6 | Turns of history injected into context |
| `agents.synthesis_mode` | weighted | `weighted`/`concat`/`llm_merge` |

---

## Free Local Stack — What Replaces What

| Production tool | Local replacement | Notes |
|---|---|---|
| AWS S3 | `data/` directory | Same path-based logic |
| Snowflake | SQLite + DuckDB | SQL Agent queries this directly |
| Pinecone / Weaviate | ChromaDB | Same LangChain vectorstore API |
| OpenAI embeddings | `all-MiniLM-L6-v2` | Free, 384-dim, ~90MB |
| GPT-4 | Ollama + Mistral 7B | Free, local, same LangChain interface |
| Cohere Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, ~80MB |
| AutoGen / CrewAI | LangGraph | Same graph-based agent orchestration |
| AWS ECS / Docker | FastAPI + uvicorn | Added in Phase 4 |
| Apache Airflow | APScheduler | Added in Phase 4 |

---

## What's Next (Phase 4)

Phase 4 adds serving, evaluation, and monitoring:
- **FastAPI** REST endpoint wrapping the AgentSystem
- **Streamlit** chat UI with streaming responses
- **RAGAS evaluation** — faithfulness, answer relevancy, context precision
- **MLflow** experiment comparison across all three phases
- **APScheduler** pipeline: data refresh → re-embedding → index update