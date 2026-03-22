# Enterprise RAG System — Phases 1 & 2

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
────────────────────────────────────────────────────────────────────
        ↓  QueryRewriter  (Ollama LLM → 3 search variants)
Multiple query variants
        ↓  ChromaVectorStore  (hybrid search on all variants)
~10 deduplicated candidate chunks
        ↓  CrossEncoderReranker  (ms-marco-MiniLM, free, offline)
Top 4 chunks scored by true relevance
        ↓  format_context()  (citation-labelled, char-budgeted)
Structured context string
        ↓  PromptTemplate  (persona injection)
Persona-aware prompt
        ↓  Ollama + Mistral 7B  (local, free)
Grounded, cited answer + RAGResponse dataclass
                                                 ← PHASE 2 COMPLETE
```

Metadata tracked in SQLite (dim_source + fact_chunk).
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

### 3. Install Ollama + pull model (Phase 2 LLM)
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
- `products.csv` — structured product catalogue
- `tech_overview.md` — cloud architecture documentation
- `employee_handbook.txt` — HR policies and benefits

### 5. Run Phase 1 — index your documents
```bash
python -m src.pipeline
```

### 6. Run Phase 2 — ask questions
```bash
python -m src.rag_pipeline --question "What is our parental leave policy?"
```

---

## Phase 1 — Ingestion & Indexing

### What it does
Loads documents → chunks them → embeds with a free local model → stores in ChromaDB with hybrid BM25 + dense search.

### CLI
```bash
# Index all files in data/raw/
python -m src.pipeline

# Index a specific directory
python -m src.pipeline --dir path/to/your/docs

# Index a single file
python -m src.pipeline --file path/to/report.pdf

# Test retrieval only (no ingestion)
python -m src.pipeline --query "What is our Q4 revenue?"
python -m src.pipeline --query "Key products" --mode mmr
python -m src.pipeline --query "Key products" --mode dense

# Check index status
python -m src.pipeline --status

# Wipe ChromaDB and re-index from scratch
python -m src.pipeline --reset
```

### Run Phase 1 tests
```bash
pytest tests/test_phase1.py -v
```

---

## Phase 2 — RAG Pipeline

### What it does
Takes your indexed ChromaDB from Phase 1 and adds:
- **Query rewriting** — rewrites your question into 3 variants to improve recall
- **Multi-query retrieval** — runs all variants against ChromaDB and deduplicates
- **Cross-encoder re-ranking** — re-scores candidates with a more accurate model
- **Persona injection** — tailors tone and format for analyst / executive / engineer / hr
- **Grounded answers** — LLM answers only from retrieved context, with citations

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

# Skip query rewriting (faster, lower recall)
python -m src.rag_pipeline --question "Q4 revenue" --no-rewrite

# Summarise all indexed documents
python -m src.rag_pipeline --summarise
python -m src.rag_pipeline --summarise --persona executive

# Compare two topics
python -m src.rag_pipeline --compare "cloud" "software"
python -m src.rag_pipeline --compare "parental leave" "remote work"

# Interactive chat loop
python -m src.rag_pipeline --interactive
```

### Interactive mode commands
Once inside `--interactive`:
```
/persona executive     # switch persona mid-session
/persona engineer
/mode mmr              # switch retrieval mode
/mode dense
/summarise             # summarise all documents
/quit                  # exit
```

### Use the RAG chain in Python
```python
from src.rag.rag_chain import RAGChain

chain = RAGChain()

# Ask a question
result = chain.ask("What is our Q4 revenue?")
print(result.answer)
print(result.sources)
print(f"Latency: {result.latency_ms:.0f}ms")

# Pretty-print to terminal
result.pretty_print()

# Change persona
result = chain.ask("Key risks?", persona="executive")

# Summarise everything
result = chain.summarise(persona="analyst")

# Compare two topics
result = chain.compare("AcmeCloud", "Acme ERP")
```

### Run Phase 2 tests (no Ollama needed)
```bash
pytest tests/test_phase2.py -v
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

Every pipeline run logs: chunk counts, embedding throughput, reranker latency, RAG latency, and retrieval metrics.

---

## Project Structure

```
rag_enterprise/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── config/
│   └── settings.yaml              # All tunable parameters (both phases)
│
├── data/
│   ├── raw/                       # Drop your documents here
│   ├── processed/                 # Reserved for Phase 3+
│   ├── chroma_db/                 # Persistent vector index (auto-created)
│   └── metadata.db                # SQLite dimensional model (auto-created)
│
├── mlruns/                        # MLflow experiment logs (auto-created)
│
├── scripts/
│   └── generate_sample_data.py    # Creates 4 realistic test documents
│
├── src/
│   ├── ingestion/                 # ── PHASE 1 ──
│   │   ├── document_loader.py     # PDF/CSV/TXT/HTML/MD loaders
│   │   └── chunker.py             # RecursiveCharacterTextSplitter + metadata
│   │
│   ├── embedding/                 # ── PHASE 1 ──
│   │   └── embedder.py            # all-MiniLM-L6-v2, free & offline
│   │
│   ├── vectorstore/               # ── PHASE 1 ──
│   │   └── chroma_store.py        # ChromaDB + BM25 hybrid + MMR
│   │
│   ├── rag/                       # ── PHASE 2 ──
│   │   ├── llm_factory.py         # Builds Ollama LLM (with stub fallback)
│   │   ├── query_rewriter.py      # LLM-based query expansion (3 variants)
│   │   ├── reranker.py            # cross-encoder/ms-marco re-ranking
│   │   ├── prompt_templates.py    # Persona prompts + context formatter
│   │   └── rag_chain.py           # Full pipeline: question → RAGResponse
│   │
│   ├── utils/
│   │   ├── config.py              # Typed settings loaded from YAML
│   │   ├── logger.py              # Logging + MLflow helpers
│   │   └── metadata_store.py      # SQLite dim_source + fact_chunk schema
│   │
│   ├── pipeline.py                # Phase 1 CLI entry point
│   └── rag_pipeline.py            # Phase 2 CLI entry point
│
└── tests/
    ├── test_phase1.py             # 15 tests: ingestion, chunking, vectorstore
    └── test_phase2.py             # 18 tests: reranker, rewriter, prompts
```

---

## Tuning Parameters

Edit `config/settings.yaml`:

### Phase 1 parameters

| Parameter | Default | Effect |
|---|---|---|
| `chunking.chunk_size` | 512 | Larger = more context per chunk |
| `chunking.chunk_overlap` | 64 | Larger = less information loss at boundaries |
| `vectorstore.dense_weight` | 0.7 | Higher = trust cosine similarity more |
| `vectorstore.sparse_weight` | 0.3 | Higher = trust BM25 keyword match more |
| `retrieval.top_k` | 10 | Candidate chunks before re-ranking |
| `retrieval.final_k` | 4 | Final chunks after MMR |
| `retrieval.mmr_lambda` | 0.5 | 0 = max diversity, 1 = max relevance |
| `embedding.device` | cpu | Change to `cuda` if GPU available |

### Phase 2 parameters

| Parameter | Default | Effect |
|---|---|---|
| `llm.model` | mistral | Any model pulled via Ollama (e.g. llama3, phi3) |
| `llm.temperature` | 0.1 | Lower = more deterministic answers |
| `reranker.top_n` | 4 | Final chunks passed to the LLM |
| `query_rewriting.num_variants` | 3 | More variants = higher recall, slower |
| `query_rewriting.enabled` | true | Set false to skip rewriting (faster) |
| `rag.default_persona` | analyst | Default persona for all queries |
| `rag.max_context_chars` | 6000 | Hard limit on context fed to LLM |

---

## Free Local Stack — What Replaces What

| Production tool | Local replacement | Notes |
|---|---|---|
| AWS S3 | `data/` directory | Same path-based logic |
| Snowflake | SQLite + DuckDB | Same SQL Agent interface in Phase 3 |
| Pinecone / Weaviate | ChromaDB | Same LangChain vectorstore API |
| OpenAI embeddings | `all-MiniLM-L6-v2` | Free, 384-dim, ~90MB |
| GPT-4 | Ollama + Mistral 7B | Free, local, same LangChain interface |
| Cohere Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, ~80MB |
| AWS ECS / Docker | FastAPI + uvicorn | Added in Phase 4 |
| Apache Airflow | APScheduler | Added in Phase 4 |

---

## What's Next (Phase 3)

Phase 3 adds multi-agent orchestration with LangGraph:
- **Retriever Agent** — queries ChromaDB
- **SQL Agent** — queries SQLite/DuckDB with natural language
- **Synthesizer Agent** — merges and cites answers from both
- **Supervisor Agent** — classifies intent and routes to the right agent