# Enterprise RAG System — Phase 1

**Multi-Agent RAG System** | Phase 1: Data Ingestion & Vector Indexing

## What Phase 1 Builds

```
Raw Documents (PDF/CSV/TXT/HTML/MD)
        ↓  DocumentLoader
LangChain Documents + rich metadata
        ↓  DocumentChunker  (RecursiveCharacterTextSplitter)
Text chunks with inherited metadata
        ↓  LocalEmbedder  (all-MiniLM-L6-v2, free, offline)
384-dim vectors
        ↓  ChromaVectorStore
Persistent ChromaDB + BM25 sparse index
        ↓  Hybrid Search + MMR
Diverse, relevant retrieved chunks
```

Metadata is tracked in a SQLite dimensional model (dim_source + fact_chunk),
mirroring a Snowflake schema. All components are logged to local MLflow.

---

## Quick Start

### 1. Clone / create the project
```bash
cd rag_enterprise
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate sample documents (no real docs needed)
```bash
python scripts/generate_sample_data.py
```
This creates 4 sample files in `data/raw/`:
- `annual_report.txt` — company narrative
- `products.csv` — structured product data
- `tech_overview.md` — technical Markdown
- `employee_handbook.txt` — HR policy text

### 5. Run the full Phase 1 pipeline
```bash
python -m src.pipeline
```

This will:
1. Load all documents from `data/raw/`
2. Chunk them with overlap
3. Download the embedding model (~90MB, one-time)
4. Index into ChromaDB (persisted to `data/chroma_db/`)
5. Track metadata in `data/metadata.db`
6. Log metrics to MLflow (`mlruns/`)
7. Run a demo hybrid search query

---

## CLI Reference

```bash
# Index a specific directory
python -m src.pipeline --dir path/to/your/docs

# Index a single file
python -m src.pipeline --file path/to/report.pdf

# Run a query (must index first)
python -m src.pipeline --query "What is our Q4 revenue?"

# Change search mode
python -m src.pipeline --query "Key products" --mode mmr
python -m src.pipeline --query "Key products" --mode dense

# Check index status
python -m src.pipeline --status

# Reset ChromaDB (wipe and re-index)
python -m src.pipeline --reset
```

---

## Run Tests

```bash
pytest tests/test_phase1.py -v
```

---

## View MLflow Experiments

```bash
mlflow ui
# Open http://localhost:5000
```

---

## Project Structure

```
rag_enterprise/
├── config/
│   └── settings.yaml          # All tunable parameters
├── data/
│   ├── raw/                   # Drop your documents here
│   ├── processed/             # Reserved for Phase 2+
│   ├── chroma_db/             # Persistent vector index (auto-created)
│   └── metadata.db            # SQLite dimensional model (auto-created)
├── mlruns/                    # MLflow experiment logs (auto-created)
├── scripts/
│   └── generate_sample_data.py
├── src/
│   ├── ingestion/
│   │   ├── document_loader.py # PDF/CSV/TXT/HTML/MD loaders
│   │   └── chunker.py         # RecursiveCharacterTextSplitter + metadata
│   ├── embedding/
│   │   └── embedder.py        # all-MiniLM-L6-v2, free & offline
│   ├── vectorstore/
│   │   └── chroma_store.py    # ChromaDB + BM25 hybrid + MMR
│   ├── utils/
│   │   ├── config.py          # Typed settings loaded from YAML
│   │   ├── logger.py          # Logging + MLflow helpers
│   │   └── metadata_store.py  # SQLite dim_source + fact_chunk schema
│   └── pipeline.py            # Phase 1 orchestrator (CLI entry point)
├── tests/
│   └── test_phase1.py
└── requirements.txt
```

---

## Tuning Parameters

Edit `config/settings.yaml`:

| Parameter | Default | Effect |
|---|---|---|
| `chunking.chunk_size` | 512 | Larger = more context per chunk |
| `chunking.chunk_overlap` | 64 | Larger = less information loss at boundaries |
| `vectorstore.dense_weight` | 0.7 | Higher = trust dense search more |
| `vectorstore.sparse_weight` | 0.3 | Higher = trust BM25 keyword match more |
| `retrieval.top_k` | 10 | Candidates before MMR |
| `retrieval.final_k` | 4 | Final chunks returned |
| `retrieval.mmr_lambda` | 0.5 | 0 = max diversity, 1 = max relevance |
| `embedding.device` | cpu | Change to `cuda` if GPU available |

---

## What's Next (Phase 2)

Phase 2 adds:
- LangChain RAG pipeline with query rewriting
- Cross-encoder re-ranking (free HuggingFace model)
- Ollama + Mistral 7B as the local LLM generator
- User persona injection and prompt templates
- Cited, grounded answers