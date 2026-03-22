import os
import sys

# Disable chroma telemetry before importing
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'false'

import traceback

print("1. Starting test with telemetry disabled...", flush=True)

try:
    from src.embedding.embedder import LocalEmbedder
    print("2. Embedder imported", flush=True)
    
    embedder = LocalEmbedder()
    print("3. Embedder initialized", flush=True)
    
    lc_emb = embedder.as_langchain_embeddings()
    print("4. LangChain embeddings ready", flush=True)
    
    print("5. About to import ChromaVectorStore...", flush=True)
    from src.vectorstore.chroma_store import ChromaVectorStore
    print("5b. ChromaVectorStore imported", flush=True)
    
    print("6. About to create ChromaVectorStore instance...", flush=True)
    store = ChromaVectorStore(lc_emb)
    print("7. ChromaVectorStore initialized", flush=True)
    
except SystemExit as se:
    print(f"SYSTEM EXIT: {se}", flush=True)
    raise
except Exception as e:
    print(f"EXCEPTION: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)
    
print("SUCCESS!", flush=True)
