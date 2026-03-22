import sys

try:
    print("1. Starting imports...")
    from src.embedding.embedder import LocalEmbedder
    print("2. Embedder imported")
    embedder = LocalEmbedder()
    print("3. Embedder initialized")
    lc_emb = embedder.as_langchain_embeddings()
    print("4. LangChain embeddings ready")
    from src.vectorstore.chroma_store import ChromaVectorStore
    print("5. ChromaVectorStore imported")
    store = ChromaVectorStore(lc_emb)
    print("6. ChromaVectorStore initialized")
    count = store.collection_count()
    print(f"7. Collection count: {count}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print("SUCCESS!")
