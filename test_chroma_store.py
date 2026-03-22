import sys
from pathlib import Path

try:
    print("1. Testing mock embedder and chromadb...")
    
    # Patch sys.modules as we do in chroma_store
    import types
    if 'onnxruntime' not in sys.modules:
        sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')
        sys.modules['onnxruntime.capi'] = types.ModuleType('onnxruntime.capi')
        sys.modules['onnxruntime.capi._pybind_state'] = types.ModuleType('onnxruntime.capi._pybind_state')
    
    print("2. Creating mock embedding function...")
    # Simple mock for embedding function
    class MockEmbeddings:
        def embed_query(self, text):
            return [0.0] * 384
        def embed_documents(self, docs):
            return [[0.0] * 384 for _ in docs]
    
    mock_emb = MockEmbeddings()
    
    print("3. Importing chromadb...")
    import chromadb
    from langchain_chroma import Chroma
    
    print("4. Creating persistent client...")
    client = chromadb.PersistentClient(path="./data/chroma_db")
    
    print("5. Creating Chroma store...")
    store = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=mock_emb,
    )
    
    print("6. Chroma store created successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("SUCCESS!")
