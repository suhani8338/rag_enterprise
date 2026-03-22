import sys

try:
    print("1. Testing chromadb initialization...")
    
    # Patch sys.modules as we do in chroma_store
    import types
    if 'onnxruntime' not in sys.modules:
        sys.modules['onnxruntime'] = types.ModuleType('onnxruntime')
        sys.modules['onnxruntime.capi'] = types.ModuleType('onnxruntime.capi')
        sys.modules['onnxruntime.capi._pybind_state'] = types.ModuleType('onnxruntime.capi._pybind_state')
    
    print("2. Mocked onnxruntime")
    import chromadb
    print("3. chromadb imported")
    
    client = chromadb.PersistentClient(path="./data/chroma_db")
    print("4. Persistent client created")
    
    from langchain_chroma import Chroma
    print("5. langchain_chroma imported")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
print("SUCCESS!")
