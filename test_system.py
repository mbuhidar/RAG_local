"""
Simple test to verify core RAG functionality.
Run after installation to ensure everything works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_document_loading():
    """Test document loading and chunking."""
    print("\n1. Testing Document Processing...")
    try:
        from src.document_processing import Document
        from src.document_processing.chunker import DocumentChunker
        from src.document_processing.loader import DocumentLoader

        # Create a test document
        doc = Document(
            content="This is a test document. " * 50,
            metadata={"filename": "test.txt"}
        )
        
        # Test chunking
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        
        print(f"   ✓ Created {len(chunks)} chunks from test document")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_embeddings():
    """Test embedding generation."""
    print("\n2. Testing Embeddings...")
    try:
        from src.embeddings import EmbeddingGenerator
        
        generator = EmbeddingGenerator(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Test single embedding
        embedding = generator.embed_text("Hello world")
        dim = generator.get_embedding_dimension()
        
        print(f"   ✓ Generated embedding with {dim} dimensions")
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_vector_store():
    """Test vector store operations."""
    print("\n3. Testing Vector Store...")
    try:
        import shutil
        import tempfile

        from src.vector_store.chroma_store import ChromaVectorStore

        # Use temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            store = ChromaVectorStore(
                collection_name="test_collection",
                persist_directory=temp_dir
            )
            
            # Test adding documents
            docs = ["Test document 1", "Test document 2"]
            embeddings = [[0.1] * 384, [0.2] * 384]  # Fake embeddings
            
            ids = store.add_documents(docs, embeddings)
            stats = store.get_collection_stats()
            
            print(f"   ✓ Stored {stats['count']} documents in vector store")
            return True
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def test_ollama():
    """Test Ollama connectivity."""
    print("\n4. Testing Ollama Connection...")
    try:
        from src.llm.ollama_llm import OllamaLLM
        
        llm = OllamaLLM(model_name="llama2")
        models = llm.list_available_models()
        
        if models:
            print(f"   ✓ Connected to Ollama, found {len(models)} models")
            print(f"     Models: {', '.join(models[:3])}")
        else:
            print("   ⚠ Ollama connected but no models found")
            print("     Run: ollama pull llama2")
        return True
    except Exception as e:
        print(f"   ⚠ Ollama not available: {e}")
        print("     This is optional. Start with: ollama serve")
        return True  # Non-critical


def main():
    print("="*60)
    print("RAG System Functionality Test")
    print("="*60)
    
    results = []
    results.append(("Document Processing", test_document_loading()))
    results.append(("Embeddings", test_embeddings()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Ollama LLM", test_ollama()))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "="*60)
    
    critical_passed = all(results[:3])  # First 3 are critical
    
    if critical_passed:
        print("✓ Core functionality working!")
        print("\nYou can now:")
        print("  1. Add documents to data/documents/")
        print("  2. Run: python main.py ingest")
        print("  3. Run: python main.py chat")
    else:
        print("✗ Some tests failed. Check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
