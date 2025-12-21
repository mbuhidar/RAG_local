"""
Installation verification script.
Tests that all core dependencies are properly installed.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    
    tests = [
        ("chromadb", "ChromaDB vector database"),
        ("sentence_transformers", "Sentence Transformers"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("ollama", "Ollama client"),
        ("gradio", "Gradio interface"),
        ("pypdf", "PDF processing"),
        ("docx", "DOCX processing"),
        ("yaml", "YAML configuration"),
    ]
    
    failed = []
    for package, description in tests:
        try:
            __import__(package)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"✗ {description}: {e}")
            failed.append(package)
    
    print("\n" + "="*50)
    if not failed:
        print("✓ All packages installed successfully!")
        return True
    else:
        print(f"✗ Failed to import: {', '.join(failed)}")
        return False


def test_ollama_connection():
    """Test Ollama server connection."""
    print("\nTesting Ollama connection...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama server is running")
            models = response.json().get("models", [])
            if models:
                print(f"  Available models: {', '.join([m['name'] for m in models])}")
            else:
                print("  ⚠ No models downloaded. Run: ollama pull llama2")
            return True
        else:
            print("✗ Ollama server returned error")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return False


if __name__ == "__main__":
    print("="*50)
    print("RAG System Installation Verification")
    print("="*50 + "\n")
    
    imports_ok = test_imports()
    ollama_ok = test_ollama_connection()
    
    print("\n" + "="*50)
    if imports_ok:
        print("\n✓ Installation successful!")
        if not ollama_ok:
            print("\n⚠ Note: Start Ollama to use the LLM features")
            print("  Run: ollama serve")
            print("  Then: ollama pull llama2")
    else:
        print("\n✗ Installation incomplete. Please install missing packages.")
        print("  Run: pip install -r requirements.txt")
