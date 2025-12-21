# RAG System Setup Summary

## Installation Issue & Resolution

**Problem**: The pip dependency resolver had a known bug causing installation failures with complex dependency trees.

**Solution**: Implemented a phased installation approach:

1. Upgrade pip/wheel/setuptools
2. Install PyTorch CPU version separately
3. Install packages without dependencies first
4. Install remaining dependencies

## Quick Start

### Option 1: Automated Setup
```bash
./setup.sh
```

### Option 2: Manual Steps
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA/GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install packages
pip install sentence-transformers scikit-learn scipy
pip install --no-deps chromadb transformers
pip install ollama requests pypdf python-docx gradio pyyaml python-dotenv tqdm
pip install bcrypt build grpcio jsonschema overrides posthog pypika tenacity tokenizers
```

### Verify Installation
```bash
python verify_install.py
```

## Project Structure

```
RAG_local/
├── config/
│   └── config.yaml           # System configuration
├── src/
│   ├── vector_store/         # ChromaDB implementation
│   ├── document_processing/   # Loaders & chunking
│   ├── embeddings/           # Sentence transformers
│   ├── llm/                  # Ollama integration
│   ├── rag/                  # Pipeline orchestrator
│   └── chat/                 # Gradio interface
├── data/
│   ├── documents/            # Your documents go here
│   └── vectorstore/          # Vector DB persistence
├── tests/                    # Unit tests
├── examples/                 # Usage examples
├── main.py                   # CLI entry point
├── setup.sh                  # Setup script
├── verify_install.py         # Installation verifier
└── QUICKSTART.md            # Getting started guide
```

## Usage

### 1. Start Ollama
```bash
ollama serve
ollama pull llama2
```

### 2. Ingest Documents
```bash
# Add docs to data/documents/
cp ~/my-docs/*.pdf data/documents/

# Index them
python main.py ingest
```

### 3. Query the System
```bash
# CLI query
python main.py query "What is the main topic?"

# Launch web interface
python main.py chat
# Open http://127.0.0.1:7860
```

### 4. Other Commands
```bash
# View statistics
python main.py stats

# Clear database
python main.py clear --confirm
```

## Architecture Highlights

### Local-First Design
- ✅ No cloud dependencies
- ✅ Complete privacy
- ✅ Works offline
- ✅ No API costs

### Components
1. **Document Processing**: Multi-format support (PDF, DOCX, TXT, MD)
2. **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
3. **Vector Store**: ChromaDB with persistence
4. **LLM**: Ollama (llama2, mistral, phi, etc.)
5. **Interface**: Gradio web UI with conversation history

### Key Features
- Intelligent chunking with overlap
- Metadata preservation
- Similarity search with filtering
- Source attribution
- Streaming responses
- Conversation history
- Configurable via YAML

## Configuration

Edit `config/config.yaml` to customize:

- **Chunk size**: 1000 (default)
- **Chunk overlap**: 200
- **Top-k retrieval**: 5
- **LLM model**: llama2
- **Temperature**: 0.7
- **Embedding model**: all-MiniLM-L6-v2

## Troubleshooting

### Ollama Connection Error
```bash
# Start Ollama
ollama serve

# Check it's running
curl http://localhost:11434/api/tags
```

### Out of Memory
- Reduce `batch_size` in config.yaml
- Use smaller model: `phi` instead of `llama2`
- Reduce `chunk_size`

### Slow Performance
- Use smaller embedding model
- Reduce `top_k` retrieval
- Enable caching in config

### Import Errors
```bash
# Reinstall dependencies
./setup.sh

# Or manually
pip install --force-reinstall chromadb sentence-transformers
```

## Next Steps

1. **Add Documents**: Place your documents in `data/documents/`
2. **Index**: Run `python main.py ingest`
3. **Chat**: Run `python main.py chat`
4. **Customize**: Edit `config/config.yaml` for your needs
5. **Extend**: Add new document loaders or LLM providers

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Run examples
python examples/usage_examples.py
```

## Performance Tips

1. **Batch Processing**: Enabled by default for embeddings
2. **Caching**: Enable in config for faster repeated queries
3. **Model Selection**: 
   - Fast: phi (1.3B params)
   - Balanced: llama2 (7B params)
   - Quality: mistral (7B params)
4. **Hardware**: Use GPU if available (update config device to "cuda")

## Resources

- **Ollama**: https://ollama.ai
- **ChromaDB**: https://www.trychroma.com
- **Sentence Transformers**: https://www.sbert.net
- **Gradio**: https://gradio.app

---

**Status**: ✅ Architecture Complete, Installation Instructions Updated

The RAG system is fully implemented and ready to use. Follow the Quick Start guide to begin!
