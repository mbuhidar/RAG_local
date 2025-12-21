# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **Ollama** installed and running (for local LLM)
   ```bash
   # Install Ollama: https://ollama.ai/download
   # Start Ollama
   ollama serve
   
   # Pull a model (e.g., llama2)
   ollama pull llama2
   ```

## Installation

### Quick Setup (Linux/Mac)

```bash
# Run the setup script
./setup.sh
```

### Manual Setup

1. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate  # On Windows
   ```

2. **Upgrade pip**
   ```bash
   python -m pip install --upgrade pip wheel
   ```

3. **Install PyTorch (CUDA version for GPU)**
   ```bash
   # For CUDA 12.1 (most common)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Or for CPU only (if no GPU)
   # pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Install dependencies**
   ```bash
   # Core ML packages
   pip install sentence-transformers scikit-learn scipy

   # Application packages
   pip install --no-deps chromadb transformers
   pip install ollama requests pypdf python-docx gradio pyyaml python-dotenv tqdm

   # ChromaDB dependencies
   pip install bcrypt build grpcio jsonschema overrides posthog pypika tenacity tokenizers
   ```

5. **Verify installation**
   ```bash
   python verify_install.py
   ```

## Usage

### 1. Ingest Documents

Add your documents to `data/documents/` then run:

```bash
python main.py ingest --path data/documents
```

### 2. Query via CLI

```bash
python main.py query "Your question here?"
```

### 3. Launch Chat Interface

```bash
python main.py chat
```

Then open http://127.0.0.1:7860 in your browser.

### 4. View Statistics

```bash
python main.py stats
```

## Example Workflow

```bash
# 1. Start Ollama (in separate terminal)
ollama serve

# 2. Activate environment
source venv/bin/activate

# 3. Add documents to data/documents/
cp ~/my-docs/*.pdf data/documents/

# 4. Ingest documents
python main.py ingest

# 5. Launch chat
python main.py chat
```

## Troubleshooting

### Ollama Connection Error
- Make sure Ollama is running: `ollama serve`
- Check base URL in `config/config.yaml`

### Out of Memory
- Reduce batch_size in `config/config.yaml`
- Use a smaller model (e.g., `phi` instead of `llama2`)

### Slow Inference
- Use GPU if available (set device to "cuda")
- Try quantized models in Ollama
- Reduce max_tokens in config

## Configuration

Edit `config/config.yaml` to customize:
- LLM model and parameters
- Embedding model
- Chunk size and overlap
- Retrieval settings
- Chat interface settings

## Project Structure

```
RAG_local/
├── config/
│   └── config.yaml          # Main configuration
├── data/
│   ├── documents/           # Put your documents here
│   └── vectorstore/         # Vector DB persistence
├── src/
│   ├── chat/               # Chat interface
│   ├── document_processing/ # Document loading & chunking
│   ├── embeddings/         # Embedding generation
│   ├── llm/                # Local LLM integration
│   ├── rag/                # RAG pipeline
│   └── vector_store/       # Vector database
├── main.py                 # CLI entry point
└── requirements.txt        # Dependencies
```
