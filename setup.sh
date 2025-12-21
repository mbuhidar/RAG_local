#!/bin/bash

# Setup script for RAG Local application

set -e

echo "===================================================="
echo "RAG Local Application - Setup Script"
echo "===================================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip wheel

echo "Installing dependencies (this may take a while)..."
echo "Step 1/3: Installing PyTorch (CUDA version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Step 2/3: Installing core ML packages..."
pip install sentence-transformers scikit-learn scipy networkx regex safetensors

echo "Step 3/3: Installing application packages..."
pip install --no-deps chromadb transformers
pip install ollama requests pypdf python-docx gradio pyyaml python-dotenv tqdm

echo "Step 4/3: Installing remaining chromadb dependencies..."
pip install bcrypt build grpcio importlib-resources jsonschema kubernetes mmh3 onnxruntime opentelemetry-api opentelemetry-exporter-otlp-proto-grpc opentelemetry-sdk overrides posthog pybase64 pypika tenacity tokenizers

echo "Step 5/3: Installing development tools (optional)..."
pip install pytest black flake8 || echo "Warning: Dev tools installation failed (non-critical)"

echo ""
echo "===================================================="
echo "✓ Installation complete!"
echo "===================================================="
echo ""
echo "Next steps:"
echo "1. Start Ollama:  ollama serve"
echo "2. Pull a model:  ollama pull llama2"
echo "3. Verify setup:  python verify_install.py"
echo ""
