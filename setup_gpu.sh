#!/bin/bash

# GPU Setup Script - Updates existing installation for CUDA/GPU support

set -e

echo "===================================================="
echo "RAG Local Application - GPU Setup"
echo "===================================================="

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "⚠ Warning: nvidia-smi not found. Make sure you have NVIDIA drivers installed."
    echo "  Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Activating virtual environment..."
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi
source venv/bin/activate

echo ""
echo "Uninstalling CPU version of PyTorch..."
pip uninstall -y torch torchvision torchaudio || true

echo ""
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Verifying CUDA installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "===================================================="
echo "✓ GPU setup complete!"
echo "===================================================="
echo ""
echo "Your configuration has been updated to use CUDA."
echo "Check config/config.yaml - device is set to 'cuda'"
echo ""
echo "To verify GPU is working:"
echo "  python test_system.py"
echo ""
