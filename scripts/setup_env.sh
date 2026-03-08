#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-vljepa}"
PYTHON_VERSION="3.11"

echo "=== VL-JEPA Research Lab Environment Setup ==="
echo "Environment name: $ENV_NAME"

# Detect platform
if [[ "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="macos"
    echo "Platform: macOS (Apple Silicon MPS backend)"
else
    PLATFORM="linux"
    echo "Platform: Linux (CUDA backend expected)"
fi

# Check for conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: conda or mamba not found. Install miniforge:"
    echo "  https://github.com/conda-forge/miniforge#install"
    exit 1
fi

echo "Using: $CONDA_CMD"

# Create environment
echo ""
echo "=== Creating conda environment ==="
$CONDA_CMD create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

# Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install PyTorch
echo ""
echo "=== Installing PyTorch ==="
if [[ "$PLATFORM" == "macos" ]]; then
    pip install torch torchvision torchaudio
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install project dependencies
echo ""
echo "=== Installing project dependencies ==="
pip install -e ".[dev,notebooks]"

# Install distributed training deps on Linux
if [[ "$PLATFORM" == "linux" ]]; then
    pip install -e ".[distributed]"
fi

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if hasattr(torch.backends, 'mps'):
    print(f'MPS available: {torch.backends.mps.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print('Installation successful!')
"

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Run smoke test: make train-ijepa-tiny"
