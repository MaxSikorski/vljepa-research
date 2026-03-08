#!/bin/bash
# Quick-start script for cloud GPU training
#
# USAGE:
#   1. SSH into your cloud GPU instance (Vast.ai, RunPod, Lambda)
#   2. Clone/upload this repo
#   3. Run: bash scripts/run_cloud_training.sh
#
# REQUIREMENTS:
#   - Docker with NVIDIA GPU support (nvidia-docker / --gpus all)
#   - At least 1 GPU with 16GB+ VRAM (A100, A10G, RTX 4090, etc.)
#
# CIFAR-10 is auto-downloaded on first run (~170MB).

set -e

echo "=== VL-JEPA Research Lab: Cloud Training ==="
echo ""

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Make sure you have NVIDIA GPU drivers installed."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Build training image
echo "Building Docker training image..."
docker build -f Dockerfile.train -t vljepa-train . 2>&1 | tail -3
echo ""

# Create output directory
mkdir -p output

# Run training
echo "Starting I-JEPA training on CIFAR-10..."
echo "Config: configs/ijepa/cifar10_a100.yaml"
echo "Output: ./output/"
echo ""

docker run --gpus all --rm \
    -v "$(pwd)/output:/output" \
    -v "$(pwd)/data:/data" \
    vljepa-train \
    python -m src.ijepa.train --config configs/ijepa/cifar10_a100.yaml

echo ""
echo "=== Training Complete ==="
echo "Checkpoints saved to: ./output/"
