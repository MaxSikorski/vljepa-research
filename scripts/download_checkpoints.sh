#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_ROOT:-./data/checkpoints}"
mkdir -p "$CHECKPOINT_DIR"

echo "=== Downloading VL-JEPA Research Checkpoints ==="
echo "Saving to: $CHECKPOINT_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# V-JEPA 2 pretrained model (used as X-Encoder in VL-JEPA)
echo ""
echo "=== Downloading V-JEPA 2 ViT-G ==="
echo "Model: facebook/vjepa2-vitg-fpc64-384"
echo "This is the frozen vision encoder used in VL-JEPA"
huggingface-cli download facebook/vjepa2-vitg-fpc64-384 \
    --local-dir "$CHECKPOINT_DIR/vjepa2-vitg" \
    --local-dir-use-symlinks False || {
    echo "WARNING: V-JEPA 2 download failed. You may need to accept the model license at:"
    echo "  https://huggingface.co/facebook/vjepa2-vitg-fpc64-384"
}

# Llama-3.2-1B (predictor initialization for VL-JEPA)
echo ""
echo "=== Downloading Llama-3.2-1B ==="
echo "Used to initialize the VL-JEPA predictor (last 8 layers)"
huggingface-cli download meta-llama/Llama-3.2-1B \
    --local-dir "$CHECKPOINT_DIR/llama-3.2-1b" \
    --local-dir-use-symlinks False || {
    echo "WARNING: Llama-3.2-1B download failed. You need to:"
    echo "  1. Accept the license at: https://huggingface.co/meta-llama/Llama-3.2-1B"
    echo "  2. Set HF_TOKEN environment variable"
}

# Gemma embedding model (Y-Encoder for VL-JEPA)
echo ""
echo "=== Downloading Gemma Embedding Model ==="
echo "Used as the Y-Encoder (target text embedding space) in VL-JEPA"
huggingface-cli download google/gemma-2b \
    --local-dir "$CHECKPOINT_DIR/gemma-2b" \
    --local-dir-use-symlinks False || {
    echo "WARNING: Gemma download failed. You may need to accept the license at:"
    echo "  https://huggingface.co/google/gemma-2b"
}

echo ""
echo "=== Download Summary ==="
echo "Checkpoints saved to: $CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR/" 2>/dev/null || echo "No checkpoints downloaded yet."
echo ""
echo "NOTE: Some models require license acceptance on HuggingFace."
echo "Visit the model pages linked above and set HF_TOKEN in your .env file."
