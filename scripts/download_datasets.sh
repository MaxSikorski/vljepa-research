#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_ROOT:-./data}"
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/processed" "$DATA_DIR/evaluation"

echo "=== VL-JEPA Research: Dataset Download Guide ==="
echo ""
echo "Due to the large size of video datasets, this script provides"
echo "download instructions rather than automatic downloads."
echo ""

echo "=== Phase 1: I-JEPA (Image Datasets) ==="
echo ""
echo "1. CIFAR-10 (for smoke tests) - Auto-downloaded by torchvision"
echo "   Size: ~170MB"
echo "   No manual action needed."
echo ""
echo "2. ImageNet-1K (for full I-JEPA training)"
echo "   Size: ~150GB"
echo "   Download from: https://image-net.org/download.php"
echo "   Place in: $DATA_DIR/raw/imagenet/"
echo "   Structure: imagenet/train/n01440764/*.JPEG"
echo ""

echo "=== Phase 2: V-JEPA (Video Datasets) ==="
echo ""
echo "3. Kinetics-400 (video classification)"
echo "   Size: ~450GB"
echo "   Download: https://github.com/cvdfoundation/kinetics-dataset"
echo "   Place in: $DATA_DIR/raw/kinetics400/"
echo ""
echo "4. Something-Something V2 (motion understanding)"
echo "   Size: ~20GB"
echo "   Download: https://developer.qualcomm.com/software/ai-datasets/something-something"
echo "   Place in: $DATA_DIR/raw/ssv2/"
echo ""

echo "=== Phase 3: VL-JEPA (Vision-Language Datasets) ==="
echo ""
echo "5. CC3M - Conceptual Captions 3M (image-text pairs)"
echo "   Size: ~40GB"
echo "   Download: https://ai.google.com/research/ConceptualCaptions/"
echo "   Place in: $DATA_DIR/raw/cc3m/"
echo ""
echo "6. WebVid-2M (video-text pairs)"
echo "   Size: ~200GB"
echo "   Download: https://maxbain.com/webvid-dataset/"
echo "   Place in: $DATA_DIR/raw/webvid2m/"
echo ""
echo "7. MSR-VTT (video retrieval evaluation)"
echo "   Size: ~6GB"
echo "   Download: https://www.microsoft.com/en-us/research/publication/msr-vtt/"
echo "   Place in: $DATA_DIR/evaluation/msrvtt/"
echo ""
echo "8. GQA (Visual QA evaluation)"
echo "   Size: ~20GB"
echo "   Download: https://cs.stanford.edu/people/dorarad/gqa/"
echo "   Place in: $DATA_DIR/evaluation/gqa/"
echo ""

echo "=== Quick Start: Download CIFAR-10 only ==="
echo "For immediate smoke testing, run:"
echo "  python -c \"import torchvision; torchvision.datasets.CIFAR10('$DATA_DIR/raw', download=True)\""
echo ""

# Auto-download CIFAR-10 for smoke testing
read -p "Download CIFAR-10 now for smoke testing? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -c "
import torchvision
torchvision.datasets.CIFAR10('$DATA_DIR/raw', download=True)
print('CIFAR-10 downloaded successfully!')
"
fi
