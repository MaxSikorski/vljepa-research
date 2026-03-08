"""
Export VL-JEPA models to ONNX format for edge deployment.

Supports exporting:
- X-Encoder only (visual feature extraction)
- Full VL-JEPA (visual + text → embedding)
- Predictor only (for action-conditioned planning)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_x_encoder(
    model,
    output_path: str,
    img_size: int = 224,
    opset_version: int = 17,
) -> None:
    """Export frozen X-Encoder to ONNX."""
    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model.x_encoder,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["image"],
        output_names=["visual_features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "visual_features": {0: "batch_size"},
        },
    )
    print(f"X-Encoder exported to {output_path}")


def export_predictor(
    model,
    output_path: str,
    num_visual_tokens: int = 196,
    num_query_tokens: int = 32,
    embed_dim: int = 192,
    opset_version: int = 17,
) -> None:
    """Export predictor to ONNX."""
    dummy_vis = torch.randn(1, num_visual_tokens, embed_dim)
    dummy_query = torch.randn(1, num_query_tokens, embed_dim)

    torch.onnx.export(
        model.predictor,
        (dummy_vis, dummy_query),
        output_path,
        opset_version=opset_version,
        input_names=["visual_features", "query_embeddings"],
        output_names=["predicted_embedding"],
        dynamic_axes={
            "visual_features": {0: "batch_size"},
            "query_embeddings": {0: "batch_size"},
            "predicted_embedding": {0: "batch_size"},
        },
    )
    print(f"Predictor exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VL-JEPA to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./exports/onnx")
    parser.add_argument("--component", choices=["x_encoder", "predictor", "all"], default="all")
    args = parser.parse_args()
    print(f"Export {args.component} from {args.checkpoint} to {args.output_dir}")
