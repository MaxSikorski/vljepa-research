#!/usr/bin/env python3
"""
Export SALT student ViT encoder to ONNX format for browser inference.

Creates an ONNX-friendly wrapper around VisionTransformer that:
- Replaces einops.rearrange with native PyTorch ops (for ONNX tracing)
- Runs forward pass without masking (full image encoding)
- Outputs patch-level embeddings (B, num_patches, embed_dim)

The exported model can be loaded via ONNX Runtime Web (WASM backend).

Usage:
    python scripts/export_onnx.py \
        --checkpoint /path/to/student.pt \
        --output docs/models/salt-student.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ijepa.models.encoder import (
    Attention,
    MLP,
    TransformerBlock,
    VisionTransformer,
)


class OnnxPatchEmbed(nn.Module):
    """ONNX-friendly PatchEmbed that avoids einops."""

    def __init__(self, img_size: int = 32, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        # Replace einops rearrange("b c h w -> b (h w) c") with native ops
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class OnnxViT(nn.Module):
    """ONNX-exportable ViT wrapper. No masking, no RoPE, no optional args."""

    def __init__(self, vit: VisionTransformer):
        super().__init__()
        self.embed_dim = vit.embed_dim
        self.num_patches = vit.num_patches

        # Replace PatchEmbed with ONNX-friendly version
        self.patch_embed = OnnxPatchEmbed(
            img_size=vit.patch_embed.img_size,
            patch_size=vit.patch_embed.patch_size,
            in_channels=3,
            embed_dim=vit.embed_dim,
        )
        # Copy weights from original
        self.patch_embed.proj.weight.data.copy_(vit.patch_embed.proj.weight.data)
        self.patch_embed.proj.bias.data.copy_(vit.patch_embed.proj.bias.data)

        # Copy positional embedding
        if vit.pos_embed is not None:
            self.pos_embed = nn.Parameter(vit.pos_embed.data.clone())
        else:
            self.pos_embed = None

        # Copy transformer blocks (no RoPE — blocks work without it)
        self.blocks = vit.blocks
        self.norm = vit.norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B, 3, H, W) -> (B, num_patches, embed_dim)"""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    img_size: int = 32,
    patch_size: int = 8,
    embed_dim: int = 192,
    depth: int = 4,
    num_heads: int = 3,
    opset: int = 17,
    fp16: bool = True,
):
    """Export a ViT checkpoint to ONNX format."""
    from src.ijepa.models.encoder import build_encoder

    config = {
        "img_size": img_size,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "mlp_ratio": 4.0,
    }
    vit = build_encoder(config)

    # Load checkpoint
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "student" in state:
        state = state["student"]
    vit.load_state_dict(state, strict=True)
    vit.eval()

    # Wrap for ONNX
    onnx_model = OnnxViT(vit)
    onnx_model.eval()

    # Dummy input
    dummy = torch.randn(1, 3, img_size, img_size)

    # Verify output
    with torch.no_grad():
        orig_out = vit(dummy)
        onnx_out = onnx_model(dummy)
        diff = (orig_out - onnx_out).abs().max().item()
        print(f"Max output difference (original vs ONNX wrapper): {diff:.2e}")
        assert diff < 1e-5, f"Output mismatch: {diff}"

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        onnx_model,
        dummy,
        output_path,
        opset_version=14,
        input_names=["image"],
        output_names=["embeddings"],
        dynamic_axes={
            "image": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        dynamo=False,  # Force legacy TorchScript exporter for ONNX Runtime Web compat
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model: {output_path} ({size_mb:.1f} MB)")

    # Optional: convert to float16 for smaller file
    if fp16:
        try:
            import onnx
            from onnx import numpy_helper
            import numpy as np

            model = onnx.load(output_path)
            for initializer in model.graph.initializer:
                if initializer.data_type == onnx.TensorProto.FLOAT:
                    arr = numpy_helper.to_array(initializer).astype(np.float16)
                    new_init = numpy_helper.from_array(arr, name=initializer.name)
                    initializer.CopyFrom(new_init)

            fp16_path = output_path.replace(".onnx", "-fp16.onnx")
            onnx.save(model, fp16_path)
            fp16_size = Path(fp16_path).stat().st_size / (1024 * 1024)
            print(f"Exported FP16 model: {fp16_path} ({fp16_size:.1f} MB)")
        except ImportError:
            print("onnx package not available, skipping FP16 conversion")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="docs/models/salt-student.onnx")
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        fp16=not args.no_fp16,
    )
