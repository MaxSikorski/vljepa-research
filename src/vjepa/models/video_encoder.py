"""
Video Vision Transformer for V-JEPA.

Extends the image ViT with temporal modeling:
- 3D patch embedding (tubelet embedding) for spatiotemporal patches
- Temporal position embeddings added to spatial position embeddings
- Same transformer backbone processes spatiotemporal tokens

Key difference from image ViT: patches are now 3D (t, h, w) and
positional embeddings include a temporal component.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from src.ijepa.models.encoder import TransformerBlock


class TubeletEmbed(nn.Module):
    """3D patch embedding — converts video clips into spatiotemporal tokens."""

    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        tubelet_size: int = 2,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.img_size = img_size

        self.num_temporal_patches = num_frames // tubelet_size
        self.num_spatial_patches_h = img_size // patch_size
        self.num_spatial_patches_w = img_size // patch_size
        self.num_patches = (
            self.num_temporal_patches
            * self.num_spatial_patches_h
            * self.num_spatial_patches_w
        )

        # 3D convolution for tubelet embedding
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video tensor

        Returns:
            (B, num_patches, embed_dim)
        """
        # Rearrange to (B, C, T, H, W) for Conv3d
        x = rearrange(x, "b t c h w -> b c t h w")
        x = self.proj(x)  # (B, D, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")
        return x


class VideoVisionTransformer(nn.Module):
    """
    Video ViT for V-JEPA.

    Processes video clips as sequences of spatiotemporal tokens.
    """

    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        tubelet_size: int = 2,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Tubelet embedding
        self.patch_embed = TubeletEmbed(
            img_size, num_frames, tubelet_size, patch_size, in_channels, embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.num_temporal_patches = self.patch_embed.num_temporal_patches
        self.num_spatial_patches_h = self.patch_embed.num_spatial_patches_h
        self.num_spatial_patches_w = self.patch_embed.num_spatial_patches_w

        # Learnable positional embeddings (spatial + temporal)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video tensor
            mask_indices: Optional (B, M) indices to select
        Returns:
            (B, N, D) or (B, M, D) patch representations
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed

        if mask_indices is not None:
            x = torch.gather(
                x, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x


def build_video_encoder(config: dict) -> VideoVisionTransformer:
    """Build a video ViT encoder from config."""
    patch_size_cfg = config.get("patch_size", [2, 16, 16])
    if isinstance(patch_size_cfg, list):
        tubelet_size = patch_size_cfg[0]
        patch_size = patch_size_cfg[1]
    else:
        tubelet_size = 2
        patch_size = patch_size_cfg

    return VideoVisionTransformer(
        img_size=config.get("img_size", 224),
        num_frames=config.get("num_frames", 16),
        tubelet_size=tubelet_size,
        patch_size=patch_size,
        embed_dim=config.get("embed_dim", 768),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 12),
        mlp_ratio=config.get("mlp_ratio", 4.0),
    )
