"""
Y-Encoder for VL-JEPA: Target text embedding space.

In the paper, this is initialized from EmbeddingGemma-300M and produces
the ground-truth text embeddings that the predictor learns to match.

Key details:
- Produces embeddings in 1536-d shared space
- Fine-tuned with reduced learning rate (0.05x the predictor LR)
- Provides the "answer" side of the bi-directional InfoNCE loss
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ijepa.models.encoder import TransformerBlock


class YEncoder(nn.Module):
    """
    Target text encoder for VL-JEPA.

    Encodes target text (answers, descriptions, labels) into the
    shared embedding space. The predictor is trained to match these
    embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 2048,
        depth: int = 12,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 128,
        shared_embedding_dim: int = 1536,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_embedding_dim = shared_embedding_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Project to shared embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, shared_embedding_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode text into shared embedding space.

        Args:
            token_ids: (B, L) token indices
            attention_mask: (B, L) boolean mask (True = attend)

        Returns:
            Text embedding (B, shared_embedding_dim)
        """
        B, L = token_ids.shape

        # Token + positional embedding
        x = self.token_embedding(token_ids) + self.pos_embedding[:, :L, :]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Mean pooling with attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Project and normalize
        output = self.output_proj(pooled)
        output = F.normalize(output, dim=-1)

        return output


class SmallYEncoder(nn.Module):
    """Small Y-Encoder for smoke testing."""

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        max_seq_len: int = 128,
        shared_embedding_dim: int = 192,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, shared_embedding_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L = token_ids.shape
        x = self.token_embedding(token_ids) + self.pos_embedding[:, :L, :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return F.normalize(self.output_proj(pooled), dim=-1)


def build_y_encoder(config: dict) -> nn.Module:
    """Build Y-Encoder from config."""
    embed_dim = config.get("embed_dim", 192)

    if embed_dim > 512:
        return YEncoder(
            vocab_size=config.get("vocab_size", 32000),
            embed_dim=embed_dim,
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 16),
            shared_embedding_dim=config.get("shared_embedding_dim", 1536),
        )
    else:
        return SmallYEncoder(
            vocab_size=config.get("vocab_size", 32000),
            embed_dim=embed_dim,
            depth=config.get("depth", 4),
            num_heads=config.get("num_heads", 3),
            shared_embedding_dim=config.get("shared_embedding_dim", embed_dim),
        )
