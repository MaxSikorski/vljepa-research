"""
VL-JEPA Predictor: Transformer bridge between vision and language.

In the paper, the predictor is initialized from the last 8 Transformer
layers of Llama-3.2-1B (490M trainable parameters). Key design choices:

1. CAUSAL ATTENTION IS DISABLED — vision and query embeddings are jointly
   attended bidirectionally. This is a critical departure from standard LLMs.

2. The predictor takes concatenated [visual_tokens, query_tokens] as input
   and outputs a predicted embedding in the shared 1536-d space.

3. Projection heads map predictor outputs to the shared embedding space
   where the bi-directional InfoNCE loss operates.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ijepa.models.encoder import TransformerBlock


class VLJEPAPredictor(nn.Module):
    """
    Vision-Language predictor for VL-JEPA.

    Takes visual embeddings from X-Encoder and text query embeddings,
    processes them jointly (no causal mask), and projects to shared space.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        depth: int = 8,
        num_heads: int = 32,
        mlp_ratio: float = 4.0,
        shared_embedding_dim: int = 1536,
        max_visual_tokens: int = 577,  # 384/16 = 24, 24*24 + 1 CLS = 577
        max_query_tokens: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.shared_embedding_dim = shared_embedding_dim

        # Transformer blocks (bidirectional attention — no causal mask)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Input projections (map encoder outputs to predictor dimension)
        self.visual_proj = nn.Linear(embed_dim, embed_dim)  # May need different input dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Positional embeddings for visual and query tokens
        self.visual_pos = nn.Parameter(
            torch.zeros(1, max_visual_tokens, embed_dim)
        )
        self.query_pos = nn.Parameter(
            torch.zeros(1, max_query_tokens, embed_dim)
        )

        # Output projection to shared embedding space
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, shared_embedding_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.visual_pos, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target text embedding from visual + query input.

        Args:
            visual_embeddings: (B, N_vis, D_vis) from X-Encoder
            query_embeddings: (B, N_query, D_query) text query tokens

        Returns:
            Predicted embedding (B, shared_embedding_dim) in shared space
        """
        B = visual_embeddings.shape[0]
        N_vis = visual_embeddings.shape[1]
        N_query = query_embeddings.shape[1]

        # Project to predictor dimension
        vis = self.visual_proj(visual_embeddings)
        query = self.query_proj(query_embeddings)

        # Add positional embeddings
        vis = vis + self.visual_pos[:, :N_vis, :]
        query = query + self.query_pos[:, :N_query, :]

        # Concatenate visual + query tokens for joint processing
        # NO CAUSAL MASK — bidirectional attention (key VL-JEPA design)
        x = torch.cat([vis, query], dim=1)  # (B, N_vis + N_query, D)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Pool: mean over all tokens, then project to shared space
        pooled = x.mean(dim=1)  # (B, D)
        output = self.output_proj(pooled)  # (B, shared_embedding_dim)

        # L2 normalize for InfoNCE
        output = F.normalize(output, dim=-1)

        return output


class SmallPredictor(nn.Module):
    """Smaller predictor for smoke testing."""

    def __init__(
        self,
        input_dim: int = 192,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        shared_embedding_dim: int = 192,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.visual_proj = nn.Linear(input_dim, embed_dim)
        self.query_proj = nn.Linear(input_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, shared_embedding_dim)

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        vis = self.visual_proj(visual_embeddings)
        query = self.query_proj(query_embeddings)
        x = torch.cat([vis, query], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        pooled = x.mean(dim=1)
        output = self.output_proj(pooled)
        return F.normalize(output, dim=-1)


def build_predictor(config: dict) -> nn.Module:
    """Build VL-JEPA predictor from config."""
    name = config.get("name", "transformer")

    if name in ("llama3_last8", "transformer") and config.get("embed_dim", 192) > 512:
        return VLJEPAPredictor(
            embed_dim=config.get("embed_dim", 2048),
            depth=config.get("depth", 8),
            num_heads=config.get("num_heads", 32),
            shared_embedding_dim=config.get("shared_embedding_dim", 1536),
        )
    else:
        return SmallPredictor(
            input_dim=config.get("embed_dim", 192),
            embed_dim=config.get("embed_dim", 192),
            depth=config.get("depth", 4),
            num_heads=config.get("num_heads", 3),
            shared_embedding_dim=config.get("shared_embedding_dim", 192),
        )
