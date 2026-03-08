"""
JEPA Predictor module.

The predictor takes context patch representations and positional tokens for
target locations, then predicts the target representations in latent space.

This is the core of the JEPA architecture — prediction happens in abstract
representation space, NOT in pixel space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ijepa.models.encoder import TransformerBlock


class JEPAPredictor(nn.Module):
    """
    Predicts target representations from context representations.

    The predictor:
    1. Takes context patch embeddings from the context encoder
    2. Adds learnable positional tokens for target patch locations
    3. Processes through transformer blocks
    4. Outputs predicted representations for target patches

    Key design: The predictor is narrower than the encoder (smaller embed_dim)
    to prevent it from simply copying the encoder's representations.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.predictor_embed_dim = predictor_embed_dim
        self.num_patches = num_patches

        # Project from encoder space to predictor space
        self.input_proj = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # Learnable positional embeddings for predictor
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim)
        )

        # Mask token: learnable embedding for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                predictor_embed_dim,
                num_heads,
                mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_embed_dim)

        # Project back to encoder space for loss computation
        self.output_proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.predictor_pos_embed, std=0.02)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict target representations from context.

        Args:
            context_embeddings: (B, M_ctx, encoder_embed_dim) from context encoder
            context_indices: (B, M_ctx) patch indices of context blocks
            target_indices: (B, M_tgt) patch indices of target blocks

        Returns:
            Predicted target representations (B, M_tgt, encoder_embed_dim)
        """
        B = context_embeddings.shape[0]
        M_ctx = context_indices.shape[1]
        M_tgt = target_indices.shape[1]

        # Project context to predictor dimension
        context = self.input_proj(context_embeddings)  # (B, M_ctx, pred_dim)

        # Add positional embeddings for context patches
        ctx_pos = torch.gather(
            self.predictor_pos_embed.expand(B, -1, -1),
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim),
        )
        context = context + ctx_pos

        # Create mask tokens for target positions with positional embeddings
        mask_tokens = self.mask_token.expand(B, M_tgt, -1)
        tgt_pos = torch.gather(
            self.predictor_pos_embed.expand(B, -1, -1),
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_embed_dim),
        )
        targets = mask_tokens + tgt_pos

        # Concatenate context + target tokens for joint processing
        x = torch.cat([context, targets], dim=1)  # (B, M_ctx + M_tgt, pred_dim)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract only the target predictions
        target_preds = x[:, M_ctx:]  # (B, M_tgt, pred_dim)

        # Project back to encoder space
        target_preds = self.output_proj(target_preds)  # (B, M_tgt, encoder_dim)

        return target_preds


def build_predictor(config: dict, num_patches: int) -> JEPAPredictor:
    """Build a predictor from config."""
    return JEPAPredictor(
        num_patches=num_patches,
        encoder_embed_dim=config.get("embed_dim", 768),
        predictor_embed_dim=config.get("predictor_embed_dim", 384),
        depth=config.get("depth", 6),
        num_heads=config.get("num_heads", 6),
        mlp_ratio=config.get("mlp_ratio", 4.0),
    )
