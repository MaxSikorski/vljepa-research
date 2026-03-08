"""
MAE Decoder for SALT Stage 1 (V-Pixel Teacher Training).

A lightweight transformer decoder that reconstructs pixels from
encoded visible patches. After Stage 1 training, the decoder is
discarded — only the encoder weights are kept as the frozen teacher.

Architecture follows VideoMAE / MAE with multi-block masking from V-JEPA.

Reference: SALT (Apple, ICLR 2025) — https://arxiv.org/abs/2509.24317
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.ijepa.models.encoder import TransformerBlock


class MAEDecoder(nn.Module):
    """
    Masked Autoencoder Decoder for SALT Stage 1.

    Takes encoded visible patches + learnable mask tokens for masked positions,
    processes through lightweight transformer blocks, and projects to pixel space.

    After Stage 1 training, the decoder is discarded.
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 4,
        decoder_num_heads: int = 16,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Project from encoder space to decoder space
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learnable mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Learnable positional embeddings for ALL patch positions
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )

        # Transformer decoder blocks (lightweight)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio=4.0,
            )
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        # Pixel prediction head: project to pixel space
        self.pred = nn.Linear(
            decoder_embed_dim,
            patch_size * patch_size * in_channels,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.pred.weight)
        nn.init.zeros_(self.pred.bias)

    def forward(
        self,
        visible_embeddings: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode masked patches from visible patch embeddings.

        Args:
            visible_embeddings: (B, M_vis, encoder_embed_dim) from encoder
            visible_indices: (B, M_vis) indices of visible patches
            masked_indices: (B, M_mask) indices of masked patches

        Returns:
            Predicted pixels for masked patches: (B, M_mask, patch_size^2 * C)
        """
        B = visible_embeddings.shape[0]
        M_vis = visible_indices.shape[1]
        M_mask = masked_indices.shape[1]

        # Project encoder embeddings to decoder space
        visible = self.decoder_embed(visible_embeddings)  # (B, M_vis, dec_dim)

        # Create mask tokens for masked positions
        mask_tokens = self.mask_token.expand(B, M_mask, -1)  # (B, M_mask, dec_dim)

        # Add positional embeddings
        vis_pos = torch.gather(
            self.decoder_pos_embed.expand(B, -1, -1),
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
        )
        mask_pos = torch.gather(
            self.decoder_pos_embed.expand(B, -1, -1),
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_embed_dim),
        )

        visible = visible + vis_pos
        mask_tokens = mask_tokens + mask_pos

        # Concatenate: [visible, masked] → full sequence
        x = torch.cat([visible, mask_tokens], dim=1)  # (B, M_vis + M_mask, dec_dim)

        # Process through decoder transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract only masked predictions (last M_mask tokens)
        masked_preds = x[:, M_vis:]  # (B, M_mask, dec_dim)

        # Project to pixel space
        pixel_preds = self.pred(masked_preds)  # (B, M_mask, P*P*C)

        return pixel_preds


def build_mae_decoder(config: dict, num_patches: int) -> MAEDecoder:
    """Build MAE decoder from config."""
    return MAEDecoder(
        num_patches=num_patches,
        encoder_embed_dim=config.get("encoder_embed_dim", 768),
        decoder_embed_dim=config.get("decoder_embed_dim", 512),
        decoder_depth=config.get("decoder_depth", 4),
        decoder_num_heads=config.get("decoder_num_heads", 16),
        patch_size=config.get("patch_size", 16),
        in_channels=config.get("in_channels", 3),
    )
