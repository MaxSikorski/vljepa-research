"""
Pixel Reconstruction Loss for SALT Stage 1.

MSE loss computed only on masked patches (not visible ones).
Optionally normalizes each patch to zero mean / unit variance before loss
(per-patch normalization, as in MAE).

Reference: SALT (Apple, ICLR 2025) — https://arxiv.org/abs/2509.24317
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PixelReconstructionLoss(nn.Module):
    """
    Pixel reconstruction loss for SALT Stage 1 (V-Pixel).

    Computes MSE between predicted and target pixels for masked patches only.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, norm_pix: bool = True):
        """
        Args:
            patch_size: Size of each patch (assumed square)
            in_channels: Number of image channels (3 for RGB)
            norm_pix: If True, normalize each target patch to zero mean / unit var
                      before computing loss (following MAE)
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.norm_pix = norm_pix

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch tokens.

        Args:
            images: (B, C, H, W)

        Returns:
            patches: (B, N, patch_size^2 * C)
        """
        p = self.patch_size
        B, C, H, W = images.shape
        assert H % p == 0 and W % p == 0, f"Image {H}x{W} not divisible by patch {p}"

        patches = rearrange(
            images, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=p, p2=p,
        )
        return patches

    def forward(
        self,
        predictions: torch.Tensor,
        images: torch.Tensor,
        masked_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pixel reconstruction loss on masked patches.

        Args:
            predictions: (B, M_mask, patch_size^2 * C) predicted pixel values
            images: (B, C, H, W) original images
            masked_indices: (B, M_mask) indices of masked patches

        Returns:
            Scalar loss value
        """
        # Convert images to patches
        all_patches = self.patchify(images)  # (B, N, P*P*C)

        # Gather target patches at masked positions
        target = torch.gather(
            all_patches,
            1,
            masked_indices.unsqueeze(-1).expand(-1, -1, all_patches.shape[-1]),
        )  # (B, M_mask, P*P*C)

        if self.norm_pix:
            # Per-patch normalization (MAE-style)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss on masked patches only
        loss = F.mse_loss(predictions, target)
        return loss
