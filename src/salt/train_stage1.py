"""
SALT Stage 1: V-Pixel Teacher Training.

Trains a Vision Transformer encoder with pixel reconstruction (MAE-style)
under V-JEPA's multi-block masking. After training, the encoder is saved
as the frozen teacher for Stage 2.

Key differences from I-JEPA training:
- NO EMA (no target encoder, no momentum scheduling)
- NO stop-gradient
- Pixel-space loss (MSE on masked patches), not latent-space loss
- Decoder is discarded after training

The encoder + decoder are jointly trained. Only the encoder is kept.

Reference: SALT (Apple, ICLR 2025) — https://arxiv.org/abs/2509.24317
"""

from __future__ import annotations

import logging
import math
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.salt.models.mae_decoder import MAEDecoder, build_mae_decoder
from src.salt.losses.pixel_loss import PixelReconstructionLoss

logger = logging.getLogger(__name__)


class SALTStage1Trainer:
    """
    SALT Stage 1: Train a MAE teacher encoder with pixel reconstruction.

    No EMA, no target encoder — just a simple encoder + decoder pipeline.
    The decoder is discarded after training.
    """

    def __init__(
        self,
        encoder: VisionTransformer,
        decoder: MAEDecoder,
        loss_fn: PixelReconstructionLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device = torch.device("cpu"),
        gradient_clip: float = 1.0,
    ):
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.gradient_clip = gradient_clip

    def train_step(
        self,
        images: torch.Tensor,
        grid_size: int,
        num_targets: int = 4,
    ) -> float:
        """
        Single training step.

        1. Generate multi-block masks
        2. Encoder processes visible (context) patches only
        3. Decoder reconstructs masked patches from encoder output
        4. MSE loss on masked pixels

        Returns:
            loss value (float)
        """
        self.encoder.train()
        self.decoder.train()

        B = images.shape[0]

        # Generate masks (same as I-JEPA multi-block masking)
        context_indices, target_list = generate_masks(
            batch_size=B,
            num_patches_h=grid_size,
            num_patches_w=grid_size,
            num_targets=num_targets,
            device=self.device,
        )

        # Combine all target indices into one set of masked indices
        all_target_indices = set()
        for tgt in target_list:
            all_target_indices.update(tgt[0].tolist())
        masked_indices = torch.tensor(
            sorted(all_target_indices), device=self.device
        ).unsqueeze(0).expand(B, -1)

        # Encoder: process visible (context) patches only
        visible_embeddings = self.encoder(images, mask_indices=context_indices)

        # Decoder: reconstruct masked patches
        pixel_predictions = self.decoder(
            visible_embeddings, context_indices, masked_indices
        )

        # Pixel reconstruction loss on masked patches
        loss = self.loss_fn(pixel_predictions, images, masked_indices)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.gradient_clip,
            )

        self.optimizer.step()

        return loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        grid_size: int,
        epoch: int,
        log_every: int = 50,
    ) -> float:
        """Train for one epoch. Returns average loss."""
        total_loss = 0.0
        num_steps = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(self.device)
            loss = self.train_step(images, grid_size)

            total_loss += loss
            num_steps += 1

            if (batch_idx + 1) % log_every == 0:
                logger.info(
                    f"  Stage 1 Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss:.4f}"
                )
                print(
                    f"  Stage 1 Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss:.4f}"
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_steps, 1)
        return avg_loss


def build_stage1(config: dict, device: torch.device = torch.device("cpu")):
    """
    Build SALT Stage 1 components from config.

    Returns:
        (encoder, decoder, loss_fn, optimizer, scheduler, trainer)
    """
    enc_config = config.get("encoder", {})
    dec_config = config.get("decoder", {})
    train_config = config.get("training", {})

    # Build encoder (same ViT as I-JEPA)
    encoder = build_encoder(enc_config)

    # Build MAE decoder
    dec_config["encoder_embed_dim"] = enc_config.get("embed_dim", 192)
    dec_config["patch_size"] = enc_config.get("patch_size", 16)
    decoder = build_mae_decoder(dec_config, encoder.num_patches)

    # Pixel reconstruction loss
    loss_fn = PixelReconstructionLoss(
        patch_size=enc_config.get("patch_size", 16),
        in_channels=3,
        norm_pix=train_config.get("norm_pix", True),
    )

    # Optimizer (encoder + decoder jointly)
    opt_config = train_config.get("optimizer", {})
    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=opt_config.get("lr", 1.5e-4),
        weight_decay=opt_config.get("weight_decay", 0.04),
        betas=tuple(opt_config.get("betas", [0.9, 0.95])),
    )

    # Scheduler
    epochs = train_config.get("epochs", 2)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=opt_config.get("min_lr", 1e-6),
    )

    # Build trainer
    trainer = SALTStage1Trainer(
        encoder=encoder,
        decoder=decoder,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip=train_config.get("gradient_clip", 1.0),
    )

    return encoder, decoder, loss_fn, optimizer, scheduler, trainer
