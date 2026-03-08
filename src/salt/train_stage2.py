"""
SALT Stage 2: Frozen-Teacher JEPA Student Training.

Freezes the Stage 1 teacher encoder, then trains a student encoder +
JEPA predictor to predict the frozen teacher's latent representations
on masked regions.

Key differences from I-JEPA training:
- NO EMA — teacher is completely frozen from Stage 1
- NO stop-gradient needed — teacher has no parameters to update
- Teacher and student CAN be different sizes ("weak teacher, strong student")
- Student loss correlates with downstream accuracy (R^2 = 0.951)

This is the core innovation of SALT: decoupling teacher and student training.

Reference: SALT (Apple, ICLR 2025) — https://arxiv.org/abs/2509.24317
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor

logger = logging.getLogger(__name__)


class SALTStage2Trainer:
    """
    SALT Stage 2: Train student + predictor with frozen teacher.

    The teacher encoder is frozen (from Stage 1). The student encoder +
    predictor are trained to predict frozen teacher latents on masked regions.
    """

    def __init__(
        self,
        teacher: VisionTransformer,
        student: VisionTransformer,
        predictor: JEPAPredictor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        device: torch.device = torch.device("cpu"),
        gradient_clip: float = 1.0,
    ):
        # Freeze teacher completely
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student.to(device)
        self.predictor = predictor.to(device)
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
        2. Frozen teacher encodes full image → target latents
        3. Student encodes visible (context) patches only
        4. Predictor predicts teacher latents for masked regions
        5. L1 loss between predictions and frozen teacher targets

        Returns:
            loss value (float)
        """
        self.student.train()
        self.predictor.train()

        B = images.shape[0]

        # Generate masks (same multi-block masking as I-JEPA)
        context_indices, target_list = generate_masks(
            batch_size=B,
            num_patches_h=grid_size,
            num_patches_w=grid_size,
            num_targets=num_targets,
            device=self.device,
        )

        # Frozen teacher: encode full image (no gradient, no EMA)
        with torch.no_grad():
            teacher_embeddings = self.teacher(images)  # (B, N, D_teacher)

        # Student: encode visible (context) patches
        student_context = self.student(images, mask_indices=context_indices)

        # Predict each target block and compute L1 loss
        total_loss = 0.0
        for target_indices in target_list:
            # Gather teacher targets at masked positions
            teacher_targets = torch.gather(
                teacher_embeddings,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, teacher_embeddings.shape[-1]),
            )  # (B, M_tgt, D_teacher)

            # Predictor: student context → predicted teacher targets
            predicted = self.predictor(
                student_context, context_indices, target_indices
            )  # (B, M_tgt, D_teacher)

            # L1 loss (SALT uses L1, not smooth_l1 or MSE)
            total_loss += F.l1_loss(predicted, teacher_targets.detach())

        total_loss /= len(target_list)

        # Backward pass (only student + predictor get gradients)
        self.optimizer.zero_grad()
        total_loss.backward()

        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.predictor.parameters()),
                self.gradient_clip,
            )

        self.optimizer.step()

        return total_loss.item()

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
                    f"  Stage 2 Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss:.4f}"
                )
                print(
                    f"  Stage 2 Epoch {epoch+1}, Step {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss:.4f}"
                )

        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(num_steps, 1)
        return avg_loss


def build_stage2(
    config: dict,
    teacher: VisionTransformer,
    device: torch.device = torch.device("cpu"),
):
    """
    Build SALT Stage 2 components from config.

    Args:
        config: Stage 2 configuration
        teacher: Pre-trained teacher encoder from Stage 1 (will be frozen)
        device: Target device

    Returns:
        (student, predictor, optimizer, scheduler, trainer)
    """
    enc_config = config.get("encoder", {})
    pred_config = config.get("predictor", {})
    train_config = config.get("training", {})

    # Build student encoder (can be same or different size than teacher)
    student = build_encoder(enc_config)

    # Build JEPA predictor (reuses I-JEPA predictor)
    # Input is student embed_dim, output is teacher embed_dim
    pred_config.setdefault("embed_dim", enc_config.get("embed_dim", 192))
    pred_config.setdefault("predictor_embed_dim", enc_config.get("embed_dim", 192) // 2)
    predictor = build_predictor(pred_config, student.num_patches)

    # Optimizer (student + predictor, NOT teacher)
    opt_config = train_config.get("optimizer", {})
    optimizer = AdamW(
        list(student.parameters()) + list(predictor.parameters()),
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
    trainer = SALTStage2Trainer(
        teacher=teacher,
        student=student,
        predictor=predictor,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clip=train_config.get("gradient_clip", 1.0),
    )

    return student, predictor, optimizer, scheduler, trainer
