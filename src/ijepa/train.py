"""
I-JEPA training loop.

Implements the core JEPA training procedure:
1. Sample multi-block masks (context + targets)
2. Context encoder processes context patches
3. Target encoder (EMA) processes full image, extract target patches
4. Predictor predicts target representations from context
5. Loss: MSE between predicted and actual target representations
6. Update context encoder + predictor via gradient descent
7. Update target encoder via Exponential Moving Average (EMA)
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.common.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.common.config import load_config, save_config
from src.common.data_utils import build_dataloader, get_image_transforms
from src.common.distributed import distributed_context, get_device, get_world_size, is_main_process, wrap_model_distributed
from src.common.logging import MetricsLogger, get_logger, setup_logger
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor


class IJEPATrainer:
    """I-JEPA training orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()

        # Build models
        enc_config = config["model"]["encoder"]
        pred_config = config["model"]["predictor"]

        self.context_encoder = build_encoder(enc_config).to(self.device)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)  # Target encoder is EMA, no gradients

        num_patches = self.context_encoder.num_patches
        self.predictor = build_predictor(pred_config, num_patches).to(self.device)

        # Distributed wrapping (DDP or FSDP)
        dist_strategy = config["training"].get("distributed_strategy", "ddp")
        mp_config = config["training"].get("mixed_precision", "none")
        if get_world_size() > 1:
            self.context_encoder = wrap_model_distributed(
                self.context_encoder, strategy=dist_strategy, mixed_precision=mp_config,
            )
            self.predictor = wrap_model_distributed(
                self.predictor, strategy=dist_strategy, mixed_precision=mp_config,
            )
            # Note: target_encoder is NOT wrapped — it's updated via EMA, never in backward pass

        # Grid dimensions for masking
        grid_size = int(num_patches ** 0.5)
        self.grid_h = grid_size
        self.grid_w = grid_size

        # Mixed precision setup
        mp_config = config["training"].get("mixed_precision", "none")
        self.use_amp = mp_config != "none" and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if mp_config == "bf16" else torch.float16
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(mp_config == "fp16"))
        else:
            self.amp_dtype = torch.float32
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Optimizer (context encoder + predictor only)
        train_config = config["training"]
        opt_config = train_config["optimizer"]
        self.optimizer = AdamW(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
            betas=tuple(opt_config.get("betas", [0.9, 0.95])),
        )

        # Scheduler
        sched_config = train_config["scheduler"]
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=train_config["epochs"],
            eta_min=sched_config.get("min_lr", 1e-6),
        )

        # EMA momentum schedule
        self.ema_momentum_start = train_config.get("ema_momentum", 0.996)
        self.ema_momentum_end = train_config.get("ema_momentum_end", 1.0)
        self.total_epochs = train_config["epochs"]

        # Logging
        log_config = config["logging"]
        self.output_dir = Path(log_config["output_dir"])
        self.logger = setup_logger(self.output_dir)
        self.metrics_logger = MetricsLogger(
            self.output_dir,
            wandb_config=log_config.get("wandb"),
        )

        self.log_every = log_config.get("log_every", 10)
        self.save_every = log_config.get("save_every", 50)

        # Parameter counts
        ctx_params = sum(p.numel() for p in self.context_encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        self.logger.info(f"Context encoder params: {ctx_params:,}")
        self.logger.info(f"Predictor params: {pred_params:,}")
        self.logger.info(f"Target encoder params: {ctx_params:,} (EMA, no grad)")
        self.logger.info(f"Device: {self.device}")

    def _get_ema_momentum(self, epoch: int) -> float:
        """Cosine schedule for EMA momentum (increases over training)."""
        progress = epoch / max(self.total_epochs - 1, 1)
        return self.ema_momentum_end - (self.ema_momentum_end - self.ema_momentum_start) * (
            1 + math.cos(math.pi * progress)
        ) / 2

    @torch.no_grad()
    def _update_target_encoder(self, momentum: float) -> None:
        """Update target encoder via EMA."""
        for param_t, param_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1 - momentum)

    def train_step(self, images: torch.Tensor, epoch: int) -> dict[str, float]:
        """Single training step with mixed precision support."""
        B = images.shape[0]
        mask_config = self.config["masking"]

        # Generate masks
        context_indices, target_indices_list = generate_masks(
            batch_size=B,
            num_patches_h=self.grid_h,
            num_patches_w=self.grid_w,
            num_targets=mask_config.get("num_targets", 4),
            min_target_scale=mask_config.get("min_target_scale", 0.15),
            max_target_scale=mask_config.get("max_target_scale", 0.2),
            min_context_scale=mask_config.get("min_context_scale", 0.85),
            max_context_scale=mask_config.get("max_context_scale", 1.0),
            device=self.device,
        )

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            # Context encoder: process only context patches
            context_embeddings = self.context_encoder(images, mask_indices=context_indices)

            # Target encoder: process full image (no masking), extract targets
            with torch.no_grad():
                full_target_embeddings = self.target_encoder(images)  # (B, N, D)

            # Compute loss for each target block
            total_loss = 0.0
            num_targets = 0

            for target_indices in target_indices_list:
                # Extract target representations from target encoder
                target_repr = torch.gather(
                    full_target_embeddings,
                    1,
                    target_indices.unsqueeze(-1).expand(-1, -1, full_target_embeddings.shape[-1]),
                )

                # Predict target representations
                predicted_repr = self.predictor(
                    context_embeddings, context_indices, target_indices
                )

                # L2 loss in representation space (the core of JEPA)
                loss = F.smooth_l1_loss(predicted_repr, target_repr.detach())
                total_loss += loss
                num_targets += 1

            total_loss = total_loss / num_targets

        # Backward pass with gradient scaling (for fp16 stability)
        self.optimizer.zero_grad()
        self.grad_scaler.scale(total_loss).backward()

        # Gradient clipping (unscale first for correct norm)
        self.grad_scaler.unscale_(self.optimizer)
        grad_clip = self.config["training"].get("gradient_clip", 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            grad_clip,
        )

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # EMA update of target encoder
        momentum = self._get_ema_momentum(epoch)
        self._update_target_encoder(momentum)

        return {
            "loss": total_loss.item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "ema_momentum": momentum,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def train(self) -> None:
        """Full training loop."""
        train_config = self.config["training"]

        # Build dataset
        data_config = self.config["data"]
        if data_config["dataset"] == "cifar10":
            from torchvision.datasets import CIFAR10
            transform = get_image_transforms(
                img_size=self.config["model"]["encoder"]["img_size"],
                is_train=True,
            )
            dataset = CIFAR10(data_config["root"], train=True, download=True, transform=transform)
        else:
            raise NotImplementedError(f"Dataset {data_config['dataset']} not yet implemented")

        dataloader = build_dataloader(
            dataset,
            batch_size=train_config["batch_size"],
            num_workers=data_config.get("num_workers", 4),
            is_train=True,
        )

        # Check for checkpoint to resume from
        resume_info = None
        latest_ckpt = find_latest_checkpoint(self.output_dir)
        if latest_ckpt is not None:
            self.logger.info(f"Resuming from {latest_ckpt}")
            # Would load checkpoint here

        self.logger.info(f"Starting I-JEPA training for {train_config['epochs']} epochs")
        self.logger.info(f"Dataset: {len(dataset)} samples, Batch size: {train_config['batch_size']}")

        global_step = 0
        best_loss = float("inf")

        for epoch in range(train_config["epochs"]):
            self.context_encoder.train()
            self.predictor.train()
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_idx, (images, _) in enumerate(dataloader):
                images = images.to(self.device)
                metrics = self.train_step(images, epoch)

                epoch_loss += metrics["loss"]
                epoch_steps += 1
                global_step += 1

                if global_step % self.log_every == 0:
                    self.metrics_logger.log(metrics, step=global_step)
                    if is_main_process():
                        self.logger.info(
                            f"Epoch [{epoch+1}/{train_config['epochs']}] "
                            f"Step [{batch_idx+1}/{len(dataloader)}] "
                            f"Loss: {metrics['loss']:.4f} "
                            f"LR: {metrics['lr']:.2e}"
                        )

            # End of epoch
            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.scheduler.step()

            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            if (epoch + 1) % self.save_every == 0 or is_best:
                save_checkpoint(
                    model=nn.ModuleDict({
                        "context_encoder": self.context_encoder,
                        "predictor": self.predictor,
                        "target_encoder": self.target_encoder,
                    }),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    config=self.config,
                    output_dir=self.output_dir,
                    metrics={"avg_loss": avg_loss},
                    is_best=is_best,
                )

        self.logger.info(f"Training complete. Best loss: {best_loss:.4f}")
        self.metrics_logger.finish()


def main():
    parser = argparse.ArgumentParser(description="I-JEPA Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = IJEPATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
