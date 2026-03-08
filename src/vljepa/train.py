"""
VL-JEPA training loop.

Two-stage training:
  Stage 1 (Pretraining): Constant LR, large-scale image-text + video-text data
  Stage 2 (SFT): Cosine annealing, task-specific data (VQA, classification, retrieval)

Key training details:
- X-Encoder is FROZEN (no gradients, no optimizer)
- Predictor trained at full learning rate
- Y-Encoder trained at 0.05x learning rate
- Loss: Bi-directional InfoNCE in 1536-d shared space
- Mixed precision: BF16 on A100s, FP32 on MPS
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from src.common.checkpointing import save_checkpoint
from src.common.config import load_config
from src.common.data_utils import DummyImageTextDataset, build_dataloader
from src.common.distributed import get_device, get_world_size, is_main_process, wrap_model_distributed
from src.common.logging import MetricsLogger, setup_logger
from src.vljepa.losses.infonce import build_loss
from src.vljepa.models.vljepa import VLJEPA, build_vljepa


class VLJEPATrainer:
    """VL-JEPA training orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()

        # Build model
        self.model = build_vljepa(config["model"]).to(self.device)

        # Distributed wrapping (DDP or FSDP) — must happen before optimizer creation
        # Only wrap trainable sub-modules; x_encoder is frozen
        dist_strategy = config["training"].get("distributed_strategy", "ddp")
        mp_config_str = config["training"].get("mixed_precision", "none")
        if get_world_size() > 1:
            self.model.predictor = wrap_model_distributed(
                self.model.predictor, strategy=dist_strategy, mixed_precision=mp_config_str,
            )
            self.model.y_encoder = wrap_model_distributed(
                self.model.y_encoder, strategy=dist_strategy, mixed_precision=mp_config_str,
            )

        # Build loss
        self.criterion = build_loss(config["loss"]).to(self.device)

        # Mixed precision setup
        mp_config = config["training"].get("mixed_precision", "none")
        self.use_amp = mp_config != "none" and self.device.type == "cuda"
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if mp_config == "bf16" else torch.float16
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=(mp_config == "fp16"))
        else:
            self.amp_dtype = torch.float32
            self.grad_scaler = torch.amp.GradScaler("cuda", enabled=False)

        # Optimizer with separate LR groups
        train_config = config["training"]
        opt_config = train_config["optimizer"]
        y_lr_mult = config["model"]["y_encoder"].get("lr_multiplier", 0.05)

        param_groups = self.model.get_param_groups(
            base_lr=opt_config["lr"],
            y_encoder_lr_mult=y_lr_mult,
        )
        self.optimizer = AdamW(
            param_groups,
            weight_decay=opt_config["weight_decay"],
            betas=tuple(opt_config.get("betas", [0.9, 0.95])),
        )

        # Scheduler
        sched_config = train_config["scheduler"]
        if sched_config["name"] == "constant":
            warmup_steps = sched_config.get("warmup_steps", 2000)
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min(1.0, step / max(warmup_steps, 1)),
            )
        elif sched_config["name"] == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=train_config["epochs"],
                eta_min=sched_config.get("min_lr", 1e-6),
            )
        elif sched_config["name"] == "warmup_constant_decay":
            # V-JEPA 2 schedule: warmup → constant → linear decay
            warmup_steps = sched_config.get("warmup_steps", 12000)
            constant_steps = sched_config.get("constant_steps", 228000)
            decay_steps = sched_config.get("decay_steps", 12000)

            def _wcd_lambda(step: int) -> float:
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                elif step < warmup_steps + constant_steps:
                    return 1.0
                else:
                    progress = (step - warmup_steps - constant_steps) / max(decay_steps, 1)
                    return max(0.0, 1.0 - progress)

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=_wcd_lambda)
        else:
            raise ValueError(f"Unknown scheduler: {sched_config['name']}")

        # Logging
        log_config = config["logging"]
        self.output_dir = Path(log_config["output_dir"])
        self.logger = setup_logger(self.output_dir)
        self.metrics_logger = MetricsLogger(self.output_dir, log_config.get("wandb"))

        self.logger.info(f"VL-JEPA model built successfully")
        self.logger.info(f"  Total params: {self.model.total_params:,}")
        self.logger.info(f"  Trainable params: {self.model.trainable_params:,}")
        self.logger.info(f"  Frozen params: {self.model.total_params - self.model.trainable_params:,}")
        self.logger.info(f"  Predictor LR: {opt_config['lr']}")
        self.logger.info(f"  Y-Encoder LR: {opt_config['lr'] * y_lr_mult}")
        self.logger.info(f"  Device: {self.device}")

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Single training step with mixed precision support."""
        images = batch["image"].to(self.device)
        text_ids = batch["text_ids"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)

        # Forward pass (query = first half of text, target = full text)
        # In practice, query/target split depends on the task
        mid = text_ids.shape[1] // 2
        query_ids = text_ids[:, :mid]
        target_ids = text_ids
        target_mask = text_mask

        with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            outputs = self.model.forward_train(
                images=images,
                query_ids=query_ids,
                query_mask=text_mask[:, :mid],
                target_ids=target_ids,
                target_mask=target_mask,
            )

            # Compute loss
            loss_dict = self.criterion(
                outputs["predicted_embedding"],
                outputs["target_embedding"],
            )

        # Backward with gradient scaling
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss_dict["loss"]).backward()

        self.grad_scaler.unscale_(self.optimizer)
        grad_clip = self.config["training"].get("gradient_clip", 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            grad_clip,
        )

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.scheduler.step()

        return {
            "loss": loss_dict["loss"].item(),
            "loss_v2t": loss_dict["loss_v2t"].item(),
            "loss_t2v": loss_dict["loss_t2v"].item(),
            "acc_v2t": loss_dict["accuracy_v2t"].item(),
            "acc_t2v": loss_dict["accuracy_t2v"].item(),
            "temperature": loss_dict["temperature"].item(),
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr_predictor": self.optimizer.param_groups[0]["lr"],
            "lr_y_encoder": self.optimizer.param_groups[1]["lr"],
        }

    def train(self) -> None:
        """Full training loop."""
        train_config = self.config["training"]
        log_config = self.config["logging"]

        # Build dataset (dummy for smoke test)
        img_size = self.config["model"]["x_encoder"].get("img_size", 224)
        dataset = DummyImageTextDataset(size=1000, img_size=img_size)
        dataloader = build_dataloader(
            dataset, batch_size=train_config["batch_size"], is_train=True
        )

        self.logger.info(f"Starting VL-JEPA training ({train_config.get('stage', 'pretrain')})")
        self.logger.info(f"  Epochs: {train_config['epochs']}")
        self.logger.info(f"  Batch size: {train_config['batch_size']}")

        global_step = 0
        best_loss = float("inf")

        for epoch in range(train_config["epochs"]):
            self.model.predictor.train()
            self.model.y_encoder.train()

            for batch_idx, batch in enumerate(dataloader):
                metrics = self.train_step(batch)
                global_step += 1

                if global_step % log_config.get("log_every", 5) == 0:
                    self.metrics_logger.log(metrics, step=global_step)
                    if is_main_process():
                        self.logger.info(
                            f"Epoch [{epoch+1}/{train_config['epochs']}] "
                            f"Step [{batch_idx+1}/{len(dataloader)}] "
                            f"Loss: {metrics['loss']:.4f} "
                            f"V2T Acc: {metrics['acc_v2t']:.2%} "
                            f"T2V Acc: {metrics['acc_t2v']:.2%}"
                        )

            # Save checkpoint
            avg_loss = metrics["loss"]  # Simplified
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss

            if (epoch + 1) % log_config.get("save_every", 10) == 0 or is_best:
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    step=global_step,
                    config=self.config,
                    output_dir=self.output_dir,
                    metrics={"loss": avg_loss},
                    is_best=is_best,
                )

        self.logger.info(f"VL-JEPA training complete. Best loss: {best_loss:.4f}")
        self.metrics_logger.finish()


def main():
    parser = argparse.ArgumentParser(description="VL-JEPA Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = VLJEPATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
