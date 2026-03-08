"""
V-JEPA training loop.

Extends I-JEPA training to video with:
- Spatiotemporal masking (tube masks)
- Video data loading and frame sampling
- Same EMA target encoder pattern
- Prediction in latent space over space AND time
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from src.common.checkpointing import save_checkpoint
from src.common.config import load_config
from src.common.data_utils import DummyVideoTextDataset, build_dataloader
from src.common.distributed import get_device, is_main_process
from src.common.logging import MetricsLogger, get_logger, setup_logger
from src.ijepa.models.predictor import JEPAPredictor
from src.vjepa.masks.spatiotemporal import generate_tube_masks
from src.vjepa.models.video_encoder import VideoVisionTransformer, build_video_encoder


class VJEPATrainer:
    """V-JEPA training orchestrator."""

    def __init__(self, config: dict):
        self.config = config
        self.device = get_device()

        enc_config = config["model"]["encoder"]
        pred_config = config["model"]["predictor"]

        # Build video encoder
        self.context_encoder = build_video_encoder(enc_config).to(self.device)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.target_encoder.requires_grad_(False)

        # Predictor
        num_patches = self.context_encoder.num_patches
        self.predictor = JEPAPredictor(
            num_patches=num_patches,
            encoder_embed_dim=enc_config["embed_dim"],
            predictor_embed_dim=pred_config.get("predictor_embed_dim", enc_config["embed_dim"] // 2),
            depth=pred_config.get("depth", 6),
            num_heads=pred_config.get("num_heads", 6),
        ).to(self.device)

        # Grid info
        self.num_temporal = self.context_encoder.num_temporal_patches
        self.num_spatial_h = self.context_encoder.num_spatial_patches_h
        self.num_spatial_w = self.context_encoder.num_spatial_patches_w

        # Optimizer
        train_config = config["training"]
        opt_config = train_config["optimizer"]
        self.optimizer = AdamW(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
        )

        # Logging
        log_config = config["logging"]
        self.output_dir = log_config["output_dir"]
        self.logger = setup_logger(self.output_dir)
        self.metrics_logger = MetricsLogger(self.output_dir, log_config.get("wandb"))

        ctx_params = sum(p.numel() for p in self.context_encoder.parameters())
        pred_params = sum(p.numel() for p in self.predictor.parameters())
        self.logger.info(f"V-JEPA Video Encoder params: {ctx_params:,}")
        self.logger.info(f"V-JEPA Predictor params: {pred_params:,}")

    @torch.no_grad()
    def _update_target_encoder(self, momentum: float) -> None:
        for p_t, p_c in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)

    def train_step(self, video: torch.Tensor) -> dict[str, float]:
        B = video.shape[0]
        mask_config = self.config["masking"]

        ctx_idx, tgt_list = generate_tube_masks(
            batch_size=B,
            num_temporal=self.num_temporal,
            num_spatial_h=self.num_spatial_h,
            num_spatial_w=self.num_spatial_w,
            num_targets=mask_config.get("num_targets", 8),
            device=self.device,
        )

        # Context encoder on masked input
        ctx_emb = self.context_encoder(video, mask_indices=ctx_idx)

        # Target encoder on full video
        with torch.no_grad():
            full_emb = self.target_encoder(video)

        total_loss = 0.0
        for tgt_idx in tgt_list:
            target_repr = torch.gather(
                full_emb, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_emb.shape[-1])
            )
            pred_repr = self.predictor(ctx_emb, ctx_idx, tgt_idx)
            total_loss += F.l1_loss(pred_repr, target_repr.detach())

        total_loss /= len(tgt_list)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()),
            self.config["training"].get("gradient_clip", 1.0),
        )
        self.optimizer.step()
        self._update_target_encoder(self.config["training"].get("ema_momentum", 0.996))

        return {"loss": total_loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def train(self) -> None:
        train_config = self.config["training"]
        data_config = self.config["data"]

        # Use dummy dataset for smoke testing
        dataset = DummyVideoTextDataset(
            size=500,
            num_frames=data_config.get("num_frames", 16),
            img_size=data_config.get("crop_size", 64),
        )
        dataloader = build_dataloader(dataset, batch_size=train_config["batch_size"], is_train=True)

        self.logger.info(f"Starting V-JEPA training for {train_config['epochs']} epochs")

        for epoch in range(train_config["epochs"]):
            self.context_encoder.train()
            self.predictor.train()

            for batch_idx, batch in enumerate(dataloader):
                video = batch["video"].to(self.device)
                metrics = self.train_step(video)

                if (batch_idx + 1) % self.config["logging"].get("log_every", 10) == 0:
                    self.logger.info(
                        f"Epoch [{epoch+1}] Step [{batch_idx+1}] Loss: {metrics['loss']:.4f}"
                    )

        self.logger.info("V-JEPA training complete")
        self.metrics_logger.finish()


def main():
    parser = argparse.ArgumentParser(description="V-JEPA Training")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    trainer = VJEPATrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
