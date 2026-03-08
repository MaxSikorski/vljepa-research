#!/usr/bin/env python3
"""
SALT End-to-End Smoke Test on CIFAR-10.

Implements the full SALT two-stage pipeline (Apple, ICLR 2025):
  Stage 1: Train MAE teacher with pixel reconstruction
  Stage 2: Freeze teacher, train student with JEPA latent prediction

Then evaluates the student encoder via k-NN and linear probe,
and compares against an I-JEPA baseline trained with the same
compute budget.

Reference: https://arxiv.org/abs/2509.24317

Usage (in Docker):
    docker build -t vljepa-salt -f Dockerfile.salt .
    docker run --rm vljepa-salt
"""

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.data_utils import get_image_transforms
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import build_encoder
from src.ijepa.models.predictor import build_predictor
from src.ijepa.eval import extract_features, knn_evaluate, train_linear_probe
from src.salt.models.mae_decoder import build_mae_decoder
from src.salt.losses.pixel_loss import PixelReconstructionLoss
from src.salt.train_stage1 import SALTStage1Trainer
from src.salt.train_stage2 import SALTStage2Trainer


# ---- Config ----
ENC_CONFIG = {
    "img_size": 32, "patch_size": 8, "embed_dim": 192,
    "depth": 4, "num_heads": 3, "mlp_ratio": 4.0,
}
DEC_CONFIG = {
    "encoder_embed_dim": 192, "decoder_embed_dim": 96,
    "decoder_depth": 2, "decoder_num_heads": 3,
    "patch_size": 8, "in_channels": 3,
}
PRED_CONFIG = {
    "embed_dim": 192, "predictor_embed_dim": 96,
    "depth": 2, "num_heads": 3,
}


def run_ijepa_baseline(train_loader, eval_train_loader, eval_test_loader, device, epochs=2):
    """Run I-JEPA baseline with same compute budget for comparison."""
    print("\n[Baseline] Training I-JEPA for comparison...")
    encoder = build_encoder(ENC_CONFIG).to(device)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.requires_grad_(False)
    predictor = build_predictor(PRED_CONFIG, encoder.num_patches).to(device)

    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1.5e-4, weight_decay=0.05,
    )
    grid_size = int(encoder.num_patches ** 0.5)

    t0 = time.time()
    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        for images, _ in train_loader:
            images = images.to(device)
            B = images.shape[0]
            ctx_idx, tgt_list = generate_masks(
                batch_size=B, num_patches_h=grid_size, num_patches_w=grid_size,
                num_targets=4, device=device,
            )
            ctx_emb = encoder(images, mask_indices=ctx_idx)
            with torch.no_grad():
                full_emb = target_encoder(images)
            loss = 0.0
            for tgt_idx in tgt_list:
                target = torch.gather(
                    full_emb, 1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, full_emb.shape[-1]),
                )
                pred = predictor(ctx_emb, ctx_idx, tgt_idx)
                loss += F.smooth_l1_loss(pred, target.detach())
            loss /= len(tgt_list)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()
            with torch.no_grad():
                for p_t, p_c in zip(target_encoder.parameters(), encoder.parameters()):
                    p_t.data.mul_(0.996).add_(p_c.data, alpha=0.004)

    baseline_time = time.time() - t0

    encoder.eval()
    encoder.requires_grad_(False)
    train_feats, train_labels = extract_features(encoder, eval_train_loader, device)
    test_feats, test_labels = extract_features(encoder, eval_test_loader, device)
    knn_acc = knn_evaluate(train_feats, train_labels, test_feats, test_labels, k=20)
    probe = train_linear_probe(
        encoder, eval_train_loader, eval_test_loader,
        embed_dim=192, num_classes=10, device=device, epochs=10, lr=0.01,
    )
    return knn_acc, probe["test_acc"], baseline_time


def main():
    print("=" * 70)
    print("SALT End-to-End Smoke Test (Apple, ICLR 2025)")
    print("Static-teacher Asymmetric Latent Training")
    print("=" * 70)

    device = torch.device("cpu")
    data_root = "/tmp/cifar10_data"
    t_start = time.time()

    # ---- Data ----
    print("\n[Step 1] Downloading CIFAR-10...")
    train_transform = get_image_transforms(img_size=32, is_train=True)
    eval_transform = get_image_transforms(img_size=32, is_train=False)
    train_dataset = CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_root, train=False, download=True, transform=eval_transform)
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    eval_train = DataLoader(Subset(train_dataset, range(2000)), batch_size=256, num_workers=2)
    eval_test = DataLoader(Subset(test_dataset, range(500)), batch_size=256, num_workers=2)

    grid_size = 4  # 32/8 = 4
    num_patches = grid_size * grid_size

    # ================================================================
    # SALT STAGE 1: V-Pixel Teacher Training
    # ================================================================
    print("\n[Step 2] SALT Stage 1: Training MAE Teacher...")
    print("  (NO EMA — pixel reconstruction with multi-block masking)")

    teacher = build_encoder(ENC_CONFIG).to(device)
    decoder = build_mae_decoder(DEC_CONFIG, num_patches).to(device)
    pixel_loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=True)

    opt1 = AdamW(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=1.5e-4, weight_decay=0.04,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=2, eta_min=1e-6)

    trainer1 = SALTStage1Trainer(
        teacher, decoder, pixel_loss_fn, opt1, sched1, device
    )

    stage1_losses = []
    t_s1 = time.time()
    for epoch in range(2):
        avg_loss = trainer1.train_epoch(train_loader, grid_size, epoch, log_every=100)
        stage1_losses.append(avg_loss)
        print(f"  Stage 1 Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    stage1_time = time.time() - t_s1
    print(f"  Stage 1 done in {stage1_time:.1f}s")
    print(f"  Pixel loss trend: {stage1_losses[0]:.4f} → {stage1_losses[-1]:.4f}")

    # ================================================================
    # SALT STAGE 2: Frozen-Teacher JEPA Student Training
    # ================================================================
    print("\n[Step 3] SALT Stage 2: Training Student with Frozen Teacher...")
    print("  (NO EMA — frozen teacher, L1 latent prediction)")

    # Freeze teacher — this is the SALT innovation
    frozen_teacher = copy.deepcopy(teacher)
    frozen_teacher.eval()
    for p in frozen_teacher.parameters():
        p.requires_grad = False

    student = build_encoder(ENC_CONFIG).to(device)
    predictor = build_predictor(PRED_CONFIG, num_patches).to(device)

    opt2 = AdamW(
        list(student.parameters()) + list(predictor.parameters()),
        lr=1.5e-4, weight_decay=0.04,
    )
    sched2 = CosineAnnealingLR(opt2, T_max=2, eta_min=1e-6)

    trainer2 = SALTStage2Trainer(
        frozen_teacher, student, predictor, opt2, sched2, device
    )

    stage2_losses = []
    t_s2 = time.time()
    for epoch in range(2):
        avg_loss = trainer2.train_epoch(train_loader, grid_size, epoch, log_every=100)
        stage2_losses.append(avg_loss)
        print(f"  Stage 2 Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    stage2_time = time.time() - t_s2
    print(f"  Stage 2 done in {stage2_time:.1f}s")
    print(f"  Latent loss trend: {stage2_losses[0]:.4f} → {stage2_losses[-1]:.4f}")

    # ================================================================
    # EVALUATION
    # ================================================================
    print("\n[Step 4] Evaluating SALT student encoder...")
    student.eval()
    student.requires_grad_(False)

    train_feats, train_labels = extract_features(student, eval_train, device)
    test_feats, test_labels = extract_features(student, eval_test, device)

    salt_knn = knn_evaluate(train_feats, train_labels, test_feats, test_labels, k=20)
    print(f"  SALT k-NN (k=20): {salt_knn:.4f}")

    salt_probe = train_linear_probe(
        student, eval_train, eval_test,
        embed_dim=192, num_classes=10, device=device, epochs=10, lr=0.01,
    )
    print(f"  SALT linear probe: {salt_probe['test_acc']:.4f}")

    # ================================================================
    # I-JEPA BASELINE COMPARISON
    # ================================================================
    print("\n[Step 5] Running I-JEPA baseline for comparison...")
    baseline_knn, baseline_probe, baseline_time = run_ijepa_baseline(
        train_loader, eval_train, eval_test, device, epochs=2
    )

    salt_total_time = stage1_time + stage2_time

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("SALT vs I-JEPA COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<25} {'SALT':>10} {'I-JEPA':>10} {'Winner':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    knn_winner = "SALT" if salt_knn > baseline_knn else "I-JEPA"
    probe_winner = "SALT" if salt_probe['test_acc'] > baseline_probe else "I-JEPA"
    time_winner = "SALT" if salt_total_time < baseline_time else "I-JEPA"

    print(f"  {'k-NN accuracy':<25} {salt_knn:>9.4f} {baseline_knn:>9.4f} {knn_winner:>10}")
    print(f"  {'Linear probe accuracy':<25} {salt_probe['test_acc']:>9.4f} {baseline_probe:>9.4f} {probe_winner:>10}")
    print(f"  {'Training time (s)':<25} {salt_total_time:>9.1f} {baseline_time:>9.1f} {time_winner:>10}")
    print(f"  {'Has EMA?':<25} {'No':>10} {'Yes':>10}")
    print(f"  {'Stage 1 time (s)':<25} {stage1_time:>9.1f} {'N/A':>10}")
    print(f"  {'Stage 2 time (s)':<25} {stage2_time:>9.1f} {'N/A':>10}")

    # Verify basic correctness
    print(f"\n  Stage 1 loss decreasing: {'YES' if stage1_losses[-1] < stage1_losses[0] else 'NO'}")
    print(f"  Stage 2 loss decreasing: {'YES' if stage2_losses[-1] < stage2_losses[0] else 'NO'}")
    print(f"  SALT above random (10%): {'YES' if salt_knn > 0.10 else 'NO'}")
    print(f"  Total time: {time.time() - t_start:.1f}s")

    all_ok = (
        stage1_losses[-1] < stage1_losses[0]
        and stage2_losses[-1] < stage2_losses[0]
        and salt_knn > 0.10
    )
    print(f"\n  Status: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
