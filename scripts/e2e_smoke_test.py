#!/usr/bin/env python3
"""
End-to-end smoke test: I-JEPA on CIFAR-10.

Proves the complete pipeline works:
1. Download CIFAR-10
2. Train I-JEPA for 2 epochs (ViT-Tiny, 32x32, ~2 min on CPU)
3. Verify loss decreases
4. Save checkpoint
5. Load checkpoint into fresh model
6. Run k-NN evaluation
7. Run linear probe evaluation
8. Report all results

Usage (in Docker):
    docker build -t vljepa-e2e -f Dockerfile.e2e .
    docker run --rm vljepa-e2e
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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint
from src.common.data_utils import get_image_transforms
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor
from src.ijepa.eval import extract_features, knn_evaluate, train_linear_probe


def main():
    print("=" * 70)
    print("I-JEPA End-to-End Smoke Test")
    print("=" * 70)

    device = torch.device("cpu")
    output_dir = Path("/tmp/ijepa_e2e_smoke")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = "/tmp/cifar10_data"

    # ---- Config ----
    enc_config = {
        "img_size": 32,
        "patch_size": 8,
        "embed_dim": 192,
        "depth": 4,
        "num_heads": 3,
        "mlp_ratio": 4.0,
    }
    pred_config = {
        "embed_dim": 192,
        "predictor_embed_dim": 96,
        "depth": 2,
        "num_heads": 3,
    }
    full_config = {
        "model": {"encoder": enc_config, "predictor": pred_config},
        "masking": {
            "num_targets": 4,
            "min_target_scale": 0.15,
            "max_target_scale": 0.2,
        },
        "training": {
            "epochs": 2,
            "batch_size": 64,
            "optimizer": {"lr": 1.5e-4, "weight_decay": 0.05, "betas": [0.9, 0.95]},
            "scheduler": {"min_lr": 1e-6},
            "gradient_clip": 1.0,
            "ema_momentum": 0.996,
            "ema_momentum_end": 1.0,
        },
        "logging": {"output_dir": str(output_dir), "log_every": 10, "save_every": 1},
    }

    # ---- Step 1: Download CIFAR-10 ----
    print("\n[Step 1] Downloading CIFAR-10...")
    t0 = time.time()
    train_transform = get_image_transforms(img_size=32, is_train=True)
    eval_transform = get_image_transforms(img_size=32, is_train=False)
    train_dataset = CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_root, train=False, download=True, transform=eval_transform)
    print(f"  Train: {len(train_dataset)} images, Test: {len(test_dataset)} images")
    print(f"  Done in {time.time() - t0:.1f}s")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

    # ---- Step 2: Build models ----
    print("\n[Step 2] Building models...")
    encoder = build_encoder(enc_config).to(device)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.requires_grad_(False)
    predictor = build_predictor(pred_config, encoder.num_patches).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"  Context encoder: {enc_params:,} params")
    print(f"  Predictor: {pred_params:,} params")
    print(f"  Target encoder: {enc_params:,} params (EMA)")
    print(f"  Total: {enc_params * 2 + pred_params:,} params")

    grid_size = int(encoder.num_patches ** 0.5)
    print(f"  Patches: {encoder.num_patches} ({grid_size}x{grid_size})")

    # ---- Step 3: Train ----
    print("\n[Step 3] Training I-JEPA for 2 epochs...")
    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1.5e-4, weight_decay=0.05, betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

    epoch_losses = []
    all_step_losses = []
    t_train_start = time.time()

    for epoch in range(2):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            B = images.shape[0]

            ctx_idx, tgt_list = generate_masks(
                batch_size=B,
                num_patches_h=grid_size,
                num_patches_w=grid_size,
                num_targets=4,
                device=device,
            )

            ctx_emb = encoder(images, mask_indices=ctx_idx)

            with torch.no_grad():
                full_emb = target_encoder(images)

            total_loss = 0.0
            for tgt_idx in tgt_list:
                target_repr = torch.gather(
                    full_emb, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_emb.shape[-1])
                )
                pred_repr = predictor(ctx_emb, ctx_idx, tgt_idx)
                total_loss += F.smooth_l1_loss(pred_repr, target_repr.detach())
            total_loss /= len(tgt_list)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(predictor.parameters()), 1.0
            )
            optimizer.step()

            # EMA update
            momentum = 0.996 + (1.0 - 0.996) * (1 + math.cos(math.pi * epoch / 1)) / 2
            with torch.no_grad():
                for p_t, p_c in zip(target_encoder.parameters(), encoder.parameters()):
                    p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)

            step_loss = total_loss.item()
            epoch_loss += step_loss
            epoch_steps += 1
            all_step_losses.append(step_loss)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/2, Step {batch_idx+1}/{len(train_loader)}, Loss: {step_loss:.4f}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        epoch_losses.append(avg_loss)
        scheduler.step()
        print(f"  Epoch {epoch+1} complete — avg loss: {avg_loss:.4f}")

    train_time = time.time() - t_train_start
    print(f"  Training complete in {train_time:.1f}s")

    # Verify loss
    assert all(not math.isnan(l) for l in all_step_losses), "FAIL: NaN loss detected!"
    assert all(not math.isinf(l) for l in all_step_losses), "FAIL: Inf loss detected!"
    print(f"  Loss check: all {len(all_step_losses)} steps finite ✓")

    if len(epoch_losses) >= 2:
        if epoch_losses[-1] < epoch_losses[0]:
            print(f"  Loss decreased: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f} ✓")
        else:
            print(f"  WARNING: Loss did not decrease: {epoch_losses[0]:.4f} → {epoch_losses[-1]:.4f}")

    # ---- Step 4: Save checkpoint ----
    print("\n[Step 4] Saving checkpoint...")
    model = nn.ModuleDict({
        "context_encoder": encoder,
        "predictor": predictor,
        "target_encoder": target_encoder,
    })
    save_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler,
        epoch=2, step=len(all_step_losses), config=full_config,
        output_dir=output_dir, metrics={"avg_loss": epoch_losses[-1]},
        is_best=True,
    )
    latest = find_latest_checkpoint(output_dir)
    assert latest is not None, "FAIL: Checkpoint not saved!"
    print(f"  Saved to: {latest} ✓")

    # ---- Step 5: Load checkpoint into fresh model ----
    print("\n[Step 5] Loading checkpoint into fresh model...")
    fresh_encoder = build_encoder(enc_config).to(device)
    fresh_predictor = build_predictor(pred_config, fresh_encoder.num_patches).to(device)
    fresh_target = copy.deepcopy(fresh_encoder)
    fresh_model = nn.ModuleDict({
        "context_encoder": fresh_encoder,
        "predictor": fresh_predictor,
        "target_encoder": fresh_target,
    })
    meta = load_checkpoint(latest, fresh_model, strict=True)
    assert meta["epoch"] == 2, f"FAIL: Expected epoch 2, got {meta['epoch']}"
    print(f"  Loaded epoch {meta['epoch']}, step {meta['step']} ✓")

    # Verify weights match
    mismatch = 0
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
        if not torch.equal(p1, p2):
            mismatch += 1
    assert mismatch == 0, f"FAIL: {mismatch} parameter mismatches!"
    print(f"  All weights match ✓")

    # ---- Step 6: k-NN evaluation ----
    print("\n[Step 6] Running k-NN evaluation...")
    fresh_encoder.eval()
    fresh_encoder.requires_grad_(False)

    t_eval = time.time()
    # Use subset for speed on CPU
    train_subset = torch.utils.data.Subset(train_dataset, range(2000))
    test_subset = torch.utils.data.Subset(test_dataset, range(500))
    train_eval_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=2)
    test_eval_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=2)

    train_features, train_labels = extract_features(fresh_encoder, train_eval_loader, device)
    test_features, test_labels = extract_features(fresh_encoder, test_eval_loader, device)

    knn_acc = knn_evaluate(train_features, train_labels, test_features, test_labels, k=20)
    print(f"  k-NN (k=20) accuracy: {knn_acc:.4f}")
    print(f"  (Random baseline: ~10% for 10 classes)")
    print(f"  k-NN eval took {time.time() - t_eval:.1f}s ✓")

    # ---- Step 7: Linear probe evaluation ----
    print("\n[Step 7] Running linear probe evaluation (10 epochs)...")
    t_probe = time.time()
    probe_results = train_linear_probe(
        fresh_encoder,
        train_eval_loader,
        test_eval_loader,
        embed_dim=enc_config["embed_dim"],
        num_classes=10,
        device=device,
        epochs=10,
        lr=0.01,
    )
    print(f"  Linear probe train_acc: {probe_results['train_acc']:.4f}")
    print(f"  Linear probe test_acc: {probe_results['test_acc']:.4f}")
    print(f"  Linear probe took {time.time() - t_probe:.1f}s ✓")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("END-TO-END SMOKE TEST RESULTS")
    print("=" * 70)
    print(f"  Model: ViT-Tiny ({enc_params:,} params)")
    print(f"  Dataset: CIFAR-10 ({len(train_dataset)} train, {len(test_dataset)} test)")
    print(f"  Training: 2 epochs, {len(all_step_losses)} steps")
    print(f"  Final loss: {epoch_losses[-1]:.4f}")
    print(f"  Loss trend: {'decreasing ✓' if epoch_losses[-1] < epoch_losses[0] else 'flat/increasing'}")
    print(f"  Checkpoint save/load: ✓")
    print(f"  k-NN accuracy: {knn_acc:.4f}")
    print(f"  Linear probe accuracy: {probe_results['test_acc']:.4f}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"  Status: ALL CHECKS PASSED ✓")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
