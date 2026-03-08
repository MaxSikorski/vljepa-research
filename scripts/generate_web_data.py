#!/usr/bin/env python3
"""
Generate all precomputed data for the SALT interactive showcase website.

Outputs (written to /app/web_output/):
  1. salt-student.onnx        — Exported tiny ViT student model
  2. training-data.json        — Per-step loss curves for both SALT stages
  3. cifar10-samples.json      — 100 sample images (10 per class) as base64
  4. embeddings.json            — 2000 embeddings + labels for k-NN demo

This script runs the full SALT pipeline (Stage 1 + Stage 2) at small scale,
saves the student checkpoint, exports to ONNX, and generates all web assets.

Usage (in Docker):
    docker build -t vljepa-webdata -f Dockerfile.webdata .
    docker run --rm -v $(pwd)/docs:/app/web_output vljepa-webdata
"""

import base64
import copy
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image

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

OUTPUT_DIR = Path("/app/web_output")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Same configs as salt_e2e_smoke_test.py
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


def generate_cifar10_samples(data_root: str, num_per_class: int = 10) -> dict:
    """Extract sample images as base64 PNG strings, organized by class."""
    dataset = CIFAR10(data_root, train=False, download=True, transform=None)
    samples = {cls: [] for cls in CIFAR10_CLASSES}
    counts = {cls: 0 for cls in CIFAR10_CLASSES}

    for img, label in dataset:
        cls_name = CIFAR10_CLASSES[label]
        if counts[cls_name] < num_per_class:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            samples[cls_name].append({
                "base64": b64,
                "label": label,
                "class": cls_name,
            })
            counts[cls_name] += 1
        if all(c >= num_per_class for c in counts.values()):
            break

    return samples


def train_salt_with_logging(train_loader, eval_train, eval_test, device):
    """Run SALT pipeline, logging per-step losses for charts."""
    grid_size = 4
    num_patches = 16

    # ---- Stage 1: MAE Teacher ----
    print("[Stage 1] Training MAE teacher...")
    teacher = build_encoder(ENC_CONFIG).to(device)
    decoder = build_mae_decoder(DEC_CONFIG, num_patches).to(device)
    pixel_loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=True)

    opt1 = AdamW(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=1.5e-4, weight_decay=0.04,
    )
    sched1 = CosineAnnealingLR(opt1, T_max=10, eta_min=1e-6)

    trainer1 = SALTStage1Trainer(teacher, decoder, pixel_loss_fn, opt1, sched1, device)

    stage1_steps = []
    global_step = 0
    for epoch in range(10):
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            loss = trainer1.train_step(images, grid_size)
            global_step += 1
            if global_step % 10 == 0:  # Log every 10 steps
                stage1_steps.append({"step": global_step, "loss": round(loss, 4)})
            if (batch_idx + 1) % 100 == 0:
                print(f"  Stage 1 Epoch {epoch+1}, Step {batch_idx+1}, Loss: {loss:.4f}")
        sched1.step()

    print(f"  Stage 1 complete. Final loss: {stage1_steps[-1]['loss']}")

    # ---- Stage 2: Frozen Teacher JEPA ----
    print("[Stage 2] Training student with frozen teacher...")
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
    sched2 = CosineAnnealingLR(opt2, T_max=10, eta_min=1e-6)

    trainer2 = SALTStage2Trainer(frozen_teacher, student, predictor, opt2, sched2, device)

    stage2_steps = []
    global_step = 0
    for epoch in range(10):
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            loss = trainer2.train_step(images, grid_size)
            global_step += 1
            if global_step % 10 == 0:
                stage2_steps.append({"step": global_step, "loss": round(loss, 4)})
            if (batch_idx + 1) % 100 == 0:
                print(f"  Stage 2 Epoch {epoch+1}, Step {batch_idx+1}, Loss: {loss:.4f}")
        sched2.step()

    print(f"  Stage 2 complete. Final loss: {stage2_steps[-1]['loss']}")

    # ---- Evaluate ----
    print("[Eval] Evaluating student encoder...")
    student.eval()
    student.requires_grad_(False)

    train_feats, train_labels = extract_features(student, eval_train, device)
    test_feats, test_labels = extract_features(student, eval_test, device)

    salt_knn = knn_evaluate(train_feats, train_labels, test_feats, test_labels, k=20)
    salt_probe = train_linear_probe(
        student, eval_train, eval_test,
        embed_dim=192, num_classes=10, device=device, epochs=10, lr=0.01,
    )
    print(f"  SALT k-NN: {salt_knn:.4f}, Linear probe: {salt_probe['test_acc']:.4f}")

    # ---- I-JEPA baseline ----
    print("[Baseline] Running I-JEPA comparison...")
    ijepa_encoder = build_encoder(ENC_CONFIG).to(device)
    target_encoder = copy.deepcopy(ijepa_encoder)
    target_encoder.requires_grad_(False)
    ijepa_predictor = build_predictor(PRED_CONFIG, num_patches).to(device)

    ijepa_opt = AdamW(
        list(ijepa_encoder.parameters()) + list(ijepa_predictor.parameters()),
        lr=1.5e-4, weight_decay=0.05,
    )

    ijepa_steps = []
    global_step = 0
    for epoch in range(10):
        ijepa_encoder.train()
        ijepa_predictor.train()
        for images, _ in train_loader:
            images = images.to(device)
            B = images.shape[0]
            ctx_idx, tgt_list = generate_masks(
                batch_size=B, num_patches_h=grid_size, num_patches_w=grid_size,
                num_targets=4, device=device,
            )
            ctx_emb = ijepa_encoder(images, mask_indices=ctx_idx)
            with torch.no_grad():
                full_emb = target_encoder(images)
            loss = 0.0
            for tgt_idx in tgt_list:
                target = torch.gather(
                    full_emb, 1,
                    tgt_idx.unsqueeze(-1).expand(-1, -1, full_emb.shape[-1]),
                )
                pred = ijepa_predictor(ctx_emb, ctx_idx, tgt_idx)
                loss += F.smooth_l1_loss(pred, target.detach())
            loss /= len(tgt_list)
            ijepa_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ijepa_encoder.parameters()) + list(ijepa_predictor.parameters()), 1.0
            )
            ijepa_opt.step()
            with torch.no_grad():
                for p_t, p_c in zip(target_encoder.parameters(), ijepa_encoder.parameters()):
                    p_t.data.mul_(0.996).add_(p_c.data, alpha=0.004)
            global_step += 1
            if global_step % 10 == 0:
                ijepa_steps.append({"step": global_step, "loss": round(loss.item(), 4)})

    ijepa_encoder.eval()
    ijepa_encoder.requires_grad_(False)
    ijepa_train_feats, ijepa_train_labels = extract_features(ijepa_encoder, eval_train, device)
    ijepa_test_feats, ijepa_test_labels = extract_features(ijepa_encoder, eval_test, device)
    ijepa_knn = knn_evaluate(ijepa_train_feats, ijepa_train_labels, ijepa_test_feats, ijepa_test_labels, k=20)
    ijepa_probe = train_linear_probe(
        ijepa_encoder, eval_train, eval_test,
        embed_dim=192, num_classes=10, device=device, epochs=10, lr=0.01,
    )
    print(f"  I-JEPA k-NN: {ijepa_knn:.4f}, Linear probe: {ijepa_probe['test_acc']:.4f}")

    training_data = {
        "salt": {
            "stage1": stage1_steps,
            "stage2": stage2_steps,
            "knn": round(salt_knn, 4),
            "linear_probe": round(salt_probe["test_acc"], 4),
        },
        "ijepa": {
            "steps": ijepa_steps,
            "knn": round(ijepa_knn, 4),
            "linear_probe": round(ijepa_probe["test_acc"], 4),
        },
    }

    return student, train_feats, train_labels, training_data


def generate_embeddings(student, eval_loader, device, max_samples=2000):
    """Generate embeddings for k-NN demo in browser."""
    student.eval()
    all_embeddings = []
    all_labels = []
    count = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            if count >= max_samples:
                break
            images = images.to(device)
            feats = student(images).mean(dim=1)  # Global avg pool
            feats = F.normalize(feats, dim=1)
            all_embeddings.append(feats.cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            count += images.shape[0]

    embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
    labels = all_labels[:max_samples]

    return {
        "embeddings": embeddings.tolist(),
        "labels": labels,
        "classes": CIFAR10_CLASSES,
        "embed_dim": embeddings.shape[1],
        "count": len(labels),
    }


def main():
    print("=" * 60)
    print("SALT Web Data Generator")
    print("=" * 60)

    device = torch.device("cpu")
    data_root = "/tmp/cifar10_data"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "data").mkdir(exist_ok=True)

    # ---- 1. CIFAR-10 samples ----
    print("\n[1/4] Generating CIFAR-10 sample images...")
    samples = generate_cifar10_samples(data_root, num_per_class=10)
    total_samples = sum(len(v) for v in samples.values())
    with open(OUTPUT_DIR / "data" / "cifar10-samples.json", "w") as f:
        json.dump(samples, f)
    print(f"  Saved {total_samples} samples")

    # ---- 2. Train SALT + I-JEPA, collect losses ----
    print("\n[2/4] Training SALT pipeline + I-JEPA baseline...")
    train_transform = get_image_transforms(img_size=32, is_train=True)
    eval_transform = get_image_transforms(img_size=32, is_train=False)
    train_dataset = CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_root, train=False, download=True, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    eval_train = DataLoader(Subset(train_dataset, range(2000)), batch_size=256, num_workers=2)
    eval_test = DataLoader(Subset(test_dataset, range(500)), batch_size=256, num_workers=2)

    student, train_feats, train_labels, training_data = train_salt_with_logging(
        train_loader, eval_train, eval_test, device
    )

    with open(OUTPUT_DIR / "data" / "training-data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    print("  Saved training data")

    # ---- 3. Save student checkpoint + export ONNX ----
    print("\n[3/4] Exporting student to ONNX...")
    ckpt_path = "/tmp/salt_student.pt"
    torch.save(student.state_dict(), ckpt_path)

    from scripts.export_onnx import export_to_onnx
    export_to_onnx(
        checkpoint_path=ckpt_path,
        output_path=str(OUTPUT_DIR / "models" / "salt-student.onnx"),
        img_size=32, patch_size=8, embed_dim=192, depth=4, num_heads=3,
        fp16=False,  # Keep fp32 for browser compatibility
    )

    # ---- 4. Embeddings for k-NN demo ----
    print("\n[4/4] Generating embeddings for k-NN demo...")
    embed_loader = DataLoader(
        Subset(CIFAR10(data_root, train=True, download=False, transform=eval_transform), range(2000)),
        batch_size=256, num_workers=2,
    )
    embeddings = generate_embeddings(student, embed_loader, device, max_samples=2000)
    with open(OUTPUT_DIR / "data" / "embeddings.json", "w") as f:
        json.dump(embeddings, f)
    print(f"  Saved {embeddings['count']} embeddings")

    print("\n" + "=" * 60)
    print("All web data generated successfully!")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
