#!/usr/bin/env python3
"""
Quick ONNX export + embeddings generation.
Trains a minimal student (1 epoch) just to get weights, exports to ONNX,
and generates embeddings for the k-NN demo.

This is a fast version (~10 min) that skips the full comparison pipeline.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
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
from src.ijepa.eval import extract_features
from src.salt.models.mae_decoder import build_mae_decoder
from src.salt.losses.pixel_loss import PixelReconstructionLoss
from src.salt.train_stage1 import SALTStage1Trainer
from src.salt.train_stage2 import SALTStage2Trainer

OUTPUT_DIR = Path("/app/web_output")
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

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


def main():
    print("=" * 60)
    print("ONNX Export + Embeddings Generator")
    print("=" * 60)

    device = torch.device("cpu")
    data_root = "/tmp/cifar10_data"
    grid_size = 4
    num_patches = 16

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)
    (OUTPUT_DIR / "data").mkdir(exist_ok=True)

    # Data
    train_transform = get_image_transforms(img_size=32, is_train=True)
    eval_transform = get_image_transforms(img_size=32, is_train=False)
    train_dataset = CIFAR10(data_root, train=True, download=True, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=True)

    # Stage 1: Quick MAE teacher (1 epoch)
    print("\n[1/4] Quick Stage 1 training (1 epoch)...")
    teacher = build_encoder(ENC_CONFIG).to(device)
    decoder = build_mae_decoder(DEC_CONFIG, num_patches).to(device)
    pixel_loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=True)

    opt1 = AdamW(
        list(teacher.parameters()) + list(decoder.parameters()),
        lr=1.5e-4, weight_decay=0.04,
    )
    trainer1 = SALTStage1Trainer(teacher, decoder, pixel_loss_fn, opt1, None, device)
    avg = trainer1.train_epoch(train_loader, grid_size, 0, log_every=200)
    print(f"  Stage 1 loss: {avg:.4f}")

    # Stage 2: Quick student training (1 epoch)
    print("\n[2/4] Quick Stage 2 training (1 epoch)...")
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
    trainer2 = SALTStage2Trainer(frozen_teacher, student, predictor, opt2, None, device)
    avg = trainer2.train_epoch(train_loader, grid_size, 0, log_every=200)
    print(f"  Stage 2 loss: {avg:.4f}")

    # Export ONNX
    print("\n[3/4] Exporting student to ONNX...")
    ckpt_path = "/tmp/salt_student.pt"
    torch.save(student.state_dict(), ckpt_path)

    from scripts.export_onnx import export_to_onnx
    export_to_onnx(
        checkpoint_path=ckpt_path,
        output_path=str(OUTPUT_DIR / "models" / "salt-student.onnx"),
        img_size=32, patch_size=8, embed_dim=192, depth=4, num_heads=3,
        fp16=False,
    )

    # Generate embeddings
    print("\n[4/4] Generating embeddings...")
    student.eval()
    student.requires_grad_(False)
    embed_dataset = CIFAR10(data_root, train=True, download=False, transform=eval_transform)
    embed_loader = DataLoader(Subset(embed_dataset, range(2000)), batch_size=256, num_workers=2)

    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in embed_loader:
            feats = student(images).mean(dim=1)
            feats = F.normalize(feats, dim=1)
            all_embeddings.append(feats.numpy())
            all_labels.extend(labels.numpy().tolist())

    embeddings = np.concatenate(all_embeddings, axis=0)
    embed_data = {
        "embeddings": embeddings.tolist(),
        "labels": all_labels,
        "classes": CIFAR10_CLASSES,
        "embed_dim": embeddings.shape[1],
        "count": len(all_labels),
    }
    with open(OUTPUT_DIR / "data" / "embeddings.json", "w") as f:
        json.dump(embed_data, f)
    print(f"  Saved {len(all_labels)} embeddings")

    print("\n" + "=" * 60)
    print("Done! ONNX model + embeddings generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
