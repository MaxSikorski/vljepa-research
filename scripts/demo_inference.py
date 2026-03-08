#!/usr/bin/env python3
"""
Inference demo: Load a trained I-JEPA model and extract visual features.

Demonstrates:
1. Load trained I-JEPA checkpoint
2. Extract features from CIFAR-10 images
3. Visualize feature similarity (nearest neighbors)
4. Show k-NN classification in action

Also includes a VL-JEPA inference demo (with dummy weights) showing the
full vision-language pipeline: image + text query -> predicted embedding.

Usage (in Docker):
    docker build -t vljepa-e2e -f Dockerfile.e2e .
    docker run --rm vljepa-e2e python scripts/demo_inference.py --checkpoint /path/to/checkpoint.pt

If no checkpoint is provided, trains a quick model first.
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.checkpointing import load_checkpoint
from src.common.data_utils import get_image_transforms
from src.ijepa.eval import extract_features, knn_evaluate
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import build_encoder
from src.ijepa.models.predictor import build_predictor


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def quick_train(encoder, predictor, target_encoder, train_loader, device, steps=200):
    """Quick training for demo when no checkpoint is available."""
    print(f"\n  Quick training for {steps} steps...")
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1.5e-4, weight_decay=0.05,
    )
    grid_size = int(encoder.num_patches ** 0.5)
    encoder.train()
    predictor.train()

    step = 0
    for images, _ in train_loader:
        if step >= steps:
            break
        images = images.to(device)
        B = images.shape[0]

        ctx_idx, tgt_list = generate_masks(
            batch_size=B, num_patches_h=grid_size, num_patches_w=grid_size,
            num_targets=4, device=device,
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

        with torch.no_grad():
            for p_t, p_c in zip(target_encoder.parameters(), encoder.parameters()):
                p_t.data.mul_(0.996).add_(p_c.data, alpha=0.004)

        step += 1
        if step % 50 == 0:
            print(f"    Step {step}/{steps}, Loss: {total_loss.item():.4f}")

    print(f"  Quick training done.")


def demo_feature_similarity(encoder, dataset, device, n_query=5, n_neighbors=5):
    """Show nearest-neighbor retrieval using learned features."""
    print("\n--- Feature Similarity Demo ---")

    # Extract features from a subset
    subset = Subset(dataset, range(1000))
    loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=0)
    features, labels = extract_features(encoder, loader, device)
    features = F.normalize(features, dim=1)

    # Pick query images
    query_indices = torch.randperm(1000)[:n_query]

    for i, qi in enumerate(query_indices):
        query_feat = features[qi:qi+1]
        query_label = labels[qi].item()

        # Compute similarity to all other images
        sim = (query_feat @ features.T).squeeze(0)
        sim[qi] = -1  # exclude self

        # Top neighbors
        topk_sim, topk_idx = sim.topk(n_neighbors)
        neighbor_labels = [labels[idx].item() for idx in topk_idx]
        neighbor_sims = [s.item() for s in topk_sim]

        print(f"\n  Query {i+1}: {CIFAR10_CLASSES[query_label]} (index {qi.item()})")
        for j, (nl, ns, ni) in enumerate(zip(neighbor_labels, neighbor_sims, topk_idx)):
            match = "MATCH" if nl == query_label else ""
            print(f"    Neighbor {j+1}: {CIFAR10_CLASSES[nl]:12s} (sim={ns:.3f}) {match}")

    # Compute class-level statistics
    print("\n  Class-level similarity statistics:")
    for cls_idx in range(10):
        cls_mask = labels == cls_idx
        if cls_mask.sum() < 2:
            continue
        cls_features = features[cls_mask]
        intra_sim = (cls_features @ cls_features.T).fill_diagonal_(0)
        mean_sim = intra_sim.sum() / (cls_mask.sum() * (cls_mask.sum() - 1))
        print(f"    {CIFAR10_CLASSES[cls_idx]:12s}: mean intra-class similarity = {mean_sim:.3f}")


def demo_knn_classification(encoder, train_dataset, test_dataset, device):
    """Demonstrate k-NN classification with learned features."""
    print("\n--- k-NN Classification Demo ---")

    train_sub = Subset(train_dataset, range(2000))
    test_sub = Subset(test_dataset, range(500))
    train_loader = DataLoader(train_sub, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_sub, batch_size=256, shuffle=False, num_workers=0)

    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)

    # Try different k values
    print("\n  k-NN accuracy for different k values:")
    for k in [1, 5, 10, 20]:
        acc = knn_evaluate(train_features, train_labels, test_features, test_labels, k=k)
        print(f"    k={k:3d}: accuracy = {acc:.4f} ({acc*100:.1f}%)")

    # Per-class accuracy with k=10
    train_features_norm = F.normalize(train_features, dim=1)
    test_features_norm = F.normalize(test_features, dim=1)
    sim = test_features_norm @ train_features_norm.T
    _, topk_indices = sim.topk(10, dim=1)
    topk_labels = train_labels[topk_indices]
    predictions = topk_labels.mode(dim=1).values

    print("\n  Per-class accuracy (k=10):")
    for cls_idx in range(10):
        cls_mask = test_labels == cls_idx
        if cls_mask.sum() == 0:
            continue
        cls_correct = (predictions[cls_mask] == cls_idx).float().mean().item()
        print(f"    {CIFAR10_CLASSES[cls_idx]:12s}: {cls_correct:.3f} ({cls_correct*100:.0f}%)")


def demo_vljepa_inference():
    """Demonstrate VL-JEPA inference pipeline with dummy weights."""
    print("\n--- VL-JEPA Inference Pipeline Demo (Dummy Weights) ---")

    from src.vljepa.models.vljepa import build_vljepa

    # All embed_dim values must match (64) so projections are compatible
    config = {
        "x_encoder": {
            "name": "vit_tiny",
            "img_size": 32,
            "patch_size": 8,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 2,
        },
        "predictor": {
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 2,
            "shared_embedding_dim": 64,
        },
        "y_encoder": {
            "name": "tiny",
            "embed_dim": 64,
            "vocab_size": 1000,
            "max_seq_len": 32,
            "depth": 2,
            "num_heads": 2,
        },
    }

    model = build_vljepa(config)
    model.eval()

    # Simulate inference
    dummy_image = torch.randn(2, 3, 32, 32)
    dummy_query = torch.randint(0, 1000, (2, 8))  # "What is in this image?"
    dummy_candidates = torch.randint(0, 1000, (5, 8))  # 5 candidate answers
    dummy_candidate_masks = torch.ones(5, 8, dtype=torch.bool)

    print("\n  1. Embedding prediction (fast, no decoder):")
    with torch.no_grad():
        embedding = model.forward_embed(dummy_image, dummy_query)
        print(f"     Input: image (2, 3, 32, 32) + query (2, 8)")
        print(f"     Output: predicted embedding {tuple(embedding.shape)}")
        print(f"     Embedding norm: {embedding.norm(dim=1).mean():.3f}")

    print("\n  2. Retrieval (rank candidates by similarity):")
    with torch.no_grad():
        scores = model.forward_retrieve(
            dummy_image, dummy_query, dummy_candidates, dummy_candidate_masks
        )
        print(f"     Input: image (2, 3, 32, 32) + query (2, 8) + 5 candidates")
        print(f"     Output: similarity scores {tuple(scores.shape)}")
        print(f"     Best candidate per image: {scores.argmax(dim=1).tolist()}")

    print("\n  3. Training forward pass:")
    dummy_target = torch.randint(0, 1000, (2, 8))
    dummy_target_mask = torch.ones(2, 8, dtype=torch.bool)
    dummy_query_mask = torch.ones(2, 8, dtype=torch.bool)
    result = model.forward_train(
        dummy_image, dummy_query, dummy_query_mask, dummy_target, dummy_target_mask
    )
    print(f"     Predicted embedding: {tuple(result['predicted_embedding'].shape)}")
    print(f"     Target embedding: {tuple(result['target_embedding'].shape)}")

    # Compute info-nce loss
    from src.vljepa.losses.infonce import BidirectionalInfoNCE
    loss_fn = BidirectionalInfoNCE(temperature=0.07)
    loss_dict = loss_fn(result["predicted_embedding"], result["target_embedding"])
    print(f"     Bi-directional InfoNCE loss: {loss_dict['loss'].item():.4f}")
    print(f"     V→T loss: {loss_dict['loss_v2t'].item():.4f}, T→V loss: {loss_dict['loss_t2v'].item():.4f}")

    print("\n  VL-JEPA pipeline verified.")


def main():
    parser = argparse.ArgumentParser(description="I-JEPA / VL-JEPA Inference Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to I-JEPA checkpoint")
    parser.add_argument("--skip-vljepa", action="store_true", help="Skip VL-JEPA demo")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("=" * 70)
    print("I-JEPA / VL-JEPA Inference Demo")
    print("=" * 70)

    # ---- Config ----
    enc_config = {
        "img_size": 32, "patch_size": 8, "embed_dim": 192,
        "depth": 4, "num_heads": 3, "mlp_ratio": 4.0,
    }
    pred_config = {
        "embed_dim": 192, "predictor_embed_dim": 96,
        "depth": 2, "num_heads": 3,
    }

    # ---- Load or train model ----
    encoder = build_encoder(enc_config).to(device)
    predictor = build_predictor(pred_config, encoder.num_patches).to(device)
    target_encoder = copy.deepcopy(encoder)
    target_encoder.requires_grad_(False)

    data_root = "/tmp/cifar10_data"
    eval_transform = get_image_transforms(img_size=32, is_train=False)
    train_transform = get_image_transforms(img_size=32, is_train=True)

    from torchvision.datasets import CIFAR10
    train_dataset = CIFAR10(data_root, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(data_root, train=False, download=True, transform=eval_transform)

    if args.checkpoint:
        print(f"\n[Loading checkpoint: {args.checkpoint}]")
        model = nn.ModuleDict({
            "context_encoder": encoder,
            "predictor": predictor,
            "target_encoder": target_encoder,
        })
        meta = load_checkpoint(args.checkpoint, model)
        print(f"  Loaded epoch {meta.get('epoch')}, step {meta.get('step')}")
    else:
        print("\n[No checkpoint provided — quick training for demo]")
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        quick_train(encoder, predictor, target_encoder, train_loader, device, steps=200)

    encoder.eval()
    encoder.requires_grad_(False)

    # ---- Demos ----
    eval_dataset = CIFAR10(data_root, train=False, download=False, transform=eval_transform)

    demo_feature_similarity(encoder, eval_dataset, device)
    demo_knn_classification(encoder, train_dataset, test_dataset, device)

    if not args.skip_vljepa:
        demo_vljepa_inference()

    print("\n" + "=" * 70)
    print("ALL DEMOS COMPLETE")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
