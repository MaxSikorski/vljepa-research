"""
I-JEPA evaluation via linear probing and k-NN classification.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.common.config import load_config
from src.common.distributed import get_device
from src.common.logging import get_logger, setup_logger
from src.ijepa.models.encoder import build_encoder


class LinearProbe(nn.Module):
    """Linear probe on top of frozen encoder for evaluation."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global average pooling over patch tokens
        x = x.mean(dim=1)  # (B, embed_dim)
        return self.linear(x)


@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features from frozen encoder."""
    encoder.eval()
    all_features = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        features = encoder(images)
        features = features.mean(dim=1)  # Global average pool
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def knn_evaluate(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
) -> float:
    """k-NN classification accuracy."""
    # Normalize features
    train_features = torch.nn.functional.normalize(train_features, dim=1)
    test_features = torch.nn.functional.normalize(test_features, dim=1)

    # Compute cosine similarity
    sim = test_features @ train_features.T  # (N_test, N_train)
    _, topk_indices = sim.topk(k, dim=1)

    # Majority vote
    topk_labels = train_labels[topk_indices]  # (N_test, k)
    predictions = topk_labels.mode(dim=1).values

    accuracy = (predictions == test_labels).float().mean().item()
    return accuracy


def train_linear_probe(
    encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    embed_dim: int,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.001,
) -> dict[str, float]:
    """
    Train and evaluate a linear probe on frozen encoder features.

    Returns dict with train_acc, test_acc, and test_loss.
    """
    probe = LinearProbe(embed_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    logger = get_logger()

    best_acc = 0.0
    for epoch in range(epochs):
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = encoder(images)
            logits = probe(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        train_acc = correct / total
        if (epoch + 1) % 10 == 0:
            logger.info(f"Probe epoch {epoch+1}/{epochs} — loss: {total_loss/total:.4f}, train_acc: {train_acc:.4f}")

    # Evaluate on test set
    probe.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = encoder(images)
            logits = probe(features)
            loss = criterion(logits, labels)
            test_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    test_acc = correct / total
    logger.info(f"Linear probe — test_acc: {test_acc:.4f}")

    return {"train_acc": train_acc, "test_acc": test_acc, "test_loss": test_loss / total}


def main():
    parser = argparse.ArgumentParser(description="I-JEPA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval-mode", choices=["linear", "knn", "both"], default="both")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lr", type=float, default=0.001)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logger(config["logging"]["output_dir"])

    # Build encoder
    encoder_config = config["model"]["encoder"]
    encoder = build_encoder(encoder_config).to(device)
    embed_dim = encoder_config.get("embed_dim", 768)

    # Load checkpoint
    from src.common.checkpointing import load_checkpoint
    load_checkpoint(args.checkpoint, encoder, strict=False)
    encoder.eval()
    encoder.requires_grad_(False)
    logger.info(f"Loaded encoder from {args.checkpoint}")

    # Load dataset
    data_config = config.get("data", {})
    dataset_name = data_config.get("dataset", "cifar10")
    data_root = data_config.get("root", "./data/raw")
    img_size = encoder_config.get("img_size", 32)

    from src.common.data_utils import get_image_transforms
    train_transform = get_image_transforms(img_size, is_train=False)  # No augmentation for eval
    test_transform = get_image_transforms(img_size, is_train=False)

    if dataset_name == "cifar10":
        from torchvision.datasets import CIFAR10
        train_dataset = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root=data_root, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset_name == "cifar100":
        from torchvision.datasets import CIFAR100
        train_dataset = CIFAR100(root=data_root, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(root=data_root, train=False, transform=test_transform, download=True)
        num_classes = 100
    else:
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder(root=f"{data_root}/train", transform=train_transform)
        test_dataset = ImageFolder(root=f"{data_root}/val", transform=test_transform)
        num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info(f"Dataset: {dataset_name}, classes: {num_classes}, train: {len(train_dataset)}, test: {len(test_dataset)}")

    results = {}

    # k-NN evaluation
    if args.eval_mode in ("knn", "both"):
        logger.info("Extracting features for k-NN...")
        train_features, train_labels = extract_features(encoder, train_loader, device)
        test_features, test_labels = extract_features(encoder, test_loader, device)

        knn_acc = knn_evaluate(train_features, train_labels, test_features, test_labels, k=args.knn_k)
        results["knn_accuracy"] = knn_acc
        logger.info(f"k-NN (k={args.knn_k}) accuracy: {knn_acc:.4f}")

    # Linear probe evaluation
    if args.eval_mode in ("linear", "both"):
        logger.info("Training linear probe...")
        probe_results = train_linear_probe(
            encoder, train_loader, test_loader,
            embed_dim=embed_dim,
            num_classes=num_classes,
            device=device,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
        )
        results.update(probe_results)

    logger.info(f"Final results: {results}")
    return results


if __name__ == "__main__":
    main()
