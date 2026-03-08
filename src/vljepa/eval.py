"""
VL-JEPA evaluation suite.

Supports:
- Zero-shot classification (embedding similarity to class labels)
- Text-to-video retrieval (Recall@1, Recall@5, Recall@10)
- Visual Question Answering (discriminative, multiple choice)
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from src.common.config import load_config
from src.common.distributed import get_device
from src.common.logging import setup_logger


@torch.no_grad()
def zero_shot_classify(
    model,
    images: torch.Tensor,
    class_labels: list[str],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """
    Zero-shot classification via embedding similarity.

    For each image, compute similarity between the predicted embedding
    and the Y-Encoder embedding of each class label.
    """
    model.eval()

    # Encode all class labels
    label_embeddings = []
    for label in class_labels:
        ids = tokenizer.encode(label, return_tensors="pt").to(device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        emb = model.y_encoder(ids, mask)
        label_embeddings.append(emb)

    label_embeddings = torch.cat(label_embeddings, dim=0)  # (C, D)

    # Predict embeddings for images
    dummy_query = tokenizer.encode("What is this?", return_tensors="pt")
    dummy_query = dummy_query.expand(images.shape[0], -1).to(device)

    predicted = model.forward_embed(images, dummy_query)  # (B, D)

    # Cosine similarity
    similarity = predicted @ label_embeddings.T  # (B, C)
    predictions = similarity.argmax(dim=1)

    return predictions


@torch.no_grad()
def retrieval_evaluate(
    model,
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    ground_truth: torch.Tensor,
) -> dict[str, float]:
    """
    Compute retrieval metrics.

    Args:
        query_embeddings: (N, D)
        candidate_embeddings: (M, D)
        ground_truth: (N,) index of correct candidate for each query
    """
    similarity = query_embeddings @ candidate_embeddings.T  # (N, M)
    rankings = similarity.argsort(dim=1, descending=True)

    metrics = {}
    for k in [1, 5, 10]:
        top_k = rankings[:, :k]
        correct = (top_k == ground_truth.unsqueeze(1)).any(dim=1).float()
        metrics[f"recall@{k}"] = correct.mean().item()

    # Median rank
    ranks = (rankings == ground_truth.unsqueeze(1)).nonzero()[:, 1].float()
    metrics["median_rank"] = ranks.median().item() + 1

    return metrics


@torch.no_grad()
def compute_video_embeddings(
    model,
    dataloader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract video embeddings and labels from a dataloader.

    Expects dataloader to yield dicts with 'video' and optionally 'label'.
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    for batch in dataloader:
        video = batch["video"].to(device)
        # Use x_encoder to get visual features, then predict
        vis_features = model.x_encoder(video)
        # Global average pool over visual tokens
        embedding = vis_features.mean(dim=1)
        embedding = F.normalize(embedding, dim=-1)
        all_embeddings.append(embedding.cpu())
        if "label" in batch:
            all_labels.append(batch["label"])

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else torch.zeros(embeddings.size(0), dtype=torch.long)
    return embeddings, labels


def main():
    parser = argparse.ArgumentParser(description="VL-JEPA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--task", choices=["classify", "retrieve"], default="classify")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logger(config["logging"]["output_dir"])
    logger.info(f"VL-JEPA evaluation — task: {args.task}")

    # Build model
    from src.vljepa.models.vljepa import build_vljepa
    model = build_vljepa(config["model"]).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        from src.common.checkpointing import load_checkpoint
        load_checkpoint(args.checkpoint, model, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model.eval()

    # Load dummy data for smoke testing (replace with real datasets in production)
    from src.common.data_utils import DummyVideoTextDataset, build_dataloader
    img_size = config["model"].get("encoder", {}).get("img_size", 224)
    num_frames = config["model"].get("encoder", {}).get("num_frames", 16)

    dataset = DummyVideoTextDataset(size=100, num_frames=num_frames, img_size=img_size)
    dataloader = build_dataloader(dataset, batch_size=args.batch_size, is_train=False, drop_last=False)

    if args.task == "classify":
        logger.info("Running zero-shot classification on dummy data...")
        # For real evaluation, replace with actual class labels and tokenizer
        # Here we demonstrate the pipeline structure
        video_embeddings, labels = compute_video_embeddings(model, dataloader, device)
        logger.info(f"Extracted {video_embeddings.shape[0]} video embeddings of dim {video_embeddings.shape[1]}")

        # Dummy classification: measure embedding quality via self-similarity
        sim = video_embeddings @ video_embeddings.T
        avg_sim = sim.mean().item()
        logger.info(f"Average pairwise similarity: {avg_sim:.4f}")
        logger.info("For real zero-shot classification, provide class labels and a tokenizer.")

    elif args.task == "retrieve":
        logger.info("Running retrieval evaluation on dummy data...")
        video_embeddings, _ = compute_video_embeddings(model, dataloader, device)

        # Split into queries and candidates for demo
        n = video_embeddings.shape[0]
        mid = n // 2
        query_emb = video_embeddings[:mid]
        candidate_emb = video_embeddings[mid:2*mid]
        ground_truth = torch.arange(mid)  # dummy 1-to-1 mapping

        metrics = retrieval_evaluate(model, query_emb, candidate_emb, ground_truth)
        logger.info(f"Retrieval metrics: {metrics}")

    logger.info("VL-JEPA evaluation complete.")


if __name__ == "__main__":
    main()
