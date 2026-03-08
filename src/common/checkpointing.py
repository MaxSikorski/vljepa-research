"""
Checkpoint save/load utilities with config snapshot for reproducibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.common.config import save_config
from src.common.distributed import is_main_process
from src.common.logging import get_logger


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    config: dict[str, Any],
    output_dir: str | Path,
    metrics: dict[str, float] | None = None,
    is_best: bool = False,
) -> Path | None:
    """
    Save a training checkpoint with metadata.

    Only saves on the main process in distributed training.
    """
    if not is_main_process():
        return None

    output_dir = Path(output_dir) / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    # Save periodic checkpoint
    path = output_dir / f"checkpoint_epoch{epoch:04d}_step{step:08d}.pt"
    torch.save(checkpoint, path)
    get_logger().info(f"Saved checkpoint: {path}")

    # Save as 'latest' for easy resumption
    latest_path = output_dir / "latest.pt"
    torch.save(checkpoint, latest_path)

    # Save as 'best' if flagged
    if is_best:
        best_path = output_dir / "best.pt"
        torch.save(checkpoint, best_path)
        get_logger().info(f"New best checkpoint saved: {best_path}")

    # Save config snapshot alongside checkpoint
    save_config(config, output_dir / "config_snapshot.yaml")

    # Save run metadata
    metadata = {
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
        "timestamp": checkpoint["timestamp"],
        "config_path": config.get("_config_path"),
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path


def _extract_model_state(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """
    Extract model state dict from various checkpoint formats.

    Supports:
    - Our format: checkpoint["model_state_dict"]
    - V-JEPA 2 FAIR format: checkpoint["encoder"] or checkpoint["target_encoder"]
    - Raw state dict (no wrapper key)
    - HuggingFace format: checkpoint["model"] or direct state dict
    """
    # Try known wrapper keys in priority order
    for key in ("model_state_dict", "encoder", "target_encoder", "model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            state = checkpoint[key]
            get_logger().info(f"Found model weights under '{key}' key")
            return state

    # Check if it's already a raw state dict (keys look like layer names)
    sample_keys = list(checkpoint.keys())[:5]
    if sample_keys and any("." in k for k in sample_keys):
        # Looks like a raw state dict
        get_logger().info("Checkpoint appears to be a raw state dict")
        return checkpoint

    raise KeyError(
        f"Could not find model weights in checkpoint. "
        f"Available keys: {list(checkpoint.keys())}"
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    device: torch.device | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """
    Load a checkpoint and restore model/optimizer/scheduler state.

    Supports multiple checkpoint formats (our format, FAIR V-JEPA 2, HuggingFace).

    Returns metadata dict with epoch, step, metrics.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device or "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    state_dict = _extract_model_state(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        get_logger().warning(f"Missing keys when loading checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        get_logger().warning(f"Unexpected keys when loading checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
    get_logger().info(f"Loaded model from {checkpoint_path}")

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def load_encoder_checkpoint(
    checkpoint_path: str | Path,
    encoder: nn.Module,
    device: torch.device | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """
    Load only encoder weights from a checkpoint (e.g., for X-Encoder).

    More permissive than load_checkpoint — uses strict=False by default
    since encoder checkpoints often come from different training frameworks.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    map_location = device or "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    state_dict = _extract_model_state(checkpoint)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=strict)
    get_logger().info(
        f"Loaded encoder from {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in an experiment directory."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    return None
