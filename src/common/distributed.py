"""
Distributed training utilities for multi-GPU and multi-node training.

Supports:
- PyTorch DDP (DistributedDataParallel)
- PyTorch FSDP (Fully Sharded Data Parallel) — required for 1B+ param models
- Single GPU / MPS (Apple Silicon)
- SLURM cluster scheduling via submitit
"""

from __future__ import annotations

import functools
import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn


def setup_distributed() -> tuple[int, int, int]:
    """
    Initialize distributed training from environment variables.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size

    # Single process fallback
    return 0, 0, 1


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return torch.device(f"cuda:{local_rank}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_world_size() -> int:
    """Get the total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


@contextmanager
def distributed_context():
    """Context manager for distributed training setup/cleanup."""
    rank, local_rank, world_size = setup_distributed()
    try:
        yield rank, local_rank, world_size
    finally:
        cleanup_distributed()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a tensor and compute mean across processes."""
    if not dist.is_initialized() or get_world_size() == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor


# ---------------------------------------------------------------------------
# FSDP Wrapping (required for V-JEPA 2 / VL-JEPA full-scale training)
# ---------------------------------------------------------------------------

def wrap_model_distributed(
    model: nn.Module,
    strategy: str = "ddp",
    mixed_precision: str = "none",
    device_id: int | None = None,
) -> nn.Module:
    """
    Wrap a model with DDP or FSDP for distributed training.

    Args:
        model: The model to wrap
        strategy: "ddp" or "fsdp"
        mixed_precision: "none", "fp16", or "bf16"
        device_id: Local GPU device ID

    Returns:
        Wrapped model ready for distributed training
    """
    if not dist.is_initialized() or get_world_size() == 1:
        return model

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    if strategy == "ddp":
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=False,
        )

    elif strategy == "fsdp":
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        # Import the block type for auto-wrapping
        from src.ijepa.models.encoder import TransformerBlock

        # Configure mixed precision for FSDP
        if mixed_precision == "bf16":
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif mixed_precision == "fp16":
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        else:
            mp_policy = None

        # Auto-wrap policy: shard at the TransformerBlock level
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )

        return FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=device_id,
            use_orig_params=True,  # Required for torch.compile compatibility
        )

    else:
        raise ValueError(f"Unknown distributed strategy: {strategy}. Use 'ddp' or 'fsdp'.")
