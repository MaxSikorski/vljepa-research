"""
Multi-block masking strategy for I-JEPA.

The masking strategy is critical for JEPA's success:
- Target blocks must be large enough to require semantic understanding
- Context blocks must be spatially distributed to provide informative signal
- This guides the model toward learning semantic rather than low-level features

Based on the I-JEPA paper (Assran et al., CVPR 2023).
"""

from __future__ import annotations

import math
import random

import torch


def sample_block_mask(
    num_patches_h: int,
    num_patches_w: int,
    min_scale: float,
    max_scale: float,
    aspect_ratio: tuple[float, float] = (0.75, 1.5),
) -> tuple[list[int], int, int, int, int]:
    """
    Sample a rectangular block mask.

    Returns:
        (patch_indices, top, left, height, width)
    """
    num_patches = num_patches_h * num_patches_w

    # Sample scale and aspect ratio
    scale = random.uniform(min_scale, max_scale)
    ar = random.uniform(aspect_ratio[0], aspect_ratio[1])

    # Compute block dimensions
    num_block_patches = int(num_patches * scale)
    block_h = max(1, int(math.sqrt(num_block_patches / ar)))
    block_w = max(1, int(math.sqrt(num_block_patches * ar)))

    # Clamp to grid size
    block_h = min(block_h, num_patches_h)
    block_w = min(block_w, num_patches_w)

    # Random position
    top = random.randint(0, num_patches_h - block_h)
    left = random.randint(0, num_patches_w - block_w)

    # Collect patch indices
    indices = []
    for i in range(top, top + block_h):
        for j in range(left, left + block_w):
            indices.append(i * num_patches_w + j)

    return indices, top, left, block_h, block_w


def generate_masks(
    batch_size: int,
    num_patches_h: int,
    num_patches_w: int,
    num_targets: int = 4,
    min_target_scale: float = 0.15,
    max_target_scale: float = 0.2,
    min_context_scale: float = 0.85,
    max_context_scale: float = 1.0,
    aspect_ratio: tuple[float, float] = (0.75, 1.5),
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Generate multi-block masks for a batch.

    The same mask pattern is used across the batch for efficiency.

    Returns:
        context_indices: (B, M_ctx) indices of context patches
        target_indices_list: list of (B, M_tgt_i) for each target block
    """
    num_patches = num_patches_h * num_patches_w
    all_indices = set(range(num_patches))

    # Sample target blocks
    target_blocks = []
    used_indices = set()

    for _ in range(num_targets):
        indices, *_ = sample_block_mask(
            num_patches_h, num_patches_w,
            min_target_scale, max_target_scale,
            aspect_ratio,
        )
        target_blocks.append(indices)
        used_indices.update(indices)

    # Context is the complement of all targets (potentially with some overlap allowed)
    # The context can also be a block mask itself
    context_indices_set, *_ = sample_block_mask(
        num_patches_h, num_patches_w,
        min_context_scale, max_context_scale,
        aspect_ratio=(1.0, 1.0),  # Square-ish context
    )

    # Remove target indices from context
    context_indices_set = [i for i in context_indices_set if i not in used_indices]

    # Ensure we have at least some context
    if len(context_indices_set) < num_patches // 4:
        context_indices_set = sorted(all_indices - used_indices)

    # Convert to tensors and expand across batch
    context_indices = torch.tensor(sorted(context_indices_set), device=device)
    context_indices = context_indices.unsqueeze(0).expand(batch_size, -1)

    target_indices_list = []
    for target in target_blocks:
        target_idx = torch.tensor(sorted(target), device=device)
        target_idx = target_idx.unsqueeze(0).expand(batch_size, -1)
        target_indices_list.append(target_idx)

    return context_indices, target_indices_list
