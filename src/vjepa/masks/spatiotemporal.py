"""
Spatiotemporal masking strategies for V-JEPA.

Two main strategies:
1. Tube masking: Mask entire spatial regions across all time steps
2. Random spatiotemporal: Mask random 3D blocks in space-time

The temporal dimension adds complexity — target blocks can span
different temporal extents, requiring the predictor to learn
temporal dynamics.
"""

from __future__ import annotations

import random

import torch


def generate_tube_masks(
    batch_size: int,
    num_temporal: int,
    num_spatial_h: int,
    num_spatial_w: int,
    num_targets: int = 8,
    target_spatial_scale: tuple[float, float] = (0.15, 0.25),
    temporal_extent: float = 0.5,
    context_scale: float = 0.9,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Generate tube masks for V-JEPA.

    Tube masks extend spatially across the full temporal range,
    forcing the predictor to understand spatial content over time.

    Returns:
        context_indices: (B, M_ctx)
        target_indices_list: list of (B, M_tgt_i)
    """
    num_spatial = num_spatial_h * num_spatial_w
    total_patches = num_temporal * num_spatial
    all_indices = set(range(total_patches))

    target_blocks = []
    used_indices = set()

    for _ in range(num_targets):
        # Sample spatial region
        scale = random.uniform(target_spatial_scale[0], target_spatial_scale[1])
        num_spatial_target = max(1, int(num_spatial * scale))

        block_h = max(1, int((num_spatial_target ** 0.5)))
        block_w = max(1, num_spatial_target // block_h)
        block_h = min(block_h, num_spatial_h)
        block_w = min(block_w, num_spatial_w)

        top = random.randint(0, num_spatial_h - block_h)
        left = random.randint(0, num_spatial_w - block_w)

        # Temporal extent
        num_temporal_target = max(1, int(num_temporal * temporal_extent))
        t_start = random.randint(0, num_temporal - num_temporal_target)

        # Collect 3D indices
        indices = []
        for t in range(t_start, t_start + num_temporal_target):
            for h in range(top, top + block_h):
                for w in range(left, left + block_w):
                    spatial_idx = h * num_spatial_w + w
                    idx = t * num_spatial + spatial_idx
                    indices.append(idx)

        target_blocks.append(sorted(indices))
        used_indices.update(indices)

    # Context: everything not in targets
    context_indices = sorted(all_indices - used_indices)

    # Subsample context if needed
    if len(context_indices) > int(total_patches * context_scale):
        context_indices = sorted(random.sample(context_indices, int(total_patches * context_scale)))

    # Convert to tensors
    ctx = torch.tensor(context_indices, device=device).unsqueeze(0).expand(batch_size, -1)
    tgt_list = [
        torch.tensor(t, device=device).unsqueeze(0).expand(batch_size, -1)
        for t in target_blocks
    ]

    return ctx, tgt_list
