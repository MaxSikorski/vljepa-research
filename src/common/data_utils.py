"""
Common data loading utilities for images, video, and text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

from src.common.distributed import get_world_size


def get_image_transforms(
    img_size: int = 224,
    is_train: bool = True,
    use_rand_augment: bool = False,
    rand_augment_magnitude: int = 9,
    rand_augment_num_ops: int = 2,
    use_rand_erase: bool = False,
    rand_erase_prob: float = 0.25,
) -> transforms.Compose:
    """
    Standard image transforms following FAIR conventions.

    Args:
        img_size: Target image size.
        is_train: Whether training transforms (with augmentation) or eval.
        use_rand_augment: Enable RandAugment (V-JEPA 2 uses this).
        rand_augment_magnitude: RandAugment magnitude (default 9).
        rand_augment_num_ops: Number of RandAugment ops per image (default 2).
        use_rand_erase: Enable RandomErasing (applied after ToTensor).
        rand_erase_prob: Probability of random erasing (default 0.25).
    """
    if is_train:
        augments = [
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_rand_augment:
            augments.append(
                transforms.RandAugment(
                    num_ops=rand_augment_num_ops,
                    magnitude=rand_augment_magnitude,
                )
            )
        augments.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        if use_rand_erase:
            augments.append(
                transforms.RandomErasing(p=rand_erase_prob)
            )
        return transforms.Compose(augments)
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    is_train: bool = True,
    distributed: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Build a DataLoader with optional distributed sampling."""
    sampler = None
    shuffle = is_train

    if distributed and get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last and is_train,
        persistent_workers=num_workers > 0,
    )


class DummyImageTextDataset(Dataset):
    """Dummy dataset for smoke testing VL-JEPA."""

    def __init__(self, size: int = 1000, img_size: int = 224, max_text_len: int = 32):
        self.size = size
        self.img_size = img_size
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = torch.randn(3, self.img_size, self.img_size)
        text_ids = torch.randint(0, 32000, (self.max_text_len,))
        text_mask = torch.ones(self.max_text_len, dtype=torch.bool)
        return {
            "image": image,
            "text_ids": text_ids,
            "text_mask": text_mask,
        }


class DummyVideoTextDataset(Dataset):
    """Dummy dataset for smoke testing V-JEPA / VL-JEPA with video."""

    def __init__(
        self,
        size: int = 500,
        num_frames: int = 16,
        img_size: int = 224,
        max_text_len: int = 32,
    ):
        self.size = size
        self.num_frames = num_frames
        self.img_size = img_size
        self.max_text_len = max_text_len

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        video = torch.randn(self.num_frames, 3, self.img_size, self.img_size)
        text_ids = torch.randint(0, 32000, (self.max_text_len,))
        text_mask = torch.ones(self.max_text_len, dtype=torch.bool)
        return {
            "video": video,
            "text_ids": text_ids,
            "text_mask": text_mask,
        }
