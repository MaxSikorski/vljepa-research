"""Shared test fixtures for VL-JEPA research tests."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def tiny_encoder_config():
    """Minimal ViT config for fast CPU tests."""
    return {
        "img_size": 32,
        "patch_size": 8,
        "embed_dim": 64,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 4.0,
    }


@pytest.fixture
def random_images():
    """Batch of 2 tiny random images."""
    return torch.randn(2, 3, 32, 32)
