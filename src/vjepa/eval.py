"""
V-JEPA evaluation: frozen backbone + attentive probes for video classification.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from src.common.config import load_config
from src.common.distributed import get_device
from src.common.logging import setup_logger


class AttentiveProbe(nn.Module):
    """Attentive probe for video classification evaluation."""

    def __init__(self, embed_dim: int, num_classes: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) from frozen encoder
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)
        attn_out, _ = self.attn(query, x, x)
        return self.classifier(attn_out.squeeze(1))


def main():
    parser = argparse.ArgumentParser(description="V-JEPA Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    logger = setup_logger(config["logging"]["output_dir"])
    logger.info("V-JEPA evaluation ready. Load checkpoint and run attentive probe.")


if __name__ == "__main__":
    main()
