"""
Bi-directional InfoNCE loss for VL-JEPA.

This is the core training objective. It operates in the shared 1536-d
embedding space, computing contrastive loss in both directions:
- Vision-to-Text: does the predicted embedding match the right target?
- Text-to-Vision: does the target embedding match the right prediction?

Key properties:
- No explicit negative mining needed (in-batch negatives)
- Temperature parameter controls the sharpness of the distribution
- Bi-directional formulation provides natural anti-collapse regularization

Mathematical formulation:
    L = 0.5 * (L_v2t + L_t2v)

    L_v2t = -log(exp(sim(p_i, t_i)/τ) / Σ_j exp(sim(p_i, t_j)/τ))
    L_t2v = -log(exp(sim(t_i, p_i)/τ) / Σ_j exp(sim(t_i, p_j)/τ))

where p_i is predicted embedding, t_i is target embedding, τ is temperature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalInfoNCE(nn.Module):
    """
    Bi-directional InfoNCE loss for VL-JEPA training.

    Computes symmetric contrastive loss between predicted and target
    embeddings using in-batch negatives.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
    ):
        super().__init__()

        if learnable_temperature:
            # Learnable log-temperature (log scale for stability)
            self.log_temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer(
                "log_temperature", torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute bi-directional InfoNCE loss.

        Args:
            predicted: (B, D) predicted embeddings from predictor (L2-normalized)
            target: (B, D) target embeddings from Y-encoder (L2-normalized)

        Returns:
            dict with 'loss', 'loss_v2t', 'loss_t2v', 'accuracy_v2t', 'accuracy_t2v'
        """
        B = predicted.shape[0]

        # Ensure L2 normalization
        predicted = F.normalize(predicted, dim=-1)
        target = F.normalize(target, dim=-1)

        # Compute similarity matrix
        # (B, B) where [i, j] = cosine_sim(predicted_i, target_j) / temperature
        logits = (predicted @ target.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(B, device=logits.device)

        # Vision-to-Text loss: predicted → target
        loss_v2t = F.cross_entropy(logits, labels)

        # Text-to-Vision loss: target → predicted
        loss_t2v = F.cross_entropy(logits.T, labels)

        # Total loss (symmetric)
        loss = 0.5 * (loss_v2t + loss_t2v)

        # Accuracy metrics (for monitoring)
        with torch.no_grad():
            acc_v2t = (logits.argmax(dim=1) == labels).float().mean()
            acc_t2v = (logits.T.argmax(dim=1) == labels).float().mean()

        return {
            "loss": loss,
            "loss_v2t": loss_v2t,
            "loss_t2v": loss_t2v,
            "accuracy_v2t": acc_v2t,
            "accuracy_t2v": acc_t2v,
            "temperature": self.temperature.detach(),
        }


def build_loss(config: dict) -> BidirectionalInfoNCE:
    """Build InfoNCE loss from config."""
    return BidirectionalInfoNCE(
        temperature=config.get("temperature", 0.07),
        learnable_temperature=config.get("learnable_temperature", False),
    )
