"""
VL-JEPA: Full model assembly.

Combines the three components:
1. X-Encoder (frozen V-JEPA 2 ViT-L) — visual understanding
2. Predictor (Llama-3 layers) — bridges vision and language
3. Y-Encoder (EmbeddingGemma-300M) — target text embedding space

The model operates in embedding space:
- X-Encoder extracts visual features (frozen, no gradients)
- Predictor takes visual features + text query → predicted embedding
- Y-Encoder produces target text embedding → ground truth
- Loss: Bi-directional InfoNCE between predicted and target embeddings

Three inference modes:
1. Embedding prediction (no decoder) — classification, retrieval
2. Captioning — with lightweight text decoder
3. Selective decoding — Ward clustering + targeted decoding
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.vljepa.models.predictor import build_predictor
from src.vljepa.models.x_encoder import build_x_encoder
from src.vljepa.models.y_encoder import build_y_encoder


class VLJEPA(nn.Module):
    """
    VL-JEPA: Vision-Language Joint Embedding Predictive Architecture.

    This is the primary model class that assembles all components
    and handles the different forward pass modes.
    """

    def __init__(
        self,
        x_encoder: nn.Module,
        predictor: nn.Module,
        y_encoder: nn.Module,
    ):
        super().__init__()
        self.x_encoder = x_encoder  # Frozen
        self.predictor = predictor  # Trainable (full LR)
        self.y_encoder = y_encoder  # Trainable (0.05x LR)

    @property
    def trainable_params(self) -> int:
        """Count trainable parameters (predictor + y_encoder only)."""
        return sum(
            p.numel()
            for p in list(self.predictor.parameters()) + list(self.y_encoder.parameters())
            if p.requires_grad
        )

    @property
    def total_params(self) -> int:
        """Count all parameters including frozen X-Encoder."""
        return sum(p.numel() for p in self.parameters())

    def get_param_groups(self, base_lr: float, y_encoder_lr_mult: float = 0.05) -> list[dict]:
        """
        Get parameter groups with different learning rates.

        The Y-Encoder uses a reduced LR (0.05x by default) to prevent
        the target embedding space from changing too rapidly.
        """
        return [
            {
                "params": list(self.predictor.parameters()),
                "lr": base_lr,
                "name": "predictor",
            },
            {
                "params": [p for p in self.y_encoder.parameters() if p.requires_grad],
                "lr": base_lr * y_encoder_lr_mult,
                "name": "y_encoder",
            },
        ]

    def forward_train(
        self,
        images: torch.Tensor,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            images: (B, C, H, W) or (B, T, C, H, W) visual input
            query_ids: (B, L_q) query text token IDs
            query_mask: (B, L_q) query attention mask
            target_ids: (B, L_t) target text token IDs
            target_mask: (B, L_t) target attention mask

        Returns:
            dict with 'predicted_embedding' and 'target_embedding'
        """
        # 1. Extract visual features (frozen, no grad)
        visual_features = self.x_encoder(images)  # (B, N_vis, D_vis)

        # 2. Create query embeddings from Y-Encoder's token embedding
        # (reuse embedding layer for query tokenization)
        query_emb = self.y_encoder.token_embedding(query_ids)  # (B, L_q, D)

        # 3. Predictor: visual + query → predicted target embedding
        predicted = self.predictor(visual_features, query_emb)  # (B, shared_dim)

        # 4. Y-Encoder: target text → ground-truth embedding
        target = self.y_encoder(target_ids, target_mask)  # (B, shared_dim)

        return {
            "predicted_embedding": predicted,
            "target_embedding": target,
        }

    @torch.no_grad()
    def forward_embed(
        self,
        images: torch.Tensor,
        query_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference: predict embedding without decoding.

        Used for classification, retrieval, and discriminative tasks.
        This is where VL-JEPA gets its 2.85x speedup — no decoder needed.
        """
        self.eval()
        visual_features = self.x_encoder(images)
        query_emb = self.y_encoder.token_embedding(query_ids)
        return self.predictor(visual_features, query_emb)

    @torch.no_grad()
    def forward_retrieve(
        self,
        images: torch.Tensor,
        query_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieval: rank candidates by similarity to predicted embedding.

        Args:
            images: (B, C, H, W)
            query_ids: (B, L_q)
            candidate_ids: (N_cand, L_c) candidate text token IDs
            candidate_masks: (N_cand, L_c)

        Returns:
            Similarity scores (B, N_cand)
        """
        self.eval()

        # Predict embedding for the visual input
        predicted = self.forward_embed(images, query_ids)  # (B, D)

        # Encode all candidates
        candidate_embs = self.y_encoder(candidate_ids, candidate_masks)  # (N, D)

        # Cosine similarity
        similarity = predicted @ candidate_embs.T  # (B, N)
        return similarity


def build_vljepa(config: dict) -> VLJEPA:
    """Build complete VL-JEPA model from config."""
    x_encoder = build_x_encoder(config["x_encoder"])
    predictor = build_predictor(config["predictor"])
    y_encoder = build_y_encoder(config["y_encoder"])

    model = VLJEPA(x_encoder, predictor, y_encoder)

    # Log parameter counts
    total = model.total_params
    trainable = model.trainable_params
    print(f"VL-JEPA total params: {total:,}")
    print(f"VL-JEPA trainable params: {trainable:,}")
    print(f"VL-JEPA frozen params: {total - trainable:,}")

    return model
