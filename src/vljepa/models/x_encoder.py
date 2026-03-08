"""
X-Encoder for VL-JEPA: Frozen V-JEPA 2 Vision Transformer.

The X-Encoder processes visual input (images or video) and produces
visual token embeddings. It remains COMPLETELY FROZEN during VL-JEPA
training — all visual understanding comes from V-JEPA 2 pretraining.

In the paper:
- Architecture: V-JEPA 2 ViT-L
- Parameters: 304M (all frozen)
- Input: Video frames → patch embeddings
- Output: Sequence of visual tokens for the predictor

Supports three modes:
- 'vit_tiny': Small ViT for smoke testing (no pretrained weights)
- 'vjepa2_vitl': Load V-JEPA 2 ViT-L from local checkpoint
- 'vjepa2_hf': Load V-JEPA 2 from HuggingFace (recommended)

HuggingFace models available:
- facebook/vjepa2-vitl-fpc64-256 (ViT-L, 64 frames, 256px)
- facebook/vjepa2-vitl-fpc16-256 (ViT-L, 16 frames, 256px)
- facebook/vjepa2-vitg-fpc64-384 (ViT-g, 64 frames, 384px)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from src.ijepa.models.encoder import VisionTransformer
from src.vjepa.models.video_encoder import VideoVisionTransformer

logger = logging.getLogger(__name__)


class HuggingFaceVJEPA2Encoder(nn.Module):
    """
    Wrapper around HuggingFace VJEPA2Model encoder for feature extraction.

    Loads the full HF model but only uses the encoder (skips predictor).
    Handles both image (B, C, H, W) and video (B, T, C, H, W) inputs.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model
        self.embed_dim = hf_model.config.hidden_size
        self.frames_per_clip = hf_model.config.frames_per_clip
        self.tubelet_size = hf_model.config.tubelet_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract encoder features.

        Args:
            x: Image (B, C, H, W) or Video (B, T, C, H, W)

        Returns:
            Visual token embeddings (B, N_visual, D)
        """
        if x.dim() == 4:
            # Image input: expand to video by repeating frames
            # Need at least tubelet_size frames for the 3D conv
            n_repeats = max(self.tubelet_size, 2)
            x = x.unsqueeze(1).expand(-1, n_repeats, -1, -1, -1)

        outputs = self.hf_model(pixel_values_videos=x, skip_predictor=True)
        return outputs.last_hidden_state


def _load_hf_vjepa2(repo_id: str, **kwargs) -> HuggingFaceVJEPA2Encoder:
    """
    Load V-JEPA 2 from HuggingFace hub.

    Args:
        repo_id: HuggingFace model ID (e.g., 'facebook/vjepa2-vitl-fpc64-256')
        **kwargs: Extra args passed to from_pretrained (torch_dtype, device_map, etc.)

    Returns:
        HuggingFaceVJEPA2Encoder wrapping the loaded model
    """
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "HuggingFace transformers is required for loading V-JEPA 2 models. "
            "Install with: pip install transformers"
        )

    logger.info(f"Loading V-JEPA 2 from HuggingFace: {repo_id}")
    hf_model = AutoModel.from_pretrained(repo_id, **kwargs)
    return HuggingFaceVJEPA2Encoder(hf_model)


class XEncoder(nn.Module):
    """
    X-Encoder wrapper for VL-JEPA.

    Wraps either an image or video ViT encoder with:
    - Automatic freezing of all parameters
    - Optional visual token compression/projection
    - Support for loading pretrained V-JEPA 2 weights
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim

        # Optional projection to match predictor input dimension
        if output_dim is not None and output_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, output_dim)
        else:
            self.proj = nn.Identity()

        # Freeze everything
        self.freeze()

    def freeze(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features.

        Args:
            x: Image (B, C, H, W) or Video (B, T, C, H, W)

        Returns:
            Visual token embeddings (B, N_visual, D)
        """
        features = self.encoder(x)
        features = self.proj(features)
        return features

    def train(self, mode: bool = True):
        """Override to keep encoder in eval mode always."""
        super().train(mode)
        self.encoder.eval()
        return self

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "facebook/vjepa2-vitl-fpc64-256",
        output_dim: int | None = None,
        **kwargs,
    ) -> "XEncoder":
        """
        Load XEncoder with pretrained V-JEPA 2 weights from HuggingFace.

        Args:
            repo_id: HuggingFace model ID
            output_dim: Optional projection dimension
            **kwargs: Extra args for from_pretrained (torch_dtype, device_map, etc.)

        Returns:
            Frozen XEncoder ready for VL-JEPA training
        """
        hf_encoder = _load_hf_vjepa2(repo_id, **kwargs)
        embed_dim = hf_encoder.embed_dim
        return cls(hf_encoder, embed_dim, output_dim)


def build_x_encoder(config: dict) -> XEncoder:
    """
    Build X-Encoder from config.

    Supports:
    - 'vit_tiny': Small ViT for smoke testing
    - 'vjepa2_vitl': Load V-JEPA 2 ViT-L from local checkpoint
    - 'vjepa2_hf': Load V-JEPA 2 from HuggingFace (recommended for real use)
    """
    name = config.get("name", "vit_tiny")
    embed_dim = config.get("embed_dim", 192)

    if name == "vit_tiny":
        encoder = VisionTransformer(
            img_size=config.get("img_size", 224),
            patch_size=config.get("patch_size", 16),
            embed_dim=embed_dim,
            depth=config.get("depth", 12),
            num_heads=config.get("num_heads", 3),
        )
    elif name == "vjepa2_vitl":
        # Load from local checkpoint
        checkpoint_path = config.get("checkpoint")
        encoder = VisionTransformer(
            img_size=config.get("img_size", 384),
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Handle different checkpoint formats
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            elif "encoder" in state:
                state = state["encoder"]
            encoder.load_state_dict(state, strict=False)
        embed_dim = 1024
    elif name == "vjepa2_hf":
        # Load from HuggingFace (recommended)
        repo_id = config.get("hf_repo", "facebook/vjepa2-vitl-fpc64-256")
        hf_kwargs = {}
        if config.get("torch_dtype"):
            dtype_str = config["torch_dtype"]
            hf_kwargs["torch_dtype"] = getattr(torch, dtype_str, torch.float32)
        output_dim = config.get("output_dim")
        return XEncoder.from_pretrained(repo_id, output_dim=output_dim, **hf_kwargs)
    elif name == "salt":
        # Load SALT Stage 2 student encoder (Apple, ICLR 2025)
        # SALT-trained encoders are licensing-independent (no Meta weights)
        checkpoint_path = config.get("checkpoint")
        encoder = VisionTransformer(
            img_size=config.get("img_size", 384),
            patch_size=config.get("patch_size", 16),
            embed_dim=config.get("embed_dim", 1024),
            depth=config.get("depth", 24),
            num_heads=config.get("num_heads", 16),
        )
        if checkpoint_path and Path(checkpoint_path).exists():
            from src.common.checkpointing import _extract_model_state
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = _extract_model_state(state)
            encoder.load_state_dict(state, strict=False)
            logger.info(f"Loaded SALT encoder from {checkpoint_path}")
        embed_dim = config.get("embed_dim", 1024)
    else:
        raise ValueError(f"Unknown X-Encoder: {name}")

    output_dim = config.get("output_dim")
    return XEncoder(encoder, embed_dim, output_dim)
