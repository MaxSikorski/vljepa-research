"""
Vision Transformer (ViT) encoder for I-JEPA.

Implements both the context encoder and target encoder.
The target encoder uses Exponential Moving Average (EMA) of the context encoder weights.

Architecture follows "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
adapted for the JEPA framework.

Supports two positional encoding modes:
- Learnable sinusoidal (original I-JEPA)
- 3D RoPE (V-JEPA 2) — applied per-layer inside attention, enabling variable resolution
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# 3D Rotary Position Embeddings (RoPE) for V-JEPA 2
# ---------------------------------------------------------------------------

def _build_rope_freqs(dim: int, max_len: int = 2048, theta: float = 10000.0) -> torch.Tensor:
    """Build 1D RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, dim//2)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (max_len, dim//2, 2)


def apply_rope_1d(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply 1D RoPE to input tensor.

    Args:
        x: (..., seq_len, dim) — dim must be even
        freqs: (seq_len, dim//2, 2) — cos/sin frequencies
    """
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)  # (..., seq, dim//2, 2)
    cos_f = freqs[..., 0]  # (seq, dim//2)
    sin_f = freqs[..., 1]  # (seq, dim//2)
    # Broadcast over batch/head dims
    for _ in range(x_pairs.dim() - freqs.dim()):
        cos_f = cos_f.unsqueeze(0)
        sin_f = sin_f.unsqueeze(0)
    x_rot = torch.stack([
        x_pairs[..., 0] * cos_f - x_pairs[..., 1] * sin_f,
        x_pairs[..., 0] * sin_f + x_pairs[..., 1] * cos_f,
    ], dim=-1)
    return x_rot.reshape(x.shape).to(x.dtype)


class RoPE3D(nn.Module):
    """
    3D Rotary Position Embeddings for spatiotemporal tokens.

    Partitions head_dim into 3 segments for (temporal, height, width) axes.
    Used by V-JEPA 2 instead of learnable positional embeddings.
    """

    def __init__(self, head_dim: int, max_t: int = 64, max_h: int = 32, max_w: int = 32, theta: float = 10000.0):
        super().__init__()
        # Partition dim into 3 roughly equal parts, each must be even
        # (apply_rope_1d reshapes into pairs of 2)
        dim_t = (head_dim // 3) // 2 * 2  # round down to even
        dim_h = (head_dim // 3) // 2 * 2
        dim_w = head_dim - dim_t - dim_h  # remainder goes to width (also even if head_dim is even)

        self.dim_t = dim_t
        self.dim_h = dim_h
        self.dim_w = dim_w

        # Pre-compute frequencies for each axis (not learnable)
        self.register_buffer("freqs_t", _build_rope_freqs(dim_t, max_t, theta), persistent=False)
        self.register_buffer("freqs_h", _build_rope_freqs(dim_h, max_h, theta), persistent=False)
        self.register_buffer("freqs_w", _build_rope_freqs(dim_w, max_w, theta), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        grid_thw: tuple[int, int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D RoPE to query and key tensors.

        Args:
            q, k: (B, num_heads, N, head_dim)
            grid_thw: (T, H, W) grid dimensions. If None, assumes 2D (T=1).

        Returns:
            Rotated (q, k) with same shape
        """
        B, H, N, D = q.shape

        if grid_thw is None:
            # 2D image: T=1, H=W=sqrt(N)
            gh = gw = int(math.sqrt(N))
            gt = 1
        else:
            gt, gh, gw = grid_thw

        # Build position indices for each token
        t_idx = torch.arange(gt, device=q.device).unsqueeze(1).unsqueeze(2).expand(gt, gh, gw).reshape(-1)
        h_idx = torch.arange(gh, device=q.device).unsqueeze(0).unsqueeze(2).expand(gt, gh, gw).reshape(-1)
        w_idx = torch.arange(gw, device=q.device).unsqueeze(0).unsqueeze(1).expand(gt, gh, gw).reshape(-1)

        # Gather frequencies for each token position
        freq_t = self.freqs_t[t_idx]  # (N, dim_t//2, 2)
        freq_h = self.freqs_h[h_idx]  # (N, dim_h//2, 2)
        freq_w = self.freqs_w[w_idx]  # (N, dim_w//2, 2)

        # Split q/k along head_dim and apply RoPE per axis
        q_t, q_h, q_w = q.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)
        k_t, k_h, k_w = k.split([self.dim_t, self.dim_h, self.dim_w], dim=-1)

        q_t = apply_rope_1d(q_t, freq_t)
        q_h = apply_rope_1d(q_h, freq_h)
        q_w = apply_rope_1d(q_w, freq_w)

        k_t = apply_rope_1d(k_t, freq_t)
        k_h = apply_rope_1d(k_h, freq_h)
        k_w = apply_rope_1d(k_w, freq_w)

        q = torch.cat([q_t, q_h, q_w], dim=-1)
        k = torch.cat([k_t, k_h, k_w], dim=-1)

        return q, k


class PatchEmbed(nn.Module):
    """Convert image patches into embeddings."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional RoPE."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope: RoPE3D | None = None, grid_thw: tuple[int, int, int] | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply RoPE if provided (V-JEPA 2 mode)
        if rope is not None:
            q, k = rope(q, k, grid_thw)

        # Use PyTorch 2.0 efficient attention when available
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm (LayerNorm before attention/MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, rope: RoPE3D | None = None, grid_thw: tuple[int, int, int] | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope, grid_thw=grid_thw)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for I-JEPA.

    Used as both the context encoder and target encoder.
    The target encoder is an EMA copy of the context encoder.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.use_rope = use_rope

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)

        # Positional embedding: either learnable sincos OR 3D RoPE
        if use_rope:
            head_dim = embed_dim // num_heads
            grid_size = img_size // patch_size
            self.rope = RoPE3D(head_dim, max_t=64, max_h=grid_size, max_w=grid_size)
            self.pos_embed = None
        else:
            self.rope = None
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize positional embeddings with sincos (only when not using RoPE)
        if self.pos_embed is not None:
            pos_embed = self._get_sincos_pos_embed(self.embed_dim, int(self.num_patches ** 0.5))
            self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    @staticmethod
    def _get_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
        """Generate 2D sinusoidal positional embeddings."""
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0).reshape(2, -1)

        emb_h = VisionTransformer._get_1d_sincos(embed_dim // 2, grid[0])
        emb_w = VisionTransformer._get_1d_sincos(embed_dim // 2, grid[1])
        return torch.cat([emb_h, emb_w], dim=1)

    @staticmethod
    def _get_1d_sincos(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
        out = pos.unsqueeze(1) * omega.unsqueeze(0)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        mask_indices: torch.Tensor | None = None,
        grid_thw: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input images (B, C, H, W)
            mask_indices: Optional patch indices to select (for context encoder)
            grid_thw: Optional (T, H, W) grid dimensions for 3D RoPE (video).
                       If None and using RoPE, assumes 2D image.

        Returns:
            Patch-level representations (B, N, embed_dim) or (B, M, embed_dim) if masked
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add positional encoding (only when not using RoPE)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Apply mask (select only context patches)
        if mask_indices is not None:
            x = torch.gather(
                x, 1, mask_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

        # Transformer blocks (RoPE is applied per-layer inside attention)
        for block in self.blocks:
            x = block(x, rope=self.rope, grid_thw=grid_thw)

        x = self.norm(x)
        return x


def build_encoder(config: dict) -> VisionTransformer:
    """Build a ViT encoder from config."""
    return VisionTransformer(
        img_size=config.get("img_size", 224),
        patch_size=config.get("patch_size", 16),
        embed_dim=config.get("embed_dim", 768),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 12),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        use_rope=config.get("use_rope", False),
    )
