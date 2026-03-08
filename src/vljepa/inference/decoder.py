"""
Lightweight text decoder for VL-JEPA.

Only invoked when generative text output is needed (captioning, open-ended VQA).
For most tasks, VL-JEPA operates purely in embedding space without this decoder.

This decoder is small and optional — the key insight of VL-JEPA is that
most vision-language tasks can be solved without generation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    """Causal (masked) multi-head attention for autoregressive decoding."""

    def __init__(self, dim: int, num_heads: int = 8, max_seq_len: int = 256):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:N, :N].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DecoderBlock(nn.Module):
    """Transformer decoder block with causal attention + cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, max_seq_len: int = 256):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalAttention(dim, num_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x_norm = self.norm2(x)
        cross_out, _ = self.cross_attn(x_norm, context, context)
        x = x + cross_out
        x = x + self.mlp(self.norm3(x))
        return x


class TextDecoder(nn.Module):
    """
    Lightweight autoregressive text decoder.

    Only used when VL-JEPA needs to generate text output.
    For most tasks, this is unnecessary.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 128,
        context_dim: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Project context (predicted embedding) to decoder dim
        context_dim = context_dim or embed_dim
        self.context_proj = nn.Linear(context_dim, embed_dim)

        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, max_seq_len)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (B, L) input tokens for teacher forcing
            context: (B, D_ctx) predicted embedding from VL-JEPA predictor

        Returns:
            logits: (B, L, vocab_size)
        """
        B, L = token_ids.shape

        x = self.token_embedding(token_ids) + self.pos_embedding[:, :L, :]
        ctx = self.context_proj(context).unsqueeze(1)  # (B, 1, D)

        for block in self.blocks:
            x = block(x, ctx)

        x = self.norm(x)
        return self.head(x)

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        max_length: int = 64,
        temperature: float = 0.8,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive text generation."""
        B = context.shape[0]
        device = context.device

        tokens = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_length - 1):
            logits = self.forward(tokens, context)
            next_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return tokens
