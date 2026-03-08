"""Tests for 3D Rotary Position Embeddings (RoPE)."""

import torch
import pytest

from src.ijepa.models.encoder import (
    RoPE3D,
    VisionTransformer,
    _build_rope_freqs,
    apply_rope_1d,
    build_encoder,
)


class TestBuildRopeFreqs:
    def test_output_shape(self):
        freqs = _build_rope_freqs(dim=16, max_len=64)
        assert freqs.shape == (64, 8, 2)  # (max_len, dim//2, 2)

    def test_values_bounded(self):
        freqs = _build_rope_freqs(dim=16, max_len=64)
        # cos/sin values should be in [-1, 1]
        assert freqs.abs().max() <= 1.0 + 1e-6


class TestApplyRope1D:
    def test_shape_preserved(self):
        x = torch.randn(2, 4, 8, 16)  # (B, heads, seq, dim)
        freqs = _build_rope_freqs(dim=16, max_len=8)
        out = apply_rope_1d(x, freqs)
        assert out.shape == x.shape

    def test_not_identity(self):
        x = torch.randn(2, 4, 8, 16)
        freqs = _build_rope_freqs(dim=16, max_len=8)
        out = apply_rope_1d(x, freqs)
        # RoPE should actually rotate — output differs from input
        assert not torch.allclose(out, x, atol=1e-5)

    def test_dtype_preserved(self):
        x = torch.randn(2, 4, 8, 16)
        freqs = _build_rope_freqs(dim=16, max_len=8)
        out = apply_rope_1d(x, freqs)
        assert out.dtype == x.dtype


class TestRoPE3D:
    def test_output_shapes(self):
        head_dim = 12  # divisible into 3 segments (4+4+4)
        rope = RoPE3D(head_dim=head_dim, max_t=4, max_h=4, max_w=4)
        B, H, N, D = 2, 4, 16, head_dim  # N = 1*4*4
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        q_out, k_out = rope(q, k, grid_thw=(1, 4, 4))
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_2d_fallback(self):
        """When grid_thw is None, assumes T=1 and H=W=sqrt(N)."""
        head_dim = 12
        rope = RoPE3D(head_dim=head_dim, max_t=4, max_h=4, max_w=4)
        B, H, N, D = 2, 4, 16, head_dim
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        q_out, k_out = rope(q, k, grid_thw=None)
        assert q_out.shape == q.shape

    def test_rotation_applied(self):
        head_dim = 12
        rope = RoPE3D(head_dim=head_dim, max_t=2, max_h=4, max_w=4)
        q = torch.randn(1, 2, 32, head_dim)  # N=2*4*4=32
        k = torch.randn(1, 2, 32, head_dim)
        q_out, k_out = rope(q, k, grid_thw=(2, 4, 4))
        assert not torch.allclose(q_out, q, atol=1e-5)


class TestVisionTransformerRoPE:
    def test_use_rope_true(self, random_images, tiny_encoder_config):
        config = {**tiny_encoder_config, "use_rope": True}
        vit = build_encoder(config)
        out = vit(random_images)
        num_patches = (32 // 8) ** 2
        assert out.shape == (2, num_patches, 64)
        assert vit.pos_embed is None
        assert vit.rope is not None

    def test_use_rope_false(self, random_images, tiny_encoder_config):
        config = {**tiny_encoder_config, "use_rope": False}
        vit = build_encoder(config)
        out = vit(random_images)
        num_patches = (32 // 8) ** 2
        assert out.shape == (2, num_patches, 64)
        assert vit.pos_embed is not None
        assert vit.rope is None

    def test_rope_with_mask(self, random_images, tiny_encoder_config):
        config = {**tiny_encoder_config, "use_rope": True}
        vit = build_encoder(config)
        mask = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        out = vit(random_images, mask_indices=mask)
        assert out.shape == (2, 4, 64)
