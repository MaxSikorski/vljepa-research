"""Tests for I-JEPA model components."""

import torch
import pytest

from src.ijepa.models.encoder import Attention, RoPE3D, VisionTransformer, PatchEmbed, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor
from src.ijepa.masks.multiblock import generate_masks


class TestPatchEmbed:
    def test_output_shape(self):
        pe = PatchEmbed(img_size=32, patch_size=8, embed_dim=192)
        x = torch.randn(2, 3, 32, 32)
        out = pe(x)
        assert out.shape == (2, 16, 192)  # (32/8)^2 = 16 patches

    def test_num_patches(self):
        pe = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
        assert pe.num_patches == 196  # (224/16)^2


class TestVisionTransformer:
    def test_forward_shape(self):
        vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=192, depth=4, num_heads=3)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        assert out.shape == (2, 16, 192)

    def test_masked_forward(self):
        vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=192, depth=4, num_heads=3)
        x = torch.randn(2, 3, 32, 32)
        mask = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        out = vit(x, mask_indices=mask)
        assert out.shape == (2, 5, 192)

    def test_build_encoder(self):
        config = {"img_size": 32, "patch_size": 8, "embed_dim": 192, "depth": 4, "num_heads": 3}
        enc = build_encoder(config)
        assert isinstance(enc, VisionTransformer)

    def test_vit_with_rope(self):
        vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=192, depth=2, num_heads=3, use_rope=True)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x)
        assert out.shape == (2, 16, 192)
        assert vit.rope is not None

    def test_vit_with_rope_and_grid_thw(self):
        vit = VisionTransformer(img_size=32, patch_size=8, embed_dim=192, depth=2, num_heads=3, use_rope=True)
        x = torch.randn(2, 3, 32, 32)
        out = vit(x, grid_thw=(1, 4, 4))
        assert out.shape == (2, 16, 192)

    def test_attention_with_rope(self):
        attn = Attention(dim=192, num_heads=3)
        rope = RoPE3D(head_dim=64, max_t=2, max_h=4, max_w=4)
        x = torch.randn(2, 16, 192)
        out = attn(x, rope=rope, grid_thw=(1, 4, 4))
        assert out.shape == (2, 16, 192)


class TestPredictor:
    def test_forward_shape(self):
        pred = JEPAPredictor(
            num_patches=16,
            encoder_embed_dim=192,
            predictor_embed_dim=96,
            depth=4,
            num_heads=3,
        )
        context = torch.randn(2, 10, 192)
        ctx_idx = torch.arange(10).unsqueeze(0).expand(2, -1)
        tgt_idx = torch.tensor([[10, 11, 12, 13], [10, 11, 12, 13]])
        out = pred(context, ctx_idx, tgt_idx)
        assert out.shape == (2, 4, 192)

    def test_output_matches_encoder_dim(self):
        pred = JEPAPredictor(num_patches=16, encoder_embed_dim=192, predictor_embed_dim=96, depth=2, num_heads=3)
        context = torch.randn(1, 8, 192)
        ctx_idx = torch.arange(8).unsqueeze(0)
        tgt_idx = torch.tensor([[8, 9, 10]])
        out = pred(context, ctx_idx, tgt_idx)
        assert out.shape[-1] == 192


class TestMasking:
    def test_mask_generation(self):
        ctx, tgts = generate_masks(batch_size=4, num_patches_h=4, num_patches_w=4, num_targets=2)
        assert ctx.shape[0] == 4
        assert len(tgts) == 2
        for tgt in tgts:
            assert tgt.shape[0] == 4

    def test_no_overlap(self):
        ctx, tgts = generate_masks(batch_size=1, num_patches_h=8, num_patches_w=8, num_targets=2)
        ctx_set = set(ctx[0].tolist())
        for tgt in tgts:
            tgt_set = set(tgt[0].tolist())
            # Context should not contain target indices
            assert len(ctx_set & tgt_set) == 0
