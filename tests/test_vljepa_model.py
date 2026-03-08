"""Tests for VL-JEPA model components."""

import torch
import pytest

from src.vljepa.models.x_encoder import XEncoder, build_x_encoder
from src.vljepa.models.predictor import SmallPredictor, build_predictor
from src.vljepa.models.y_encoder import SmallYEncoder, build_y_encoder
from src.vljepa.models.vljepa import VLJEPA, build_vljepa
from src.vljepa.losses.infonce import BidirectionalInfoNCE, build_loss


class TestXEncoder:
    def test_frozen(self):
        config = {"name": "vit_tiny", "embed_dim": 192, "img_size": 32, "patch_size": 8, "depth": 4, "num_heads": 3}
        x_enc = build_x_encoder(config)
        for param in x_enc.encoder.parameters():
            assert not param.requires_grad

    def test_output_shape(self):
        config = {"name": "vit_tiny", "embed_dim": 192, "img_size": 32, "patch_size": 8, "depth": 4, "num_heads": 3}
        x_enc = build_x_encoder(config)
        images = torch.randn(2, 3, 32, 32)
        out = x_enc(images)
        assert out.shape == (2, 16, 192)


class TestPredictor:
    def test_small_predictor(self):
        pred = SmallPredictor(input_dim=192, embed_dim=192, depth=2, num_heads=3, shared_embedding_dim=192)
        vis = torch.randn(2, 16, 192)
        query = torch.randn(2, 8, 192)
        out = pred(vis, query)
        assert out.shape == (2, 192)
        # Should be L2 normalized
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_build_predictor(self):
        config = {"name": "transformer", "embed_dim": 192, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192}
        pred = build_predictor(config)
        assert isinstance(pred, SmallPredictor)


class TestYEncoder:
    def test_small_y_encoder(self):
        enc = SmallYEncoder(vocab_size=1000, embed_dim=192, depth=2, num_heads=3, shared_embedding_dim=192)
        ids = torch.randint(0, 1000, (2, 16))
        out = enc(ids)
        assert out.shape == (2, 192)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestBidirectionalInfoNCE:
    def test_loss_computation(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07)
        predicted = torch.nn.functional.normalize(torch.randn(8, 192), dim=-1)
        target = torch.nn.functional.normalize(torch.randn(8, 192), dim=-1)
        result = loss_fn(predicted, target)
        assert "loss" in result
        assert "loss_v2t" in result
        assert "loss_t2v" in result
        assert "accuracy_v2t" in result
        assert "accuracy_t2v" in result
        assert result["loss"].item() > 0

    def test_perfect_match(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07)
        emb = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
        result = loss_fn(emb, emb.clone())
        # Perfect match should give high accuracy
        assert result["accuracy_v2t"].item() == 1.0
        assert result["accuracy_t2v"].item() == 1.0

    def test_symmetric(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07)
        a = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
        b = torch.nn.functional.normalize(torch.randn(4, 192), dim=-1)
        r1 = loss_fn(a, b)
        r2 = loss_fn(b, a)
        # Loss should be approximately symmetric
        assert abs(r1["loss"].item() - r2["loss"].item()) < 0.1


class TestVLJEPA:
    def test_build_and_forward(self):
        config = {
            "x_encoder": {"name": "vit_tiny", "embed_dim": 192, "img_size": 32, "patch_size": 8, "depth": 4, "num_heads": 3},
            "predictor": {"name": "transformer", "embed_dim": 192, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
            "y_encoder": {"name": "text_encoder_tiny", "embed_dim": 192, "vocab_size": 1000, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
        }
        model = build_vljepa(config)
        assert model.total_params > 0
        assert model.trainable_params > 0

    def test_param_groups(self):
        config = {
            "x_encoder": {"name": "vit_tiny", "embed_dim": 192, "img_size": 32, "patch_size": 8, "depth": 4, "num_heads": 3},
            "predictor": {"name": "transformer", "embed_dim": 192, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
            "y_encoder": {"name": "text_encoder_tiny", "embed_dim": 192, "vocab_size": 1000, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
        }
        model = build_vljepa(config)
        groups = model.get_param_groups(base_lr=1e-4, y_encoder_lr_mult=0.05)
        assert len(groups) == 2
        assert groups[0]["lr"] == 1e-4
        assert groups[1]["lr"] == 1e-4 * 0.05

    def test_forward_train(self):
        config = {
            "x_encoder": {"name": "vit_tiny", "embed_dim": 192, "img_size": 32, "patch_size": 8, "depth": 4, "num_heads": 3},
            "predictor": {"name": "transformer", "embed_dim": 192, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
            "y_encoder": {"name": "text_encoder_tiny", "embed_dim": 192, "vocab_size": 1000, "depth": 2, "num_heads": 3, "shared_embedding_dim": 192},
        }
        model = build_vljepa(config)
        out = model.forward_train(
            images=torch.randn(2, 3, 32, 32),
            query_ids=torch.randint(0, 1000, (2, 8)),
            query_mask=torch.ones(2, 8, dtype=torch.bool),
            target_ids=torch.randint(0, 1000, (2, 16)),
            target_mask=torch.ones(2, 16, dtype=torch.bool),
        )
        assert "predicted_embedding" in out
        assert "target_embedding" in out
        assert out["predicted_embedding"].shape == (2, 192)
        assert out["target_embedding"].shape == (2, 192)
