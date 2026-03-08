"""
Tests for checkpoint loading with various formats.

Covers:
- Our training checkpoint format (model_state_dict key)
- V-JEPA 2 FAIR format (encoder key)
- Raw state dict format
- XEncoder with different backends (vit_tiny, vjepa2_vitl)
- Checkpoint save/load roundtrip
- load_encoder_checkpoint permissive loading
"""

import pytest
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from src.common.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    load_encoder_checkpoint,
    find_latest_checkpoint,
    _extract_model_state,
)
from src.vljepa.models.x_encoder import XEncoder, build_x_encoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class SimpleModel(nn.Module):
    """Tiny model for checkpoint tests."""
    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(self.linear(x))


@pytest.fixture
def simple_model():
    return SimpleModel(dim=32)


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# _extract_model_state tests
# ---------------------------------------------------------------------------

class TestExtractModelState:
    def test_model_state_dict_key(self, simple_model):
        """Our standard format: checkpoint['model_state_dict']"""
        ckpt = {"model_state_dict": simple_model.state_dict(), "epoch": 5}
        state = _extract_model_state(ckpt)
        assert "linear.weight" in state

    def test_encoder_key(self, simple_model):
        """V-JEPA 2 FAIR format: checkpoint['encoder']"""
        ckpt = {"encoder": simple_model.state_dict(), "epoch": 10}
        state = _extract_model_state(ckpt)
        assert "linear.weight" in state

    def test_target_encoder_key(self, simple_model):
        """V-JEPA 2 target encoder format."""
        ckpt = {"target_encoder": simple_model.state_dict()}
        state = _extract_model_state(ckpt)
        assert "linear.weight" in state

    def test_raw_state_dict(self, simple_model):
        """Raw state dict with no wrapper key."""
        state = simple_model.state_dict()
        result = _extract_model_state(state)
        assert "linear.weight" in result

    def test_state_dict_key(self, simple_model):
        """PyTorch Lightning format: checkpoint['state_dict']"""
        ckpt = {"state_dict": simple_model.state_dict()}
        state = _extract_model_state(ckpt)
        assert "linear.weight" in state

    def test_unknown_format_raises(self):
        """Checkpoint with no recognizable keys should raise."""
        ckpt = {"metadata": {"epoch": 5}, "config": {"lr": 0.001}}
        with pytest.raises(KeyError, match="Could not find model weights"):
            _extract_model_state(ckpt)


# ---------------------------------------------------------------------------
# load_checkpoint tests
# ---------------------------------------------------------------------------

class TestLoadCheckpoint:
    def test_roundtrip(self, simple_model, tmp_dir):
        """Save and reload produces identical weights."""
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        config = {"model": {"dim": 32}}

        # Save
        save_checkpoint(
            model=simple_model,
            optimizer=optimizer,
            scheduler=None,
            epoch=5,
            step=1000,
            config=config,
            output_dir=tmp_dir,
            metrics={"loss": 0.42},
        )

        # Reload into fresh model
        fresh_model = SimpleModel(dim=32)
        latest = find_latest_checkpoint(tmp_dir)
        assert latest is not None

        meta = load_checkpoint(latest, fresh_model)
        assert meta["epoch"] == 5
        assert meta["step"] == 1000
        assert meta["metrics"]["loss"] == 0.42

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            simple_model.named_parameters(), fresh_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_load_with_encoder_key_format(self, simple_model, tmp_dir):
        """Load checkpoint saved in V-JEPA 2 encoder key format."""
        ckpt_path = tmp_dir / "vjepa2_ckpt.pt"
        torch.save({"encoder": simple_model.state_dict(), "epoch": 42}, ckpt_path)

        fresh = SimpleModel(dim=32)
        meta = load_checkpoint(ckpt_path, fresh)
        assert meta["epoch"] == 42

        for (n1, p1), (n2, p2) in zip(
            simple_model.named_parameters(), fresh.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Mismatch in {n1}"

    def test_missing_file_raises(self, simple_model):
        """Loading from nonexistent path should raise."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path.pt", simple_model)

    def test_strict_mode(self, tmp_dir):
        """strict=True should fail on mismatched keys."""
        model_a = SimpleModel(dim=32)
        ckpt_path = tmp_dir / "ckpt.pt"
        torch.save({"model_state_dict": model_a.state_dict()}, ckpt_path)

        # Model with different architecture
        model_b = nn.Linear(64, 64)
        with pytest.raises(RuntimeError):
            load_checkpoint(ckpt_path, model_b, strict=True)

    def test_non_strict_mode(self, tmp_dir):
        """strict=False should not raise on mismatched keys."""
        model_a = SimpleModel(dim=32)
        ckpt_path = tmp_dir / "ckpt.pt"
        torch.save({"model_state_dict": model_a.state_dict()}, ckpt_path)

        model_b = nn.Linear(64, 64)
        # Should not raise
        meta = load_checkpoint(ckpt_path, model_b, strict=False)
        assert meta["epoch"] == 0  # no epoch in checkpoint


# ---------------------------------------------------------------------------
# load_encoder_checkpoint tests
# ---------------------------------------------------------------------------

class TestLoadEncoderCheckpoint:
    def test_permissive_load(self, simple_model, tmp_dir):
        """load_encoder_checkpoint uses strict=False by default."""
        ckpt_path = tmp_dir / "encoder.pt"
        torch.save({"encoder": simple_model.state_dict()}, ckpt_path)

        # Load into model with extra params — should not raise
        class BiggerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(32, 32)
                self.norm = nn.LayerNorm(32)
                self.extra = nn.Linear(32, 10)  # extra param not in checkpoint
            def forward(self, x):
                return self.extra(self.norm(self.linear(x)))

        bigger = BiggerModel()
        meta = load_encoder_checkpoint(ckpt_path, bigger)
        assert meta["epoch"] == 0


# ---------------------------------------------------------------------------
# XEncoder / build_x_encoder tests
# ---------------------------------------------------------------------------

class TestXEncoder:
    def test_build_vit_tiny(self):
        """Build XEncoder with vit_tiny backend."""
        config = {
            "name": "vit_tiny",
            "img_size": 32,
            "patch_size": 8,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
        }
        encoder = build_x_encoder(config)
        assert isinstance(encoder, XEncoder)
        assert encoder.embed_dim == 64

        # All params should be frozen
        for p in encoder.parameters():
            assert not p.requires_grad

        # Forward pass
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 16, 64)  # 32/8=4, 4*4=16 patches

    def test_build_with_output_projection(self):
        """XEncoder with output_dim projects to different dimension."""
        config = {
            "name": "vit_tiny",
            "img_size": 32,
            "patch_size": 8,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "output_dim": 128,
        }
        encoder = build_x_encoder(config)
        assert encoder.embed_dim == 64
        assert isinstance(encoder.proj, nn.Linear)
        assert encoder.proj.out_features == 128

        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (1, 16, 128)

    def test_encoder_stays_frozen_after_train(self):
        """XEncoder.train() should keep encoder in eval mode."""
        config = {
            "name": "vit_tiny",
            "img_size": 32,
            "patch_size": 8,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
        }
        encoder = build_x_encoder(config)
        encoder.train()
        assert not encoder.encoder.training

    def test_unknown_name_raises(self):
        """Unknown encoder name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown X-Encoder"):
            build_x_encoder({"name": "nonexistent_model"})

    def test_vjepa2_vitl_without_checkpoint(self):
        """vjepa2_vitl without checkpoint path should build with random weights."""
        config = {
            "name": "vjepa2_vitl",
            "img_size": 224,
        }
        encoder = build_x_encoder(config)
        assert encoder.embed_dim == 1024

    def test_vjepa2_vitl_with_checkpoint(self, tmp_dir):
        """vjepa2_vitl with a valid checkpoint should load weights."""
        # Create a checkpoint matching the ViT-L architecture
        from src.ijepa.models.encoder import VisionTransformer
        ref_encoder = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        )
        ckpt_path = tmp_dir / "vjepa2.pt"
        torch.save({"encoder": ref_encoder.state_dict()}, ckpt_path)

        config = {
            "name": "vjepa2_vitl",
            "img_size": 224,
            "checkpoint": str(ckpt_path),
        }
        encoder = build_x_encoder(config)
        assert encoder.embed_dim == 1024


# ---------------------------------------------------------------------------
# find_latest_checkpoint tests
# ---------------------------------------------------------------------------

class TestFindLatestCheckpoint:
    def test_finds_latest(self, simple_model, tmp_dir):
        """Finds latest.pt after save."""
        optimizer = torch.optim.Adam(simple_model.parameters())
        save_checkpoint(
            simple_model, optimizer, None, 1, 100,
            {}, tmp_dir,
        )
        assert find_latest_checkpoint(tmp_dir) is not None

    def test_returns_none_when_empty(self, tmp_dir):
        """Returns None when no checkpoints exist."""
        assert find_latest_checkpoint(tmp_dir) is None
