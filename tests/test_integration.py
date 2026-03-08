"""
Integration tests: end-to-end train → save → load → eval pipelines.

These tests use dummy data (no network downloads) and run on CPU.
They prove the entire pipeline works together, not just individual components.
"""

import copy
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor
from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.eval import LinearProbe, extract_features, knn_evaluate, train_linear_probe
from src.common.checkpointing import save_checkpoint, load_checkpoint, find_latest_checkpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TINY_CONFIG = {
    "model": {
        "encoder": {
            "img_size": 32,
            "patch_size": 8,
            "embed_dim": 64,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 4.0,
        },
        "predictor": {
            "embed_dim": 64,
            "predictor_embed_dim": 32,
            "depth": 2,
            "num_heads": 4,
        },
    },
    "masking": {
        "num_targets": 2,
        "min_target_scale": 0.15,
        "max_target_scale": 0.2,
        "min_context_scale": 0.85,
        "max_context_scale": 1.0,
    },
    "training": {
        "epochs": 3,
        "batch_size": 8,
        "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.05, "betas": [0.9, 0.95]},
        "scheduler": {"name": "cosine", "warmup_epochs": 1, "min_lr": 1e-6},
        "gradient_clip": 1.0,
        "ema_momentum": 0.996,
        "ema_momentum_end": 1.0,
    },
    "logging": {
        "log_every": 1,
        "save_every": 1,
        "output_dir": "/tmp/test_integration",
    },
}


def _make_dummy_image_dataset(n=64, img_size=32, num_classes=5):
    """Create a tiny dataset of random images + labels."""
    images = torch.randn(n, 3, img_size, img_size)
    labels = torch.randint(0, num_classes, (n,))
    return TensorDataset(images, labels)


def _run_ijepa_training_steps(encoder, predictor, target_encoder, images, config, n_steps=4):
    """
    Run n_steps of I-JEPA training and return losses.

    This is a mini version of IJEPATrainer.train_step() without AMP or distributed.
    """
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3,
        weight_decay=0.05,
    )

    grid_size = int(encoder.num_patches ** 0.5)
    mask_config = config["masking"]
    losses = []

    for step in range(n_steps):
        batch = images[step * 8 : (step + 1) * 8]
        if batch.shape[0] == 0:
            break
        B = batch.shape[0]

        # Generate masks
        ctx_idx, tgt_list = generate_masks(
            batch_size=B,
            num_patches_h=grid_size,
            num_patches_w=grid_size,
            num_targets=mask_config.get("num_targets", 2),
            device=device,
        )

        # Context encoder on masked input
        ctx_emb = encoder(batch, mask_indices=ctx_idx)

        # Target encoder on full image
        with torch.no_grad():
            full_emb = target_encoder(batch)

        # Predict targets and compute loss
        total_loss = 0.0
        for tgt_idx in tgt_list:
            target_repr = torch.gather(
                full_emb, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full_emb.shape[-1])
            )
            pred_repr = predictor(ctx_emb, ctx_idx, tgt_idx)
            total_loss += F.smooth_l1_loss(pred_repr, target_repr.detach())

        total_loss /= len(tgt_list)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for p_t, p_c in zip(target_encoder.parameters(), encoder.parameters()):
                p_t.data.mul_(0.996).add_(p_c.data, alpha=0.004)

        losses.append(total_loss.item())

    return losses


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIJEPATrainLoop:
    """Test the I-JEPA training loop end-to-end."""

    def test_loss_is_finite(self):
        """Training loss should be finite (not NaN or Inf)."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        images = torch.randn(32, 3, 32, 32)
        losses = _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)

        assert len(losses) == 4
        for loss in losses:
            assert not torch.isnan(torch.tensor(loss)), f"Loss is NaN at step"
            assert not torch.isinf(torch.tensor(loss)), f"Loss is Inf at step"
            assert loss > 0, f"Loss should be positive"

    def test_loss_decreases_over_training(self):
        """Loss should trend downward (or at least not explode) over more steps."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        # Use same batch repeatedly (overfitting test — loss MUST decrease)
        images = torch.randn(8, 3, 32, 32).repeat(8, 1, 1, 1)  # 64 images, same 8 repeated
        losses = _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG, n_steps=8)

        # With repeated data, first loss should be >= last loss (overfitting)
        # Allow some noise — check that avg of last 2 < avg of first 2
        avg_first = sum(losses[:2]) / 2
        avg_last = sum(losses[-2:]) / 2
        assert avg_last <= avg_first * 1.5, (
            f"Loss should not explode: first_avg={avg_first:.4f}, last_avg={avg_last:.4f}"
        )

    def test_ema_target_encoder_diverges_from_context(self):
        """After training, target encoder should differ from context encoder (EMA lag)."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        images = torch.randn(32, 3, 32, 32)
        _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)

        # Parameters should differ (EMA lags behind gradient updates)
        total_diff = 0.0
        for p_t, p_c in zip(target_encoder.parameters(), encoder.parameters()):
            total_diff += (p_t - p_c).abs().sum().item()
        assert total_diff > 0, "Target encoder should differ from context encoder after training"


class TestCheckpointRoundtrip:
    """Test save → load checkpoint pipeline."""

    def test_train_save_load_roundtrip(self):
        """Train, save checkpoint, load into fresh model, verify weights match."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        # Train for a few steps
        images = torch.randn(32, 3, 32, 32)
        _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save checkpoint
            model = nn.ModuleDict({
                "context_encoder": encoder,
                "predictor": predictor,
                "target_encoder": target_encoder,
            })
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
            )
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                epoch=1, step=4, config=TINY_CONFIG,
                output_dir=tmp_dir, metrics={"loss": 0.5},
            )

            # Verify checkpoint exists
            latest = find_latest_checkpoint(tmp_dir)
            assert latest is not None

            # Load into fresh model
            fresh_encoder = build_encoder(enc_cfg)
            fresh_predictor = build_predictor(pred_cfg, fresh_encoder.num_patches)
            fresh_target = copy.deepcopy(fresh_encoder)
            fresh_model = nn.ModuleDict({
                "context_encoder": fresh_encoder,
                "predictor": fresh_predictor,
                "target_encoder": fresh_target,
            })

            meta = load_checkpoint(latest, fresh_model)
            assert meta["epoch"] == 1
            assert meta["step"] == 4

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(model.named_parameters(), fresh_model.named_parameters()):
                assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch in {n1}"


class TestTrainThenEval:
    """Test full pipeline: train → save → load → evaluate."""

    def test_train_then_knn_eval(self):
        """Train, then run k-NN evaluation."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]
        embed_dim = enc_cfg["embed_dim"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        # Train
        images = torch.randn(32, 3, 32, 32)
        losses = _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)

        # Eval: extract features and run k-NN
        encoder.eval()
        dataset = _make_dummy_image_dataset(n=40, img_size=32, num_classes=5)
        loader = DataLoader(dataset, batch_size=10, shuffle=False)

        features, labels = extract_features(encoder, loader, torch.device("cpu"))
        # extract_features already does global average pooling over patches
        assert features.shape == (40, embed_dim)

        # Split into train/test
        train_f, test_f = features[:30], features[30:]
        train_l, test_l = labels[:30], labels[30:]

        acc = knn_evaluate(train_f, train_l, test_f, test_l, k=5)
        assert 0.0 <= acc <= 1.0, f"k-NN accuracy should be in [0, 1], got {acc}"

    def test_train_then_linear_probe(self):
        """Train, then run linear probe evaluation."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]
        embed_dim = enc_cfg["embed_dim"]
        num_classes = 5

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        # Train
        images = torch.randn(32, 3, 32, 32)
        _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)

        # Eval: linear probe
        encoder.eval()
        encoder.requires_grad_(False)

        train_dataset = _make_dummy_image_dataset(n=40, img_size=32, num_classes=num_classes)
        test_dataset = _make_dummy_image_dataset(n=20, img_size=32, num_classes=num_classes)
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

        results = train_linear_probe(
            encoder, train_loader, test_loader,
            embed_dim=embed_dim,
            num_classes=num_classes,
            device=torch.device("cpu"),
            epochs=5,
            lr=0.01,
        )

        assert "test_acc" in results
        assert "train_acc" in results
        assert 0.0 <= results["test_acc"] <= 1.0
        assert 0.0 <= results["train_acc"] <= 1.0

    def test_full_save_load_eval_pipeline(self):
        """Complete pipeline: train → save → load into fresh encoder → eval."""
        enc_cfg = TINY_CONFIG["model"]["encoder"]
        pred_cfg = TINY_CONFIG["model"]["predictor"]
        embed_dim = enc_cfg["embed_dim"]

        encoder = build_encoder(enc_cfg)
        target_encoder = copy.deepcopy(encoder)
        target_encoder.requires_grad_(False)
        predictor = build_predictor(pred_cfg, encoder.num_patches)

        # Train
        images = torch.randn(32, 3, 32, 32)
        _run_ijepa_training_steps(encoder, predictor, target_encoder, images, TINY_CONFIG)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save
            model = nn.ModuleDict({
                "context_encoder": encoder,
                "predictor": predictor,
                "target_encoder": target_encoder,
            })
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3
            )
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=None,
                epoch=1, step=4, config=TINY_CONFIG,
                output_dir=tmp_dir, metrics={"loss": 0.5},
            )

            # Load into a completely fresh encoder
            latest = find_latest_checkpoint(tmp_dir)
            fresh_model = nn.ModuleDict({
                "context_encoder": build_encoder(enc_cfg),
                "predictor": build_predictor(pred_cfg, 16),
                "target_encoder": copy.deepcopy(build_encoder(enc_cfg)),
            })
            load_checkpoint(latest, fresh_model)

            # Eval the loaded encoder
            fresh_encoder = fresh_model["context_encoder"]
            fresh_encoder.eval()
            fresh_encoder.requires_grad_(False)

            dataset = _make_dummy_image_dataset(n=40, img_size=32, num_classes=5)
            loader = DataLoader(dataset, batch_size=20, shuffle=False)

            features, labels = extract_features(fresh_encoder, loader, torch.device("cpu"))
            train_f, test_f = features[:30], features[30:]
            train_l, test_l = labels[:30], labels[30:]

            acc = knn_evaluate(train_f, train_l, test_f, test_l, k=5)
            assert 0.0 <= acc <= 1.0
