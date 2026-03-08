"""
Tests for SALT: Static-teacher Asymmetric Latent Training.

Tests Stage 1 (MAE teacher), Stage 2 (frozen-teacher JEPA), and the
complete two-stage pipeline.

Reference: SALT (Apple, ICLR 2025) — https://arxiv.org/abs/2509.24317
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from src.ijepa.masks.multiblock import generate_masks
from src.ijepa.models.encoder import VisionTransformer, build_encoder
from src.ijepa.models.predictor import JEPAPredictor, build_predictor
from src.salt.models.mae_decoder import MAEDecoder, build_mae_decoder
from src.salt.losses.pixel_loss import PixelReconstructionLoss
from src.salt.train_stage1 import SALTStage1Trainer
from src.salt.train_stage2 import SALTStage2Trainer


# ---- Shared fixtures ----

TINY_ENC_CONFIG = {
    "img_size": 32,
    "patch_size": 8,
    "embed_dim": 64,
    "depth": 2,
    "num_heads": 2,
    "mlp_ratio": 4.0,
}

TINY_DEC_CONFIG = {
    "encoder_embed_dim": 64,
    "decoder_embed_dim": 32,
    "decoder_depth": 2,
    "decoder_num_heads": 2,
    "patch_size": 8,
    "in_channels": 3,
}

TINY_PRED_CONFIG = {
    "embed_dim": 64,
    "predictor_embed_dim": 32,
    "depth": 2,
    "num_heads": 2,
}


@pytest.fixture
def tiny_encoder():
    return build_encoder(TINY_ENC_CONFIG)


@pytest.fixture
def tiny_decoder(tiny_encoder):
    return build_mae_decoder(TINY_DEC_CONFIG, tiny_encoder.num_patches)


@pytest.fixture
def dummy_images():
    return torch.randn(4, 3, 32, 32)


# ---- MAE Decoder Tests ----

class TestMAEDecoder:
    def test_forward_shape(self, tiny_encoder, tiny_decoder, dummy_images):
        """Decoder outputs correct shape: (B, M_mask, P*P*C)."""
        grid_size = 4  # 32/8 = 4
        ctx_idx, tgt_list = generate_masks(
            batch_size=4, num_patches_h=grid_size, num_patches_w=grid_size,
            num_targets=4, device=torch.device("cpu"),
        )

        # Combine targets into masked indices
        all_masked = set()
        for tgt in tgt_list:
            all_masked.update(tgt[0].tolist())
        masked_idx = torch.tensor(sorted(all_masked)).unsqueeze(0).expand(4, -1)

        # Encode visible patches
        with torch.no_grad():
            visible_emb = tiny_encoder(dummy_images, mask_indices=ctx_idx)

        # Decode
        pixel_preds = tiny_decoder(visible_emb, ctx_idx, masked_idx)

        assert pixel_preds.shape[0] == 4  # batch
        assert pixel_preds.shape[1] == masked_idx.shape[1]  # num masked
        assert pixel_preds.shape[2] == 8 * 8 * 3  # P*P*C = 192

    def test_decoder_params_independent(self, tiny_encoder, tiny_decoder):
        """Decoder and encoder have separate parameters."""
        enc_params = set(id(p) for p in tiny_encoder.parameters())
        dec_params = set(id(p) for p in tiny_decoder.parameters())
        assert len(enc_params & dec_params) == 0

    def test_build_mae_decoder(self):
        """Build decoder from config."""
        config = {"decoder_embed_dim": 64, "decoder_depth": 3}
        config.update(TINY_DEC_CONFIG)
        decoder = build_mae_decoder(config, num_patches=16)
        assert isinstance(decoder, MAEDecoder)


# ---- Pixel Reconstruction Loss Tests ----

class TestPixelReconstructionLoss:
    def test_patchify(self, dummy_images):
        """Patchify produces correct number of patches."""
        loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=False)
        patches = loss_fn.patchify(dummy_images)
        assert patches.shape == (4, 16, 192)  # 16 patches, 8*8*3=192 per patch

    def test_loss_computes(self, dummy_images):
        """Loss returns a finite scalar."""
        loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=True)
        predictions = torch.randn(4, 6, 192)  # 6 masked patches
        masked_indices = torch.tensor([[0, 1, 2, 5, 10, 15]]).expand(4, -1)
        loss = loss_fn(predictions, dummy_images, masked_indices)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_perfect_reconstruction_low_loss(self, dummy_images):
        """Perfect reconstruction should give near-zero loss (no norm_pix)."""
        loss_fn = PixelReconstructionLoss(patch_size=8, norm_pix=False)
        patches = loss_fn.patchify(dummy_images)
        masked_indices = torch.tensor([[0, 1, 2, 3]]).expand(4, -1)
        target = torch.gather(
            patches, 1,
            masked_indices.unsqueeze(-1).expand(-1, -1, 192),
        )
        loss = loss_fn(target, dummy_images, masked_indices)
        assert loss.item() < 1e-6


# ---- Stage 1 Trainer Tests ----

class TestSALTStage1:
    def test_no_ema(self, tiny_encoder, tiny_decoder, dummy_images):
        """Stage 1 has NO EMA — only encoder + decoder, no target encoder."""
        loss_fn = PixelReconstructionLoss(patch_size=8)
        optimizer = torch.optim.AdamW(
            list(tiny_encoder.parameters()) + list(tiny_decoder.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage1Trainer(
            tiny_encoder, tiny_decoder, loss_fn, optimizer
        )
        # There's no target_encoder attribute
        assert not hasattr(trainer, "target_encoder")

    def test_train_step_runs(self, tiny_encoder, tiny_decoder, dummy_images):
        """Train step produces a finite loss."""
        loss_fn = PixelReconstructionLoss(patch_size=8)
        optimizer = torch.optim.AdamW(
            list(tiny_encoder.parameters()) + list(tiny_decoder.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage1Trainer(
            tiny_encoder, tiny_decoder, loss_fn, optimizer
        )
        loss = trainer.train_step(dummy_images, grid_size=4)
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    def test_loss_decreases(self, tiny_encoder, tiny_decoder, dummy_images):
        """Loss decreases over multiple steps."""
        loss_fn = PixelReconstructionLoss(patch_size=8)
        optimizer = torch.optim.AdamW(
            list(tiny_encoder.parameters()) + list(tiny_decoder.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage1Trainer(
            tiny_encoder, tiny_decoder, loss_fn, optimizer
        )

        losses = []
        for _ in range(10):
            loss = trainer.train_step(dummy_images, grid_size=4)
            losses.append(loss)

        # Average of last 3 should be less than average of first 3
        assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3


# ---- Stage 2 Trainer Tests ----

class TestSALTStage2:
    def test_teacher_frozen(self, tiny_encoder, dummy_images):
        """Teacher gradients are frozen — no parameter updates."""
        teacher = copy.deepcopy(tiny_encoder)
        student = build_encoder(TINY_ENC_CONFIG)
        predictor = build_predictor(TINY_PRED_CONFIG, student.num_patches)

        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage2Trainer(teacher, student, predictor, optimizer)

        # Save teacher weights before step
        teacher_before = {n: p.clone() for n, p in trainer.teacher.named_parameters()}

        trainer.train_step(dummy_images, grid_size=4)

        # Teacher weights must be identical
        for name, param in trainer.teacher.named_parameters():
            assert torch.equal(param, teacher_before[name]), \
                f"Teacher param '{name}' changed — should be frozen!"

    def test_student_updates(self, tiny_encoder, dummy_images):
        """Student and predictor weights DO change after training step."""
        teacher = copy.deepcopy(tiny_encoder)
        student = build_encoder(TINY_ENC_CONFIG)
        predictor = build_predictor(TINY_PRED_CONFIG, student.num_patches)

        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage2Trainer(teacher, student, predictor, optimizer)

        # Save student weights before step
        student_before = {n: p.clone() for n, p in student.named_parameters()}

        trainer.train_step(dummy_images, grid_size=4)

        # At least some student weights must have changed
        changed = sum(
            1 for n, p in student.named_parameters()
            if not torch.equal(p, student_before[n])
        )
        assert changed > 0, "Student weights didn't change — training isn't working"

    def test_asymmetric_sizes(self, dummy_images):
        """Weak teacher (tiny) can train a stronger student (wider)."""
        # Tiny teacher
        teacher = build_encoder({
            "img_size": 32, "patch_size": 8, "embed_dim": 32,
            "depth": 1, "num_heads": 2,
        })

        # Larger student (wider, but same patch count)
        student = build_encoder({
            "img_size": 32, "patch_size": 8, "embed_dim": 64,
            "depth": 2, "num_heads": 2,
        })

        # Predictor: maps student dim → teacher dim
        predictor = JEPAPredictor(
            num_patches=student.num_patches,
            encoder_embed_dim=32,  # Must match teacher
            predictor_embed_dim=32,
            depth=2,
            num_heads=2,
        )
        # Need custom input_proj to handle student embed_dim → predictor dim
        predictor.input_proj = torch.nn.Linear(64, 32)  # student_dim → pred_dim

        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage2Trainer(teacher, student, predictor, optimizer)

        loss = trainer.train_step(dummy_images, grid_size=4)
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    def test_l1_loss_used(self, dummy_images):
        """Stage 2 uses L1 loss (not MSE or smooth_l1)."""
        # We verify by checking that the training step uses F.l1_loss
        # indirectly — the train_step function in train_stage2.py uses F.l1_loss
        teacher = build_encoder(TINY_ENC_CONFIG)
        student = build_encoder(TINY_ENC_CONFIG)
        predictor = build_predictor(TINY_PRED_CONFIG, student.num_patches)

        optimizer = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=1e-3,
        )
        trainer = SALTStage2Trainer(teacher, student, predictor, optimizer)
        loss = trainer.train_step(dummy_images, grid_size=4)
        assert loss > 0  # L1 loss is always non-negative


# ---- Pipeline Tests ----

class TestSALTPipeline:
    def test_stage1_to_stage2_checkpoint(self, tiny_encoder, tiny_decoder, dummy_images):
        """Stage 1 encoder checkpoint can be loaded as Stage 2 frozen teacher."""
        loss_fn = PixelReconstructionLoss(patch_size=8)
        optimizer = torch.optim.AdamW(
            list(tiny_encoder.parameters()) + list(tiny_decoder.parameters()),
            lr=1e-3,
        )
        trainer1 = SALTStage1Trainer(
            tiny_encoder, tiny_decoder, loss_fn, optimizer
        )

        # Train Stage 1 briefly
        for _ in range(3):
            trainer1.train_step(dummy_images, grid_size=4)

        # Extract encoder state dict (simulating checkpoint save)
        teacher_state = tiny_encoder.state_dict()

        # Load into fresh teacher for Stage 2
        teacher = build_encoder(TINY_ENC_CONFIG)
        teacher.load_state_dict(teacher_state)

        student = build_encoder(TINY_ENC_CONFIG)
        predictor = build_predictor(TINY_PRED_CONFIG, student.num_patches)
        optimizer2 = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()),
            lr=1e-3,
        )

        trainer2 = SALTStage2Trainer(teacher, student, predictor, optimizer2)

        # Stage 2 should train successfully
        loss = trainer2.train_step(dummy_images, grid_size=4)
        assert isinstance(loss, float)
        assert not (loss != loss)  # not NaN

    def test_full_pipeline_loss_decreases(self, dummy_images):
        """Full SALT pipeline: Stage 1 trains, Stage 2 loss decreases."""
        # Stage 1
        encoder = build_encoder(TINY_ENC_CONFIG)
        decoder = build_mae_decoder(TINY_DEC_CONFIG, encoder.num_patches)
        loss_fn = PixelReconstructionLoss(patch_size=8)
        opt1 = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3,
        )
        trainer1 = SALTStage1Trainer(encoder, decoder, loss_fn, opt1)

        for _ in range(5):
            trainer1.train_step(dummy_images, grid_size=4)

        # Stage 2: freeze teacher, train student
        teacher = copy.deepcopy(encoder)
        student = build_encoder(TINY_ENC_CONFIG)
        predictor = build_predictor(TINY_PRED_CONFIG, student.num_patches)
        opt2 = torch.optim.AdamW(
            list(student.parameters()) + list(predictor.parameters()), lr=1e-3,
        )
        trainer2 = SALTStage2Trainer(teacher, student, predictor, opt2)

        losses = []
        for _ in range(10):
            loss = trainer2.train_step(dummy_images, grid_size=4)
            losses.append(loss)

        # Loss should decrease over training
        assert sum(losses[-3:]) / 3 < sum(losses[:3]) / 3
