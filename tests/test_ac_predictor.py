"""Tests for action-conditioned predictor and robotics losses."""

import torch
import pytest

from src.robotics.ac_predictor import (
    ACPredictorLoss,
    ActionConditionedPredictor,
    GoalConditionedPlanner,
    InverseDynamicsModel,
    TemporalSimilarityLoss,
    VICRegLoss,
)


class TestVICRegLoss:
    def test_collapsed_input_high_variance_loss(self):
        """Constant input should trigger high variance loss."""
        loss_fn = VICRegLoss(gamma=1.0)
        z = torch.ones(32, 64)  # all same — zero variance
        result = loss_fn(z)
        assert result["var_loss"].item() > 0.5  # should be ~gamma

    def test_diverse_input_low_variance_loss(self):
        """High-variance input should have low variance loss."""
        loss_fn = VICRegLoss(gamma=1.0)
        z = torch.randn(32, 64) * 5.0  # high variance
        result = loss_fn(z)
        assert result["var_loss"].item() < 0.1

    def test_uncorrelated_low_covariance_loss(self):
        """Random input should have low off-diagonal covariance."""
        loss_fn = VICRegLoss()
        torch.manual_seed(42)
        z = torch.randn(256, 16)  # large batch, small dim for stats
        result = loss_fn(z)
        assert result["cov_loss"].item() < 1.0

    def test_output_scalars(self):
        loss_fn = VICRegLoss()
        z = torch.randn(16, 32)
        result = loss_fn(z)
        assert result["var_loss"].dim() == 0
        assert result["cov_loss"].dim() == 0


class TestTemporalSimilarityLoss:
    def test_single_timestep_returns_zero(self):
        loss_fn = TemporalSimilarityLoss()
        z = torch.randn(4, 1, 32)  # T=1
        assert loss_fn(z).item() == 0.0

    def test_constant_sequence_returns_zero(self):
        loss_fn = TemporalSimilarityLoss()
        z = torch.ones(4, 5, 32)  # constant over time
        assert loss_fn(z).item() == 0.0

    def test_varying_sequence_positive(self):
        loss_fn = TemporalSimilarityLoss()
        z = torch.randn(4, 5, 32)
        assert loss_fn(z).item() > 0.0


class TestInverseDynamicsModel:
    def test_forward_shape(self):
        idm = InverseDynamicsModel(embed_dim=64, action_dim=7)
        z_t = torch.randn(8, 64)
        z_next = torch.randn(8, 64)
        actions_gt = torch.randn(8, 7)
        loss = idm(z_t, z_next, actions_gt)
        assert loss.dim() == 0  # scalar

    def test_zero_loss_when_perfect(self):
        """IDM should be able to fit trivial patterns."""
        idm = InverseDynamicsModel(embed_dim=4, action_dim=2, hidden_dim=16)
        # Not testing convergence, just that forward works
        z_t = torch.randn(4, 4)
        z_next = torch.randn(4, 4)
        actions = torch.randn(4, 2)
        loss = idm(z_t, z_next, actions)
        assert loss.item() >= 0.0


class TestACPredictorLoss:
    def test_combined_loss(self):
        loss_fn = ACPredictorLoss(embed_dim=32, action_dim=7)
        pred = torch.randn(4, 5, 32)
        target = torch.randn(4, 5, 32)
        actions = torch.randn(4, 5, 7)
        result = loss_fn(pred, target, actions)
        assert "loss" in result
        assert "pred_loss" in result
        assert "var_loss" in result
        assert "cov_loss" in result
        assert "sim_loss" in result
        assert "idm_loss" in result
        assert result["loss"].dim() == 0

    def test_without_actions(self):
        loss_fn = ACPredictorLoss(embed_dim=32, action_dim=7)
        pred = torch.randn(4, 5, 32)
        target = torch.randn(4, 5, 32)
        result = loss_fn(pred, target, actions=None)
        assert result["idm_loss"].item() == 0.0
        assert result["loss"].item() > 0.0

    def test_coefficients_applied(self):
        """Total loss should reflect weighted sum."""
        loss_fn = ACPredictorLoss(
            embed_dim=32, action_dim=7,
            var_coeff=0.0, cov_coeff=0.0, sim_coeff=0.0, idm_coeff=0.0,
        )
        pred = torch.randn(4, 3, 32)
        target = torch.randn(4, 3, 32)
        result = loss_fn(pred, target)
        # With all coefficients zero, total should equal pred_loss
        assert torch.allclose(result["loss"], result["pred_loss"], atol=1e-5)


class TestActionConditionedPredictor:
    def test_forward_shape(self):
        model = ActionConditionedPredictor(
            embed_dim=64, action_dim=7, depth=2, num_heads=4, prediction_horizon=8,
        )
        vis = torch.randn(2, 16, 64)  # (B, N_patches, D)
        actions = torch.randn(2, 5, 7)  # (B, T, action_dim)
        out = model(vis, actions)
        assert out.shape == (2, 5, 64)

    def test_predict_single_step(self):
        model = ActionConditionedPredictor(
            embed_dim=64, action_dim=7, depth=2, num_heads=4,
        )
        vis = torch.randn(2, 16, 64)
        action = torch.randn(2, 7)  # single action
        out = model.predict_single_step(vis, action)
        assert out.shape == (2, 64)

    def test_recursive_rollout(self):
        model = ActionConditionedPredictor(
            embed_dim=64, action_dim=7, depth=2, num_heads=4,
        )
        vis = torch.randn(2, 16, 64)
        actions = torch.randn(2, 4, 7)  # 4-step rollout
        out = model.recursive_rollout(vis, actions)
        assert out.shape == (2, 4, 64)


class TestGoalConditionedPlanner:
    def test_plan_shape(self):
        from src.ijepa.models.encoder import VisionTransformer

        encoder = VisionTransformer(
            img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4,
        )
        predictor = ActionConditionedPredictor(
            embed_dim=64, action_dim=7, depth=2, num_heads=4, prediction_horizon=8,
        )
        planner = GoalConditionedPlanner(
            predictor=predictor,
            encoder=encoder,
            action_dim=7,
            horizon=8,
            num_samples=32,
            num_elites=8,
            num_iterations=2,
        )
        obs = torch.randn(1, 3, 32, 32)
        goal = torch.randn(1, 64)
        plan = planner.plan(obs, goal, device=torch.device("cpu"))
        assert plan.shape == (8, 7)

    def test_actions_in_range(self):
        from src.ijepa.models.encoder import VisionTransformer

        encoder = VisionTransformer(
            img_size=32, patch_size=8, embed_dim=64, depth=2, num_heads=4,
        )
        predictor = ActionConditionedPredictor(
            embed_dim=64, action_dim=7, depth=2, num_heads=4,
        )
        planner = GoalConditionedPlanner(
            predictor=predictor, encoder=encoder,
            action_dim=7, horizon=4, num_samples=16, num_elites=4,
            num_iterations=2, action_range=(-1.0, 1.0),
        )
        obs = torch.randn(1, 3, 32, 32)
        goal = torch.randn(1, 64)
        plan = planner.plan(obs, goal, device=torch.device("cpu"))
        assert plan.min() >= -1.0
        assert plan.max() <= 1.0
