"""
Action-Conditioned Predictor for Robotics.

Extends the JEPA predictor to take actions as input:
  (current_visual_state, action) → predicted_future_visual_state

This enables world modeling for robots:
- Given current camera observation + proposed action
- Predict what the world will look like after the action
- Use this for planning (Model Predictive Control)

Based on V-JEPA 2-AC architecture, which was trained on 62 hours
of Droid robot data and demonstrated zero-shot Franka arm control.

Loss function follows EB-JEPA (arXiv:2602.03604):
  L = L_pred + 16*L_var + 8*L_cov + 12*L_sim + 1*L_IDM
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ijepa.models.encoder import TransformerBlock


# ---------------------------------------------------------------------------
# VICReg + IDM Regularization Losses (from EB-JEPA)
# ---------------------------------------------------------------------------

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance regularization (Bardes et al., 2022).

    Prevents representation collapse in the action-conditioned predictor.
    Without this, the predictor learns to map everything to the same point.

    Components:
      - Variance loss: keeps std(z) >= gamma (prevents collapse to a point)
      - Covariance loss: decorrelates feature dimensions (prevents dimensional collapse)
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Encourage each feature dimension to have variance >= gamma."""
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return F.relu(self.gamma - std).mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Decorrelate feature dimensions (off-diagonal covariance → 0)."""
        N, D = z.shape
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / max(N - 1, 1)
        # Zero out the diagonal, penalize off-diagonal
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / D

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "var_loss": self.variance_loss(z),
            "cov_loss": self.covariance_loss(z),
        }


class TemporalSimilarityLoss(nn.Module):
    """
    Encourages smooth representation trajectories over time.

    L_sim = sum(||z_t - z_{t+1}||^2) for consecutive timesteps.
    This prevents the model from learning discontinuous representations
    while still allowing gradual change.
    """

    def forward(self, z_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_sequence: (B, T, D) sequence of representations
        Returns:
            Scalar temporal similarity loss
        """
        if z_sequence.shape[1] < 2:
            return torch.tensor(0.0, device=z_sequence.device)
        diffs = z_sequence[:, 1:] - z_sequence[:, :-1]
        return diffs.pow(2).mean()


class InverseDynamicsModel(nn.Module):
    """
    Predicts actions from consecutive state representations.

    L_IDM = ||a_t - MLP(z_t, z_{t+1})||^2

    This ensures that representations retain enough information
    to infer what action caused the state transition.
    """

    def __init__(self, embed_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        z_next: torch.Tensor,
        actions_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, D) current state representations
            z_next: (B, D) next state representations
            actions_gt: (B, action_dim) ground-truth actions
        Returns:
            Scalar IDM loss
        """
        pred_actions = self.net(torch.cat([z_t, z_next], dim=-1))
        return F.mse_loss(pred_actions, actions_gt)


class ACPredictorLoss(nn.Module):
    """
    Combined loss for the action-conditioned predictor.

    L = L_pred + var_coeff*L_var + cov_coeff*L_cov + sim_coeff*L_sim + idm_coeff*L_IDM

    Default coefficients from EB-JEPA (arXiv:2602.03604):
      var_coeff=16, cov_coeff=8, sim_coeff=12, idm_coeff=1
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        var_coeff: float = 16.0,
        cov_coeff: float = 8.0,
        sim_coeff: float = 12.0,
        idm_coeff: float = 1.0,
    ):
        super().__init__()
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.sim_coeff = sim_coeff
        self.idm_coeff = idm_coeff

        self.vicreg = VICRegLoss()
        self.temporal_sim = TemporalSimilarityLoss()
        self.idm = InverseDynamicsModel(embed_dim, action_dim)

    def forward(
        self,
        predicted_states: torch.Tensor,
        target_states: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            predicted_states: (B, T, D) predicted future representations
            target_states: (B, T, D) ground-truth future representations
            actions: (B, T, action_dim) actions taken (for IDM loss)
        Returns:
            Dict with total loss and component losses
        """
        B, T, D = predicted_states.shape

        # Prediction loss (L2)
        pred_loss = F.mse_loss(predicted_states, target_states.detach())

        # VICReg on predicted representations (flatten B*T for statistics)
        z_flat = predicted_states.reshape(-1, D)
        vicreg = self.vicreg(z_flat)
        var_loss = vicreg["var_loss"]
        cov_loss = vicreg["cov_loss"]

        # Temporal similarity
        sim_loss = self.temporal_sim(predicted_states)

        # Inverse dynamics model
        idm_loss = torch.tensor(0.0, device=predicted_states.device)
        if actions is not None and T >= 2:
            # Use predicted states for consecutive pairs
            z_t = predicted_states[:, :-1].reshape(-1, D)
            z_next = predicted_states[:, 1:].reshape(-1, D)
            a_t = actions[:, :-1].reshape(-1, actions.shape[-1])
            idm_loss = self.idm(z_t, z_next, a_t)

        total_loss = (
            pred_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
            + self.sim_coeff * sim_loss
            + self.idm_coeff * idm_loss
        )

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "var_loss": var_loss,
            "cov_loss": cov_loss,
            "sim_loss": sim_loss,
            "idm_loss": idm_loss,
        }


class ActionConditionedPredictor(nn.Module):
    """
    Predicts future visual state given current state + action.

    Used for:
    - World modeling: "What happens if I do X?"
    - Goal-conditioned planning via MPC
    - Zero-shot task transfer
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        action_dim: int = 7,  # 7-DoF robot arm
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        prediction_horizon: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.prediction_horizon = prediction_horizon

        # Action embedding
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Temporal position embeddings for prediction steps
        self.temporal_pos = nn.Parameter(
            torch.zeros(1, prediction_horizon, embed_dim)
        )

        # Transformer blocks with block-causal attention
        # (autoregressive over prediction steps)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def forward(
        self,
        visual_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future visual states.

        Args:
            visual_state: (B, N_patches, D) current visual encoding
            actions: (B, T, action_dim) sequence of actions

        Returns:
            Predicted future states (B, T, D)
        """
        B = visual_state.shape[0]
        T = actions.shape[1]

        # Embed actions
        action_emb = self.action_proj(actions)  # (B, T, D)

        # Add temporal position
        action_emb = action_emb + self.temporal_pos[:, :T, :]

        # Pool visual state to single token
        vis_token = visual_state.mean(dim=1, keepdim=True)  # (B, 1, D)

        # Concatenate visual context + action sequence
        x = torch.cat([vis_token, action_emb], dim=1)  # (B, 1+T, D)

        # Process through transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract predictions (skip the visual context token)
        predictions = x[:, 1:, :]  # (B, T, D)
        predictions = self.output_proj(predictions)

        return predictions

    def predict_single_step(
        self,
        visual_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict one step ahead.

        Args:
            visual_state: (B, N, D) or (B, D) current state
            action: (B, action_dim) single action

        Returns:
            Predicted next state (B, D)
        """
        if action.dim() == 2:
            action = action.unsqueeze(1)
        if visual_state.dim() == 2:
            visual_state = visual_state.unsqueeze(1)

        pred = self.forward(visual_state, action)
        return pred[:, 0, :]

    def recursive_rollout(
        self,
        visual_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Recursive multi-step rollout for training (EB-JEPA style).

        Predicts step-by-step, feeding each prediction back as input.
        This teaches the model to handle compounding errors during planning.

        Args:
            visual_state: (B, N, D) initial visual state
            actions: (B, K, action_dim) sequence of K actions

        Returns:
            Predicted states at each step (B, K, D)
        """
        B, K = actions.shape[0], actions.shape[1]
        predictions = []
        current_state = visual_state  # (B, N, D)

        for k in range(K):
            action_k = actions[:, k:k+1, :]  # (B, 1, action_dim)
            pred_k = self.forward(current_state, action_k)  # (B, 1, D)
            predictions.append(pred_k[:, 0, :])  # (B, D)
            # Feed prediction back as next state (unsqueeze for patch dim)
            current_state = pred_k  # (B, 1, D) — acts as single-token state

        return torch.stack(predictions, dim=1)  # (B, K, D)


class GoalConditionedPlanner:
    """
    Plan actions using the world model via Model Predictive Control (MPC).

    Given a goal state (specified as an image or embedding), uses
    the action-conditioned predictor to search for action sequences
    that reach the goal.

    Algorithm: Cross-Entropy Method (CEM)
    1. Sample random action sequences
    2. Predict outcomes using world model
    3. Score by similarity to goal
    4. Refit distribution to top-k sequences
    5. Repeat for N iterations
    """

    def __init__(
        self,
        predictor: ActionConditionedPredictor,
        encoder: nn.Module,
        action_dim: int = 7,
        horizon: int = 16,
        num_samples: int = 512,
        num_elites: int = 64,
        num_iterations: int = 5,
        action_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.predictor = predictor
        self.encoder = encoder
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.action_range = action_range

    @torch.no_grad()
    def plan(
        self,
        current_obs: torch.Tensor,
        goal_embedding: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Plan an action sequence to reach the goal.

        Args:
            current_obs: (1, C, H, W) current observation image
            goal_embedding: (1, D) target state embedding

        Returns:
            Best action sequence (horizon, action_dim)
        """
        # Encode current observation
        visual_state = self.encoder(current_obs)  # (1, N, D)

        # Initialize action distribution
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device)

        for iteration in range(self.num_iterations):
            # Sample action sequences
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.num_samples, self.horizon, self.action_dim, device=device
            )
            actions = actions.clamp(*self.action_range)

            # Predict outcomes
            vs_expanded = visual_state.expand(self.num_samples, -1, -1)
            predicted_states = self.predictor(vs_expanded, actions)  # (N, T, D)

            # Score by trajectory-level cost (EB-JEPA: accumulate over all timesteps)
            # This gives +8% success rate vs final-state-only cost
            goal_expanded = F.normalize(
                goal_embedding.expand(self.num_samples, -1), dim=-1
            )  # (N, D)
            all_states_norm = F.normalize(predicted_states, dim=-1)  # (N, T, D)
            # Cosine similarity at each timestep, sum over trajectory
            per_step_sim = torch.einsum(
                "ntd,nd->nt", all_states_norm, goal_expanded
            )  # (N, T)
            scores = per_step_sim.sum(dim=-1)  # (N,)

            # Select elites
            elite_indices = scores.topk(self.num_elites).indices
            elite_actions = actions[elite_indices]

            # Refit distribution
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0).clamp(min=0.01)

        return mean  # Best action sequence
