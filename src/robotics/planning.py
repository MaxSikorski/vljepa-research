"""
Goal-conditioned planning utilities for robotics.

Integrates VL-JEPA's language understanding with the action-conditioned
world model for language-guided robot control:

"Pick up the red cup" →
  1. VL-JEPA encodes the language goal into embedding space
  2. World model predicts consequences of action sequences
  3. MPC planner finds actions that reach the goal embedding
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.robotics.ac_predictor import ActionConditionedPredictor, GoalConditionedPlanner


class LanguageConditionedPlanner:
    """
    Plan robot actions from natural language instructions.

    Combines VL-JEPA's vision-language understanding with
    the action-conditioned world model for language-guided control.
    """

    def __init__(
        self,
        vljepa_model: nn.Module,
        ac_predictor: ActionConditionedPredictor,
        action_dim: int = 7,
        horizon: int = 16,
    ):
        self.vljepa = vljepa_model
        self.planner = GoalConditionedPlanner(
            predictor=ac_predictor,
            encoder=vljepa_model.x_encoder,
            action_dim=action_dim,
            horizon=horizon,
        )

    @torch.no_grad()
    def plan_from_language(
        self,
        observation: torch.Tensor,
        instruction: str,
        tokenizer,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate action plan from language instruction.

        Args:
            observation: (1, C, H, W) current camera image
            instruction: e.g. "Pick up the red cup"
            tokenizer: text tokenizer

        Returns:
            action_sequence: (horizon, action_dim)
        """
        # Encode instruction as goal embedding using Y-Encoder
        ids = tokenizer.encode(instruction, return_tensors="pt").to(device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        goal_embedding = self.vljepa.y_encoder(ids, mask)  # (1, D)

        # Plan using MPC with world model
        actions = self.planner.plan(observation, goal_embedding, device)

        return actions

    @torch.no_grad()
    def plan_from_goal_image(
        self,
        observation: torch.Tensor,
        goal_image: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate action plan from goal image.

        Args:
            observation: (1, C, H, W) current state
            goal_image: (1, C, H, W) desired goal state

        Returns:
            action_sequence: (horizon, action_dim)
        """
        # Encode goal image
        goal_features = self.vljepa.x_encoder(goal_image)
        goal_embedding = goal_features.mean(dim=1)  # Pool to single vector
        goal_embedding = F.normalize(goal_embedding, dim=-1)

        # Plan
        actions = self.planner.plan(observation, goal_embedding, device)
        return actions
