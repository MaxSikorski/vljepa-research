"""Tests for bi-directional InfoNCE loss."""

import torch
import pytest

from src.vljepa.losses.infonce import BidirectionalInfoNCE


class TestInfoNCEGradients:
    def test_gradients_flow(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07)
        predicted = torch.randn(4, 128, requires_grad=True)
        target = torch.randn(4, 128, requires_grad=True)
        result = loss_fn(predicted, target)
        result["loss"].backward()
        assert predicted.grad is not None
        assert target.grad is not None

    def test_learnable_temperature(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07, learnable_temperature=True)
        assert loss_fn.log_temperature.requires_grad
        predicted = torch.randn(4, 128)
        target = torch.randn(4, 128)
        result = loss_fn(predicted, target)
        result["loss"].backward()
        assert loss_fn.log_temperature.grad is not None

    def test_batch_size_one(self):
        loss_fn = BidirectionalInfoNCE(temperature=0.07)
        predicted = torch.randn(1, 128)
        target = torch.randn(1, 128)
        result = loss_fn(predicted, target)
        assert not torch.isnan(result["loss"])
