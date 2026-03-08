"""Tests for training utilities: schedulers, EMA, AMP setup."""

import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


class TestWarmupConstantDecayScheduler:
    """Test the warmup_constant_decay scheduler from vljepa/train.py."""

    def _make_scheduler(self, warmup=100, constant=300, decay=100):
        model = torch.nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=1.0)

        def _wcd_lambda(step: int) -> float:
            if step < warmup:
                return step / max(warmup, 1)
            elif step < warmup + constant:
                return 1.0
            else:
                progress = (step - warmup - constant) / max(decay, 1)
                return max(0.0, 1.0 - progress)

        scheduler = LambdaLR(optimizer, lr_lambda=_wcd_lambda)
        return optimizer, scheduler

    def test_warmup_phase(self):
        opt, sched = self._make_scheduler(warmup=100, constant=300, decay=100)
        # At step 0, LR factor = 0
        assert opt.param_groups[0]["lr"] == 0.0
        # Step through warmup
        for _ in range(50):
            sched.step()
        assert abs(opt.param_groups[0]["lr"] - 0.5) < 0.02

    def test_constant_phase(self):
        opt, sched = self._make_scheduler(warmup=100, constant=300, decay=100)
        for _ in range(100):
            sched.step()
        # Should be at full LR
        assert abs(opt.param_groups[0]["lr"] - 1.0) < 0.02
        # Stay constant
        for _ in range(200):
            sched.step()
        assert abs(opt.param_groups[0]["lr"] - 1.0) < 0.02

    def test_decay_phase(self):
        opt, sched = self._make_scheduler(warmup=100, constant=300, decay=100)
        for _ in range(500):
            sched.step()
        # After warmup(100) + constant(300) + decay(100) = 500 steps, LR = 0
        assert opt.param_groups[0]["lr"] < 0.02


class TestEMAMomentumSchedule:
    """Test the EMA momentum schedule from ijepa/train.py."""

    @staticmethod
    def _get_ema_momentum(epoch: int, total_epochs: int, start: float, end: float) -> float:
        progress = epoch / max(total_epochs - 1, 1)
        return end - (end - start) * (1 + math.cos(math.pi * progress)) / 2

    def test_starts_at_start_value(self):
        m = self._get_ema_momentum(0, 100, start=0.996, end=1.0)
        assert abs(m - 0.996) < 1e-6

    def test_ends_near_end_value(self):
        m = self._get_ema_momentum(99, 100, start=0.996, end=1.0)
        assert abs(m - 1.0) < 1e-6

    def test_monotonically_increases(self):
        values = [
            self._get_ema_momentum(e, 100, start=0.996, end=1.0) for e in range(100)
        ]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1] - 1e-10
