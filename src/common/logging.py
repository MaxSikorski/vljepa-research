"""
Unified logging with optional Weights & Biases integration.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.common.distributed import is_main_process

_logger: logging.Logger | None = None


def setup_logger(
    output_dir: str | Path,
    name: str = "vljepa",
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up logger with file and console handlers (main process only)."""
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    if is_main_process():
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File handler
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger."""
    if _logger is None:
        return logging.getLogger("vljepa")
    return _logger


class MetricsLogger:
    """Log metrics to file and optionally to W&B."""

    def __init__(
        self,
        output_dir: str | Path,
        wandb_config: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.wandb_run = None

        # Initialize W&B if configured and on main process
        if wandb_config and wandb_config.get("enabled") and is_main_process():
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_config.get("project", "vljepa-research"),
                    tags=wandb_config.get("tags", []),
                    config=wandb_config.get("full_config", {}),
                )
            except ImportError:
                get_logger().warning("wandb not installed, skipping W&B logging")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Log metrics to file and W&B."""
        if not is_main_process():
            return

        metrics["step"] = step

        # Append to JSONL file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        # Log to W&B
        if self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Finish logging."""
        if self.wandb_run is not None:
            import wandb

            wandb.finish()
