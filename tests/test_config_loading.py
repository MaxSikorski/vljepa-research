"""Tests for configuration loading utilities."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.common.config import load_config, save_config, get_nested, merge_configs


class TestConfigLoading:
    def test_load_config(self, tmp_path):
        config = {"model": {"embed_dim": 768}, "training": {"epochs": 100}}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        loaded = load_config(config_path)
        assert loaded["model"]["embed_dim"] == 768
        assert loaded["training"]["epochs"] == 100

    def test_save_config(self, tmp_path):
        config = {"model": {"embed_dim": 768}, "_config_path": "/tmp/test.yaml"}
        save_config(config, tmp_path / "saved.yaml")
        loaded = load_config(tmp_path / "saved.yaml")
        assert "_config_path" not in loaded or loaded.get("_config_path") != "/tmp/test.yaml"

    def test_get_nested(self):
        config = {"model": {"encoder": {"embed_dim": 768}}}
        assert get_nested(config, "model.encoder.embed_dim") == 768
        assert get_nested(config, "model.missing.key", default=42) == 42

    def test_merge_configs(self):
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 10}, "e": 5}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"]["c"] == 10
        assert merged["b"]["d"] == 3
        assert merged["e"] == 5

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")
