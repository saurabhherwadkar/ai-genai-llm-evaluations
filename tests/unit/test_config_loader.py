"""Unit tests for the ConfigLoader utility module."""

from pathlib import Path  # Path handling
from unittest.mock import mock_open, patch  # Mocking utilities

import pytest  # Test framework

from src.utils.config_loader import ConfigLoader  # Module under test


class TestConfigLoader:
    """Test suite for ConfigLoader configuration reading and merging."""

    def test_load_base_config_successfully(self, tmp_path: Path) -> None:
        """Verify base configuration file is loaded correctly."""
        # Arrange: create a temporary YAML config file
        config_content = "logging:\n  level: DEBUG\nllm:\n  model: gpt-4\n"
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Act: load configuration from the temporary file
        loader = ConfigLoader(config_path=config_file)

        # Assert: verify values are accessible
        assert loader.get("logging.level") == "DEBUG"
        assert loader.get("llm.model") == "gpt-4"

    def test_get_nested_value_with_dot_notation(self, tmp_path: Path) -> None:
        """Verify dot-notation access works for deeply nested values."""
        # Arrange: create config with nested structure
        config_content = "a:\n  b:\n    c: deep_value\n"
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Act: load and access nested value
        loader = ConfigLoader(config_path=config_file)

        # Assert: dot notation resolves to the correct value
        assert loader.get("a.b.c") == "deep_value"

    def test_get_returns_default_for_missing_key(self, tmp_path: Path) -> None:
        """Verify default value is returned when key path does not exist."""
        # Arrange: create minimal config
        config_content = "logging:\n  level: INFO\n"
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Act: request a non-existent key with a default
        loader = ConfigLoader(config_path=config_file)
        result = loader.get("nonexistent.key", "fallback_value")

        # Assert: the default is returned
        assert result == "fallback_value"

    def test_get_returns_none_for_missing_key_without_default(self, tmp_path: Path) -> None:
        """Verify None is returned when key is missing and no default given."""
        # Arrange: create minimal config
        config_content = "logging:\n  level: INFO\n"
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Act: request non-existent key without default
        loader = ConfigLoader(config_path=config_file)
        result = loader.get("nonexistent.key")

        # Assert: None is returned
        assert result is None

    def test_environment_override_merges_correctly(self, tmp_path: Path) -> None:
        """Verify environment-specific config overrides base values."""
        # Arrange: create base and environment configs
        base_content = "logging:\n  level: INFO\n  format: basic\n"
        env_content = "logging:\n  level: DEBUG\n"
        base_file = tmp_path / "settings.yaml"
        env_file = tmp_path / "settings.test.yaml"
        base_file.write_text(base_content, encoding="utf-8")
        env_file.write_text(env_content, encoding="utf-8")

        # Act: load with test environment
        loader = ConfigLoader(config_path=base_file, environment="test")

        # Assert: override value takes precedence, base values preserved
        assert loader.get("logging.level") == "DEBUG"
        assert loader.get("logging.format") == "basic"

    def test_raises_file_not_found_for_missing_config(self) -> None:
        """Verify FileNotFoundError is raised when config file doesn't exist."""
        # Arrange: use a path that does not exist
        nonexistent_path = Path("/nonexistent/path/settings.yaml")

        # Act & Assert: verify the exception is raised
        with pytest.raises(FileNotFoundError):
            ConfigLoader(config_path=nonexistent_path)

    def test_config_property_returns_copy(self, tmp_path: Path) -> None:
        """Verify config property returns a copy, not the internal state."""
        # Arrange: create a simple config
        config_content = "key: value\n"
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Act: get config and modify the returned dict
        loader = ConfigLoader(config_path=config_file)
        config_copy = loader.config
        config_copy["key"] = "modified"

        # Assert: internal state is not affected
        assert loader.get("key") == "value"

    def test_deep_merge_handles_nested_dicts(self) -> None:
        """Verify deep merge recursively combines nested dictionaries."""
        # Arrange: two nested dictionaries
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}

        # Act: perform deep merge
        result = ConfigLoader._deep_merge(base, override)

        # Assert: merged correctly with override taking precedence
        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}

    def test_empty_yaml_file_returns_empty_dict(self, tmp_path: Path) -> None:
        """Verify an empty YAML file results in empty configuration."""
        # Arrange: create an empty file
        config_file = tmp_path / "settings.yaml"
        config_file.write_text("", encoding="utf-8")

        # Act: load the empty config
        loader = ConfigLoader(config_path=config_file)

        # Assert: config is an empty dict
        assert loader.config == {}
