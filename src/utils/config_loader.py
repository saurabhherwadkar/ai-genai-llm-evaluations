"""Configuration loader module for reading and merging YAML settings files."""

import os  # File path operations
from pathlib import Path  # Cross-platform path handling
from typing import Any  # Type annotation support

import yaml  # YAML parsing library

from src.utils.logger import get_logger  # Application logger


# Initialize module-level logger instance
logger = get_logger(__name__)


class ConfigLoader:
    """Loads and merges configuration from YAML files with environment awareness.

    Supports base settings with environment-specific overrides.
    Reads the APP_ENV environment variable to determine which overlay to apply.
    """

    # Default path to the base configuration file
    _DEFAULT_CONFIG_PATH = Path("config/settings.yaml")

    def __init__(self, config_path: Path | None = None, environment: str | None = None) -> None:
        """Initialize the configuration loader.

        Args:
            config_path: Optional override for base config file location.
            environment: Optional environment name (e.g., 'test', 'production').
        """
        # Use provided path or fall back to default location
        self._config_path = config_path or self._DEFAULT_CONFIG_PATH
        # Read environment from parameter or APP_ENV variable
        self._environment = environment or os.getenv("APP_ENV", "")
        # Dictionary to hold the merged configuration
        self._config: dict[str, Any] = {}
        # Load configuration on initialization
        self._load_config()

    def _load_config(self) -> None:
        """Load base config and merge with environment-specific overrides."""
        # Load the base configuration file
        self._config = self._read_yaml_file(self._config_path)
        logger.info("Loaded base configuration from %s", self._config_path)

        # Check if an environment-specific override file exists
        if self._environment:
            # Construct the environment-specific filename
            env_config_path = self._config_path.parent / f"settings.{self._environment}.yaml"
            # Only merge if the file actually exists on disk
            if env_config_path.exists():
                # Read the override configuration
                env_config = self._read_yaml_file(env_config_path)
                # Deep merge override values into base config
                self._config = self._deep_merge(self._config, env_config)
                logger.info("Merged environment config from %s", env_config_path)
            else:
                # Log warning if expected override file is missing
                logger.warning("Environment config not found: %s", env_config_path)

    @staticmethod
    def _read_yaml_file(file_path: Path) -> dict[str, Any]:
        """Read and parse a YAML file into a dictionary.

        Args:
            file_path: Path to the YAML file to read.

        Returns:
            Parsed dictionary from the YAML content.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            yaml.YAMLError: If the file contains invalid YAML syntax.
        """
        # Verify the file exists before attempting to read
        if not file_path.exists():
            logger.error("Configuration file not found: %s", file_path)
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        # Open and parse the YAML file safely
        with open(file_path, "r", encoding="utf-8") as config_file:
            try:
                # Use safe_load to prevent arbitrary code execution
                content = yaml.safe_load(config_file)
                # Return empty dict if file is empty
                return content if content is not None else {}
            except yaml.YAMLError as parse_error:
                logger.error("Failed to parse YAML file %s: %s", file_path, parse_error)
                raise

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge override dictionary into base dictionary.

        Args:
            base: The base dictionary to merge into.
            override: The dictionary with values to overlay.

        Returns:
            Merged dictionary with override values taking precedence.
        """
        # Create a copy to avoid mutating the original base dict
        merged = base.copy()
        # Iterate through each key in the override dictionary
        for key, value in override.items():
            # If both values are dicts, recurse into them
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader._deep_merge(merged[key], value)
            else:
                # Otherwise override the value directly
                merged[key] = value
        # Return the fully merged dictionary
        return merged

    def get(self, key_path: str, default: Any = None) -> Any:
        """Retrieve a configuration value using dot-notation path.

        Args:
            key_path: Dot-separated path to the desired value (e.g., 'logging.level').
            default: Value to return if the key path does not exist.

        Returns:
            The configuration value or the provided default.
        """
        # Split the dot-notation path into individual keys
        keys = key_path.split(".")
        # Start traversal from the root config dictionary
        current = self._config
        # Walk through each key in the path
        for key in keys:
            # Check if current level is a dict with the required key
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                # Key not found, return the default value
                logger.debug("Config key '%s' not found, using default: %s", key_path, default)
                return default
        # Return the found value
        return current

    @property
    def config(self) -> dict[str, Any]:
        """Provide read-only access to the full configuration dictionary.

        Returns:
            Complete merged configuration dictionary.
        """
        return self._config.copy()
