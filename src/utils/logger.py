"""Logging utility module providing centralized logger configuration.

Reads log level and format from the settings YAML file.
Supports both console and rotating file output handlers.
"""

import logging  # Standard library logging framework
import os  # File system operations
from logging.handlers import RotatingFileHandler  # Size-based log rotation
from pathlib import Path  # Cross-platform file paths

import yaml  # YAML configuration parsing


def _load_log_settings() -> dict:
    """Load logging configuration from the settings YAML file.

    Returns:
        Dictionary containing logging configuration values.
    """
    # Default logging configuration as fallback
    defaults = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_dir": "logs",
        "log_file": "evaluation.log",
        "max_file_size": 5242880,
        "backup_count": 3,
    }

    # Attempt to read settings from the config file
    config_path = Path("config/settings.yaml")
    # Check if config file exists on disk
    if not config_path.exists():
        return defaults

    try:
        # Open and parse the YAML config file
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        # Extract logging section or use empty dict
        log_config = config.get("logging", {})
        # Merge file settings with defaults for any missing keys
        for key in defaults:
            if key not in log_config:
                log_config[key] = defaults[key]
        return log_config
    except (yaml.YAMLError, OSError):
        # Return defaults if file cannot be read or parsed
        return defaults


def get_logger(name: str) -> logging.Logger:
    """Create and configure a logger instance with console and file handlers.

    Args:
        name: The name for the logger, typically __name__ of the calling module.

    Returns:
        Configured Logger instance ready for use.
    """
    # Create a logger with the given module name
    logger = logging.getLogger(name)

    # Only configure handlers if they haven't been added yet
    if not logger.handlers:
        # Load settings from configuration file
        settings = _load_log_settings()

        # Set the minimum log level from configuration
        log_level = getattr(logging, settings["level"].upper(), logging.INFO)
        logger.setLevel(log_level)

        # Create a consistent format for all handlers
        formatter = logging.Formatter(settings["format"])

        # Configure console handler for terminal output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Configure rotating file handler for persistent logs
        log_dir = Path(settings["log_dir"])
        # Create log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)
        # Build full path to the log file
        log_file_path = log_dir / settings["log_file"]

        # Create handler with size-based rotation
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=settings["max_file_size"],
            backupCount=settings["backup_count"],
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent log messages from propagating to the root logger
        logger.propagate = False

    return logger
