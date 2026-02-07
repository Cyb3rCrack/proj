"""Configuration management system for ACE.

Supports YAML/JSON config files with environment variable overrides.
Provides centralized configuration for all ACE components.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("ZYPHERUS.Config")


class ConfigManager:
    """Centralized configuration management with env var support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config manager.
        
        Args:
            config_path: Path to config file (YAML or JSON). If None, uses defaults.
        """
        self._config: Dict[str, Any] = {}
        self._defaults = self._get_defaults()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.info("Using default configuration")
        
        # Apply environment variable overrides
        self._apply_env_overrides()

    def _get_defaults(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            # Embedding configuration
            "embedding": {
                "model": os.getenv("ACE_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
                "batch_size": int(os.getenv("ACE_EMBEDDING_BATCH_SIZE", "32")),
                "cache_embeddings": os.getenv("ACE_CACHE_EMBEDDINGS", "true").lower() == "true",
            },
            # LLM configuration
            "llm": {
                "api_endpoint": os.getenv("ACE_LLM_ENDPOINT", "http://localhost:11434"),
                "model": os.getenv("ACE_LLM_MODEL", "mistral"),
                "timeout_s": int(os.getenv("ACE_LLM_TIMEOUT_S", "120")),
                "max_retries": int(os.getenv("ACE_LLM_MAX_RETRIES", "3")),
                "retry_backoff_factor": float(os.getenv("ACE_LLM_RETRY_BACKOFF", "1.5")),
            },
            # Cache configuration
            "cache": {
                "max_size": int(os.getenv("ACE_CACHE_MAX_SIZE", "10000")),
                "ttl_hours": int(os.getenv("ACE_CACHE_TTL_HOURS", "24")),
                "enabled": os.getenv("ACE_CACHE_ENABLED", "true").lower() == "true",
            },
            # Circuit breaker
            "circuit_breaker": {
                "failure_threshold": int(os.getenv("ACE_CB_FAILURE_THRESHOLD", "5")),
                "cooldown_s": int(os.getenv("ACE_CB_COOLDOWN_S", "30")),
            },
            # Memory/storage
            "memory": {
                "storage_path": os.getenv("ACE_STORAGE_PATH", "./data/"),
                "auto_save": os.getenv("ACE_AUTO_SAVE", "true").lower() == "true",
                "save_interval_s": int(os.getenv("ACE_SAVE_INTERVAL_S", "60")),
            },
            # Dialogue
            "dialogue": {
                "max_turns": int(os.getenv("ACE_DIALOGUE_MAX_TURNS", "6")),
                "context_window": int(os.getenv("ACE_DIALOGUE_CONTEXT", "10")),
            },
            # Logging
            "logging": {
                "level": os.getenv("ACE_LOG_LEVEL", "INFO"),
                "format": os.getenv("ACE_LOG_FORMAT", "standard"),  # "standard" or "json"
                "file": os.getenv("ACE_LOG_FILE", None),
            },
            # Performance/optimization
            "performance": {
                "enable_metrics": os.getenv("ACE_METRICS", "true").lower() == "true",
                "enable_profiling": os.getenv("ACE_PROFILING", "false").lower() == "true",
                "embedding_cache_size_mb": int(os.getenv("ACE_EMB_CACHE_MB", "500")),
            },
        }

    def _load_config_file(self, path: str) -> None:
        """Load configuration from YAML or JSON file."""
        try:
            path_obj = Path(path)
            if path_obj.suffix.lower() == ".yaml" or path_obj.suffix.lower() == ".yml":
                self._load_yaml(path)
            elif path_obj.suffix.lower() == ".json":
                self._load_json(path)
            else:
                logger.warning(f"Unsupported config format: {path_obj.suffix}")
        except Exception as e:
            logger.error(f"Failed to load config file {path}: {e}")

    def _load_json(self, path: str) -> None:
        """Load JSON configuration."""
        with open(path, "r") as f:
            file_config = json.load(f)
            self._merge_configs(self._defaults, file_config)

    def _load_yaml(self, path: str) -> None:
        """Load YAML configuration."""
        try:
            import yaml
            with open(path, "r") as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_configs(self._defaults, file_config)
        except ImportError:
            logger.warning("PyYAML not installed. Install with: pip install pyyaml")
        except Exception as e:
            logger.error(f"Failed to load YAML: {e}")

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override config into base config."""
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
        self._config = base

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to loaded config."""
        # This is already done in _get_defaults for top-level settings
        pass

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path like "llm.timeout_s"
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config or self._defaults
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., "llm", "embedding")
            
        Returns:
            Dictionary of configuration for that section
        """
        return (self._config or self._defaults).get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary."""
        return self._config or self._defaults

    def __repr__(self) -> str:
        """String representation (safe - no secrets)."""
        return f"ConfigManager({len(self._config or self._defaults)} settings)"


# Global config instance
_global_config: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create global config manager.
    
    Args:
        config_path: Path to config file (only used on first call)
        
    Returns:
        Global ConfigManager instance
    """
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    return _global_config


__all__ = ["ConfigManager", "get_config"]
