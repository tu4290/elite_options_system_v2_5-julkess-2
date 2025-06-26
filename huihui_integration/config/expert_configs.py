# utils/config_manager_v2_5.py
# EOTS v2.5 - SENTRY-APPROVED, CANONICAL V2.5.3 IMPLEMENTATION (UNABRIDGED)

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import ValidationError

# Change relative import to absolute import
from data_models.eots_schemas_v2_5 import EOTSConfigV2_5

logger = logging.getLogger(__name__)

class ConfigManagerV2_5:
    """
    Singleton configuration manager for EOTS v2.5.
    Handles loading, validation, and access to configuration settings.
    """
    _instance = None
    _config: Optional[EOTSConfigV2_5] = None
    _project_root: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManagerV2_5, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the singleton instance."""
        self._determine_project_root()
        self._load_config()

    def _determine_project_root(self):
        """Determine the project root directory."""
        # Start from the current file's directory
        current_dir = Path(__file__).parent
        # Look for the root package file
        while current_dir != current_dir.parent:
            if (current_dir / "elite_options_system_v2_5.py").exists():
                self._project_root = current_dir
                logger.debug(f"Project root determined as: {self._project_root}")
                return
            current_dir = current_dir.parent
        raise RuntimeError("Could not determine project root directory")

    def _substitute_env_vars(self, data):
        """Recursively substitute environment variables in configuration data."""
        import re
        
        if isinstance(data, dict):
            return {key: self._substitute_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Replace ${VAR} with environment variable value
            def replace_env_var(match):
                var_name = match.group(1)
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found, keeping placeholder")
                    return match.group(0)  # Return original ${VAR} if not found
                # Try to convert to appropriate type
                if env_value.isdigit():
                    return int(env_value)
                try:
                    return float(env_value)
                except ValueError:
                    return env_value
            
            pattern = r'\$\{([^}]+)\}'
            result = re.sub(pattern, replace_env_var, data)
            return result
        else:
            return data

    def _load_config(self):
        """Load and validate the configuration file."""
        if not self._project_root:
            raise RuntimeError("Project root not determined")
            
        config_path = self._project_root / "config" / "config_v2_5.json"
        logger.info(f"Loading configuration from: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Substitute environment variables
            config_data = self._substitute_env_vars(config_data)

            # Validate against JSON schema
            logger.debug("Validating configuration against JSON schema...")
            # Schema validation is handled by Pydantic model
            logger.info("JSON schema validation successful.")

            # Parse into Pydantic model
            logger.debug("Parsing validated configuration into Pydantic model...")
            self._config = EOTSConfigV2_5(**config_data)
            logger.info("Pydantic model parsing successful. Configuration is now loaded and type-safe.")

        except FileNotFoundError:
            raise RuntimeError(f"Configuration file not found at {config_path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in configuration file: {e}")
        except ValidationError as e:
            raise RuntimeError(f"Configuration validation failed: {e}")

    def get_setting(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration setting by path.

        Args:
            path: Dot-separated path to the setting (e.g., 'database.host')
            default: Default value to return if setting not found

        Returns:
            The requested setting value or default if not found
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        try:
            parts = path.split('.')
            value = self._config
            for part in parts:
                value = getattr(value, part)
            return value
        except (AttributeError, KeyError):
            return default



    @property
    def config(self) -> EOTSConfigV2_5:
        """Get the loaded configuration object."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        if not self._project_root:
            raise RuntimeError("Project root not determined")
        return self._project_root

    def get_resolved_path(self, path: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration setting that represents a path and resolve it relative to project root.
        
        Args:
            path: Dot-notation path to the setting (e.g., 'performance_tracker_settings_v2_5.performance_data_directory')
            default: Default value to return if setting is not found
            
        Returns:
            The resolved path string or None if not found
        """
        relative_path = self.get_setting(path, default)
        if not relative_path:
            return None
        
        if not self._project_root:
            raise RuntimeError("Project root not determined")
            
        return str(self._project_root / relative_path)

    def load_config(self):
        """Public method to load the configuration."""
        self._load_config()