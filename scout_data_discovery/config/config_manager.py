"""
Configuration Management for Scout Data Discovery

Handles loading, validation, and merging of configuration files.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from ..src.exceptions import ConfigurationError


class ConfigManager:
    """
    Configuration manager for Scout Data Discovery.

    Handles loading configuration from multiple sources:
    - Default configuration file
    - User configuration file
    - Environment variables
    - Direct dictionary input
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to user configuration file
        """
        self.config_dir = Path(__file__).parent
        self.default_config_path = self.config_dir / "default_config.yaml"
        self.user_config_path = Path(config_path) if config_path else None

        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from all sources"""
        try:
            # Start with default configuration
            config = self._load_default_config()

            # Merge user configuration if provided
            if self.user_config_path and self.user_config_path.exists():
                user_config = self._load_user_config()
                config = self._merge_configs(config, user_config)

            # Apply environment variable overrides
            config = self._apply_env_overrides(config)

            # Validate configuration
            self._validate_config(config)

            self.logger.info("Configuration loaded successfully")
            return config

        except Exception as e:
            error_msg = f"Failed to load configuration: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        if not self.default_config_path.exists():
            raise ConfigurationError(f"Default configuration file not found: {self.default_config_path}")

        with open(self.default_config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_user_config(self) -> Dict[str, Any]:
        """Load user configuration"""
        try:
            with open(self.user_config_path, 'r') as f:
                if self.user_config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    return yaml.safe_load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load user config from {self.user_config_path}: {str(e)}")

    def _merge_configs(self, base_config: Dict, user_config: Dict) -> Dict[str, Any]:
        """
        Recursively merge user configuration into base configuration.
        User config values override base config values.
        """
        merged = base_config.copy()

        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides.
        Environment variables use the format: SCOUT_<SECTION>_<KEY>
        """
        env_prefix = "SCOUT_"

        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                # Parse environment variable name
                config_path = env_key[len(env_prefix):].lower().split('_')

                if len(config_path) >= 2:
                    # Convert environment variable value to appropriate type
                    typed_value = self._convert_env_value(env_value)

                    # Set the value in config
                    self._set_nested_config(config, config_path, typed_value)

        return config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # JSON conversion (for lists/dicts)
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass

        # Return as string
        return value

    def _set_nested_config(self, config: Dict, path: List[str], value: Any):
        """Set a nested configuration value"""
        current = config

        # Navigate to the parent of the target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Can't traverse further, skip this override
                return
            current = current[key]

        # Set the final value
        current[path[-1]] = value

    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration values"""
        errors = []

        # Validate required sections
        required_sections = ['api', 'cache', 'data', 'portals', 'quality']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: {section}")

        # Validate API settings
        if 'api' in config:
            api_config = config['api']
            if api_config.get('rate_limit_delay', 0) < 0:
                errors.append("API rate_limit_delay must be non-negative")
            if api_config.get('request_timeout', 1) < 1:
                errors.append("API request_timeout must be at least 1 second")
            if api_config.get('retry_attempts', 1) < 1:
                errors.append("API retry_attempts must be at least 1")

        # Validate cache settings
        if 'cache' in config:
            cache_config = config['cache']
            if cache_config.get('duration_hours', 1) < 0:
                errors.append("Cache duration_hours must be non-negative")

        # Validate data settings
        if 'data' in config:
            data_config = config['data']
            if data_config.get('default_sample_size', 1) < 1:
                errors.append("Data default_sample_size must be at least 1")
            if not (0 <= data_config.get('quality_threshold', 70) <= 100):
                errors.append("Data quality_threshold must be between 0 and 100")

        # Validate quality weights sum to 1.0
        if 'quality' in config and 'weights' in config['quality']:
            weights = config['quality']['weights']
            weight_sum = sum(weights.values())
            if abs(weight_sum - 1.0) > 0.01:  # Allow small floating point errors
                errors.append(f"Quality weights must sum to 1.0, got {weight_sum}")

        if errors:
            raise ConfigurationError("Configuration validation failed:\n" + "\n".join(errors))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'api.rate_limit_delay')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self.config

        try:
            for k in keys:
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'api.rate_limit_delay')
            value: Value to set
        """
        keys = key.split('.')
        current = self.config

        # Navigate to parent of target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get an entire configuration section.

        Args:
            section: Section name

        Returns:
            Configuration section dictionary
        """
        return self.config.get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

    def save_user_config(self, filepath: Union[str, Path], format: str = 'yaml'):
        """
        Save current configuration to user config file.

        Args:
            filepath: Path to save configuration
            format: Format to save ('yaml' or 'json')
        """
        filepath = Path(filepath)

        try:
            with open(filepath, 'w') as f:
                if format.lower() == 'json':
                    json.dump(self.config, f, indent=2, default=str)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)

            self.logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            error_msg = f"Failed to save configuration to {filepath}: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e

    def create_profile(self, profile_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a configuration profile with specific overrides.

        Args:
            profile_name: Name of the profile
            overrides: Configuration overrides for this profile

        Returns:
            Profile configuration dictionary
        """
        profile_config = self.config.copy()
        profile_config = self._merge_configs(profile_config, overrides)
        profile_config['_profile_name'] = profile_name

        return profile_config

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        return {
            'max_workers': self.get('api.max_concurrent_requests', 5),
            'rate_limit_delay': self.get('api.rate_limit_delay', 0.5),
            'chunk_size': self.get('performance.chunk_size_for_large_datasets', 10000),
            'max_memory_mb': self.get('performance.max_memory_usage_mb', 1000),
            'enable_parallel': self.get('performance.enable_parallel_processing', True)
        }

    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality assessment configuration"""
        return {
            'weights': self.get_section('quality').get('weights', {}),
            'thresholds': self.get_section('quality').get('thresholds', {}),
            'outlier_detection': self.get_section('quality').get('outlier_detection', {}),
            'default_threshold': self.get('data.quality_threshold', 70)
        }

    def get_export_config(self) -> Dict[str, Any]:
        """Get export configuration"""
        return {
            'default_directory': self.get('export.default_directory', 'exports'),
            'formats': self.get('export.formats', ['csv', 'json']),
            'include_metadata': self.get('export.include_metadata', True),
            'include_summary': self.get('export.include_summary_report', True)
        }

    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration from dictionary.

        Args:
            updates: Dictionary of configuration updates
        """
        self.config = self._merge_configs(self.config, updates)
        self._validate_config(self.config)