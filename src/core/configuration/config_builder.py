"""
Advanced Configuration Management with Multiple Sources
Clean Code Implementation following SOLID Principles
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, TypeVar
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import yaml
import os
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigurationPriority(Enum):
    """Configuration source priority levels"""
    DEFAULT = 0
    FILE = 1
    ENVIRONMENT = 2
    OVERRIDE = 3


@dataclass
class ConfigurationValue:
    """Represents a configuration value with its source"""
    key: str
    value: Any
    source: str
    priority: ConfigurationPriority


class IConfigurationSource(ABC):
    """Interface for configuration sources - Interface Segregation Principle"""
    
    @abstractmethod
    def load(self) -> Dict[str, Any]:
        """Load configuration from source"""
        pass
    
    @abstractmethod
    def get_priority(self) -> ConfigurationPriority:
        """Get source priority"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get source name for debugging"""
        pass


class EnvironmentVariableSource(IConfigurationSource):
    """Load configuration from environment variables"""
    
    def __init__(self, prefix: str = "", transform_keys: bool = True):
        self.prefix = prefix
        self.transform_keys = transform_keys
    
    def load(self) -> Dict[str, Any]:
        """Load environment variables"""
        config = {}
        prefix_len = len(self.prefix)
        
        for key, value in os.environ.items():
            if self.prefix and not key.startswith(self.prefix):
                continue
            
            # Remove prefix and transform key
            config_key = key[prefix_len:] if self.prefix else key
            
            if self.transform_keys:
                # Convert UPPER_CASE to lower_case
                config_key = config_key.lower()
            
            # Try to parse JSON values
            try:
                config[config_key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                config[config_key] = value
        
        logger.debug(f"Loaded {len(config)} values from environment")
        return config
    
    def get_priority(self) -> ConfigurationPriority:
        return ConfigurationPriority.ENVIRONMENT
    
    def get_name(self) -> str:
        return f"Environment[{self.prefix or 'all'}]"


class JsonFileSource(IConfigurationSource):
    """Load configuration from JSON file"""
    
    def __init__(self, path: Path, required: bool = False):
        self.path = path
        self.required = required
    
    def load(self) -> Dict[str, Any]:
        """Load JSON file"""
        if not self.path.exists():
            if self.required:
                raise FileNotFoundError(f"Required configuration file not found: {self.path}")
            logger.debug(f"Optional configuration file not found: {self.path}")
            return {}
        
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {self.path}")
                return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.path}: {e}")
    
    def get_priority(self) -> ConfigurationPriority:
        return ConfigurationPriority.FILE
    
    def get_name(self) -> str:
        return f"JSON[{self.path.name}]"


class YamlFileSource(IConfigurationSource):
    """Load configuration from YAML file"""
    
    def __init__(self, path: Path, required: bool = False):
        self.path = path
        self.required = required
    
    def load(self) -> Dict[str, Any]:
        """Load YAML file"""
        if not self.path.exists():
            if self.required:
                raise FileNotFoundError(f"Required configuration file not found: {self.path}")
            logger.debug(f"Optional configuration file not found: {self.path}")
            return {}
        
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.path}")
                return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.path}: {e}")
    
    def get_priority(self) -> ConfigurationPriority:
        return ConfigurationPriority.FILE
    
    def get_name(self) -> str:
        return f"YAML[{self.path.name}]"


class DictionarySource(IConfigurationSource):
    """Load configuration from dictionary (for overrides)"""
    
    def __init__(self, config: Dict[str, Any], name: str = "Override"):
        self.config = config
        self.name = name
    
    def load(self) -> Dict[str, Any]:
        """Return the dictionary"""
        return self.config.copy()
    
    def get_priority(self) -> ConfigurationPriority:
        return ConfigurationPriority.OVERRIDE
    
    def get_name(self) -> str:
        return self.name


class ConfigurationBuilder:
    """Build configuration from multiple sources with priority"""
    
    def __init__(self):
        self._sources: List[IConfigurationSource] = []
        self._validators: List[callable] = []
        self._transformers: List[callable] = []
        self._cache: Optional[Dict[str, Any]] = None
    
    def add_source(self, source: IConfigurationSource) -> 'ConfigurationBuilder':
        """Add a configuration source"""
        self._sources.append(source)
        self._cache = None  # Invalidate cache
        return self
    
    def add_json_file(self, path: Path, required: bool = False) -> 'ConfigurationBuilder':
        """Add JSON file source"""
        return self.add_source(JsonFileSource(path, required))
    
    def add_yaml_file(self, path: Path, required: bool = False) -> 'ConfigurationBuilder':
        """Add YAML file source"""
        return self.add_source(YamlFileSource(path, required))
    
    def add_environment_variables(self, prefix: str = "") -> 'ConfigurationBuilder':
        """Add environment variables source"""
        return self.add_source(EnvironmentVariableSource(prefix))
    
    def add_defaults(self, defaults: Dict[str, Any]) -> 'ConfigurationBuilder':
        """Add default values"""
        source = DictionarySource(defaults, "Defaults")
        source.get_priority = lambda: ConfigurationPriority.DEFAULT
        return self.add_source(source)
    
    def add_overrides(self, overrides: Dict[str, Any]) -> 'ConfigurationBuilder':
        """Add override values"""
        return self.add_source(DictionarySource(overrides, "Overrides"))
    
    def add_validator(self, validator: callable) -> 'ConfigurationBuilder':
        """Add configuration validator"""
        self._validators.append(validator)
        return self
    
    def add_transformer(self, transformer: callable) -> 'ConfigurationBuilder':
        """Add configuration transformer"""
        self._transformers.append(transformer)
        return self
    
    def _merge_configurations(self) -> Dict[str, Any]:
        """Merge configurations based on priority"""
        # Sort sources by priority
        sorted_sources = sorted(self._sources, key=lambda s: s.get_priority().value)
        
        merged = {}
        for source in sorted_sources:
            try:
                config = source.load()
                logger.debug(f"Merging {len(config)} values from {source.get_name()}")
                merged.update(config)
            except Exception as e:
                logger.error(f"Error loading from {source.get_name()}: {e}")
                if hasattr(source, 'required') and source.required:
                    raise
        
        return merged
    
    def build(self) -> Dict[str, Any]:
        """Build the final configuration"""
        if self._cache is not None:
            return self._cache
        
        # Merge all sources
        config = self._merge_configurations()
        
        # Apply transformers
        for transformer in self._transformers:
            config = transformer(config)
        
        # Validate
        for validator in self._validators:
            validator(config)
        
        self._cache = config
        logger.info(f"Built configuration with {len(config)} values")
        return config
    
    def build_typed(self, config_class: Type[T]) -> T:
        """Build configuration as typed object"""
        config_dict = self.build()
        
        # If it's a Pydantic model
        if hasattr(config_class, 'parse_obj'):
            return config_class.parse_obj(config_dict)
        
        # If it's a dataclass
        if hasattr(config_class, '__dataclass_fields__'):
            return config_class(**config_dict)
        
        # Otherwise, just instantiate with dict
        return config_class(**config_dict)


class ConfigurationValidator:
    """Validates configuration values"""
    
    @staticmethod
    def require_keys(*keys: str) -> callable:
        """Create validator that requires specific keys"""
        def validator(config: Dict[str, Any]):
            missing = [k for k in keys if k not in config]
            if missing:
                raise ValueError(f"Missing required configuration keys: {missing}")
        return validator
    
    @staticmethod
    def validate_types(**type_map: Type) -> callable:
        """Create validator that checks value types"""
        def validator(config: Dict[str, Any]):
            for key, expected_type in type_map.items():
                if key in config and not isinstance(config[key], expected_type):
                    actual_type = type(config[key]).__name__
                    raise TypeError(
                        f"Configuration key '{key}' has wrong type. "
                        f"Expected {expected_type.__name__}, got {actual_type}"
                    )
        return validator
    
    @staticmethod
    def validate_range(key: str, min_val: Any = None, max_val: Any = None) -> callable:
        """Create validator that checks value range"""
        def validator(config: Dict[str, Any]):
            if key in config:
                value = config[key]
                if min_val is not None and value < min_val:
                    raise ValueError(f"Configuration key '{key}' value {value} is below minimum {min_val}")
                if max_val is not None and value > max_val:
                    raise ValueError(f"Configuration key '{key}' value {value} is above maximum {max_val}")
        return validator


class ConfigurationTransformer:
    """Transforms configuration values"""
    
    @staticmethod
    def expand_paths(*keys: str) -> callable:
        """Create transformer that expands paths"""
        def transformer(config: Dict[str, Any]) -> Dict[str, Any]:
            for key in keys:
                if key in config:
                    config[key] = Path(config[key]).expanduser().resolve()
            return config
        return transformer
    
    @staticmethod
    def interpolate_variables() -> callable:
        """Create transformer that interpolates ${var} references"""
        def transformer(config: Dict[str, Any]) -> Dict[str, Any]:
            def interpolate_value(value: Any) -> Any:
                if isinstance(value, str):
                    # Replace ${key} with config[key]
                    import re
                    pattern = r'\$\{([^}]+)\}'
                    
                    def replacer(match):
                        key = match.group(1)
                        if key in config:
                            return str(config[key])
                        return match.group(0)
                    
                    return re.sub(pattern, replacer, value)
                elif isinstance(value, dict):
                    return {k: interpolate_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [interpolate_value(v) for v in value]
                return value
            
            return {k: interpolate_value(v) for k, v in config.items()}
        return transformer


@lru_cache(maxsize=1)
def get_configuration() -> Dict[str, Any]:
    """Get application configuration (cached)"""
    builder = ConfigurationBuilder()
    
    # Add configuration sources in priority order
    builder.add_defaults({
        'app_name': 'teknofest-2025',
        'debug': False,
        'log_level': 'INFO',
        'database_pool_size': 10
    })
    
    # Add configuration files
    config_dir = Path(__file__).parent.parent.parent.parent / 'configs'
    builder.add_json_file(config_dir / 'default.json')
    builder.add_yaml_file(config_dir / 'local.yaml')
    
    # Add environment variables
    builder.add_environment_variables(prefix="APP_")
    
    # Add validators
    builder.add_validator(
        ConfigurationValidator.require_keys('app_name', 'log_level')
    )
    builder.add_validator(
        ConfigurationValidator.validate_types(
            debug=bool,
            database_pool_size=int
        )
    )
    builder.add_validator(
        ConfigurationValidator.validate_range('database_pool_size', min_val=1, max_val=100)
    )
    
    # Add transformers
    builder.add_transformer(
        ConfigurationTransformer.interpolate_variables()
    )
    
    return builder.build()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Build configuration
    config = get_configuration()
    
    # Print configuration
    print("Configuration loaded:")
    for key, value in config.items():
        print(f"  {key}: {value}")
