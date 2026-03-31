"""Configuration utilities for loading YAML configs and merging with CLI args."""

import logging
import os
import sys
from typing import Dict, Any, Type, TypeVar
import yaml
from dataclasses import fields, is_dataclass

log = logging.getLogger(__name__)

T = TypeVar('T')


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        return {}
    
    log.info(f"Loaded configuration from: {config_path}")
    return config


def save_yaml_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
    
    log.info(f"Saved configuration to: {config_path}")


def dataclass_from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
    """Create dataclass instance from dictionary, handling type conversions."""
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")
    
    # Get field names and types
    field_map = {f.name: f for f in fields(cls)}
    
    # Filter config to only include valid fields
    filtered_config = {}
    for key, value in config_dict.items():
        if key in field_map:
            field = field_map[key]
            # Handle type conversion if needed
            try:
                # Handle None values and optional types
                if value is None:
                    filtered_config[key] = None
                elif hasattr(field.type, '__origin__') and field.type.__origin__ is type(None):
                    # Optional type (Union[X, None])
                    filtered_config[key] = value
                else:
                    # Try to convert to the expected type
                    if field.type == bool and isinstance(value, str):
                        filtered_config[key] = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        filtered_config[key] = field.type(value) if value is not None else None
            except (ValueError, TypeError) as e:
                log.warning(f"Could not convert {key}={value} to {field.type}: {e}")
                filtered_config[key] = value
        else:
            log.warning(f"Unknown configuration parameter: {key}")
    
    return cls(**filtered_config)


def merge_config_with_cli(config_class: Type[T], config_file: str = None) -> T:
    """Merge YAML config with CLI arguments, with CLI taking precedence.
    
    Args:
        config_class: Dataclass type to create
        config_file: Path to YAML config file (optional)
    
    Returns:
        Instance of config_class with merged configuration
    """
    # Parse CLI args first to get the default/CLI values
    import tyro
    
    # Check if --config is in sys.argv and extract it
    extracted_config_file = config_file
    if "--config" in sys.argv:
        config_idx = sys.argv.index("--config")
        if config_idx + 1 < len(sys.argv):
            extracted_config_file = sys.argv[config_idx + 1]
            # Remove from sys.argv so tyro doesn't see it
            sys.argv.pop(config_idx)  # Remove --config
            sys.argv.pop(config_idx)  # Remove config file path
    
    # Load YAML config if available
    yaml_config = {}
    if extracted_config_file and os.path.exists(extracted_config_file):
        yaml_config = load_yaml_config(extracted_config_file)
        log.info(f"Merging YAML config from {extracted_config_file} with CLI args")
    elif extracted_config_file:
        log.warning(f"Config file not found: {extracted_config_file}, using CLI args only")
    
    # Create default instance to know what the defaults are
    default_instance = config_class()
    default_dict = {f.name: getattr(default_instance, f.name) for f in fields(config_class)}
    
    # Parse CLI args
    cli_instance = tyro.cli(config_class)
    cli_dict = {f.name: getattr(cli_instance, f.name) for f in fields(config_class)}
    
    # Merge configs: YAML as base, CLI overrides non-default values
    final_config = {}
    
    for field in fields(config_class):
        field_name = field.name
        default_value = default_dict[field_name] 
        yaml_value = yaml_config.get(field_name, default_value)
        cli_value = cli_dict[field_name]
        
        # Use CLI value if it's different from default, otherwise use YAML value
        if cli_value != default_value:
            final_config[field_name] = cli_value
            if yaml_value != default_value and yaml_value != cli_value:
                log.info(f"CLI override: {field_name}={cli_value} (YAML had {yaml_value})")
        else:
            final_config[field_name] = yaml_value
            if yaml_value != default_value:
                log.info(f"YAML config: {field_name}={yaml_value}")
    
    return config_class(**final_config)


def print_config(config: Any, title: str = "Configuration"):
    """Pretty print configuration."""
    log.info(f"{title}:")
    if is_dataclass(config):
        for field in fields(config):
            value = getattr(config, field.name)
            log.info(f"  {field.name}: {value}")
    else:
        for key, value in vars(config).items():
            log.info(f"  {key}: {value}")