"""
Configuration validation and utility functions.
Ensures that configuration files contain all required parameters
and provides safe access to nested configuration values.
"""

from typing import Dict, Any, List
import logging
import yaml
import os

logger = logging.getLogger(__name__)

def get_nested_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Safely get nested configuration value.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., 'data.text_col')
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default

def validate_required_keys(config: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Validate that all required keys exist in configuration.
    
    Args:
        config: Configuration dictionary
        required_keys: List of required dot-separated key paths
        
    Returns:
        List of missing keys (empty if all present)
    """
    missing_keys = []
    
    for key_path in required_keys:
        if get_nested_value(config, key_path) is None:
            missing_keys.append(key_path)
    
    return missing_keys

def validate_phishguard_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate PhishGuard configuration and add defaults.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration with defaults added
        
    Raises:
        ValueError: If required keys are missing
    """
    # Define required configuration keys
    required_keys = [
        "data.tweets_csv",
        "data.text_col", 
        "data.label_col",
        "model.model_name_or_path",
        "train.batch_size",
        "train.num_epochs",
        "train.lr",
        "loss.lambda_cls",
        "loss.lambda_adv",
        "loss.mu_prop"
    ]
    
    # Check for missing required keys
    missing_keys = validate_required_keys(config, required_keys)
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Add default values for optional keys
    defaults = {
        "seed": 42,
        "output_dir": "runs/phishguard_exp",
        
        # Data defaults
        "data.user_id_col": "user_id",
        "data.edges_csv": None,
        "data.remove_duplicates": True,
        "data.filter_non_english": True,
        "data.min_text_length": 10,
        "data.max_text_length": 512,
        "data.split.train": 0.8,
        "data.split.val": 0.1,
        "data.split.test": 0.1,
        
        # Model defaults
        "model.fallback_model": "distilbert-base-uncased",
        "model.max_length": 512,
        "model.peft": None,
        "model.lora_r": 16,
        "model.embedding_dim": 768,
        
        # Training defaults
        "train.weight_decay": 0.01,
        "train.warmup_ratio": 0.1,
        "train.grad_accum_steps": 1,
        "train.fp16": False,
        "train.gradient_checkpointing": False,
        
        # Loss defaults
        "loss.adv_eps": 1e-2,
        "loss.adv_steps": 3,
        "loss.adv_temperature": 1.0,
        
        # Propagation defaults
        "propagation.ic_samples": 100,
        "propagation.budget": 20,
        "propagation.topk_candidates": 200,
        "propagation.edge_weight_threshold": 0.01,
        "propagation.risk_threshold": 0.7
    }
    
    # Apply defaults
    validated_config = config.copy()
    for key_path, default_value in defaults.items():
        if get_nested_value(validated_config, key_path) is None:
            set_nested_value(validated_config, key_path, default_value)
    
    # Validate ranges and constraints
    _validate_config_constraints(validated_config)
    
    logger.info("Configuration validation completed successfully")
    return validated_config

def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested configuration value.
    
    Args:
        config: Configuration dictionary (modified in place)
        key_path: Dot-separated path
        value: Value to set
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to parent dict
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set final value
    current[keys[-1]] = value

def _validate_config_constraints(config: Dict[str, Any]) -> None:
    """Validate configuration value constraints.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If constraints are violated
    """
    # Training constraints
    batch_size = get_nested_value(config, "train.batch_size")
    if batch_size is not None and batch_size <= 0:
        raise ValueError("train.batch_size must be positive")
    
    num_epochs = get_nested_value(config, "train.num_epochs")
    if num_epochs is not None and num_epochs <= 0:
        raise ValueError("train.num_epochs must be positive")
    
    lr = get_nested_value(config, "train.lr")
    if lr is not None and lr <= 0:
        raise ValueError("train.lr must be positive")
    
    # Split constraints
    train_split = get_nested_value(config, "data.split.train", 0.8)
    val_split = get_nested_value(config, "data.split.val", 0.1)
    test_split = get_nested_value(config, "data.split.test", 0.1)
    
    total_split = train_split + val_split + test_split
    if abs(total_split - 1.0) > 1e-6:
        raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
    
    # Loss weight constraints
    if get_nested_value(config, "loss.lambda_cls") < 0:
        raise ValueError("loss.lambda_cls must be non-negative")
    
    if get_nested_value(config, "loss.lambda_adv") < 0:
        raise ValueError("loss.lambda_adv must be non-negative")
    
    if get_nested_value(config, "loss.mu_prop") < 0:
        raise ValueError("loss.mu_prop must be non-negative")
    
    # File existence checks
    tweets_path = get_nested_value(config, "data.tweets_csv")
    if tweets_path and not os.path.exists(tweets_path):
        logger.warning(f"Tweets file not found: {tweets_path}")
    
    edges_path = get_nested_value(config, "data.edges_csv")
    if edges_path and not os.path.exists(edges_path):
        logger.warning(f"Edges file not found: {edges_path}")

def load_and_validate_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {config_path}: {e}")
    
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a dictionary at the root level")
    
    return validate_phishguard_config(config)

def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of key configuration parameters.
    
    Args:
        config: Configuration dictionary
    """
    logger.info("=== Configuration Summary ===")
    logger.info(f"Model: {get_nested_value(config, 'model.model_name_or_path')}")
    logger.info(f"Data: {get_nested_value(config, 'data.tweets_csv')}")
    logger.info(f"Batch size: {get_nested_value(config, 'train.batch_size')}")
    logger.info(f"Epochs: {get_nested_value(config, 'train.num_epochs')}")
    logger.info(f"Learning rate: {get_nested_value(config, 'train.lr')}")
    logger.info(f"Loss weights - cls: {get_nested_value(config, 'loss.lambda_cls')}, "
                f"adv: {get_nested_value(config, 'loss.lambda_adv')}, "
                f"prop: {get_nested_value(config, 'loss.mu_prop')}")
    logger.info(f"PEFT: {get_nested_value(config, 'model.peft')}")
    logger.info("=" * 30)
