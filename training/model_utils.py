"""
Utility functions for safe model operations across different architectures.
This module provides consistent interfaces for accessing model components
regardless of whether PEFT, different transformers, or custom architectures are used.
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def safe_model_forward(model, input_ids=None, inputs_embeds=None, attention_mask=None, 
                      labels=None, return_dict=True):
    """Safe model forward pass handling different architectures.
    
    Args:
        model: The model instance (could be wrapped with PEFT)
        input_ids: Token IDs (mutually exclusive with inputs_embeds)
        inputs_embeds: Input embeddings (mutually exclusive with input_ids)
        attention_mask: Attention mask
        labels: Labels for loss computation (optional)
        return_dict: Whether to return structured output
        
    Returns:
        Model output with consistent structure
    """
    # Determine the base model
    if hasattr(model, 'model'):
        # PhishGuardClassifier or similar wrapper
        base_model = model.model
    elif hasattr(model, 'base_model'):
        # PEFT wrapped model
        base_model = model.base_model
    else:
        # Direct model
        base_model = model
    
    # Prepare kwargs
    kwargs = {
        'attention_mask': attention_mask,
        'return_dict': return_dict
    }
    
    if labels is not None:
        kwargs['labels'] = labels
    
    # Forward pass with appropriate inputs
    try:
        if input_ids is not None:
            kwargs['input_ids'] = input_ids
            return base_model(**kwargs)
        elif inputs_embeds is not None:
            kwargs['inputs_embeds'] = inputs_embeds
            return base_model(**kwargs)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
            
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        raise

def safe_get_embeddings(model, input_ids):
    """Safely get input embeddings from model.
    
    Args:
        model: Model instance
        input_ids: Token IDs
        
    Returns:
        Input embeddings tensor
    """
    try:
        # Try different embedding access patterns
        if hasattr(model, 'model') and hasattr(model.model, 'get_input_embeddings'):
            # PhishGuardClassifier wrapper
            return model.model.get_input_embeddings()(input_ids)
        elif hasattr(model, 'get_input_embeddings'):
            # Direct model with embedding method
            return model.get_input_embeddings()(input_ids)
        elif hasattr(model, 'embeddings'):
            # Some models have direct embeddings attribute
            return model.embeddings(input_ids)
        elif hasattr(model, 'base_model') and hasattr(model.base_model, 'get_input_embeddings'):
            # PEFT wrapped model
            return model.base_model.get_input_embeddings()(input_ids)
        else:
            raise AttributeError("Cannot find embedding layer in model")
            
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise

def safe_get_logits(model, input_ids=None, inputs_embeds=None, attention_mask=None):
    """Safely get logits from model with different input types.
    
    Args:
        model: Model instance  
        input_ids: Token IDs (mutually exclusive with inputs_embeds)
        inputs_embeds: Input embeddings (mutually exclusive with input_ids)
        attention_mask: Attention mask
        
    Returns:
        Logits tensor
    """
    try:
        output = safe_model_forward(
            model, 
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract logits from output
        if hasattr(output, 'logits'):
            return output.logits
        elif isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, (list, tuple)):
            return output[0]  # Assume logits are first element
        else:
            raise ValueError(f"Unexpected model output type: {type(output)}")
            
    except Exception as e:
        logger.error(f"Failed to get logits: {e}")
        raise

def safe_device_transfer(tensor, device):
    """Safely transfer tensor to device with error handling.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        Tensor on target device
    """
    if tensor is None:
        return None
        
    try:
        return tensor.to(device)
    except RuntimeError as e:
        logger.warning(f"Device transfer failed: {e}. Keeping tensor on original device.")
        return tensor

def validate_model_inputs(input_ids=None, inputs_embeds=None, attention_mask=None):
    """Validate model inputs for consistency.
    
    Args:
        input_ids: Token IDs
        inputs_embeds: Input embeddings
        attention_mask: Attention mask
        
    Raises:
        ValueError: If inputs are invalid
    """
    if input_ids is None and inputs_embeds is None:
        raise ValueError("Either input_ids or inputs_embeds must be provided")
    
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("input_ids and inputs_embeds are mutually exclusive")
    
    # Validate shapes
    if input_ids is not None and attention_mask is not None:
        if input_ids.shape[:2] != attention_mask.shape[:2]:
            raise ValueError(f"input_ids shape {input_ids.shape} doesn't match attention_mask shape {attention_mask.shape}")
    
    if inputs_embeds is not None and attention_mask is not None:
        if inputs_embeds.shape[:2] != attention_mask.shape[:2]:
            raise ValueError(f"inputs_embeds shape {inputs_embeds.shape} doesn't match attention_mask shape {attention_mask.shape}")

def compute_safe_loss(logits, labels, loss_type='cross_entropy'):
    """Compute loss with proper error handling.
    
    Args:
        logits: Model predictions
        labels: Ground truth labels
        loss_type: Type of loss to compute
        
    Returns:
        Computed loss tensor
    """
    try:
        if loss_type == 'cross_entropy':
            return F.cross_entropy(logits, labels)
        elif loss_type == 'mse':
            return F.mse_loss(logits, labels)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
    except Exception as e:
        logger.error(f"Loss computation failed: {e}")
        # Return a small positive loss to avoid breaking training
        return torch.tensor(1e-6, requires_grad=True, device=logits.device)

def debug_model_structure(model, name="model"):
    """Debug helper to print model structure.
    
    Args:
        model: Model to analyze
        name: Name for logging
    """
    logger.info(f"=== {name} Structure ===")
    logger.info(f"Type: {type(model)}")
    logger.info(f"Has 'model' attr: {hasattr(model, 'model')}")
    logger.info(f"Has 'base_model' attr: {hasattr(model, 'base_model')}")
    logger.info(f"Has 'get_input_embeddings' method: {hasattr(model, 'get_input_embeddings')}")
    
    if hasattr(model, 'model'):
        logger.info(f"model.model type: {type(model.model)}")
        logger.info(f"model.model has 'get_input_embeddings': {hasattr(model.model, 'get_input_embeddings')}")
    
    if hasattr(model, 'base_model'):
        logger.info(f"base_model type: {type(model.base_model)}")
    
    logger.info("=" * 20)
