from typing import Optional, Dict, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhishGuardClassifier(nn.Module):
    """Enhanced LLaMA-based classifier with deep semantic embedding for phishing detection.
    
    This implementation follows the research framework for semantic discrimination
    of phishing content with adversarial training capabilities.
    """
    
    def __init__(self, model_name_or_path: str, num_labels: int = 2, peft_cfg: Optional[Dict] = None):
        super().__init__()
        self.num_labels = num_labels
        self.peft_cfg = peft_cfg or {}
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Try to load LLaMA model, fallback to alternative if needed
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path, 
                num_labels=num_labels,
                torch_dtype=torch.float16 if peft_cfg.get("fp16", False) else torch.float32
            )
            logger.info(f"Successfully loaded model: {model_name_or_path}")
        except Exception as e:
            fallback = peft_cfg.get("fallback_model", "distilbert-base-uncased")
            logger.warning(f"Failed to load {model_name_or_path}: {e}. Using fallback: {fallback}")
            self.model = AutoModelForSequenceClassification.from_pretrained(fallback, num_labels=num_labels)
            
        # Configure PEFT if specified
        if peft_cfg and peft_cfg.get("peft") == "lora":
            if not PEFT_AVAILABLE:
                raise RuntimeError("peft not installed; set peft: null or install PEFT.")
            lcfg = LoraConfig(
                r=peft_cfg.get("lora_r", 16),
                lora_alpha=peft_cfg.get("lora_alpha", 32),
                lora_dropout=peft_cfg.get("lora_dropout", 0.1),
                target_modules=peft_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias="none",
                task_type="SEQ_CLS",
            )
            self.model = get_peft_model(self.model, lcfg)
            logger.info("Applied LoRA configuration for efficient fine-tuning")
            
        # Semantic embedding enhancement layers
        hidden_size = getattr(self.model.config, 'hidden_size', 768)
        self.embedding_dim = peft_cfg.get("embedding_dim", hidden_size)
        
        # Enhanced semantic projection for better phishing pattern recognition
        self.semantic_projector = nn.Sequential(
            nn.Linear(hidden_size, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.LayerNorm(self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Phishing-specific attention mechanism for URL and suspicious pattern detection
        self.attention_weights = nn.Linear(self.embedding_dim // 2, 1)
        
        # Final classification head with enhanced capacity
        self.classifier_head = nn.Sequential(
            nn.Linear(self.embedding_dim // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, num_labels)
        )

    def tokenize(self, texts, max_length: int):
        """Enhanced tokenization with padding and truncation."""
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    
    def get_semantic_embeddings(self, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract deep semantic embeddings using LLaMA backbone.
        
        Returns:
            embeddings: Enhanced semantic representations
            attention_scores: Attention weights for interpretability
        """
        # Get base model outputs (without classification head)
        with torch.cuda.amp.autocast(enabled=self.peft_cfg.get("fp16", False)):
            if hasattr(self.model, 'base_model'):
                # PEFT model
                base_outputs = self.model.base_model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            else:
                # Standard model - get transformer outputs
                if hasattr(self.model, 'transformer'):
                    base_outputs = self.model.transformer(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                elif hasattr(self.model, 'bert'):
                    base_outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                elif hasattr(self.model, 'distilbert'):
                    base_outputs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                else:
                    # Generic approach
                    base_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Use last hidden state for semantic embedding
            last_hidden_state = base_outputs.last_hidden_state if hasattr(base_outputs, 'last_hidden_state') else base_outputs.hidden_states[-1]
            
            # Apply semantic projection
            projected = self.semantic_projector(last_hidden_state)
            
            # Calculate attention weights for important tokens
            attention_scores = torch.softmax(self.attention_weights(projected).squeeze(-1), dim=-1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, 0)
            
            # Weighted pooling based on attention
            embeddings = torch.sum(projected * attention_scores.unsqueeze(-1), dim=1)
            
        return embeddings, attention_scores
    
    def forward(self, input_ids, attention_mask, labels=None, return_embeddings=False):
        """Enhanced forward pass with semantic embedding and classification.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            labels: Ground truth labels (optional)
            return_embeddings: Whether to return semantic embeddings
            
        Returns:
            Dictionary with logits, loss (if labels provided), and optionally embeddings
        """
        # Get enhanced semantic embeddings
        embeddings, attention_scores = self.get_semantic_embeddings(input_ids, attention_mask)
        
        # Classification
        logits = self.classifier_head(embeddings)
        
        outputs = {
            'logits': logits,
            'attention_scores': attention_scores
        }
        
        if return_embeddings:
            outputs['embeddings'] = embeddings
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            outputs['loss'] = loss_fct(logits, labels)
        
        return type('ModelOutput', (), outputs)()
    
    def extract_phishing_features(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        """Extract phishing-specific features for analysis.
        
        Returns:
            Dictionary containing various feature representations useful for 
            understanding phishing patterns and model decisions.
        """
        embeddings, attention_scores = self.get_semantic_embeddings(input_ids, attention_mask)
        logits = self.classifier_head(embeddings)
        
        # Calculate phishing probability
        probs = F.softmax(logits, dim=-1)
        phishing_prob = probs[:, 1] if self.num_labels > 1 else probs[:, 0]
        
        return {
            'embeddings': embeddings,
            'attention_scores': attention_scores,
            'logits': logits,
            'phishing_probability': phishing_prob,
            'risk_score': phishing_prob  # Alias for propagation analysis
        }

# Maintain backward compatibility
TextClassifier = PhishGuardClassifier
